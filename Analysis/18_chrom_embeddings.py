#!/usr/bin/env python3
"""
Extract shared_cell_embs for cCREs from the Borzoi model.

For each cCRE (chr, start, end, celltype) in a BED file, runs model inference
centered on the cCRE and extracts the shared cell type embedding for overlapping
bins. Stores results in HDF5 format.

Usage:
    python Analysis/18_chrom_embeddings.py \
        --bed test_ccres.bed \
        --output test_out.h5 \
        --checkpoint logs/basel_ganglia_complete_lora_16_v1/checkpoint.pt \
        --config logs/basel_ganglia_complete_lora_16_v1/overall_setting.yaml \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.tokenizer import FastaInterval, one_hot_reverse_complement
from model.model_utils import setup_model
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts shared_cell_embs from the model for specified cCREs."""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda:0'):
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load config
        config_path = Path(config_path)
        if config_path.exists() and config_path.is_file() and config_path.suffix in ['.yaml', '.yml']:
            logger.info(f"Loading saved config from: {config_path}")
            with open(config_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            self.cfg = OmegaConf.create(cfg_dict)
        else:
            logger.info(f"Loading config using Hydra from: {config_path}")
            self.cfg = load_config(config_path)

        # Disable compilation for inference
        if hasattr(self.cfg.model, 'use_compile'):
            self.cfg.model.use_compile = False

        # Build model skeleton (no checkpoint loaded yet)
        self.model = setup_model(self.cfg, logger)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(device)
        self.model.eval()

        # Model geometry
        self.seq_len = self.cfg.data.context_length
        self.window_size = self.cfg.data.preprocess.window_size
        self.n_window = self.cfg.model.crop_param.bins_to_return

        # Derive C and H from model config
        self.celltype_num = self.cfg.model.output_heads.regression.celltype_num
        self.celltype_hidden_dim = self.cfg.model.celltype_hidden_dim

        # Setup FASTA
        fasta_path = os.path.abspath(self.cfg.data.refer_genom)
        logger.info(f"Loading FASTA from {fasta_path}")
        self.fasta = FastaInterval(fasta_file=fasta_path, context_length=self.seq_len)

        # Load label metadata and build celltype → C-index mapping
        label_meta_path = Path(self.cfg.logging.log_dir) / "regression_label_meta.csv"
        if not label_meta_path.exists():
            raise FileNotFoundError(f"regression_label_meta.csv not found at {label_meta_path}")
        logger.info(f"Loading label metadata from {label_meta_path}")
        label_meta = pd.read_csv(label_meta_path)
        # Sort by dim to get C-dimension order; drop_duplicates keeps first occurrence per celltype
        celltypes_ordered = (
            label_meta.sort_values('dim')['cell_type']
            .drop_duplicates()
            .tolist()
        )
        self.celltypes_ordered = celltypes_ordered
        self.celltype_to_idx: dict[str, int] = {ct: i for i, ct in enumerate(celltypes_ordered)}
        logger.info(f"Loaded {len(celltypes_ordered)} cell types, hidden_dim={self.celltype_hidden_dim}")

        # Register forward hook on shared_cell_encoder
        self._cell_embs_holder: dict = {}
        self._register_hook()

        logger.info(
            f"EmbeddingExtractor ready. "
            f"seq_len={self.seq_len}, window_size={self.window_size}, "
            f"n_window={self.n_window}, C={self.celltype_num}, H={self.celltype_hidden_dim}"
        )

    def _register_hook(self) -> None:
        """Register forward hook on prediction_head.shared_cell_encoder."""
        # Navigate through potential PEFT wrapper
        if hasattr(self.model, 'base_model'):
            pred_head = self.model.base_model.model.prediction_head
        else:
            pred_head = self.model.prediction_head

        if pred_head.shared_cell_encoder is None:
            raise RuntimeError("Model does not use shared_cell_encoder (use_cell_encoder=False)")

        def _hook(module, inp, out):
            # out: [B, L, C*H] (after linear + activation, before reshape)
            self._cell_embs_holder['value'] = out

        pred_head.shared_cell_encoder.register_forward_hook(_hook)
        logger.info("Registered forward hook on shared_cell_encoder")

    def _get_celltype_idx(self, celltype: str) -> int:
        if celltype not in self.celltype_to_idx:
            raise KeyError(
                f"Cell type '{celltype}' not found in label metadata. "
                f"Available: {list(self.celltype_to_idx.keys())[:5]} ..."
            )
        return self.celltype_to_idx[celltype]

    def get_ccre_embedding(
        self,
        chrom: str,
        start: int,
        end: int,
        celltype: str,
        use_rev_aug: bool = True,
    ) -> tuple[np.ndarray, int, int]:
        """
        Extract shared_cell_embs for a cCRE.

        Args:
            chrom: Chromosome name
            start: cCRE start (0-based)
            end: cCRE end (exclusive)
            celltype: Cell type name matching label_meta cell_type column
            use_rev_aug: Average forward and RC embeddings

        Returns:
            emb: float32 array [L_bins, H]
            bin_start: inclusive bin index
            bin_end: exclusive bin index
        """
        celltype_idx = self._get_celltype_idx(celltype)
        center_pos = (start + end) // 2

        token_dict = self.fasta(
            chr_name=chrom,
            start=center_pos - self.seq_len // 2,
            end=center_pos + self.seq_len // 2,
            return_augs=False,
            return_rela_idx=True,
        )
        seq_onehot = token_dict["one_hot"]   # [L, 4]
        real_start, _ = token_dict["real_region"]

        with torch.no_grad():
            seq_fwd = seq_onehot.unsqueeze(0).permute(0, 2, 1).to(self.device)  # [1, 4, L]
            self.model(seq_fwd, 'regression')
            emb_raw_fwd = self._cell_embs_holder['value']  # [1, n_window, C*H]

            # Reshape to [1, n_window, C, H]
            emb_fwd = emb_raw_fwd.view(1, self.n_window, self.celltype_num, self.celltype_hidden_dim)

            if use_rev_aug:
                seq_rev = one_hot_reverse_complement(seq_onehot).unsqueeze(0).permute(0, 2, 1).to(self.device)
                self.model(seq_rev, 'regression')
                emb_raw_rev = self._cell_embs_holder['value']  # [1, n_window, C*H]
                emb_rev = emb_raw_rev.view(1, self.n_window, self.celltype_num, self.celltype_hidden_dim)
                # Flip L dimension for RC
                emb_rev = torch.flip(emb_rev, dims=[1])
                emb_combined = (emb_fwd + emb_rev) / 2.0
            else:
                emb_combined = emb_fwd

            # [n_window, H] for this celltype
            emb_np = emb_combined[0, :, celltype_idx, :].cpu().float().numpy()

        # Compute overlapping bins
        bin_start = (start - real_start) // self.window_size
        bin_end = (end - real_start + self.window_size - 1) // self.window_size
        bin_start = max(0, bin_start)
        bin_end = min(self.n_window, bin_end)

        if bin_start >= bin_end:
            # cCRE center is outside cropped window — return single bin at center
            center_bin = (center_pos - real_start) // self.window_size
            center_bin = max(0, min(self.n_window - 1, center_bin))
            bin_start = center_bin
            bin_end = center_bin + 1

        return emb_np[bin_start:bin_end], bin_start, bin_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract shared_cell_embs for cCREs")
    parser.add_argument("--bed", required=True, help="Input BED file (chr, start, end, celltype, ...)")
    parser.add_argument("--output", required=True, help="Output .h5 file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--use_rev_aug", type=lambda x: x.lower() != 'false', default=True,
        help="RC augmentation (default: True)"
    )
    args = parser.parse_args()

    # Load BED file (0-based coordinates assumed)
    bed = pd.read_csv(args.bed, sep='\t', header=None)
    if bed.shape[1] < 4:
        raise ValueError("BED file must have at least 4 columns: chr, start, end, celltype")
    bed.columns = list(bed.columns)  # keep numeric column names
    logger.info(f"Loaded {len(bed)} cCREs from {args.bed}")

    extractor = EmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata accumulators
    meta_chrom: list[str] = []
    meta_start: list[int] = []
    meta_end: list[int] = []
    meta_celltype: list[str] = []
    meta_celltype_idx: list[int] = []
    meta_bin_start: list[int] = []
    meta_bin_end: list[int] = []

    with h5py.File(output_path, 'w') as hf:
        emb_grp = hf.create_group("embeddings")
        meta_grp = hf.create_group("metadata")

        for i, row in tqdm(bed.iterrows(), total=len(bed), desc="Extracting embeddings"):
            chrom = str(row.iloc[0])
            start = int(row.iloc[1])
            end = int(row.iloc[2])
            celltype = str(row.iloc[3])

            try:
                emb, bin_s, bin_e = extractor.get_ccre_embedding(
                    chrom, start, end, celltype, use_rev_aug=args.use_rev_aug
                )
                celltype_idx = extractor.celltype_to_idx[celltype]
            except Exception as e:
                logger.warning(f"Skipping cCRE {i} ({chrom}:{start}-{end} {celltype}): {e}")
                # Store zero embedding to keep indices aligned
                emb = np.zeros((1, extractor.celltype_hidden_dim), dtype=np.float32)
                bin_s = 0
                bin_e = 1
                celltype_idx = -1

            emb_grp.create_dataset(str(i), data=emb.astype(np.float32), compression="gzip")

            meta_chrom.append(chrom)
            meta_start.append(start)
            meta_end.append(end)
            meta_celltype.append(celltype)
            meta_celltype_idx.append(celltype_idx)
            meta_bin_start.append(bin_s)
            meta_bin_end.append(bin_e)

        # Write metadata arrays
        dt_str = h5py.special_dtype(vlen=str)
        meta_grp.create_dataset("chrom", data=np.array(meta_chrom, dtype=object), dtype=dt_str)
        meta_grp.create_dataset("start", data=np.array(meta_start, dtype=np.int32))
        meta_grp.create_dataset("end", data=np.array(meta_end, dtype=np.int32))
        meta_grp.create_dataset("celltype", data=np.array(meta_celltype, dtype=object), dtype=dt_str)
        meta_grp.create_dataset("celltype_idx", data=np.array(meta_celltype_idx, dtype=np.int32))
        meta_grp.create_dataset("bin_start", data=np.array(meta_bin_start, dtype=np.int32))
        meta_grp.create_dataset("bin_end", data=np.array(meta_bin_end, dtype=np.int32))

        # Top-level attributes
        hf.attrs['H'] = extractor.celltype_hidden_dim
        hf.attrs['C'] = extractor.celltype_num
        hf.attrs['window_size'] = extractor.window_size
        hf.attrs['celltype_list'] = json.dumps(extractor.celltypes_ordered)
        hf.attrs['use_rev_aug'] = args.use_rev_aug
        hf.attrs['n_ccres'] = len(bed)

    logger.info(f"Saved {len(bed)} cCRE embeddings to {output_path}")


if __name__ == "__main__":
    main()
