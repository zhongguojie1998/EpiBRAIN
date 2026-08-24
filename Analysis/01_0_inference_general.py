#!/usr/bin/env python3
"""
General-purpose inference: any region -> centered 524 kb context -> tracks and/or embeddings.

Merges the two existing entry points:
  * Analysis/01_0_quick_inference_bigwig.py  (region -> one bigwig per track)
  * Analysis/18_chrom_embeddings.py          (BED    -> shared_cell_embs HDF5)

Input (mutually exclusive):
    --region chr1:100000-100500     single interval
    --bed    peaks.bed              chrom, start, end [, celltype]

Output (--output_type):
    tracks      regression-head predictions, untransformed to original scale
    embedding   shared_cell_embs from prediction_head.shared_cell_encoder
    both        both, from a SINGLE forward pass per interval

Track format (--track_format, default auto):
    bigwig  one .bw per track over the full cropped window  (auto for --region)
    h5      /predictions/{row} = [L_bins, n_tracks] restricted to the bins
            overlapping the interval                        (auto for --bed)

Model (--exp_name / --chk, or an explicit --checkpoint + --config):
    default      full_finetune_original_loss_celltype_head_dim8_linear_full_atlas, epoch 17
    alternative  full_finetune_original_loss_celltype_head_dim8_linear, epoch 20

Every interval is centered in a data.context_length (524,288 bp) window. By
default the forward and reverse-complement passes are averaged (--strand both);
--strand forward / --strand reverse runs a single pass on that strand only, for
both tracks and embeddings. Reverse-strand output is always flipped back into
forward-strand bin order.

Usage:
    # all tracks of one region as bigwigs (pyGenomeTracks / IGV), default model
    python Analysis/01_0_inference_general.py \
        --region chr1:100000-100500 --output_type tracks -o Res/infer_region

    # embeddings for every peak in a BED, all cell types, sharded over 4 GPUs
    python Analysis/01_0_inference_general.py \
        --bed peaks.bed --output_type embedding \
        -o Res/infer_peaks --rank 0 --world_size 4

    # tracks + embeddings for a BED, per-bin HDF5, basal-ganglia model instead
    python Analysis/01_0_inference_general.py \
        --bed peaks.bed --output_type both -o Res/infer_peaks \
        --exp_name full_finetune_original_loss_celltype_head_dim8_linear --chk 20

    # forward strand only (half the GPU time, no RC averaging)
    python Analysis/01_0_inference_general.py \
        --region chr1:100000-100500 --output_type both --strand forward \
        -o Res/infer_fwd
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.tokenizer import FastaInterval, one_hot_reverse_complement  # noqa: E402
from model.model_utils import setup_model  # noqa: E402
from utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHROM_SIZE = 300_000_000

STRAND_BOTH = "both"
STRAND_FORWARD = "forward"
STRAND_REVERSE = "reverse"
STRAND_CHOICES = (STRAND_BOTH, STRAND_FORWARD, STRAND_REVERSE)

# Default model: full atlas fine-tune. Alternative (basal ganglia only):
#   --exp_name full_finetune_original_loss_celltype_head_dim8_linear --chk 20
DEFAULT_EXP_NAME = "full_finetune_original_loss_celltype_head_dim8_linear_full_atlas"
DEFAULT_CHK = "17"
ALT_EXP_NAME = "full_finetune_original_loss_celltype_head_dim8_linear"
ALT_CHK = "20"


# ---------------------------------------------------------------------------
# Interval handling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    """A genomic interval (0-based, half-open) with an optional cell type."""

    chrom: str
    start: int
    end: int
    celltype: str | None = None

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    @property
    def name(self) -> str:
        return f"{self.chrom}_{self.start}_{self.end}"


@dataclass(frozen=True)
class WindowResult:
    """Model outputs for one centered context window."""

    window_start: int
    window_end: int
    pred: np.ndarray | None  # [n_window, n_tracks], label-name order, untransformed
    emb: np.ndarray | None  # [n_window, C, H]


def parse_region(region_str: str) -> Interval:
    """Parse 'chr1:100000-100500' into an Interval."""
    try:
        chrom, pos = region_str.split(":")
        start, end = (int(v.replace(",", "")) for v in pos.split("-"))
    except ValueError as exc:
        raise ValueError(f"Invalid region '{region_str}'. Expected chr1:100000-100500") from exc
    if end <= start:
        raise ValueError(f"Invalid region '{region_str}': end must be > start")
    return Interval(chrom, start, end)


def load_bed(bed_path: Path, known_celltypes: set[str]) -> tuple[list[Interval], bool]:
    """
    Read a BED file into Intervals.

    Returns:
        intervals: one per row, in file order
        has_celltype: True when the 4th column holds recognized cell types
    """
    bed = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if bed.shape[1] < 3:
        raise ValueError(f"{bed_path} needs at least 3 columns (chrom, start, end)")

    has_celltype = False
    if bed.shape[1] >= 4 and known_celltypes:
        sample = bed.iloc[: min(20, len(bed)), 3].astype(str)
        has_celltype = bool(sample.isin(known_celltypes).any())

    intervals = [
        Interval(
            chrom=str(row.iloc[0]),
            start=int(row.iloc[1]),
            end=int(row.iloc[2]),
            celltype=str(row.iloc[3]) if has_celltype else None,
        )
        for _, row in bed.iterrows()
    ]
    return intervals, has_celltype


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------
class GeneralInference:
    """
    Runs one forward pass per interval and exposes both the regression-head
    predictions and the shared cell-type embeddings from that same pass.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda",
        want_tracks: bool = True,
        want_embedding: bool = False,
        untransform: bool = True,
    ) -> None:
        self.device = device
        self.want_tracks = want_tracks
        self.want_embedding = want_embedding
        self.untransform = untransform

        self.cfg = self._load_config(config_path)
        if hasattr(self.cfg.model, "use_compile"):
            self.cfg.model.use_compile = False

        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = setup_model(self.cfg, logger, checkpoint=checkpoint)
        del checkpoint
        self.model.to(device)
        self.model.eval()

        self.seq_len: int = int(self.cfg.data.context_length)
        self.window_size: int = int(self.cfg.data.preprocess.window_size)
        self.n_window: int = int(self.cfg.model.crop_param.bins_to_return)
        self.celltype_num: int = int(self.cfg.model.output_heads.regression.celltype_num)
        self.celltype_hidden_dim: int = int(self.cfg.model.output_heads.celltype_hidden_dim)

        fasta_path = os.path.abspath(self.cfg.data.refer_genom)
        logger.info(f"Loading FASTA from {fasta_path}")
        self.fasta = FastaInterval(fasta_file=fasta_path, context_length=self.seq_len)
        self.chrom_sizes = self._load_chrom_sizes(fasta_path)

        self._load_label_meta()
        self.rc_orig_index, self.rc_swap_index = self._build_rc_swap_index()

        self._emb_holder: dict[str, torch.Tensor] = {}
        if self.want_embedding:
            self._register_embedding_hook()

        logger.info(
            f"Ready. seq_len={self.seq_len}, window_size={self.window_size}, "
            f"n_window={self.n_window}, n_tracks={len(self.label_names)}, "
            f"C={self.celltype_num}, H={self.celltype_hidden_dim}"
        )

    # -- setup ------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> DictConfig:
        path = Path(config_path)
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            logger.info(f"Loading saved config from: {path}")
            with open(path, "r") as fh:
                return OmegaConf.create(yaml.safe_load(fh))
        logger.info(f"Loading config using Hydra from: {path}")
        return load_config(path)

    @staticmethod
    def _load_chrom_sizes(fasta_path: str) -> dict[str, int]:
        """Chromosome sizes from the FASTA .fai index (needed for bigwig headers)."""
        fai = Path(f"{fasta_path}.fai")
        if not fai.exists():
            logger.warning(f"{fai} not found; bigwig headers fall back to {DEFAULT_CHROM_SIZE}")
            return {}
        sizes: dict[str, int] = {}
        with open(fai, "r") as fh:
            for line in fh:
                parts = line.split("\t")
                if len(parts) >= 2:
                    sizes[parts[0]] = int(parts[1])
        return sizes

    def _load_label_meta(self) -> None:
        """Track names, output-dim mapping and cell-type order from regression_label_meta.csv."""
        meta_path = Path(self.cfg.logging.log_dir) / "regression_label_meta.csv"
        if not meta_path.exists():
            if self.untransform or self.want_embedding:
                raise FileNotFoundError(
                    f"regression_label_meta.csv not found at {meta_path}; "
                    "required for untransform and for embedding cell-type names "
                    "(pass --no_untransform for raw track values)"
                )
            logger.warning(f"{meta_path} not found; using generic track names")
            n_tracks = int(self.cfg.model.output_heads.regression.track_num)
            self.label_meta = None
            self.label_names = [f"track_{i}" for i in range(n_tracks)]
            self.dim_index = np.arange(n_tracks, dtype=np.int64)
            self.celltypes_ordered = []
            self.celltype_to_idx = {}
            return

        logger.info(f"Loading label metadata from {meta_path}")
        meta = pd.read_csv(meta_path)
        self.label_meta = meta
        self.label_names = meta["trial"].tolist()
        self.dim_index = meta["dim"].to_numpy(dtype=np.int64)
        self.celltypes_ordered = meta.sort_values("dim")["cell_type"].drop_duplicates().tolist()
        self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes_ordered)}
        logger.info(
            f"{len(self.label_names)} tracks, {len(self.celltypes_ordered)} cell types"
        )

    def _build_rc_swap_index(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """RNAplus/RNAminus track pairing, applied after flipping the RC prediction."""
        meta = self.label_meta
        if meta is None or "modality" not in meta.columns:
            return None, None
        if not {"RNAplus", "RNAminus"}.issubset(set(meta["modality"].values)):
            return None, None

        by_ct = "cell_type" in meta.columns
        swap: list[int] = []
        for _, row in meta.iterrows():
            partner = {"RNAplus": "RNAminus", "RNAminus": "RNAplus"}.get(row["modality"])
            if partner is None:
                swap.append(int(row["dim"]))
                continue
            match = meta[meta["modality"] == partner]
            if by_ct:
                match = match[match["cell_type"] == row["cell_type"]]
            swap.append(int(match.iloc[0]["dim"]) if len(match) else int(row["dim"]))

        logger.info("Built reverse complement swap index for RNAplus/RNAminus tracks")
        return meta["dim"].to_numpy(dtype=np.int64), np.array(swap, dtype=np.int64)

    def _register_embedding_hook(self) -> None:
        """Capture prediction_head.shared_cell_encoder output: [B, L, C*H]."""
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            pred_head = self.model.base_model.model.prediction_head
        else:
            pred_head = self.model.prediction_head

        if pred_head.shared_cell_encoder is None:
            raise RuntimeError("Model does not use shared_cell_encoder (use_cell_encoder=False)")

        def _hook(_module, _inp, out: torch.Tensor) -> None:
            self._emb_holder["value"] = out

        pred_head.shared_cell_encoder.register_forward_hook(_hook)
        logger.info("Registered forward hook on shared_cell_encoder")

    # -- untransform ------------------------------------------------------
    def _untransform_track(self, values: np.ndarray, track_name: str) -> np.ndarray:
        """Undo scale / soft-clip / power transform for a single track."""
        if self.label_meta is None:
            return values
        rows = self.label_meta[self.label_meta["trial"] == track_name]
        if len(rows) == 0:
            logger.warning(f"Track {track_name} missing from label metadata; left raw")
            return values

        row = rows.iloc[0]
        scale = row.get("scale", 1.0)
        clip_soft = row.get("clip_soft", 48.0)
        sum_stat = row.get("sum_stat", "sum_three_quarter")

        out = values.copy()
        if scale != 1.0:
            out = out / scale
        if clip_soft is not None and not pd.isna(clip_soft):
            mask = out > clip_soft
            out[mask] = (clip_soft - 1) + (out[mask] - (clip_soft - 1)) ** 2
        if sum_stat == "sum_three_quarter":
            out = out ** (4.0 / 3.0)
        elif sum_stat in ("sum_sqrt", "mean_sqrt", "avg_sqrt"):
            out = (out + 1) ** 2 - 1
        elif sum_stat not in ("sum", "mean", "avg"):
            logger.warning(f"Unknown sum_stat '{sum_stat}' for {track_name}; no power transform")
        return out

    def _untransform_all(self, pred_dim: np.ndarray) -> np.ndarray:
        """
        Untransform in model-output (dim) space, then reorder columns so that
        column j corresponds to self.label_names[j].
        """
        out = pred_dim.copy()
        if self.untransform and self.label_meta is not None:
            for name, dim in zip(self.label_names, self.dim_index):
                out[:, dim] = self._untransform_track(out[:, dim], name)
                # Parity with 01_0_quick_inference_bigwig.py: BasalGanglia ATAC is x100
                if name.startswith("BasalGanglia-") and "ATAC" in name:
                    out[:, dim] = out[:, dim] * 100
        return out[:, self.dim_index]

    # -- inference --------------------------------------------------------
    def predict(self, interval: Interval, strand: str = STRAND_BOTH) -> WindowResult:
        """
        Center `interval` in a seq_len window and run the model.

        strand:
            'both'    average of the forward and reverse-complement passes (default)
            'forward' forward strand only, one pass
            'reverse' reverse-complement strand only, one pass

        Reverse-strand outputs are always flipped back into forward-strand bin
        order (and RNAplus/RNAminus tracks swapped), so every mode returns
        arrays on the same coordinate system.
        """
        if strand not in STRAND_CHOICES:
            raise ValueError(f"Unknown strand '{strand}'. Expected one of {STRAND_CHOICES}")

        token_dict = self.fasta(
            chr_name=interval.chrom,
            start=interval.center - self.seq_len // 2,
            end=interval.center + self.seq_len // 2,
            return_augs=False,
            return_rela_idx=True,
        )
        seq_onehot = token_dict["one_hot"]  # [L, 4]
        real_start, real_end = token_dict["real_region"]

        with torch.no_grad():
            passes: list[tuple[np.ndarray | None, np.ndarray | None]] = []
            if strand in (STRAND_BOTH, STRAND_FORWARD):
                passes.append(self._run_strand(seq_onehot, reverse=False))
            if strand in (STRAND_BOTH, STRAND_REVERSE):
                passes.append(self._run_strand(seq_onehot, reverse=True))

        pred = self._mean([p for p, _ in passes])
        emb = self._mean([e for _, e in passes])
        if pred is not None:
            pred = self._untransform_all(pred)
        return WindowResult(window_start=real_start, window_end=real_end, pred=pred, emb=emb)

    def _run_strand(
        self, seq_onehot: torch.Tensor, reverse: bool
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        One forward pass through the model.

        Returns (pred [n_window, n_tracks] in dim order, emb [n_window, C, H]),
        each None when not requested. Reverse-complement outputs are flipped
        back to forward-strand bin order before returning.
        """
        seq = one_hot_reverse_complement(seq_onehot) if reverse else seq_onehot
        seq = seq.unsqueeze(0).permute(0, 2, 1).to(self.device)
        out = self.model(seq, "regression")

        pred: np.ndarray | None = None
        if self.want_tracks:
            pred = out.detach().cpu().numpy().squeeze(0)
            if reverse:
                pred = np.flip(pred, axis=0).copy()
                if self.rc_swap_index is not None:
                    pred[:, self.rc_orig_index] = pred[:, self.rc_swap_index]

        emb: np.ndarray | None = None
        if self.want_embedding:
            emb = self._read_emb_hook()
            if reverse:
                emb = np.flip(emb, axis=0).copy()

        return pred, emb

    @staticmethod
    def _mean(arrays: list[np.ndarray | None]) -> np.ndarray | None:
        """Element-wise mean of the requested strand passes; None when unrequested."""
        present = [a for a in arrays if a is not None]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return sum(present) / len(present)

    def _read_emb_hook(self) -> np.ndarray:
        """Reshape the captured [1, n_window, C*H] activation to [n_window, C, H]."""
        raw = self._emb_holder["value"]
        emb = raw.view(1, self.n_window, self.celltype_num, self.celltype_hidden_dim)
        return emb[0].detach().cpu().float().numpy()

    def overlap_bins(self, interval: Interval, window_start: int) -> tuple[int, int]:
        """Half-open bin range of `interval` inside the cropped window."""
        bin_start = max(0, (interval.start - window_start) // self.window_size)
        bin_end = (interval.end - window_start + self.window_size - 1) // self.window_size
        bin_end = min(self.n_window, bin_end)
        if bin_start >= bin_end:
            center_bin = (interval.center - window_start) // self.window_size
            center_bin = max(0, min(self.n_window - 1, center_bin))
            return center_bin, center_bin + 1
        return bin_start, bin_end


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
class IntervalWriter(ABC):
    """Consumes one interval's WindowResult at a time."""

    @abstractmethod
    def add(self, row_idx: int, interval: Interval, result: WindowResult) -> None: ...

    @abstractmethod
    def add_failed(self, row_idx: int, interval: Interval) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class BigwigTrackWriter(IntervalWriter):
    """One .bw per track over the full cropped window, per interval."""

    def __init__(self, engine: GeneralInference, out_dir: Path, per_interval_subdir: bool) -> None:
        self.engine = engine
        self.out_dir = out_dir
        self.per_interval_subdir = per_interval_subdir

    def add(self, row_idx: int, interval: Interval, result: WindowResult) -> None:
        assert result.pred is not None
        base = self.out_dir / interval.name if self.per_interval_subdir else self.out_dir
        out_path = base / "pred"
        out_path.mkdir(parents=True, exist_ok=True)

        eng = self.engine
        n_bins = result.pred.shape[0]
        chrom_size = eng.chrom_sizes.get(interval.chrom, DEFAULT_CHROM_SIZE)
        starts = [result.window_start + i * eng.window_size for i in range(n_bins)]
        ends = [s + eng.window_size for s in starts]
        chroms = [interval.chrom] * n_bins

        def _write(track_idx: int, track_name: str) -> None:
            bw = pyBigWig.open(str(out_path / f"{track_name}.bw"), "w")
            try:
                bw.addHeader([(interval.chrom, chrom_size)])
                bw.addEntries(
                    chroms, starts, ends=ends,
                    values=[float(v) for v in result.pred[:, track_idx]],
                )
            finally:
                bw.close()

        max_workers = min(len(eng.label_names), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_write, i, name): name
                for i, name in enumerate(eng.label_names)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"bigwig {interval.name}"):
                fut.result()
        logger.info(f"Wrote {len(eng.label_names)} bigwigs to {out_path}")

    def add_failed(self, row_idx: int, interval: Interval) -> None:
        logger.warning(f"No bigwigs written for failed interval {interval.name}")

    def close(self) -> None:
        return


class _H5RowStore:
    """Shared machinery for the row-indexed HDF5 outputs."""

    def __init__(self, path: Path, group_name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.hf = h5py.File(path, "w")
        self.grp = self.hf.create_group(group_name)
        self.meta: dict[str, list] = {
            "chrom": [], "start": [], "end": [], "bin_start": [], "bin_end": []
        }

    def put(self, row_idx: int, data: np.ndarray, interval: Interval, bins: tuple[int, int]) -> None:
        self.grp.create_dataset(str(row_idx), data=data.astype(np.float32), compression="gzip")
        self.meta["chrom"].append(interval.chrom)
        self.meta["start"].append(interval.start)
        self.meta["end"].append(interval.end)
        self.meta["bin_start"].append(bins[0])
        self.meta["bin_end"].append(bins[1])

    def finalize(self, attrs: dict[str, object], extra_meta: dict[str, list] | None = None) -> None:
        dt_str = h5py.special_dtype(vlen=str)
        mg = self.hf.create_group("metadata")
        mg.create_dataset("chrom", data=np.array(self.meta["chrom"], dtype=object), dtype=dt_str)
        for key in ("start", "end", "bin_start", "bin_end"):
            mg.create_dataset(key, data=np.array(self.meta[key], dtype=np.int32))
        for key, values in (extra_meta or {}).items():
            if values and isinstance(values[0], str):
                mg.create_dataset(key, data=np.array(values, dtype=object), dtype=dt_str)
            else:
                mg.create_dataset(key, data=np.array(values, dtype=np.int32))
        for key, value in attrs.items():
            self.hf.attrs[key] = value
        self.hf.close()
        logger.info(f"Saved {len(self.meta['chrom'])} rows to {self.path}")


class TrackH5Writer(IntervalWriter):
    """/predictions/{row_idx} = [L_bins, n_tracks] over the interval's bins only."""

    def __init__(self, engine: GeneralInference, path: Path, attrs: dict[str, object]) -> None:
        self.engine = engine
        self.attrs = attrs
        self.store = _H5RowStore(path, "predictions")

    def add(self, row_idx: int, interval: Interval, result: WindowResult) -> None:
        assert result.pred is not None
        b0, b1 = self.engine.overlap_bins(interval, result.window_start)
        self.store.put(row_idx, result.pred[b0:b1], interval, (b0, b1))

    def add_failed(self, row_idx: int, interval: Interval) -> None:
        zeros = np.zeros((1, len(self.engine.label_names)), dtype=np.float32)
        self.store.put(row_idx, zeros, interval, (0, 1))

    def close(self) -> None:
        attrs = dict(self.attrs)
        attrs["track_list"] = json.dumps(self.engine.label_names)
        attrs["n_tracks"] = len(self.engine.label_names)
        attrs["window_size"] = self.engine.window_size
        attrs["untransformed"] = self.engine.untransform
        self.store.finalize(attrs)


class EmbeddingH5Writer(IntervalWriter):
    """
    /embeddings/{row_idx} = [L_bins, H] over the interval's bins only.

    celltype is None  -> one file per cell type (all-cell-types mode)
    celltype is a str -> a single file for that cell type
    """

    def __init__(
        self,
        engine: GeneralInference,
        out_dir: Path,
        stem: str,
        fixed_celltype: str | None,
        per_row_celltype: bool,
        attrs: dict[str, object],
    ) -> None:
        self.engine = engine
        self.attrs = attrs
        self.per_row_celltype = per_row_celltype
        self.fixed_celltype = fixed_celltype

        if per_row_celltype or fixed_celltype is not None:
            self.celltypes = [fixed_celltype] if fixed_celltype is not None else [None]
            self.stores = [_H5RowStore(out_dir / f"{stem}.h5", "embeddings")]
            self.row_celltype: dict[str, list] = {"celltype": [], "celltype_idx": []}
        else:
            self.celltypes = list(engine.celltypes_ordered)
            self.stores = [
                _H5RowStore(out_dir / f"{stem}_{_sanitize(ct)}.h5", "embeddings")
                for ct in self.celltypes
            ]
            self.row_celltype = {}

    def add(self, row_idx: int, interval: Interval, result: WindowResult) -> None:
        assert result.emb is not None
        b0, b1 = self.engine.overlap_bins(interval, result.window_start)
        emb = result.emb[b0:b1]  # [L_bins, C, H]

        if self.row_celltype != {} or self.fixed_celltype is not None:
            celltype = interval.celltype if self.per_row_celltype else self.fixed_celltype
            idx = self.engine.celltype_to_idx.get(str(celltype), -1)
            if idx < 0:
                raise KeyError(
                    f"Cell type '{celltype}' not in label metadata. "
                    f"Available: {self.engine.celltypes_ordered[:5]} ..."
                )
            self.stores[0].put(row_idx, emb[:, idx, :], interval, (b0, b1))
            self.row_celltype["celltype"].append(str(celltype))
            self.row_celltype["celltype_idx"].append(idx)
        else:
            for ci, store in enumerate(self.stores):
                store.put(row_idx, emb[:, ci, :], interval, (b0, b1))

    def add_failed(self, row_idx: int, interval: Interval) -> None:
        zeros = np.zeros((1, self.engine.celltype_hidden_dim), dtype=np.float32)
        for store in self.stores:
            store.put(row_idx, zeros, interval, (0, 1))
        if self.row_celltype:
            self.row_celltype["celltype"].append(str(interval.celltype))
            self.row_celltype["celltype_idx"].append(-1)

    def close(self) -> None:
        eng = self.engine
        for ci, store in enumerate(self.stores):
            attrs = dict(self.attrs)
            attrs.update(
                H=eng.celltype_hidden_dim,
                C=eng.celltype_num,
                window_size=eng.window_size,
                celltype_list=json.dumps(eng.celltypes_ordered),
            )
            if not self.row_celltype and self.celltypes[ci] is not None:
                attrs["celltype"] = self.celltypes[ci]
                attrs["celltype_idx"] = eng.celltype_to_idx[self.celltypes[ci]]
            store.finalize(attrs, extra_meta=self.row_celltype or None)


def _sanitize(name: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    for ch in ' /\\:*?"<>|':
        name = name.replace(ch, "_")
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_model_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Checkpoint/config from --checkpoint+--config or --exp_name+--chk."""
    if args.checkpoint:
        config_path = args.config or str(ROOT / "Model" / "config.yaml")
        return args.checkpoint, config_path

    if not (args.exp_name and args.chk):
        raise SystemExit("Provide either --checkpoint (+--config) or --exp_name and --chk")
    log_dir = Path(args.log_base) / args.exp_name
    chk_dir = Path(args.chk_base) / args.exp_name
    checkpoint = chk_dir / f"chk_epoch_{args.chk}.pt"
    config = args.config or str(log_dir / "overall_setting.yaml")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint), config


def shard(intervals: Sequence[Interval], rank: int, world_size: int) -> list[tuple[int, Interval]]:
    """Row-index-preserving stride sharding, matching 18_chrom_embeddings.py."""
    rows = list(enumerate(intervals))
    return rows[rank::world_size] if world_size > 1 else rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Centered-window inference: tracks and/or embeddings for regions or a BED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--region", type=str, help="Single region, e.g. chr1:100000-100500")
    src.add_argument("--bed", type=str, help="BED file: chrom, start, end [, celltype]")

    parser.add_argument("--output_type", choices=("tracks", "embedding", "both"), default="tracks")
    parser.add_argument(
        "--track_format", choices=("auto", "bigwig", "h5"), default="auto",
        help="auto: bigwig for --region, h5 for --bed",
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--celltype", type=str, default=None,
        help="Restrict embeddings to one cell type (default: all cell types, one file each)",
    )

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--exp_name", "-e", type=str, default=DEFAULT_EXP_NAME,
        help=f"Experiment name (default: {DEFAULT_EXP_NAME}; "
             f"alternative: {ALT_EXP_NAME} with --chk {ALT_CHK})",
    )
    parser.add_argument(
        "--chk", type=str, default=DEFAULT_CHK,
        help=f"Checkpoint epoch (default: {DEFAULT_CHK}; use {ALT_CHK} with {ALT_EXP_NAME})",
    )
    parser.add_argument("--log_base", type=str, default="./logs")
    parser.add_argument("--chk_base", type=str, default="./Chk")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--strand", choices=STRAND_CHOICES, default=STRAND_BOTH,
        help="both: average the forward and reverse-complement passes (default); "
             "forward / reverse: a single pass on that strand only. Applies to "
             "tracks and embeddings alike.",
    )
    parser.add_argument(
        "--no_untransform", action="store_true",
        help="Keep track values on the model's transformed scale",
    )
    parser.add_argument("--rank", type=int, default=0, help="0-based shard index")
    parser.add_argument("--world_size", type=int, default=1, help="Number of shards")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path, config_path = resolve_model_paths(args)

    want_tracks = args.output_type in ("tracks", "both")
    want_embedding = args.output_type in ("embedding", "both")
    track_format = args.track_format
    if track_format == "auto":
        track_format = "bigwig" if args.region else "h5"

    engine = GeneralInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device,
        want_tracks=want_tracks,
        want_embedding=want_embedding,
        untransform=not args.no_untransform,
    )

    # -- inputs ---------------------------------------------------------
    per_row_celltype = False
    if args.region:
        intervals: list[Interval] = [parse_region(args.region)]
    else:
        intervals, per_row_celltype = load_bed(Path(args.bed), set(engine.celltype_to_idx))
        logger.info(
            f"Loaded {len(intervals)} intervals from {args.bed}"
            + (" (4th column read as cell type)" if per_row_celltype else "")
        )
    if args.celltype is not None:
        per_row_celltype = False
    rows = shard(intervals, args.rank, args.world_size)
    if args.world_size > 1:
        logger.info(f"Rank {args.rank}/{args.world_size}: {len(rows)} of {len(intervals)} intervals")

    if want_tracks and track_format == "bigwig" and len(rows) > 5:
        logger.warning(
            f"--track_format bigwig with {len(rows)} intervals writes "
            f"{len(rows) * len(engine.label_names)} files; --track_format h5 is usually what you want"
        )

    # -- writers --------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    common_attrs: dict[str, object] = {
        "strand": args.strand,
        "n_intervals": len(rows),
        "total_n_intervals": len(intervals),
        "rank": args.rank,
        "world_size": args.world_size,
        "checkpoint": checkpoint_path,
        "context_length": engine.seq_len,
    }

    writers: list[IntervalWriter] = []
    if want_tracks:
        if track_format == "bigwig":
            writers.append(BigwigTrackWriter(engine, out_dir, per_interval_subdir=len(rows) > 1))
        else:
            writers.append(TrackH5Writer(engine, out_dir / f"tracks{suffix}.h5", common_attrs))
    if want_embedding:
        writers.append(
            EmbeddingH5Writer(
                engine=engine,
                out_dir=out_dir,
                stem=f"embeddings{suffix}",
                fixed_celltype=args.celltype,
                per_row_celltype=per_row_celltype,
                attrs=common_attrs,
            )
        )

    # -- run ------------------------------------------------------------
    try:
        for row_idx, interval in tqdm(rows, total=len(rows), desc="Inference"):
            try:
                result = engine.predict(interval, strand=args.strand)
                for writer in writers:
                    writer.add(row_idx, interval, result)
            except Exception as exc:  # keep row indices aligned with the input BED
                logger.warning(f"Row {row_idx} ({interval.name}) failed: {exc}")
                for writer in writers:
                    writer.add_failed(row_idx, interval)
    finally:
        for writer in writers:
            writer.close()

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
