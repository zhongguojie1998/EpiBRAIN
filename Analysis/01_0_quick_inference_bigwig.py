#!/usr/bin/env python3
"""
Quick inference and bigwig generation for visualization.

This script performs model inference on genomic regions and generates bigwig files
for visualization with pyGenomeTracks.

Input types:
1. Gene name: --gene GENE_NAME (extracts exons from GTF)
2. Region: --region chr1:1000-2000
3. Variant: --variant chr1:1000:A:T (reference allele and alternative allele)
4. Saturation mutagenesis: --sat_mutagenesis chr1:1000 or chr1:1000-2000 (all possible SNVs)

Output:
- For gene/region: pred/ and label/ folders with bigwig files
- For variant: label/, pred/ (reference), and alt/ (alternative) folders with bigwig files
- For saturation mutagenesis: TSV files with variant effect scores
  - If --gene is also provided: includes gene-specific statistics (diff_gene_sum, diff_gene_mean, gene_length)

Usage examples:
    # Gene-based inference
    python 01_0_quick_inference_bigwig.py --gene GRIN2A --output gene_GRIN2A

    # Region-based inference
    python 01_0_quick_inference_bigwig.py --region chr1:100000-200000 --output region_chr1

    # Variant effect prediction
    python 01_0_quick_inference_bigwig.py --variant chr1:154426970:G:A --output variant_rs1234

    # Saturation mutagenesis
    python 01_0_quick_inference_bigwig.py --sat_mutagenesis chr1:154426970 --output saturation_results

    # Saturation mutagenesis with gene-specific statistics
    python 01_0_quick_inference_bigwig.py --sat_mutagenesis chr12:40208963 --gene LRRK2 --output saturation_LRRK2
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.data_utils import STD_CHR, ModelSeq, annotate_unmap, get_labels
from data.tokenizer import FastaInterval, str_to_one_hot, one_hot_reverse_complement
from model.model_utils import setup_model
from utils.config import load_config

# Import pygene for GTF parsing
try:
    from pygene import GTF
except ImportError:
    sys.path.append(str(ROOT / "Analysis"))
    from pygene import GTF


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegionInference:
    """Handles model inference for genomic regions."""

    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (optional)
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load config
        if config_path is None:
            config_path = ROOT / "Model" / "config.yaml"

        # Check if config_path is a saved YAML file or a Hydra config directory
        config_path = Path(config_path)
        if config_path.exists() and config_path.is_file() and config_path.suffix in ['.yaml', '.yml']:
            # Load saved config file directly
            logger.info(f"Loading saved config from: {config_path}")
            with open(config_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            self.cfg = OmegaConf.create(cfg_dict)
        else:
            # Use Hydra's load_config for config directories
            logger.info(f"Loading config using Hydra from: {config_path}")
            self.cfg = load_config(config_path)

        # Setup model
        logger.info(f"Loading model from {checkpoint_path}")

        # Disable model compilation for inference
        if hasattr(self.cfg.model, 'use_compile'):
            self.cfg.model.use_compile = False

        # Create model
        self.model = setup_model(self.cfg, logger)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(device)
        self.model.eval()

        # Get model parameters from config
        self.seq_len = self.cfg.data.context_length
        self.window_size = self.cfg.data.preprocess.window_size
        self.n_window = self.cfg.model.crop_param.bins_to_return

        # Setup FASTA tokenizer (following 03_0_variant_effect.py)
        fasta_path = os.path.abspath(self.cfg.data.refer_genom)
        logger.info(f"Loading FASTA from {fasta_path}")
        self.fasta = FastaInterval(fasta_file=fasta_path, context_length=self.seq_len)

        # Get label names from config or metadata
        # Try to load from label metadata file first
        self.label_meta = None
        try:
            label_meta_path = Path(self.cfg.logging.log_dir) / "regression_label_meta.csv"
            if label_meta_path.exists():
                logger.info(f"Loading label metadata from {label_meta_path}")
                self.label_meta = pd.read_csv(label_meta_path)
                # Use the 'trial' column which has full track names
                # Format: "BasalGanglia-Astrocyte_ATAC", "BasalGanglia-Astrocyte_K27Ac", etc.
                self.label_names = self.label_meta['trial'].tolist()
            else:
                # Fall back to generating labels from data config
                logger.info("Label metadata not found, generating from data config")
                data_config_path = self.cfg.data.preprocess.trial_summary_path
                data_config = pd.read_csv(data_config_path, index_col=0)
                self.label_names = data_config.index.tolist()
        except Exception as e:
            logger.warning(f"Could not load label names from metadata: {e}")
            # Last resort: create generic labels
            n_tracks = self.cfg.model.output_heads.regression.track_num
            self.label_names = [f"track_{i}" for i in range(n_tracks)]

        # Build reverse complement swap index for RNAplus/RNAminus tracks
        self.reverse_complement_swap_index = self._build_rc_swap_index()

        logger.info(f"Model ready. Sequence length: {self.seq_len}, Window size: {self.window_size}, N windows: {self.n_window}")
        logger.info(f"Number of tracks: {len(self.label_names)}")

    def _build_rc_swap_index(self):
        """
        Build reverse complement swap index for RNAplus/RNAminus tracks.

        Returns:
            numpy array: Swap index where RNAplus and RNAminus are swapped
        """
        if self.label_meta is None:
            return None

        # Check if we have RNAplus and RNAminus tracks
        if 'modality' not in self.label_meta.columns or 'cell_type' not in self.label_meta.columns:
            logger.warning("Missing 'modality' or 'cell_type' columns in label metadata, skipping RC swap index")
            return None

        has_plus = 'RNAplus' in self.label_meta['modality'].values
        has_minus = 'RNAminus' in self.label_meta['modality'].values

        if not (has_plus and has_minus):
            logger.info("No RNAplus/RNAminus tracks found, skipping RC swap index")
            return None

        # Build swap index using 'dim' column (track index in predictions)
        swap_index = []
        for i, row in self.label_meta.iterrows():
            if row['modality'] == 'RNAplus':
                # Find the index of RNAminus for corresponding cell type
                matching = self.label_meta[
                    (self.label_meta['modality'] == 'RNAminus') &
                    (self.label_meta['cell_type'] == row['cell_type'])
                ]
                if len(matching) > 0:
                    # Use the 'dim' column value (track index in pred)
                    swap_index.append(int(matching.iloc[0]['dim']))
                else:
                    swap_index.append(int(row['dim']))
            elif row['modality'] == 'RNAminus':
                # Find the index of RNAplus for corresponding cell type
                matching = self.label_meta[
                    (self.label_meta['modality'] == 'RNAplus') &
                    (self.label_meta['cell_type'] == row['cell_type'])
                ]
                if len(matching) > 0:
                    # Use the 'dim' column value (track index in pred)
                    swap_index.append(int(matching.iloc[0]['dim']))
                else:
                    swap_index.append(int(row['dim']))
            else:
                # Don't change for other modalities
                swap_index.append(int(row['dim']))

        logger.info(f"Built reverse complement swap index for RNAplus/RNAminus tracks")
        return np.array(swap_index)

    def parse_region(self, region_str):
        """
        Parse region string (chr1:1000-2000).

        Returns:
            tuple: (chr, start, end)
        """
        try:
            chr_part, pos_part = region_str.split(':')
            start, end = map(int, pos_part.split('-'))
            return chr_part, start, end
        except:
            raise ValueError(f"Invalid region format: {region_str}. Expected format: chr1:1000-2000")

    def parse_variant(self, variant_str):
        """
        Parse variant string (chr1:1000:A:T).

        Returns:
            tuple: (chr, pos, ref, alt)
        """
        try:
            parts = variant_str.split(':')
            if len(parts) != 4:
                raise ValueError
            chr_part, pos, ref, alt = parts
            pos = int(pos)
            return chr_part, pos, ref.upper(), alt.upper()
        except:
            raise ValueError(f"Invalid variant format: {variant_str}. Expected format: chr1:1000:A:T")

    def parse_sat_region(self, region_str):
        """
        Parse saturation mutagenesis region string.

        Args:
            region_str: Region string in format 'chr1:1000' (single position)
                       or 'chr1:1000-2000' (range)

        Returns:
            tuple: (chr, start, end) where start==end for single position
        """
        try:
            chr_part, pos_part = region_str.split(':')
            if '-' in pos_part:
                # Range format: chr1:1000-2000
                start, end = map(int, pos_part.split('-'))
            else:
                # Single position format: chr1:1000
                pos = int(pos_part)
                start = pos
                end = pos
            return chr_part, start, end
        except:
            raise ValueError(f"Invalid region format: {region_str}. Expected format: chr1:1000 or chr1:1000-2000")

    def get_gene_exons(self, gene_name, gtf_path=None):
        """
        Extract exon regions for a gene from GTF file.

        Args:
            gene_name: Gene name (e.g., "GRIN2A")
            gtf_path: Path to GTF file

        Returns:
            list: List of (chr, start, end) tuples for exons
        """
        if gtf_path is None:
            gtf_path = ROOT / "Data" / "reference" / "gencode.v45.annotation.gtf.gz"

        logger.info(f"Loading GTF from {gtf_path}")
        gtf = GTF(str(gtf_path), trim_dot=False)

        # Find gene by name
        gene_obj = None
        gene_id = None
        for gid, gene in gtf.genes.items():
            if gene.name == gene_name:
                gene_obj = gene
                gene_id = gid
                break

        if gene_obj is None:
            raise ValueError(f"Gene '{gene_name}' not found in GTF file")

        logger.info(f"Found gene: {gene_name} ({gene_id})")
        logger.info(f"  Chromosome: {gene_obj.chrom}")
        logger.info(f"  Strand: {gene_obj.strand}")
        logger.info(f"  Transcripts: {len(gene_obj.transcripts)}")

        # Collect all unique exons across all transcripts
        exon_set = set()
        for tx in gene_obj.transcripts.values():
            for exon in tx.exons:
                exon_set.add((exon.start, exon.end))

        # Sort exons by position
        exons = sorted(list(exon_set))
        logger.info(f"  Total unique exons: {len(exons)}")

        if len(exons) == 0:
            raise ValueError(f"No exons found for gene {gene_name}")

        # Convert to (chr, start, end) tuples
        exons_with_chr = [(gene_obj.chrom, start, end) for start, end in exons]

        return exons_with_chr

    def get_sequence_and_predict(self, chr, center_pos, use_head='regression', use_rev_aug=True):
        """
        Get sequence window and run prediction with optional reverse complement augmentation.

        Args:
            chr: Chromosome
            center_pos: Center position
            use_head: Which prediction head to use
            use_rev_aug: Whether to use reverse complement augmentation (default: True)

        Returns:
            tuple: (predictions, window_start, window_end)
                predictions: numpy array (n_window, n_tracks)
        """
        # Get sequence using FastaInterval (returns dict with 'one_hot', 'real_region')
        # FastaInterval is 0-indexed, so we center on center_pos
        token_dict = self.fasta(
            chr_name=chr,
            start=center_pos - self.seq_len // 2,
            end=center_pos + self.seq_len // 2,
            return_augs=False,
            return_rela_idx=True
        )

        seq_onehot = token_dict["one_hot"]  # Shape: (L, 4)
        real_start, real_end = token_dict["real_region"]

        logger.info(f"Sequence window: {chr}:{real_start}-{real_end}")

        # Run prediction with optional reverse complement augmentation
        with torch.no_grad():
            # Permute from (L, 4) to (1, 4, L) and move to device
            seq_tensor = seq_onehot.unsqueeze(0).permute(0, 2, 1).to(self.device)
            pred_fwd = self.model(seq_tensor, use_head).detach().cpu().numpy().squeeze(0)

            if use_rev_aug:
                # Get reverse complement sequence
                seq_onehot_rev = one_hot_reverse_complement(seq_onehot)
                seq_tensor_rev = seq_onehot_rev.unsqueeze(0).permute(0, 2, 1).to(self.device)
                pred_rev = self.model(seq_tensor_rev, use_head).detach().cpu().numpy().squeeze(0)

                # Flip predictions along sequence dimension (reverse order)
                pred_rev = np.flip(pred_rev, axis=0)

                # Swap RNAplus and RNAminus tracks if needed
                if self.reverse_complement_swap_index is not None:
                    pred_rev = pred_rev[:, self.reverse_complement_swap_index]

                # Average forward and reverse predictions
                pred = (pred_fwd + pred_rev) / 2.0
            else:
                pred = pred_fwd

        return pred, real_start, real_end

    def predict_variant_effect(self, chr, pos, ref, alt, use_head='regression'):
        """
        Predict effect of variant (following 03_0_variant_effect.py).

        Args:
            chr: Chromosome
            pos: Position (1-based, VCF format)
            ref: Reference allele
            alt: Alternative allele
            use_head: Which prediction head to use

        Returns:
            tuple: (ref_pred, alt_pred, window_start, window_end)
        """
        logger.info(f"Predicting variant effect: {chr}:{pos}:{ref}>{alt}")

        # Get reference sequence (pos is 1-based in VCF, FastaInterval is 0-based)
        token_dict = self.fasta(
            chr_name=chr,
            start=pos - 1,  # Convert to 0-based
            end=pos,  # End is exclusive
            return_augs=False,
            return_rela_idx=True
        )

        s_idx, e_idx = token_dict["rela_idx"]
        wt_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]

        # Verify reference allele
        def onehot_to_str(seq_onehot):
            mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "T", (0, 0, 0, 0): "N"}
            seq_str = ""
            for vec in seq_onehot:
                key = tuple((vec > 0.5).int().tolist())
                seq_str += mapping.get(key, "N")
            return seq_str

        wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
        if ref != wt_nt_fetched:
            logger.warning(f"Ref info isn't consistent with genome. {chr}:{pos}, given {ref}, fetched {wt_nt_fetched}")

        # Create alt sequence
        alt_nt_onehot = str_to_one_hot(alt)
        mut_seq_onehot = wt_seq_onehot.clone()
        mut_seq_onehot[s_idx:e_idx] = alt_nt_onehot

        logger.info(f"Predicting reference and alternative sequences for {chr}:{real_start}-{real_end}")

        # Predict both sequences
        with torch.no_grad():
            wt_tensor = wt_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(self.device)
            ref_pred = self.model(wt_tensor, use_head).detach().cpu().numpy().squeeze(0)

            mut_tensor = mut_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(self.device)
            alt_pred = self.model(mut_tensor, use_head).detach().cpu().numpy().squeeze(0)

        return ref_pred, alt_pred, real_start, real_end

    def run_saturation_mutagenesis(self, chr, start_pos, end_pos, use_head='regression', output_dir=None,
                                   gene_exons=None, gene_name=None):
        """
        Run saturation mutagenesis on a region (all possible single nucleotide variants).

        Args:
            chr: Chromosome
            start_pos: Start position (1-based, inclusive)
            end_pos: End position (1-based, inclusive)
            use_head: Which prediction head to use
            output_dir: Output directory to save results
            gene_exons: Optional list of (start, end) tuples for gene exons (0-based)
            gene_name: Optional gene name

        Returns:
            pandas DataFrame with columns:
                - position: genomic position
                - ref: reference allele
                - alt: alternative allele
                - track_name: name of the track
                - ref_pred_mean: reference prediction (mean over all windows)
                - alt_pred_mean: alternative prediction (mean over all windows)
                - diff_mean: alt_pred_mean - ref_pred_mean (global mean)
                - ref_pred_max: max reference prediction
                - alt_pred_max: max alternative prediction
                - diff_max: max absolute difference (global)
                - diff_local_mean: mean difference in windows within +-500bp of variant
                - diff_local_max: max absolute difference in windows within +-500bp of variant
                - diff_gene_sum: sum of differences in windows overlapping gene exons (if gene provided)
                - diff_gene_mean: mean difference in windows overlapping gene exons (if gene provided)
                - gene_length: number of bins overlapping gene exons (if gene provided)
                - gene_name: name of the gene (if gene provided)
                - window_start: start of prediction window
                - window_end: end of prediction window
        """
        logger.info(f"Running saturation mutagenesis on {chr}:{start_pos}-{end_pos}")

        # Get reference sequence for the region
        token_dict = self.fasta(
            chr_name=chr,
            start=start_pos - 1,  # Convert to 0-based
            end=end_pos,  # End is exclusive
            return_augs=False,
            return_rela_idx=True
        )

        ref_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]

        # Convert one-hot to string for reference alleles
        def onehot_to_str(seq_onehot):
            mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "T", (0, 0, 0, 0): "N"}
            seq_str = ""
            for vec in seq_onehot:
                key = tuple((vec > 0.5).int().tolist())
                seq_str += mapping.get(key, "N")
            return seq_str

        # Get reference sequence string
        ref_seq_str = onehot_to_str(ref_seq_onehot)

        # Find the relative positions within the fetched sequence
        s_idx, e_idx = token_dict["rela_idx"]

        # All possible nucleotides
        nucleotides = ['A', 'C', 'G', 'T']

        # Store results
        all_results = []

        # Iterate through each position in the region
        n_positions = end_pos - start_pos + 1
        logger.info(f"Testing {n_positions} positions with up to 3 alternative alleles each")

        for i in tqdm(range(n_positions), desc="Saturation mutagenesis"):
            pos = start_pos + i
            pos_idx = s_idx + i  # Index in the fetched sequence

            # Get reference allele at this position
            ref_allele = ref_seq_str[pos_idx]

            if ref_allele == 'N':
                logger.warning(f"Skipping position {pos} with reference allele 'N'")
                continue

            # Test all alternative alleles
            for alt_allele in nucleotides:
                if alt_allele == ref_allele:
                    continue  # Skip if same as reference

                # Predict variant effect
                ref_pred, alt_pred, window_start, window_end = self.predict_variant_effect(
                    chr, pos, ref_allele, alt_allele, use_head
                )

                # Calculate difference
                diff = alt_pred - ref_pred

                # Identify local windows within +-500bp of the variant position
                # Each window has a center position
                local_window_radius = 500  # bp
                n_windows = ref_pred.shape[0]
                window_centers = []
                for w_idx in range(n_windows):
                    window_center = window_start + w_idx * self.window_size + self.window_size // 2
                    window_centers.append(window_center)

                # Find windows within +-500bp of variant position
                local_window_mask = []
                for w_center in window_centers:
                    if abs(w_center - pos) <= local_window_radius:
                        local_window_mask.append(True)
                    else:
                        local_window_mask.append(False)

                local_window_mask = np.array(local_window_mask)

                # Identify gene-overlapping windows if gene exons provided
                gene_window_mask = None
                if gene_exons is not None:
                    gene_window_mask = []
                    for w_idx in range(n_windows):
                        window_start_pos = window_start + w_idx * self.window_size
                        window_end_pos = window_start_pos + self.window_size

                        # Check if this window overlaps with any exon
                        overlaps_exon = False
                        for exon_start, exon_end in gene_exons:
                            # Check overlap: window and exon overlap if they don't not-overlap
                            # Not-overlap: window_end <= exon_start or window_start >= exon_end
                            if not (window_end_pos <= exon_start or window_start_pos >= exon_end):
                                overlaps_exon = True
                                break

                        gene_window_mask.append(overlaps_exon)

                    gene_window_mask = np.array(gene_window_mask)

                # Average predictions over windows for each track
                for track_idx, track_name in enumerate(self.label_names):
                    # Global statistics (all windows)
                    diff_track = diff[:, track_idx]

                    # Local statistics (windows within +-500bp)
                    if np.any(local_window_mask):
                        diff_local = diff_track[local_window_mask]
                        diff_local_mean = float(np.mean(diff_local))
                        diff_local_max = float(np.max(np.abs(diff_local)))
                    else:
                        # Fallback if no local windows found
                        diff_local_mean = float(np.mean(diff_track))
                        diff_local_max = float(np.max(np.abs(diff_track)))

                    # Gene statistics (windows overlapping gene exons)
                    diff_gene_sum = None
                    diff_gene_mean = None
                    gene_length_bins = None
                    if gene_window_mask is not None and np.any(gene_window_mask):
                        diff_gene = diff_track[gene_window_mask]
                        diff_gene_sum = float(np.sum(diff_gene))
                        diff_gene_mean = float(np.mean(diff_gene))
                        gene_length_bins = int(np.sum(gene_window_mask))

                    result = {
                        'position': pos,
                        'ref': ref_allele,
                        'alt': alt_allele,
                        'track_name': track_name,
                        'ref_pred_mean': float(np.mean(ref_pred[:, track_idx])),
                        'alt_pred_mean': float(np.mean(alt_pred[:, track_idx])),
                        'diff_mean': float(np.mean(diff_track)),
                        'ref_pred_max': float(np.max(ref_pred[:, track_idx])),
                        'alt_pred_max': float(np.max(alt_pred[:, track_idx])),
                        'diff_max': float(np.max(np.abs(diff_track))),
                        'diff_local_mean': diff_local_mean,
                        'diff_local_max': diff_local_max,
                        'diff_gene_sum': diff_gene_sum,
                        'diff_gene_mean': diff_gene_mean,
                        'gene_length': gene_length_bins,
                        'gene_name': gene_name,
                        'window_start': window_start,
                        'window_end': window_end,
                    }
                    all_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results if output directory is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save full results as TSV
            results_file = output_path / "saturation_mutagenesis_results.tsv"
            results_df.to_csv(results_file, sep='\t', index=False)
            logger.info(f"Saved saturation mutagenesis results to {results_file}")

            # Create summary pivot table: position x track
            # Prioritize diff_gene_sum if available, otherwise use diff_local_mean
            if 'diff_gene_sum' in results_df.columns and results_df['diff_gene_sum'].notna().any():
                # Use gene-specific sum
                summary_column = 'diff_gene_sum'
                agg_func = 'max'  # Max across alternative alleles for each position
                logger.info("Using diff_gene_sum for summary pivot table")
            else:
                # Use local mean
                summary_column = 'diff_local_mean'
                agg_func = 'max'  # Max across alternative alleles for each position
                logger.info("Using diff_local_mean for summary pivot table")

            pivot_data = results_df.groupby(['position', 'track_name'])[summary_column].agg(agg_func).reset_index()
            pivot_table = pivot_data.pivot(index='position', columns='track_name', values=summary_column)

            pivot_file = output_path / "saturation_mutagenesis_summary.tsv"
            pivot_table.to_csv(pivot_file, sep='\t')
            logger.info(f"Saved summary pivot table to {pivot_file} (using {summary_column})")

        return results_df

    def predict_region(self, chr, start, end, use_head='regression'):
        """
        Predict chromatin accessibility for a region.

        Args:
            chr: Chromosome
            start: Start position
            end: End position
            use_head: Which prediction head to use

        Returns:
            tuple: (predictions, window_start, window_end)
        """
        logger.info(f"Predicting region: {chr}:{start}-{end}")

        # Get center position and predict
        center_pos = (start + end) // 2
        pred, window_start, window_end = self.get_sequence_and_predict(chr, center_pos, use_head)

        return pred, window_start, window_end

    def get_labels_for_region(self, chr, window_start, window_end):
        """
        Get ground truth labels for a region by extracting from bigwig files.

        Args:
            chr: Chromosome
            window_start: Window start position
            window_end: Window end position

        Returns:
            numpy array: Labels (n_window, n_tracks) or None if not available
        """
        try:
            # Load trial summary CSV to get bigwig file paths
            trial_summary_path = self.cfg.data.preprocess.trial_summary_path
            if not Path(trial_summary_path).exists():
                logger.warning(f"Trial summary not found: {trial_summary_path}")
                return None

            data_config = pd.read_csv(trial_summary_path, index_col=1)

            # Create ModelSeq for this region
            mseqs = [ModelSeq(chr, window_start, window_end, "inference")]

            # Create temporary directory for label files
            import tempfile
            temp_dir = tempfile.mkdtemp()

            # Annotate unmappable regions if available
            unmap_npy_path = None
            if hasattr(self.cfg.data.preprocess, 'unmap_bed') and Path(self.cfg.data.preprocess.unmap_bed).exists():
                unmap_npy = os.path.join(temp_dir, "mseqs_unmap.npy")
                mseqs_unmap = annotate_unmap(
                    mseqs,
                    self.cfg.data.preprocess.unmap_bed,
                    self.cfg.data.preprocess.context_length,
                    self.cfg.data.preprocess.window_size,
                )
                np.save(unmap_npy, mseqs_unmap)
                unmap_npy_path = unmap_npy

            # Extract labels for each track
            all_labels = []
            for track_name in self.label_names:
                # Get track info from data_config
                if track_name not in data_config.index:
                    logger.warning(f"Track {track_name} not found in data config, skipping")
                    continue

                track_info = data_config.loc[track_name]
                genome_cov_file = track_info["file"]

                # Check if bigwig file exists
                if not Path(genome_cov_file).exists():
                    logger.warning(f"Bigwig file not found for {track_name}: {genome_cov_file}")
                    continue

                # Create temporary H5 file for this track's labels
                label_h5_file = os.path.join(temp_dir, f"{track_name}_label.h5")

                # Extract labels using get_labels function
                get_labels(
                    mseqs,
                    blacklist_bed=self.cfg.data.preprocess.blacklist_bed,
                    pool_width=self.cfg.data.preprocess.window_size,
                    kept_num_after_crop=self.cfg.data.preprocess.n_window,
                    seqs_cov_file=label_h5_file,
                    genome_cov_file=genome_cov_file,
                    umap_npy_path=unmap_npy_path,
                    **track_info[["sum_stat", "baseline_pct", "umap_pct", "scale", "clip", "clip_soft"]].to_dict(),
                )

                # Read the extracted labels
                with h5py.File(label_h5_file, 'r') as f:
                    track_labels = f["targets"][0][:]  # Shape: (n_window,)
                    all_labels.append(track_labels)

                # Clean up temp file
                os.remove(label_h5_file)

            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)

            if len(all_labels) == 0:
                logger.warning(f"No labels could be extracted for {chr}:{window_start}-{window_end}")
                return None

            # Stack labels into (n_window, n_tracks) array
            labels = np.stack(all_labels, axis=1)
            logger.info(f"Extracted labels with shape {labels.shape} for {chr}:{window_start}-{window_end}")

            return labels

        except Exception as e:
            logger.warning(f"Could not load labels: {e}")
            import traceback
            traceback.print_exc()
            return None

    def untransform_predictions(self, predictions, track_name):
        """
        Untransform predictions back to original scale.

        Args:
            predictions: Numpy array (n_window,) for a single track
            track_name: Name of the track to get transformation params

        Returns:
            Untransformed predictions
        """
        preds = predictions.copy()

        # Use cached label metadata
        if self.label_meta is None:
            # No metadata available, skip untransform
            return preds

        try:
            # Find the row for this track by matching the 'trial' column
            matching_rows = self.label_meta[self.label_meta['trial'] == track_name]
            if len(matching_rows) == 0:
                logger.warning(f"Track {track_name} not found in label metadata, skipping untransform")
                return preds

            meta_row = matching_rows.iloc[0]

            trial_scale = meta_row.get('scale', 1.0)
            trial_clip_soft = meta_row.get('clip_soft', 48.0)
            trial_sum_stat = meta_row.get('sum_stat', 'sum_three_quarter')

            # Step 1: Undo scale
            if trial_scale != 1.0:
                preds = preds / trial_scale

            # Step 2: Undo soft clip
            if trial_clip_soft is not None and not pd.isna(trial_clip_soft):
                clip_mask = preds > trial_clip_soft
                preds[clip_mask] = (trial_clip_soft - 1) + (preds[clip_mask] - (trial_clip_soft - 1)) ** 2

            # Step 3: Undo power transform
            if trial_sum_stat == "sum_three_quarter":
                preds = preds ** (4.0 / 3.0)
            elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
                preds = (preds + 1) ** 2 - 1
            elif trial_sum_stat in ['sum', 'mean', "avg"]:
                pass
            else:
                logger.warning(f"Unknown sum_stat: {trial_sum_stat}, skipping power transform")

            return preds

        except Exception as e:
            logger.warning(f"Error untransforming predictions for {track_name}: {e}")
            return predictions

    def export_to_bigwig(self, predictions, chr, window_start, track_names, output_dir, prefix="pred"):
        """
        Export predictions to bigwig files.

        Args:
            predictions: Numpy array (n_window, n_tracks)
            chr: Chromosome
            window_start: Start position of first window
            track_names: List of track names
            output_dir: Output directory
            prefix: Prefix for output folder (pred, label, alt)
        """
        output_path = Path(output_dir) / prefix
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {predictions.shape[1]} tracks to {output_path}")

        # Get chromosome sizes (need this for bigwig)
        # Use a standard chromosome sizes file
        chrom_sizes_path = ROOT / "Data" / "reference" / "hg38.chrom.sizes"
        if not chrom_sizes_path.exists():
            # Create minimal chromosome sizes for this chromosome
            # This is a fallback - ideally use actual chromosome sizes file
            chrom_sizes = {chr: window_start + predictions.shape[0] * self.window_size + 1000000}
        else:
            chrom_sizes = {}
            with open(chrom_sizes_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        chrom_sizes[parts[0]] = int(parts[1])

        # Export each track
        for track_idx, track_name in enumerate(tqdm(track_names, desc=f"Exporting {prefix}")):
            output_file = output_path / f"{track_name}.bw"

            # Untransform predictions for this track back to original scale
            track_predictions = predictions[:, track_idx]
            if prefix in ["pred", "alt"]:  # Only untransform model predictions, not labels
                track_predictions = self.untransform_predictions(track_predictions, track_name)

            # Create bigwig file
            bw = pyBigWig.open(str(output_file), "w")

            # Add header with chromosome sizes
            chr_size = chrom_sizes.get(chr, window_start + predictions.shape[0] * self.window_size + 1000000)
            bw.addHeader([(chr, chr_size)])

            # Add values for each window
            chroms = []
            starts = []
            ends = []
            values = []

            for i in range(len(track_predictions)):
                pos_start = window_start + i * self.window_size
                pos_end = pos_start + self.window_size
                value = float(track_predictions[i])

                chroms.append(chr)
                starts.append(pos_start)
                ends.append(pos_end)
                values.append(value)

            # Write entries
            bw.addEntries(chroms, starts, ends=ends, values=values)
            bw.close()

        logger.info(f"Exported {len(track_names)} bigwig files to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick inference and bigwig generation for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Method 1: Using direct checkpoint path
  python 01_0_quick_inference_bigwig.py --gene GRIN2A --checkpoint /path/to/model.ckpt --output outputs/gene_GRIN2A

  # Method 2: Using experiment name (recommended)
  python 01_0_quick_inference_bigwig.py --gene GRIN2A --exp_name my_experiment --chk best --output outputs/gene_GRIN2A

  # Region-based inference
  python 01_0_quick_inference_bigwig.py --region chr1:100000-200000 --exp_name my_exp --chk best --output outputs/region_chr1

  # Variant effect prediction
  python 01_0_quick_inference_bigwig.py --variant chr1:154426970:G:A --exp_name my_exp --chk best --output outputs/variant_rs1234

  # Saturation mutagenesis (single position)
  python 01_0_quick_inference_bigwig.py --sat_mutagenesis chr1:154426970 --exp_name my_exp --chk best --output outputs/saturation_chr1_154426970

  # Saturation mutagenesis (region)
  python 01_0_quick_inference_bigwig.py --sat_mutagenesis chr1:154426970-154426975 --exp_name my_exp --chk best --output outputs/saturation_chr1_region

  # Saturation mutagenesis with gene-specific statistics
  python 01_0_quick_inference_bigwig.py --sat_mutagenesis chr12:40208963 --gene LRRK2 --exp_name my_exp --chk best --output outputs/saturation_LRRK2
        """
    )

    # Input specification (mutually exclusive for primary modes)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--region', type=str, help='Genomic region (e.g., chr1:100000-200000)')
    input_group.add_argument('--variant', type=str, help='Variant (e.g., chr1:154426970:G:A)')
    input_group.add_argument('--sat_mutagenesis', type=str, help='Saturation mutagenesis region (e.g., chr1:154426970 or chr1:154426970-154426975)')

    # Gene can be used alone or with saturation mutagenesis
    parser.add_argument('--gene', type=str, default=None,
                       help='Gene name (e.g., GRIN2A). Can be used alone for gene inference, or with --sat_mutagenesis to calculate gene-specific statistics.')

    # Model parameters - Method 1: Direct paths
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (full path)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: Model/config.yaml)')

    # Model parameters - Method 2: Experiment name and checkpoint
    parser.add_argument('--exp_name', '-e', type=str, default=None,
                       help='Experiment name (alternative to --checkpoint)')
    parser.add_argument('--chk', type=str, default=None,
                       help='Checkpoint epoch (e.g., "best" or "100", used with --exp_name)')
    parser.add_argument('--log_base', type=str, default='./logs',
                       help='Base directory for logs (default: ./logs)')
    parser.add_argument('--chk_base', type=str, default='./Chk',
                       help='Base directory for checkpoints (default: ./Chk)')

    # GTF file (for gene mode)
    parser.add_argument('--gtf', type=str, default='Data/source/gencode.v48.annotation.gtf.gz',
                       help='Path to GTF file (default: auto-detect)')

    # Output
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--no-labels', action='store_true',
                       help='Skip exporting ground truth labels')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Validate input: need at least --gene or one of the mutually exclusive options
    if not args.gene and not args.region and not args.variant and not args.sat_mutagenesis:
        parser.error("Must specify at least one of: --gene, --region, --variant, or --sat_mutagenesis")

    # Resolve model paths
    # Method 1: Direct checkpoint path
    # Method 2: Experiment name + checkpoint epoch
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        config_path = args.config
        logger.info("Using direct checkpoint path")
    elif args.exp_name and args.chk:
        # Construct paths from experiment name and checkpoint
        checkpoint_path = f"{args.chk_base}/{args.exp_name}/chk_epoch_{args.chk}.pt"
        config_path = f"{args.log_base}/{args.exp_name}/overall_setting.yaml"
        logger.info(f"Using experiment: {args.exp_name}, checkpoint: {args.chk}")
    else:
        raise ValueError(
            "Must specify either:\n"
            "  Method 1: --checkpoint /path/to/checkpoint.pt\n"
            "  Method 2: --exp_name EXP_NAME --chk EPOCH"
        )

    # Initialize inference engine
    logger.info("="*80)
    logger.info("Quick Inference and BigWig Generation")
    logger.info("="*80)

    inference = RegionInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process based on input type
    # Check saturation mutagenesis first (can be combined with --gene)
    if args.sat_mutagenesis:
        # Saturation mutagenesis mode
        logger.info(f"Mode: Saturation mutagenesis for {args.sat_mutagenesis}")

        # Parse region
        chr, start_pos, end_pos = inference.parse_sat_region(args.sat_mutagenesis)

        # Check if gene is also provided for gene-specific statistics
        gene_exons = None
        gene_name = None
        if args.gene:
            logger.info(f"Extracting exons for gene: {args.gene}")
            exons_with_chr = inference.get_gene_exons(args.gene, args.gtf)

            # Convert to list of (start, end) tuples without chromosome
            gene_exons = [(start, end) for _, start, end in exons_with_chr]
            gene_name = args.gene
            logger.info(f"Found {len(gene_exons)} exons for gene {gene_name}")

        # Run saturation mutagenesis
        results_df = inference.run_saturation_mutagenesis(
            chr, start_pos, end_pos, output_dir=output_dir,
            gene_exons=gene_exons, gene_name=gene_name
        )

        logger.info(f"Saturation mutagenesis complete")
        logger.info(f"Region: {chr}:{start_pos}-{end_pos}")
        if gene_name:
            logger.info(f"Gene: {gene_name} with {len(gene_exons)} exons")
        logger.info(f"Total variants tested: {len(results_df) // len(inference.label_names)}")
        logger.info(f"Results saved to: {output_dir}")

    elif args.gene:
        # Gene mode (without saturation mutagenesis)
        logger.info(f"Mode: Gene-based inference for {args.gene}")

        # Get exons
        exons = inference.get_gene_exons(args.gene, args.gtf)

        # For simplicity, use the first exon region
        # You could extend this to process all exons
        chr, start, end = exons[0]
        logger.info(f"Using first exon: {chr}:{start}-{end}")

        # Predict
        predictions, window_start, window_end = inference.predict_region(chr, start, end)

        # Export predictions
        inference.export_to_bigwig(
            predictions, chr, window_start,
            inference.label_names, output_dir, prefix="pred"
        )

        # Export labels if requested
        if not args.no_labels:
            labels = inference.get_labels_for_region(chr, window_start, window_end)
            if labels is not None:
                inference.export_to_bigwig(
                    labels, chr, window_start,
                    inference.label_names, output_dir, prefix="label"
                )

        logger.info(f"Gene inference complete for {args.gene}")
        logger.info(f"Region: {chr}:{window_start}-{window_end}")

    elif args.region:
        # Region mode
        logger.info(f"Mode: Region-based inference for {args.region}")

        # Parse region
        chr, start, end = inference.parse_region(args.region)

        # Predict
        predictions, window_start, window_end = inference.predict_region(chr, start, end)

        # Export predictions
        inference.export_to_bigwig(
            predictions, chr, window_start,
            inference.label_names, output_dir, prefix="pred"
        )

        # Export labels if requested
        if not args.no_labels:
            labels = inference.get_labels_for_region(chr, window_start, window_end)
            if labels is not None:
                inference.export_to_bigwig(
                    labels, chr, window_start,
                    inference.label_names, output_dir, prefix="label"
                )

        logger.info(f"Region inference complete")
        logger.info(f"Window: {chr}:{window_start}-{window_end}")

    elif args.variant:
        # Variant mode
        logger.info(f"Mode: Variant effect prediction for {args.variant}")

        # Parse variant
        chr, pos, ref, alt = inference.parse_variant(args.variant)

        # Predict variant effect
        ref_pred, alt_pred, window_start, window_end = inference.predict_variant_effect(
            chr, pos, ref, alt
        )

        # Export reference predictions
        inference.export_to_bigwig(
            ref_pred, chr, window_start,
            inference.label_names, output_dir, prefix="pred"
        )

        # Export alternative predictions
        inference.export_to_bigwig(
            alt_pred, chr, window_start,
            inference.label_names, output_dir, prefix="alt"
        )

        # Export labels if requested
        if not args.no_labels:
            labels = inference.get_labels_for_region(chr, window_start, window_end)
            if labels is not None:
                inference.export_to_bigwig(
                    labels, chr, window_start,
                    inference.label_names, output_dir, prefix="label"
                )

        # Calculate and export difference (alt - ref)
        diff = alt_pred - ref_pred
        inference.export_to_bigwig(
            diff, chr, window_start,
            inference.label_names, output_dir, prefix="diff"
        )

        logger.info(f"Variant effect prediction complete")
        logger.info(f"Variant: {chr}:{pos}:{ref}>{alt}")
        logger.info(f"Window: {chr}:{window_start}-{window_end}")

    # Print summary
    logger.info("="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of tracks: {len(inference.label_names)}")

    # List output folders
    output_folders = [d.name for d in output_dir.iterdir() if d.is_dir()]
    logger.info(f"Output folders: {', '.join(output_folders)}")

    # Visualization command
    logger.info("")
    logger.info("To visualize the results, run:")
    print()
    if args.variant:
        print(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        print(f"    --bigwig-dir {output_dir}/pred \\")
        print(f"    --region {chr}:{window_start}-{window_end} \\")
        print(f"    --output {output_dir}/visualization.pdf")
        print()
        print("  Or compare reference vs alternative:")
        print(f"  # Reference allele")
        print(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        print(f"    --bigwig-dir {output_dir}/pred \\")
        print(f"    --region {chr}:{pos-5000}-{pos+5000} \\")
        print(f"    --output {output_dir}/ref_visualization.pdf")
        print()
        print(f"  # Alternative allele")
        print(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        print(f"    --bigwig-dir {output_dir}/alt \\")
        print(f"    --region {chr}:{pos-5000}-{pos+5000} \\")
        print(f"    --output {output_dir}/alt_visualization.pdf")
    elif args.sat_mutagenesis:
        print(f"  # View results in output directory:")
        print(f"  cat {output_dir}/saturation_mutagenesis_results.tsv")
        print()
        print(f"  # Summary matrix (position x track):")
        print(f"  cat {output_dir}/saturation_mutagenesis_summary.tsv")
    else:
        print(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        print(f"    --bigwig-dir {output_dir}/pred \\")
        print(f"    --region {chr}:{window_start}-{window_end} \\")
        print(f"    --output {output_dir}/visualization.pdf")

    print()
    logger.info("Done!")


if __name__ == "__main__":
    main()
