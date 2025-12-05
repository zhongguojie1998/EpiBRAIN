#!/usr/bin/env python3
"""
Quick inference and bigwig generation for visualization.

This script performs model inference on genomic regions and generates bigwig files
for visualization with pyGenomeTracks.

Input types:
1. Gene name: --gene GENE_NAME (extracts exons from GTF)
2. Region: --region chr1:1000-2000
3. Variant: --variant chr1:1000:A:T (reference allele and alternative allele)

Output:
- For gene/region: pred/ and label/ folders with bigwig files
- For variant: label/, pred/ (reference), and alt/ (alternative) folders with bigwig files

Usage examples:
    # Gene-based inference
    python 01_0_quick_inference_bigwig.py --gene GRIN2A --output gene_GRIN2A

    # Region-based inference
    python 01_0_quick_inference_bigwig.py --region chr1:100000-200000 --output region_chr1

    # Variant effect prediction
    python 01_0_quick_inference_bigwig.py --variant chr1:154426970:G:A --output variant_rs1234
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
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.data_utils import STD_CHR, ModelSeq, annotate_unmap, get_labels
from data.tokenizer import FastaInterval, str_to_one_hot
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
        self.cfg = load_config(config_path)

        # Setup model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = setup_model(self.cfg, checkpoint_path)
        self.model.to(device)
        self.model.eval()

        # Get model parameters
        self.seq_len = self.cfg.dataset.seq_len
        self.window_size = self.cfg.dataset.window_size
        self.n_window = self.seq_len // self.window_size

        # Setup FASTA tokenizer
        fasta_path = self.cfg.dataset.fasta_path
        logger.info(f"Loading FASTA from {fasta_path}")
        self.fasta = FastaInterval(fasta_path, return_seq_indices=False)

        # Load labels (for ground truth)
        h5_path = self.cfg.dataset.h5_path
        logger.info(f"Loading labels from {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            self.label_names = [x.decode() for x in f['label_names'][:]]

        logger.info(f"Model ready. Sequence length: {self.seq_len}, Window size: {self.window_size}")
        logger.info(f"Number of tracks: {len(self.label_names)}")

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
        gtf = GTF(str(gtf_path))

        # Get gene
        genes = gtf.get_gene(gene_name)
        if len(genes) == 0:
            raise ValueError(f"Gene '{gene_name}' not found in GTF")

        gene = genes[0]
        logger.info(f"Found gene: {gene.gene_name} ({gene.gene_id}) on {gene.seqid}:{gene.start}-{gene.end}")

        # Get exons
        exons = []
        for transcript in gene.transcripts:
            for exon in transcript.exons:
                exons.append((gene.seqid, exon.start, exon.end))

        # Merge overlapping exons
        if len(exons) == 0:
            raise ValueError(f"No exons found for gene '{gene_name}'")

        exons = sorted(set(exons))
        logger.info(f"Found {len(exons)} unique exons for {gene_name}")

        return exons

    def get_sequence_window(self, chr, center_pos):
        """
        Get sequence window centered at position.

        Args:
            chr: Chromosome
            center_pos: Center position

        Returns:
            tuple: (sequence, window_start, window_end)
        """
        # Calculate window boundaries
        half_len = self.seq_len // 2
        window_start = center_pos - half_len
        window_end = center_pos + half_len

        # Get sequence
        seq = self.fasta(chr, window_start, window_end)

        return seq, window_start, window_end

    def predict_sequence(self, seq, apply_softmax=True):
        """
        Run model prediction on sequence.

        Args:
            seq: DNA sequence string
            apply_softmax: Whether to apply softmax to output

        Returns:
            numpy array: Predictions (n_window, n_tracks)
        """
        # One-hot encode
        seq_encoded = str_to_one_hot(seq)
        seq_tensor = torch.from_numpy(seq_encoded).unsqueeze(0).float().to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(seq_tensor)
            if apply_softmax:
                output = torch.softmax(output, dim=-1)
            pred = output[0].cpu().numpy()

        return pred

    def predict_variant_effect(self, chr, pos, ref, alt):
        """
        Predict effect of variant on chromatin accessibility.

        Args:
            chr: Chromosome
            pos: Position (1-based)
            ref: Reference allele
            alt: Alternative allele

        Returns:
            tuple: (ref_pred, alt_pred, window_start, window_end)
        """
        logger.info(f"Predicting variant effect: {chr}:{pos}:{ref}>{alt}")

        # Get reference sequence
        ref_seq, window_start, window_end = self.get_sequence_window(chr, pos)

        # Verify reference allele
        center_idx = len(ref_seq) // 2
        ref_in_seq = ref_seq[center_idx:center_idx+len(ref)]
        if ref_in_seq.upper() != ref:
            logger.warning(f"Reference allele mismatch: expected {ref}, got {ref_in_seq}")

        # Create alternative sequence
        alt_seq = ref_seq[:center_idx] + alt + ref_seq[center_idx+len(ref):]

        # Predict both sequences
        logger.info("Predicting reference sequence...")
        ref_pred = self.predict_sequence(ref_seq)

        logger.info("Predicting alternative sequence...")
        alt_pred = self.predict_sequence(alt_seq)

        return ref_pred, alt_pred, window_start, window_end

    def predict_region(self, chr, start, end):
        """
        Predict chromatin accessibility for a region.

        Args:
            chr: Chromosome
            start: Start position
            end: End position

        Returns:
            tuple: (predictions, window_start, window_end)
        """
        logger.info(f"Predicting region: {chr}:{start}-{end}")

        # Get center position
        center_pos = (start + end) // 2

        # Get sequence window
        seq, window_start, window_end = self.get_sequence_window(chr, center_pos)

        # Predict
        pred = self.predict_sequence(seq)

        return pred, window_start, window_end

    def get_labels_for_region(self, chr, window_start, window_end):
        """
        Get ground truth labels for a region.

        Args:
            chr: Chromosome
            window_start: Window start position
            window_end: Window end position

        Returns:
            numpy array: Labels (n_window, n_tracks) or None if not available
        """
        try:
            # Load labels from h5 file
            h5_path = self.cfg.dataset.h5_path
            with h5py.File(h5_path, 'r') as f:
                # Get chromosome data
                if chr not in f:
                    logger.warning(f"Chromosome {chr} not found in labels")
                    return None

                chr_data = f[chr]

                # Find overlapping windows
                # Assuming labels are stored at regular intervals
                labels = chr_data[:]

                # Calculate which windows we need
                # This is simplified - adjust based on actual data structure
                window_idx_start = window_start // self.window_size
                window_idx_end = window_end // self.window_size

                if window_idx_end > len(labels):
                    logger.warning(f"Window extends beyond available labels")
                    return None

                return labels[window_idx_start:window_idx_end]

        except Exception as e:
            logger.warning(f"Could not load labels: {e}")
            return None

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

            for i in range(predictions.shape[0]):
                pos_start = window_start + i * self.window_size
                pos_end = pos_start + self.window_size
                value = float(predictions[i, track_idx])

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
        """
    )

    # Input specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--gene', type=str, help='Gene name (e.g., GRIN2A)')
    input_group.add_argument('--region', type=str, help='Genomic region (e.g., chr1:100000-200000)')
    input_group.add_argument('--variant', type=str, help='Variant (e.g., chr1:154426970:G:A)')

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
    parser.add_argument('--gtf', type=str, default=None,
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
    if args.gene:
        # Gene mode
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
    if args.variant:
        logger.info(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        logger.info(f"    --bigwig-dir {output_dir}/pred \\")
        logger.info(f"    --region {chr}:{window_start}-{window_end} \\")
        logger.info(f"    --output {output_dir}/visualization.pdf")
        logger.info("")
        logger.info("  Or compare reference vs alternative:")
        logger.info(f"  # Reference allele")
        logger.info(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        logger.info(f"    --bigwig-dir {output_dir}/pred \\")
        logger.info(f"    --region {chr}:{pos-5000}-{pos+5000} \\")
        logger.info(f"    --output {output_dir}/ref_visualization.pdf")
        logger.info("")
        logger.info(f"  # Alternative allele")
        logger.info(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        logger.info(f"    --bigwig-dir {output_dir}/alt \\")
        logger.info(f"    --region {chr}:{pos-5000}-{pos+5000} \\")
        logger.info(f"    --output {output_dir}/alt_visualization.pdf")
    else:
        logger.info(f"  python Analysis/00_visualize_data_pygenometrack.py \\")
        logger.info(f"    --bigwig-dir {output_dir}/pred \\")
        logger.info(f"    --region {chr}:{window_start}-{window_end} \\")
        logger.info(f"    --output {output_dir}/visualization.pdf")

    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
