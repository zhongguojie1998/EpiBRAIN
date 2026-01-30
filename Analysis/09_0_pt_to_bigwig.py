#!/usr/bin/env python
"""
Convert .pt attribution files from motif interpretation to BigWig format.

Usage:
    python 10_pt_to_bigwig.py -i <input_pt_file> -o <output_bw_file> --fasta <reference_fasta>
    python 10_pt_to_bigwig.py -d <directory> -o <output_directory> --fasta <reference_fasta>
"""

import os
import re
import sys
import torch
import numpy as np
import pyBigWig
import click
from pathlib import Path

# Add Model directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))

from data.tokenizer import FastaInterval, str_to_one_hot


def parse_filename(filename):
    """
    Parse filename to extract genomic coordinates.
    Expected format: chr<X>_<start>_<end>_<trial>_<baseline>.pt
    Example: chr19_49118631_49119654_MiniAtlas-L56IT_K27Ac_random.pt
    """
    basename = os.path.basename(filename)
    match = re.match(r'(chr\w+)_(\d+)_(\d+)_(.+)\.pt', basename)
    if match:
        chrom = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        trial_baseline = match.group(4)
        return chrom, start, end, trial_baseline
    else:
        raise ValueError(f"Cannot parse filename: {filename}")


def pt_to_bigwig(pt_file, output_bw, fasta_file, score_type='sum', window_size=32):
    """
    Convert a .pt attribution file to BigWig format.

    Parameters:
    -----------
    pt_file : str
        Path to input .pt file
    output_bw : str
        Path to output .bw file
    fasta_file : str
        Path to reference genome FASTA file
    score_type : str
        How to aggregate attribution scores: 'sum', 'max', 'mean', 'l2'
    window_size : int
        Size of genomic bins (default: 32bp)
    """
    # Parse filename to get genomic coordinates
    chrom, start, end, trial_baseline = parse_filename(pt_file)

    # Load attribution tensor
    # Shape: (1, sequence_length, 4) where 4 is for A, C, G, T
    attribution = torch.load(pt_file, map_location='cpu', weights_only=False)

    if attribution.dim() == 3:
        attribution = attribution.squeeze(0)  # Remove batch dimension

    # Get the reference sequence to multiply with attribution
    # This ensures we only look at the contribution from the actual nucleotides
    seq_length = attribution.shape[0]
    context_start = start - (seq_length - (end - start)) // 2

    dna_tokenizer = FastaInterval(fasta_file=fasta_file, context_length=seq_length)
    token_dict = dna_tokenizer(chr_name=chrom, start=context_start, end=context_start + seq_length,
                                return_augs=False, return_rela_idx=False)
    test_seq_onehot = token_dict["one_hot"]  # Shape: (sequence_length, 4), PyTorch tensor

    # Convert both to numpy arrays
    attribution_np = attribution.numpy() if torch.is_tensor(attribution) else attribution
    test_seq_onehot_np = test_seq_onehot.numpy() if torch.is_tensor(test_seq_onehot) else test_seq_onehot

    # Multiply attribution by one-hot encoded sequence (like line 248 in 02_motif_interpretation.py)
    # This extracts only the contribution from the reference genome nucleotides
    attribution_weighted = attribution_np * test_seq_onehot_np

    # Aggregate across nucleotides based on score_type
    if score_type == 'sum':
        scores = np.sum(attribution_weighted, axis=1)
    elif score_type == 'max':
        scores = np.max(attribution_weighted, axis=1)
    elif score_type == 'mean':
        scores = np.mean(attribution_weighted, axis=1)
    elif score_type == 'l2':
        scores = np.sqrt(np.sum(attribution_weighted ** 2, axis=1))
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    # Pool scores into bins of window_size
    n_bins = len(scores) // window_size
    pooled_scores = scores[:n_bins * window_size].reshape(n_bins, window_size).mean(axis=1)

    # Calculate actual genomic coordinates
    # The .pt file represents a context region centered on the interval
    seq_length = len(scores)
    context_start = start - (seq_length - (end - start)) // 2
    bin_starts = context_start + np.arange(n_bins) * window_size
    bin_ends = bin_starts + window_size

    # Create BigWig file
    bw = pyBigWig.open(output_bw, "w")

    # Define chromosome sizes (you may need to adjust this)
    # For now, we'll use a conservative estimate
    chrom_sizes = {chrom: int(bin_ends[-1] + 1000000)}
    bw.addHeader(list(chrom_sizes.items()))

    # Write values
    bw.addEntries(
        [chrom] * len(pooled_scores),
        bin_starts.tolist(),
        ends=bin_ends.tolist(),
        values=pooled_scores.astype(float).tolist()
    )

    bw.close()
    print(f"Created BigWig: {output_bw}")
    print(f"  Region: {chrom}:{context_start}-{bin_ends[-1]}")
    print(f"  Number of bins: {n_bins}")
    print(f"  Score range: [{pooled_scores.min():.4f}, {pooled_scores.max():.4f}]")


def process_directory(input_dir, output_dir, fasta_file, score_type='sum', window_size=32, pattern='*.pt'):
    """Process all .pt files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pt_files = list(input_path.glob(pattern))
    print(f"Found {len(pt_files)} .pt files in {input_dir}")

    for pt_file in pt_files:
        output_bw = output_path / (pt_file.stem + '.bw')
        try:
            pt_to_bigwig(str(pt_file), str(output_bw), fasta_file, score_type, window_size)
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")


@click.command()
@click.option('-i', '--input', 'input_file', type=str, help='Input .pt file')
@click.option('-d', '--input_dir', type=str, help='Input directory containing .pt files')
@click.option('-o', '--output', required=True, type=str, help='Output .bw file or directory')
@click.option('-f', '--fasta', 'fasta_file', required=True, type=str, help='Reference genome FASTA file')
@click.option('-s', '--score_type', type=click.Choice(['sum', 'max', 'mean', 'l2']), default='sum',
              help='How to aggregate attribution scores across nucleotides')
@click.option('-w', '--window_size', type=int, default=32, help='Size of genomic bins in bp')
@click.option('-p', '--pattern', type=str, default='*.pt', help='File pattern to match (for directory mode)')
def main(input_file, input_dir, output, fasta_file, score_type, window_size, pattern):
    """Convert .pt attribution files to BigWig format."""

    if input_file and input_dir:
        raise ValueError("Cannot specify both --input and --input_dir")

    if not input_file and not input_dir:
        raise ValueError("Must specify either --input or --input_dir")

    if input_file:
        # Single file mode
        pt_to_bigwig(input_file, output, fasta_file, score_type, window_size)
    else:
        # Directory mode
        process_directory(input_dir, output, fasta_file, score_type, window_size, pattern)


if __name__ == '__main__':
    main()
