#!/usr/bin/env python
"""
Extract attribution scores from .pt files for ABC enhancer-gene connections.

For each ABC connection, loads the corresponding attribution file and extracts
max, sum, and min attribution scores for the enhancer region.

Usage:
    python 09_3_ABC_screen_significant_attributions.py -f <fasta_file> -o <output.tsv> -m K27Ac
    python 09_3_ABC_screen_significant_attributions.py -f <fasta_file> -o <output.tsv> -m ATAC -j 16
"""

import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import click
from pathlib import Path
from joblib import Parallel, delayed

# Add Model directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))

from data.tokenizer import FastaInterval


def load_and_process_pt_file(pt_file, fasta_file, window_size=32):
    """
    Load a .pt file and return processed attribution scores.

    Returns:
    --------
    tuple : (file_chrom, bin_starts, pooled_scores) or None if error
    """
    if not os.path.exists(pt_file):
        return None

    try:
        # Load attribution tensor
        attribution = torch.load(pt_file, map_location='cpu', weights_only=False)
        if attribution.dim() == 3:
            attribution = attribution.squeeze(0)

        # Parse filename to get genomic coordinates
        basename = os.path.basename(pt_file)
        match = re.match(r'(chr\w+)_(\d+)_(\d+)_(.+)\.pt', basename)
        if not match:
            return None

        file_chrom = match.group(1)
        file_start = int(match.group(2))
        file_end = int(match.group(3))

        # Get the reference sequence
        seq_length = attribution.shape[0]
        # The filename start/end indicate the region of interest (midpoint)
        # The actual sequence starts earlier to provide context
        region_length = file_end - file_start
        context_start = file_start - (seq_length - region_length) // 2

        dna_tokenizer = FastaInterval(fasta_file=fasta_file, context_length=seq_length)
        token_dict = dna_tokenizer(chr_name=file_chrom, start=context_start, end=context_start + seq_length,
                                    return_augs=False, return_rela_idx=False)
        test_seq_onehot = token_dict["one_hot"]

        # Convert to numpy (make writable copies to avoid warnings)
        attribution_np = attribution.numpy().copy() if torch.is_tensor(attribution) else np.array(attribution, copy=True)
        test_seq_onehot_np = test_seq_onehot.numpy().copy() if torch.is_tensor(test_seq_onehot) else np.array(test_seq_onehot, copy=True)

        # Multiply by sequence to get actual contributions
        attribution_weighted = attribution_np * test_seq_onehot_np

        # Sum across nucleotides
        scores = np.sum(attribution_weighted, axis=1)

        # Pool into bins
        n_bins = len(scores) // window_size
        pooled_scores = scores[:n_bins * window_size].reshape(n_bins, window_size).mean(axis=1)

        # Calculate bin positions
        bin_starts = context_start + np.arange(n_bins) * window_size

        return file_chrom, bin_starts, pooled_scores

    except Exception as e:
        print(f"Error loading {pt_file}: {e}")
        return None


def extract_scores_from_processed_data(bin_starts, pooled_scores, enhancer_start, enhancer_end):
    """
    Extract max, sum, min scores for a given enhancer region from pre-processed data.
    """
    # Find bins that overlap with the enhancer region
    enhancer_mask = (bin_starts >= enhancer_start) & (bin_starts < enhancer_end)

    if not np.any(enhancer_mask):
        return None

    enhancer_scores = pooled_scores[enhancer_mask]

    return {
        'max_score': float(np.max(enhancer_scores)),
        'sum_score': float(np.sum(enhancer_scores)),
        'min_score': float(np.min(enhancer_scores)),
    }


def process_single_pt_file_group(pt_file, group_df, fasta_file):
    """
    Process a single .pt file and extract scores for all associated ABC connections.

    Parameters:
    -----------
    pt_file : str
        Path to .pt file
    group_df : pd.DataFrame
        DataFrame containing all ABC connections for this file
    fasta_file : str
        Path to reference genome FASTA file

    Returns:
    --------
    list : List of tuples (idx, max_score, sum_score, min_score, file_found, filename)
    """
    results = []
    filename = os.path.basename(pt_file)

    # Load and process the .pt file once
    processed_data = load_and_process_pt_file(pt_file, fasta_file)

    if processed_data is None:
        # File not found or error - mark all rows as not found
        for idx, row in group_df.iterrows():
            results.append((idx, np.nan, np.nan, np.nan, False, ''))
        return results

    _file_chrom, bin_starts, pooled_scores = processed_data

    # Extract scores for all enhancers in this file
    for idx, row in group_df.iterrows():
        result = extract_scores_from_processed_data(
            bin_starts, pooled_scores, row['start.x'], row['end.x']
        )

        if result is not None:
            results.append((
                idx,
                result['max_score'],
                result['sum_score'],
                result['min_score'],
                True,
                filename
            ))
        else:
            results.append((idx, np.nan, np.nan, np.nan, False, ''))

    return results


def process_abc_file(abc_file, fasta_file, output_file, n_jobs=-1, modality='K27Ac', base_dir="Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp"):
    """
    Process ABC file and extract attribution scores for each connection.

    Parameters:
    -----------
    abc_file : str
        Path to ABC connections file
    fasta_file : str
        Path to reference genome FASTA file
    output_file : str
        Path to output TSV file
    n_jobs : int
        Number of parallel jobs (default: -1 = all CPUs)
    modality : str
        Modality name (default: 'K27Ac')
    base_dir : str
        Base directory containing .pt attribution files
    """
    # Read ABC file
    print(f"Reading ABC file: {abc_file}")
    abc = pd.read_csv(abc_file, sep='\t')
    print(f"Found {len(abc)} ABC connections")

    # Add pt_file column to track which file each row uses
    # File pattern: {chrom}_{start}_{end}_MiniAtlas-{celltype}_{modality}_random.pt
    # Window is [start.y - 511, start.y + 512] = 1024bp centered on start.y
    abc['pt_file'] = abc.apply(
        lambda row: os.path.join(
            base_dir,
            f"{row['chrom.x']}_{row['start.y']-511}_{row['start.y']+512}_MiniAtlas-{row['CellType']}_{modality}_random.pt"
        ),
        axis=1
    )

    # Group by pt_file to process each file only once
    grouped = abc.groupby('pt_file')
    print(f"Modality: {modality}")
    print(f"Found {len(grouped)} unique .pt files to process")
    print(f"Average {len(abc) / len(grouped):.1f} ABC connections per file")

    # Initialize result columns
    abc['max_score'] = np.nan
    abc['sum_score'] = np.nan
    abc['min_score'] = np.nan
    abc['file_found'] = False
    abc['pt_filename'] = ''

    # Process files in parallel
    print(f"Processing .pt files in parallel with {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    # Use joblib to process file groups in parallel
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_pt_file_group)(pt_file, group_df, fasta_file)
        for pt_file, group_df in grouped
    )

    # Flatten results and update dataframe
    print("Updating results...")
    files_found = 0
    for file_results in all_results:
        if any(result[4] for result in file_results):  # Check if any file_found is True
            files_found += 1
        for idx, max_score, sum_score, min_score, file_found, filename in file_results:
            abc.at[idx, 'max_score'] = max_score
            abc.at[idx, 'sum_score'] = sum_score
            abc.at[idx, 'min_score'] = min_score
            abc.at[idx, 'file_found'] = file_found
            abc.at[idx, 'pt_filename'] = filename

    # Remove the temporary pt_file column
    abc = abc.drop(columns=['pt_file'])

    # Save results
    abc.to_csv(output_file, sep='\t', index=False)

    # Print summary
    n_found = abc['file_found'].sum()
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total ABC connections: {len(abc)}")
    print(f"  Unique .pt files: {len(grouped)}")
    print(f"  Files found: {files_found} ({files_found/len(grouped)*100:.1f}%)")
    print(f"  Connections processed: {n_found} ({n_found/len(abc)*100:.1f}%)")
    print(f"  Connections not found: {len(abc)-n_found} ({(len(abc)-n_found)/len(abc)*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print statistics for successfully processed connections
    if n_found > 0:
        valid_data = abc[abc['file_found']]
        print("\nAttribution score statistics (for found files):")
        print(f"  Max score - mean: {valid_data['max_score'].mean():.4f}, std: {valid_data['max_score'].std():.4f}")
        print(f"  Sum score - mean: {valid_data['sum_score'].mean():.4f}, std: {valid_data['sum_score'].std():.4f}")
        print(f"  Min score - mean: {valid_data['min_score'].mean():.4f}, std: {valid_data['min_score'].std():.4f}")

        print("\nTop 10 connections by max score:")
        top_max = valid_data.nlargest(10, 'max_score')
        print(top_max[['chrom.x', 'start.x', 'end.x', 'CellType', 'max_score', 'sum_score', 'min_score', 'pt_filename']].to_string(index=False))


@click.command()
@click.option('-a', '--abc_file', type=str, default='Data/source/ABC/H3K27ac_abc_filtcelltype_conns.txt',
              help='ABC connections file (default: Data/source/ABC/H3K27ac_abc_filtcelltype_conns.txt)')
@click.option('-f', '--fasta', 'fasta_file', required=True, type=str, help='Reference genome FASTA file')
@click.option('-o', '--output', type=str, required=True, help='Output TSV file')
@click.option('-j', '--n_jobs', type=int, default=-1,
              help='Number of parallel jobs (default: -1 = all CPUs)')
@click.option('-m', '--modality', type=str, default='K27Ac',
              help='Modality name for .pt files (default: K27Ac)')
@click.option('-b', '--base_dir', type=str,
              default='Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp',
              help='Base directory containing .pt attribution files')
def main(abc_file, fasta_file, output, n_jobs, modality, base_dir):
    """Extract attribution scores from .pt files for ABC enhancer-gene connections."""
    process_abc_file(abc_file, fasta_file, output, n_jobs, modality, base_dir)


if __name__ == '__main__':
    main()
