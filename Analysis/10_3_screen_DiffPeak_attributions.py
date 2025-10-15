#!/usr/bin/env python
"""
Extract attribution scores from .pt files for all peaks overlapping with attribution windows.

This script creates a unified "non_diffpeak" dataframe containing all ATAC peaks that overlap
with .pt files, along with their attribution scores and distance to TSS. Optionally annotates
peaks with DiffPeak information to distinguish differential peaks from non-differential peaks.

Workflow:
1. Scans all .pt files in the base directory
2. For each file, finds overlapping ATAC peaks from the provided BED file
3. Calculates:
   - 6 attribution scores per peak: max_score, sum_score, min_score (raw and binned)
   - Distance from peak center to TSS (extracted from .pt filename)
4. Creates the "non_diffpeak" dataframe with all results
5. (Optional) Annotates with DiffPeak info using --diffpeak_file

Output columns:
- pt_file, chr, tss, celltype, modality
- peak_chr, peak_start, peak_end, peak_center, distance_to_tss
- max_score, sum_score, min_score (raw attribution scores)
- max_bin_score, sum_bin_score, min_bin_score (binned attribution scores)
- is_diffpeak, diffpeak_gene, diffpeak_padj, diffpeak_log2fc (if DiffPeak annotation provided)

Usage:
    # Basic usage - all peaks
    python 10_3_screen_DiffPeak_attributions.py -f genome.fa -o results.tsv

    # With DiffPeak annotation
    python 10_3_screen_DiffPeak_attributions.py -f genome.fa -o results.tsv \\
        --diffpeak_file Data/source/DiffPeak/DiffPeak.overlap.DiffTss.csv
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
from glob import glob

# Add Model directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))

from data.tokenizer import FastaInterval


def load_and_process_pt_file(pt_file, fasta_file, window_size=32):
    """
    Load a .pt file and return processed attribution scores.

    Returns:
    --------
    tuple : (file_chrom, context_start, raw_scores, bin_starts, pooled_scores) or None if error
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

        # Sum across nucleotides to get raw scores
        raw_scores = np.sum(attribution_weighted, axis=1)

        # Pool into bins
        n_bins = len(raw_scores) // window_size
        pooled_scores = raw_scores[:n_bins * window_size].reshape(n_bins, window_size).mean(axis=1)

        # Calculate bin positions
        bin_starts = context_start + np.arange(n_bins) * window_size

        return file_chrom, context_start, raw_scores, bin_starts, pooled_scores

    except Exception as e:
        print(f"Error loading {pt_file}: {e}")
        return None


def extract_raw_scores(context_start, raw_scores, enhancer_start, enhancer_end):
    """
    Extract max, sum, min scores for a given enhancer region from raw (non-binned) scores.
    """
    # Calculate indices in the raw scores array
    start_idx = enhancer_start - context_start
    end_idx = enhancer_end - context_start

    # Ensure indices are within bounds
    if start_idx < 0 or end_idx > len(raw_scores) or start_idx >= end_idx:
        return None

    enhancer_scores = raw_scores[start_idx:end_idx]

    return {
        'max_score': float(np.max(enhancer_scores)),
        'sum_score': float(np.sum(enhancer_scores)),
        'min_score': float(np.min(enhancer_scores)),
    }


def extract_binned_scores(bin_starts, pooled_scores, enhancer_start, enhancer_end):
    """
    Extract max, sum, min scores for a given enhancer region from binned scores.
    """
    # Find bins that overlap with the enhancer region
    enhancer_mask = (bin_starts >= enhancer_start) & (bin_starts < enhancer_end)

    if not np.any(enhancer_mask):
        return None

    enhancer_scores = pooled_scores[enhancer_mask]

    return {
        'max_bin_score': float(np.max(enhancer_scores)),
        'sum_bin_score': float(np.sum(enhancer_scores)),
        'min_bin_score': float(np.min(enhancer_scores)),
    }


def process_single_pt_file_for_peaks(pt_file, peaks_df, fasta_file):
    """
    Process a single .pt file and extract scores for all overlapping peaks.

    Parameters:
    -----------
    pt_file : str
        Path to .pt file
    peaks_df : pd.DataFrame
        DataFrame containing all peaks (chr, start, end)
    fasta_file : str
        Path to reference genome FASTA file

    Returns:
    --------
    list : List of dicts containing peak results
    """
    results = []
    filename = os.path.basename(pt_file)

    try:
        # Parse filename to extract TSS location
        # Pattern: {chrom}_{start}_{end}_MiniAtlas-{celltype}_{modality}_all_random.pt
        match = re.match(r'(chr\w+)_(\d+)_(\d+)_MiniAtlas-(.+?)_(\w+)_all_random\.pt', filename)
        if not match:
            return results

        file_chrom = match.group(1)
        file_start = int(match.group(2))
        file_end = int(match.group(3))
        celltype = match.group(4)
        modality = match.group(5)

        # TSS is at the midpoint
        tss = (file_start + file_end) // 2

        # Load and process the .pt file
        processed_data = load_and_process_pt_file(pt_file, fasta_file)
        if processed_data is None:
            return results

        _file_chrom, context_start, raw_scores, bin_starts, pooled_scores = processed_data

        # Filter peaks to those on the same chromosome
        chrom_peaks = peaks_df[peaks_df['chr'] == file_chrom].copy()

        if len(chrom_peaks) == 0:
            return results

        # Find overlapping peaks
        # A peak overlaps if it intersects with [context_start, context_start + len(raw_scores)]
        context_end = context_start + len(raw_scores)

        for _, peak in chrom_peaks.iterrows():
            peak_start = peak['start']
            peak_end = peak['end']

            # Check for overlap
            if peak_end <= context_start or peak_start >= context_end:
                continue  # No overlap

            # Calculate peak center
            peak_center = (peak_start + peak_end) // 2

            # Calculate distance from peak center to TSS
            distance_to_tss = abs(peak_center - tss)

            # Extract raw scores
            raw_result = extract_raw_scores(
                context_start, raw_scores, peak_start, peak_end
            )

            # Extract binned scores
            binned_result = extract_binned_scores(
                bin_starts, pooled_scores, peak_start, peak_end
            )

            if raw_result is not None and binned_result is not None:
                result = {
                    'pt_file': filename,
                    'chr': file_chrom,
                    'tss_start': file_start,
                    'tss_end': file_end,
                    'tss': tss,
                    'celltype': celltype,
                    'modality': modality,
                    'peak_chr': peak['chr'],
                    'peak_start': peak_start,
                    'peak_end': peak_end,
                    'peak_center': peak_center,
                    'distance_to_tss': distance_to_tss,
                    'max_score': raw_result['max_score'],
                    'sum_score': raw_result['sum_score'],
                    'min_score': raw_result['min_score'],
                    'max_bin_score': binned_result['max_bin_score'],
                    'sum_bin_score': binned_result['sum_bin_score'],
                    'min_bin_score': binned_result['min_bin_score']
                }

                # Add peak name and score if available
                if 'name' in peak:
                    result['peak_name'] = peak['name']
                if 'score' in peak:
                    result['peak_score'] = peak['score']

                results.append(result)

    except Exception as e:
        print(f"Error processing file {pt_file}: {e}")

    return results


def _process_match_key_group_diffpeak(match_key, non_diffpeak_subset, diffpeak_rows_data):
    """
    Process a single match_key group to find DiffPeak annotations.

    Parameters:
    -----------
    match_key : str
        The match key for this group
    non_diffpeak_subset : pd.DataFrame
        Subset of non_diffpeak_df with this match_key
    diffpeak_rows_data : list of dict
        DiffPeak row data as dictionaries

    Returns:
    --------
    list : List of tuples (idx, is_diffpeak, gene, padj, log2fc, celltype)
    """
    results = []

    # Extract celltype from match_key (format: chr_start_end_celltype)
    celltype = match_key.rsplit('_', 1)[-1]

    # For each DiffPeak, find overlapping peaks
    for diffpeak_row in diffpeak_rows_data:
        # Check for peak overlap with DiffPeak [start_peak, end_peak]
        overlap_mask = (
            (non_diffpeak_subset['peak_chr'] == diffpeak_row['Chromosome']) &
            (non_diffpeak_subset['peak_start'] < diffpeak_row['end_peak']) &
            (non_diffpeak_subset['peak_end'] > diffpeak_row['start_peak'])
        )

        # Get matching indices
        matching_indices = non_diffpeak_subset.index[overlap_mask].tolist()

        for idx in matching_indices:
            results.append((
                idx,
                True,  # is_diffpeak
                diffpeak_row.get('tss_gene', ''),
                diffpeak_row.get('padj', np.nan),
                diffpeak_row.get('log2FoldChange', np.nan),
                celltype
            ))

    return results


def _process_match_key_group_diffpeak_direct(match_key, diffpeak_grouped, non_diffpeak_grouped, cols_to_extract):
    """
    Process a single match_key group to find DiffPeak annotations.
    Does data preparation and processing in one step.

    Parameters:
    -----------
    match_key : str
        The match key for this group
    diffpeak_grouped : pd.DataFrameGroupBy
        Grouped DiffPeak dataframe
    non_diffpeak_grouped : pd.DataFrameGroupBy
        Grouped non_diffpeak dataframe
    cols_to_extract : list
        Columns to extract from DiffPeak rows

    Returns:
    --------
    list : List of tuples (idx, is_diffpeak, gene, padj, log2fc, celltype)
    """
    # Get the groups for this match_key
    diffpeak_rows = diffpeak_grouped.get_group(match_key)
    non_diffpeak_subset = non_diffpeak_grouped.get_group(match_key)

    # Convert DiffPeak rows to list of dicts
    diffpeak_rows_data = diffpeak_rows[cols_to_extract].to_dict('records')

    # Call the original processing function
    return _process_match_key_group_diffpeak(match_key, non_diffpeak_subset, diffpeak_rows_data)


def annotate_with_diffpeak(non_diffpeak_df, diffpeak_file, n_jobs=-1):
    """
    Annotate the non_diffpeak dataframe with DiffPeak information.

    Matches peaks in non_diffpeak with differential peaks by:
    1. Matching chr, tss coordinates, and celltype
    2. Checking for peak coordinate overlap
    3. Using parallel processing for efficiency

    Parameters:
    -----------
    non_diffpeak_df : pd.DataFrame
        DataFrame from process_all_peaks() containing peak-level attribution scores
    diffpeak_file : str
        Path to DiffPeak.overlap.DiffTss.csv file
    n_jobs : int
        Number of parallel jobs (default: -1 = all CPUs)

    Returns:
    --------
    pd.DataFrame : non_diffpeak_df with additional DiffPeak annotation columns
    """
    print(f"\nAnnotating with DiffPeak data from: {diffpeak_file}")
    diffpeak = pd.read_csv(diffpeak_file)
    print(f"Loaded {len(diffpeak)} differential peaks")

    # Create match key from DiffPeak data
    # The .pt filename uses: {Chromosome}_{start_tss}_{end_tss}_MiniAtlas-{tss_celltype}_{modality}_all_random.pt
    diffpeak['ID'] = diffpeak['tss_celltype'] + ":" + diffpeak['Chromosome'] + ':' + \
        diffpeak['start_tss'].astype(int).astype(str) + '-' + diffpeak['end_tss'].astype(int).astype(str) + ":" + \
        diffpeak['peak_start'].astype(int).astype(str) + '-' + diffpeak['peak_end'].astype(int).astype(str)

    # Create match key in non_diffpeak_df
    non_diffpeak_df['ID'] = non_diffpeak_df['celltype'] + ":" + non_diffpeak_df['chr'] + ':' + \
        non_diffpeak_df['tss_start'].astype(int).astype(str) + '-' + non_diffpeak_df['tss_end'].astype(int).astype(str) + ":" + \
        non_diffpeak_df['peak_start'].astype(int).astype(str) + '-' + non_diffpeak_df['peak_end'].astype(int).astype(str)

    # drop duplicats in diffpeak
    diffpeak = diffpeak.drop_duplicates(subset='ID', keep='first')
    diffpeak = diffpeak.set_index('ID')
    
    # Set index for non_abc_df and remove any duplicate indices (keep first)
    non_diffpeak_df = non_diffpeak_df.set_index('ID')
    if non_diffpeak_df.index.duplicated().any():
        print(f"Warning: Found {non_diffpeak_df.index.duplicated().sum()} duplicate IDs in non_diffpeak_df, keeping first occurrence")
        non_diffpeak_df = non_diffpeak_df[~non_diffpeak_df.index.duplicated(keep='first')]
    
    # directly map by index
    non_diffpeak_df['is_diffpeak'] = non_diffpeak_df.index.isin(diffpeak.index)
    non_diffpeak_df['diffpeak_gene'] = diffpeak['tss_gene'].reindex(non_diffpeak_df.index).values
    non_diffpeak_df['diffpeak_padj'] = diffpeak['peak_adjusted p-value'].reindex(non_diffpeak_df.index).values
    non_diffpeak_df['diffpeak_log2fc'] = diffpeak['peak_log2(fold_change)'].reindex(non_diffpeak_df.index).values
    non_diffpeak_df['diffpeak_celltype'] = diffpeak['tss_celltype'].reindex(non_diffpeak_df.index).values

    return non_diffpeak_df


def process_all_peaks(bed_file, fasta_file, output_file, n_jobs=-1,
                      base_dir="Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp_diff",
                      diffpeak_file=None, test=False, modality=None):
    """
    Process all .pt files and extract attribution scores for overlapping peaks.
    Creates the "non_diffpeak" dataframe containing all peaks.
    Optionally annotates with DiffPeak information.

    Parameters:
    -----------
    bed_file : str
        Path to BED file with peaks (e.g., merged_all_peaks.bed)
    fasta_file : str
        Path to reference genome FASTA file
    output_file : str
        Path to output TSV file
    n_jobs : int
        Number of parallel jobs (default: -1 = all CPUs)
    base_dir : str
        Base directory containing .pt attribution files
    diffpeak_file : str, optional
        Path to DiffPeak.overlap.DiffTss.csv file for annotation
    test : bool, optional
        If True, only process first 1000 .pt files (default: False)
    modality : str, optional
        Filter .pt files by modality (e.g., 'ATAC', 'K27Ac'). If None, process all modalities.
    """
    # Load peaks BED file
    print(f"Loading peaks from: {bed_file}")
    peaks = pd.read_csv(bed_file, sep='\t', header=None)

    # Assign column names based on number of columns
    if len(peaks.columns) >= 5:
        peaks.columns = ['chr', 'start', 'end', 'name', 'score'] + [f'col{i}' for i in range(5, len(peaks.columns))]
    elif len(peaks.columns) >= 4:
        peaks.columns = ['chr', 'start', 'end', 'name'] + [f'col{i}' for i in range(4, len(peaks.columns))]
    else:
        peaks.columns = ['chr', 'start', 'end'] + [f'col{i}' for i in range(3, len(peaks.columns))]

    # Ensure chr is string type
    peaks['chr'] = peaks['chr'].astype(str)
    print(f"Loaded {len(peaks)} peaks")

    # Find all .pt files in base directory
    print(f"Searching for .pt files in: {base_dir}")
    pt_files = glob(os.path.join(base_dir, "*_all_random.pt"))
    print(f"Found {len(pt_files)} .pt files (all modalities)")

    # Filter by modality if specified
    if modality:
        # Filter files that contain the modality in their filename
        # Filename pattern: {chrom}_{start}_{end}_MiniAtlas-{celltype}_{modality}_all_random.pt
        pt_files = [f for f in pt_files if f"_{modality}_all_random.pt" in f]
        print(f"Filtered to {len(pt_files)} .pt files with modality '{modality}'")

    # Limit to first 1000 files if test mode
    if test and len(pt_files) > 1000:
        print(f"TEST MODE: Limiting to first 1000 .pt files")
        pt_files = pt_files[:1000]

    if len(pt_files) == 0:
        print("Warning: No .pt files found!")
        return

    # Process files in parallel
    print(f"Processing .pt files in parallel with {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    # all_results = Parallel(
    #     n_jobs=n_jobs,
    #     verbose=10,
    #     backend='loky',
    #     timeout=300  # 5 minute timeout per file
    # )(
    #     delayed(process_single_pt_file_for_peaks)(pt_file, peaks, fasta_file)
    #     for pt_file in pt_files
    # )

    # # Flatten results
    # print("Aggregating results...")
    # flat_results = []
    # for file_results in all_results:
    #     flat_results.extend(file_results)

    # # Create dataframe
    # non_diffpeak = pd.DataFrame(flat_results)

    # # Save results
    # non_diffpeak.to_csv(output_file, sep='\t', index=False)

    non_diffpeak = pd.read_csv(output_file, sep='\t')

    if len(non_diffpeak) == 0:
        print("Warning: No overlapping peaks found!")
        return non_diffpeak

    # Optionally annotate with DiffPeak data
    if diffpeak_file:
        non_diffpeak = annotate_with_diffpeak(non_diffpeak, diffpeak_file, n_jobs=n_jobs)

    # Save results
    non_diffpeak.to_csv(output_file, sep='\t', index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total .pt files processed: {len(pt_files)}")
    print(f"  Total peaks: {len(peaks)}")
    print(f"  Total overlapping peak-file pairs: {len(non_diffpeak)}")
    if len(non_diffpeak) > 0:
        print(f"  Unique peaks with overlaps: {non_diffpeak[['peak_chr', 'peak_start', 'peak_end']].drop_duplicates().shape[0]}")
        print(f"  Unique .pt files with overlaps: {non_diffpeak['pt_file'].nunique()}")
        if 'is_diffpeak' in non_diffpeak.columns:
            print(f"\n  DiffPeak annotation:")
            print(f"    DiffPeak peaks: {non_diffpeak['is_diffpeak'].sum()} ({non_diffpeak['is_diffpeak'].sum()/len(non_diffpeak)*100:.1f}%)")
            print(f"    Non-DiffPeak peaks: {(~non_diffpeak['is_diffpeak']).sum()} ({(~non_diffpeak['is_diffpeak']).sum()/len(non_diffpeak)*100:.1f}%)")
        print(f"\n  Distance to TSS statistics:")
        print(f"    Mean: {non_diffpeak['distance_to_tss'].mean():.1f} bp")
        print(f"    Median: {non_diffpeak['distance_to_tss'].median():.1f} bp")
        print(f"    Min: {non_diffpeak['distance_to_tss'].min()} bp")
        print(f"    Max: {non_diffpeak['distance_to_tss'].max()} bp")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print statistics for attribution scores
    if len(non_diffpeak) > 0:
        print("\nRaw attribution score statistics:")
        print(f"  Max score - mean: {non_diffpeak['max_score'].mean():.4f}, std: {non_diffpeak['max_score'].std():.4f}")
        print(f"  Sum score - mean: {non_diffpeak['sum_score'].mean():.4f}, std: {non_diffpeak['sum_score'].std():.4f}")
        print(f"  Min score - mean: {non_diffpeak['min_score'].mean():.4f}, std: {non_diffpeak['min_score'].std():.4f}")

        print("\nBinned attribution score statistics:")
        print(f"  Max bin score - mean: {non_diffpeak['max_bin_score'].mean():.4f}, std: {non_diffpeak['max_bin_score'].std():.4f}")
        print(f"  Sum bin score - mean: {non_diffpeak['sum_bin_score'].mean():.4f}, std: {non_diffpeak['sum_bin_score'].std():.4f}")
        print(f"  Min bin score - mean: {non_diffpeak['min_bin_score'].mean():.4f}, std: {non_diffpeak['min_bin_score'].std():.4f}")

        print("\nTop 10 peaks by max raw score:")
        top_max = non_diffpeak.nlargest(10, 'max_score')
        print(top_max[['peak_chr', 'peak_start', 'peak_end', 'celltype', 'modality', 'distance_to_tss',
                       'max_score', 'sum_score', 'min_score', 'max_bin_score', 'sum_bin_score', 'min_bin_score']].to_string(index=False))

    return non_diffpeak


@click.command()
@click.option('-d', '--diffpeak_file', type=str, default=None,
              help='DiffPeak.overlap.DiffTss.csv file for annotation (optional)')
@click.option('-f', '--fasta', 'fasta_file', required=True, type=str, help='Reference genome FASTA file')
@click.option('-o', '--output', type=str, required=True, help='Output TSV file')
@click.option('-j', '--n_jobs', type=int, default=-1,
              help='Number of parallel jobs (default: -1 = all CPUs)')
@click.option('-b', '--base_dir', type=str,
              default='Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp_diff',
              help='Base directory containing .pt attribution files')
@click.option('--bed_file', type=str, default='Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed',
              help='BED file with peaks (default: Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed)')
@click.option('-m', '--modality', type=str, default=None,
              help='Filter by modality (e.g., ATAC, K27Ac). If not specified, process all modalities.')
@click.option('--test', is_flag=True, default=False,
              help='Test mode: only process first 1000 .pt files')
def main(diffpeak_file, fasta_file, output, n_jobs, base_dir, bed_file, modality, test):
    """
    Extract attribution scores from .pt files for all peaks overlapping with attribution windows.

    This script creates a unified "non_diffpeak" dataframe containing all ATAC peaks that overlap
    with .pt files, along with their attribution scores and distance to TSS.

    If --diffpeak_file is provided, the script will annotate peaks to distinguish differential peaks
    from non-differential peaks.

    Examples:

        # Basic usage - all peaks without DiffPeak annotation
        python 10_3_screen_DiffPeak_attributions.py \\
            -f genome.fa -o results.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed

        # Filter by modality (ATAC only)
        python 10_3_screen_DiffPeak_attributions.py \\
            -f genome.fa -o results_atac.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            -m ATAC

        # With DiffPeak annotation
        python 10_3_screen_DiffPeak_attributions.py \\
            -f genome.fa -o results.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            --diffpeak_file Data/source/DiffPeak/DiffPeak.overlap.DiffTss.csv

        # Test mode - only process first 1000 .pt files
        python 10_3_screen_DiffPeak_attributions.py \\
            -f genome.fa -o results_test.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            --test
    """
    process_all_peaks(bed_file, fasta_file, output, n_jobs, base_dir, diffpeak_file, test, modality)


if __name__ == '__main__':
    main()
