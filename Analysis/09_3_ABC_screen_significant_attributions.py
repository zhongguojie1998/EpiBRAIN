#!/usr/bin/env python
"""
Extract attribution scores from .pt files for all peaks overlapping with attribution windows.

This script creates a unified "non_abc" dataframe containing all ATAC peaks that overlap
with .pt files, along with their attribution scores and distance to TSS. Optionally annotates
peaks with ABC connection information to distinguish ABC enhancers from non-ABC peaks.

Workflow:
1. Scans all .pt files in the base directory
2. For each file, finds overlapping ATAC peaks from the provided BED file
3. Calculates:
   - 6 attribution scores per peak: max_score, sum_score, min_score (raw and binned)
   - Distance from peak center to TSS (extracted from .pt filename)
4. Creates the "non_abc" dataframe with all results
5. (Optional) Annotates with ABC connection info using --abc_file

Output columns:
- pt_file, chr, tss, celltype, modality
- peak_chr, peak_start, peak_end, peak_center, distance_to_tss
- max_score, sum_score, min_score (raw attribution scores)
- max_bin_score, sum_bin_score, min_bin_score (binned attribution scores)
- is_abc, abc_gene, abc_score, abc_class (if ABC annotation provided)

Usage:
    # Basic usage - all peaks
    python 09_3_ABC_screen_significant_attributions.py -f genome.fa -o results.tsv

    # With ABC annotation
    python 09_3_ABC_screen_significant_attributions.py -f genome.fa -o results.tsv \\
        --abc_file Data/source/ABC/H3K27ac_abc_filtcelltype_conns.txt
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
    list : List of tuples (idx, max_score, sum_score, min_score, max_bin_score, sum_bin_score, min_bin_score, file_found, filename)
    """
    results = []
    filename = os.path.basename(pt_file)

    try:
        # Load and process the .pt file once
        processed_data = load_and_process_pt_file(pt_file, fasta_file)

        if processed_data is None:
            # File not found or error - mark all rows as not found
            for idx, row in group_df.iterrows():
                results.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, ''))
            return results

        _file_chrom, context_start, raw_scores, bin_starts, pooled_scores = processed_data

        # Extract scores for all enhancers in this file
        for idx, row in group_df.iterrows():
            try:
                # Extract raw scores
                raw_result = extract_raw_scores(
                    context_start, raw_scores, row['start.x'], row['end.x']
                )

                # Extract binned scores
                binned_result = extract_binned_scores(
                    bin_starts, pooled_scores, row['start.x'], row['end.x']
                )

                if raw_result is not None and binned_result is not None:
                    results.append((
                        idx,
                        raw_result['max_score'],
                        raw_result['sum_score'],
                        raw_result['min_score'],
                        binned_result['max_bin_score'],
                        binned_result['sum_bin_score'],
                        binned_result['min_bin_score'],
                        True,
                        filename
                    ))
                else:
                    results.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, ''))
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, ''))

    except Exception as e:
        print(f"Error processing file {pt_file}: {e}")
        # Mark all rows as not found
        for idx, row in group_df.iterrows():
            results.append((idx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, ''))

    return results


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
        # Pattern: {chrom}_{start}_{end}_MiniAtlas-{celltype}_{modality}_random.pt
        match = re.match(r'(chr\w+)_(\d+)_(\d+)_MiniAtlas-(.+?)_(\w+)_random\.pt', filename)
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


def annotate_with_abc(non_abc_df, abc_file, n_jobs=-1):
    """
    Annotate the non_abc dataframe with ABC connection information.

    Matches peaks in non_abc with ABC enhancers by:
    1. Matching chr, tss coordinates, celltype, and peak coordinates
    2. Using direct pandas index operations for efficiency

    This uses a fast index-based matching approach similar to annotate_with_diffpeak()
    in 10_3_screen_DiffPeak_attributions.py. By creating unique IDs that include all
    matching criteria (celltype, chr, tss, and peak coordinates), we can use pandas'
    optimized index operations (.isin() and .reindex()) instead of parallel processing
    with overlap checks. This is much faster for large datasets.

    Parameters:
    -----------
    non_abc_df : pd.DataFrame
        DataFrame from process_all_peaks() containing peak-level attribution scores
    abc_file : str
        Path to ABC connections file
    n_jobs : int
        Number of parallel jobs (default: -1 = all CPUs, currently unused but kept for API compatibility)

    Returns:
    --------
    pd.DataFrame : non_abc_df with additional ABC annotation columns
    """
    print(f"\nAnnotating with ABC connections from: {abc_file}")
    abc = pd.read_csv(abc_file, sep='\t')
    print(f"Loaded {len(abc)} ABC connections")

    # Create ID from ABC data
    # ID includes: celltype, chr, tss coordinates, and peak coordinates
    # The .pt filename uses: {chrom}_{start.y-511}_{start.y+512}_MiniAtlas-{CellType}_{modality}_random.pt
    abc['tss_start'] = abc['start.y'] - 511
    abc['tss_end'] = abc['start.y'] + 512
    abc['ID'] = abc['CellType'] + ":" + abc['chrom.x'] + ':' + \
        abc['tss_start'].astype(int).astype(str) + '-' + abc['tss_end'].astype(int).astype(str) + ":" + \
        abc['start.x'].astype(int).astype(str) + '-' + abc['end.x'].astype(int).astype(str)

    # Create ID in non_abc_df
    # Match format: celltype:chr:tss_start-tss_end:peak_start-peak_end
    non_abc_df['ID'] = non_abc_df['celltype'] + ":" + non_abc_df['chr'] + ':' + \
        (non_abc_df['tss'] - 511).astype(int).astype(str) + '-' + \
        (non_abc_df['tss'] + 512).astype(int).astype(str) + ":" + \
        non_abc_df['peak_start'].astype(int).astype(str) + '-' + \
        non_abc_df['peak_end'].astype(int).astype(str)

    # Drop duplicates in ABC based on ID (keep first occurrence)
    # Use drop_duplicates on the ID column BEFORE setting it as index to avoid duplicate index values
    abc = abc.drop_duplicates(subset='ID', keep='first')
    abc = abc.set_index('ID')

    # Set index for non_abc_df and remove any duplicate indices (keep first)
    non_abc_df = non_abc_df.set_index('ID')
    if non_abc_df.index.duplicated().any():
        print(f"Warning: Found {non_abc_df.index.duplicated().sum()} duplicate IDs in non_abc_df, keeping first occurrence")
        non_abc_df = non_abc_df[~non_abc_df.index.duplicated(keep='first')]

    # Directly map by index using pandas operations
    print("Mapping ABC annotations using direct index matching...")
    non_abc_df['is_abc'] = non_abc_df.index.isin(abc.index)
    non_abc_df['abc_gene'] = abc['TargetGene'].reindex(non_abc_df.index).values if 'TargetGene' in abc.columns else ''
    non_abc_df['abc_score'] = abc['ABC.Score'].reindex(non_abc_df.index).values if 'ABC.Score' in abc.columns else np.nan
    non_abc_df['abc_class'] = abc['CellType'].reindex(non_abc_df.index).values if 'CellType' in abc.columns else ''

    # Reset index to restore original dataframe structure
    non_abc_df = non_abc_df.reset_index(drop=True)

    matched_count = non_abc_df['is_abc'].sum()
    print(f"Matched {matched_count} peak-file pairs to ABC connections ({matched_count/len(non_abc_df)*100:.1f}%)")
    print(f"ABC peaks: {non_abc_df['is_abc'].sum()}, Non-ABC peaks: {(~non_abc_df['is_abc']).sum()}")

    return non_abc_df


def process_all_peaks(bed_file, fasta_file, output_file, n_jobs=-1, base_dir="Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp", abc_file=None, test=False, modality=None):
    """
    Process all .pt files and extract attribution scores for overlapping peaks.
    Creates the "non_abc" dataframe containing all peaks.
    Optionally annotates with ABC connection information.

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
    abc_file : str, optional
        Path to ABC connections file for annotation
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
    pt_files = glob(os.path.join(base_dir, "*.pt"))
    print(f"Found {len(pt_files)} .pt files (all modalities)")

    # Filter by modality if specified
    if modality:
        # Filter files that contain the modality in their filename
        # Filename pattern: {chrom}_{start}_{end}_MiniAtlas-{celltype}_{modality}_random.pt
        pt_files = [f for f in pt_files if f"_{modality}_random.pt" in f]
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

    all_results = Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend='loky',
        timeout=300  # 5 minute timeout per file
    )(
        delayed(process_single_pt_file_for_peaks)(pt_file, peaks, fasta_file)
        for pt_file in pt_files
    )

    # Flatten results
    print("Aggregating results...")
    flat_results = []
    for file_results in all_results:
        flat_results.extend(file_results)

    # Create dataframe
    non_abc = pd.DataFrame(flat_results)

    # Save results
    non_abc.to_csv(output_file, sep='\t', index=False)
    
    if len(non_abc) == 0:
        print("Warning: No overlapping peaks found!")
        return non_abc

    # Optionally annotate with ABC connections
    if abc_file:
        non_abc = annotate_with_abc(non_abc, abc_file, n_jobs=n_jobs)

    # Save results
    non_abc.to_csv(output_file, sep='\t', index=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total .pt files processed: {len(pt_files)}")
    print(f"  Total peaks: {len(peaks)}")
    print(f"  Total overlapping peak-file pairs: {len(non_abc)}")
    if len(non_abc) > 0:
        print(f"  Unique peaks with overlaps: {non_abc[['peak_chr', 'peak_start', 'peak_end']].drop_duplicates().shape[0]}")
        print(f"  Unique .pt files with overlaps: {non_abc['pt_file'].nunique()}")
        if 'is_abc' in non_abc.columns:
            print(f"\n  ABC annotation:")
            print(f"    ABC peaks: {non_abc['is_abc'].sum()} ({non_abc['is_abc'].sum()/len(non_abc)*100:.1f}%)")
            print(f"    Non-ABC peaks: {(~non_abc['is_abc']).sum()} ({(~non_abc['is_abc']).sum()/len(non_abc)*100:.1f}%)")
        print(f"\n  Distance to TSS statistics:")
        print(f"    Mean: {non_abc['distance_to_tss'].mean():.1f} bp")
        print(f"    Median: {non_abc['distance_to_tss'].median():.1f} bp")
        print(f"    Min: {non_abc['distance_to_tss'].min()} bp")
        print(f"    Max: {non_abc['distance_to_tss'].max()} bp")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print statistics for attribution scores
    if len(non_abc) > 0:
        print("\nRaw attribution score statistics:")
        print(f"  Max score - mean: {non_abc['max_score'].mean():.4f}, std: {non_abc['max_score'].std():.4f}")
        print(f"  Sum score - mean: {non_abc['sum_score'].mean():.4f}, std: {non_abc['sum_score'].std():.4f}")
        print(f"  Min score - mean: {non_abc['min_score'].mean():.4f}, std: {non_abc['min_score'].std():.4f}")

        print("\nBinned attribution score statistics:")
        print(f"  Max bin score - mean: {non_abc['max_bin_score'].mean():.4f}, std: {non_abc['max_bin_score'].std():.4f}")
        print(f"  Sum bin score - mean: {non_abc['sum_bin_score'].mean():.4f}, std: {non_abc['sum_bin_score'].std():.4f}")
        print(f"  Min bin score - mean: {non_abc['min_bin_score'].mean():.4f}, std: {non_abc['min_bin_score'].std():.4f}")

        print("\nTop 10 peaks by max raw score:")
        top_max = non_abc.nlargest(10, 'max_score')
        print(top_max[['peak_chr', 'peak_start', 'peak_end', 'celltype', 'modality', 'distance_to_tss',
                       'max_score', 'sum_score', 'min_score', 'max_bin_score', 'sum_bin_score', 'min_bin_score']].to_string(index=False))

    return non_abc


def process_non_abc_peaks(bed_file, fasta_file, output_file, n_jobs=-1, base_dir="Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp", modality=None):
    """
    Deprecated: Use process_all_peaks() instead.
    """
    return process_all_peaks(bed_file, fasta_file, output_file, n_jobs, base_dir, abc_file=None, test=False, modality=modality)


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
    abc['max_bin_score'] = np.nan
    abc['sum_bin_score'] = np.nan
    abc['min_bin_score'] = np.nan
    abc['file_found'] = False
    abc['pt_filename'] = ''

    # Process files in parallel
    print(f"Processing .pt files in parallel with {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    # Use joblib to process file groups in parallel with memory management
    # Use loky backend with timeout to handle memory issues better
    all_results = Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend='loky',
        timeout=300  # 5 minute timeout per file
    )(
        delayed(process_single_pt_file_group)(pt_file, group_df, fasta_file)
        for pt_file, group_df in grouped
    )

    # Flatten results and update dataframe
    print("Updating results...")
    files_found = 0
    for file_results in all_results:
        if any(result[7] for result in file_results):  # Check if any file_found is True
            files_found += 1
        for idx, max_score, sum_score, min_score, max_bin_score, sum_bin_score, min_bin_score, file_found, filename in file_results:
            abc.at[idx, 'max_score'] = max_score
            abc.at[idx, 'sum_score'] = sum_score
            abc.at[idx, 'min_score'] = min_score
            abc.at[idx, 'max_bin_score'] = max_bin_score
            abc.at[idx, 'sum_bin_score'] = sum_bin_score
            abc.at[idx, 'min_bin_score'] = min_bin_score
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
        print("\nRaw attribution score statistics (for found files):")
        print(f"  Max score - mean: {valid_data['max_score'].mean():.4f}, std: {valid_data['max_score'].std():.4f}")
        print(f"  Sum score - mean: {valid_data['sum_score'].mean():.4f}, std: {valid_data['sum_score'].std():.4f}")
        print(f"  Min score - mean: {valid_data['min_score'].mean():.4f}, std: {valid_data['min_score'].std():.4f}")

        print("\nBinned attribution score statistics (for found files):")
        print(f"  Max bin score - mean: {valid_data['max_bin_score'].mean():.4f}, std: {valid_data['max_bin_score'].std():.4f}")
        print(f"  Sum bin score - mean: {valid_data['sum_bin_score'].mean():.4f}, std: {valid_data['sum_bin_score'].std():.4f}")
        print(f"  Min bin score - mean: {valid_data['min_bin_score'].mean():.4f}, std: {valid_data['min_bin_score'].std():.4f}")

        print("\nTop 10 connections by max raw score:")
        top_max = valid_data.nlargest(10, 'max_score')
        print(top_max[['chrom.x', 'start.x', 'end.x', 'CellType', 'max_score', 'sum_score', 'min_score',
                       'max_bin_score', 'sum_bin_score', 'min_bin_score', 'pt_filename']].to_string(index=False))


@click.command()
@click.option('-a', '--abc_file', type=str, default=None,
              help='ABC connections file for annotation (optional)')
@click.option('-f', '--fasta', 'fasta_file', required=True, type=str, help='Reference genome FASTA file')
@click.option('-o', '--output', type=str, required=True, help='Output TSV file')
@click.option('-j', '--n_jobs', type=int, default=-1,
              help='Number of parallel jobs (default: -1 = all CPUs)')
@click.option('-b', '--base_dir', type=str,
              default='Res/basal_ganglia_miniatlas_drop_celltype_v1/analysis_150/raw_data/interp',
              help='Base directory containing .pt attribution files')
@click.option('--bed_file', type=str, default='Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed',
              help='BED file with peaks (default: Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed)')
@click.option('-m', '--modality', type=str, default=None,
              help='Filter by modality (e.g., ATAC, K27Ac). If not specified, process all modalities.')
@click.option('--test', is_flag=True, default=False,
              help='Test mode: only process first 1000 .pt files')
def main(abc_file, fasta_file, output, n_jobs, base_dir, bed_file, modality, test):
    """
    Extract attribution scores from .pt files for all peaks overlapping with attribution windows.

    This script creates a unified "non_abc" dataframe containing all ATAC peaks that overlap
    with .pt files, along with their attribution scores and distance to TSS.

    If --abc_file is provided, the script will annotate peaks to distinguish ABC enhancers
    from non-ABC peaks.

    Examples:

        # Basic usage - all peaks without ABC annotation
        python 09_3_ABC_screen_significant_attributions.py \\
            -f genome.fa -o results.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed

        # Filter by modality (ATAC only)
        python 09_3_ABC_screen_significant_attributions.py \\
            -f genome.fa -o results_atac.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            -m ATAC

        # With ABC annotation and K27Ac modality
        python 09_3_ABC_screen_significant_attributions.py \\
            -f genome.fa -o results_k27ac.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            --abc_file Data/source/ABC/H3K27ac_abc_filtcelltype_conns.txt \\
            -m K27Ac

        # Test mode - only process first 1000 .pt files
        python 09_3_ABC_screen_significant_attributions.py \\
            -f genome.fa -o results_test.tsv \\
            --bed_file Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed \\
            --test
    """
    process_all_peaks(bed_file, fasta_file, output, n_jobs, base_dir, abc_file, test, modality)


if __name__ == '__main__':
    main()
