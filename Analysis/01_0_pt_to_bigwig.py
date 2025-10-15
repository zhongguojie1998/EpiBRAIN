#!/usr/bin/env python
"""
Convert model predictions from .pt files to BigWig format at 32bp resolution.

This script reads prediction files from the testing dataset (or train/valid) and
writes the predictions to BigWig format for visualization in genome browsers.

Usage:
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --chk 250 -s Test --res_base ./Res --log_base ./logs \
        -o ./bigwig_output --fasta /path/to/reference.fa \
        --trials "MiniAtlas-L56IT_K27Ac,MiniAtlas-MSN_ATAC"
"""

import os
import sys
import click
import torch
import pandas as pd
import pyBigWig
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# Add Model directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))


def get_chrom_sizes(fasta_file):
    """
    Get chromosome sizes from FASTA file.

    Parameters:
    -----------
    fasta_file : str
        Path to reference genome FASTA file

    Returns:
    --------
    dict : Dictionary mapping chromosome names to sizes
    """
    chrom_sizes = {}
    current_chrom = None
    current_size = 0

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Save previous chromosome
                if current_chrom is not None:
                    chrom_sizes[current_chrom] = current_size
                # Start new chromosome
                current_chrom = line.strip().split()[0][1:]  # Remove '>'
                current_size = 0
            else:
                current_size += len(line.strip())

        # Save last chromosome
        if current_chrom is not None:
            chrom_sizes[current_chrom] = current_size

    return chrom_sizes


def write_predictions_to_bigwig(predictions, regions_df, trial_idx, trial_name,
                                 output_dir, chrom_sizes, window_size=32):
    """
    Write predictions for a specific trial to BigWig files.

    Parameters:
    -----------
    predictions : torch.Tensor
        Predictions tensor of shape (n_samples, n_windows, n_trials)
    regions_df : pd.DataFrame
        DataFrame with columns ['chr', 'start', 'end', 'split'] containing genomic coordinates
    trial_idx : int
        Index of the trial to extract from predictions
    trial_name : str
        Name of the trial (used for output filename)
    output_dir : str
        Output directory for BigWig files
    chrom_sizes : dict
        Dictionary mapping chromosome names to sizes
    window_size : int
        Size of genomic bins in bp (default: 32)
    """
    # Extract predictions for this trial
    trial_preds = predictions[:, :, trial_idx].numpy()  # shape: (n_samples, n_windows)

    # Group by chromosome for efficient BigWig writing
    chroms = regions_df['chr'].unique()

    for chrom in tqdm(chroms, desc=f"Processing {trial_name}", leave=False):
        # Get all regions for this chromosome
        chrom_regions = regions_df[regions_df['chr'] == chrom].reset_index(drop=True)

        if len(chrom_regions) == 0:
            continue

        # Create BigWig file for this chromosome
        output_bw = os.path.join(output_dir, f"{trial_name}_{chrom}.bw")
        bw = pyBigWig.open(output_bw, "w")
        bw.addHeader([(chrom, chrom_sizes.get(chrom, 300000000))])  # Use 300Mb as default if unknown

        # Collect all intervals and values for this chromosome
        all_starts = []
        all_ends = []
        all_values = []

        for idx, row in chrom_regions.iterrows():
            # Get predictions for this region
            region_idx = row.name  # This is the index in the original dataframe
            region_preds = trial_preds[region_idx]  # shape: (n_windows,)

            # Calculate genomic coordinates for each window
            region_start = row['start']
            region_end = row['end']
            n_windows = len(region_preds)

            # Calculate the actual genomic range covered by predictions
            # Assuming the predictions cover the entire region uniformly
            pred_length = n_windows * window_size

            # Center the predictions on the region
            pred_start = region_start + (region_end - region_start - pred_length) // 2

            # Create coordinates for each window
            for i, value in enumerate(region_preds):
                window_start = pred_start + i * window_size
                window_end = window_start + window_size

                # Only include windows that fall within reasonable chromosome bounds
                if window_start >= 0 and window_end <= chrom_sizes.get(chrom, 300000000):
                    all_starts.append(int(window_start))
                    all_ends.append(int(window_end))
                    all_values.append(float(value))

        # Sort by start position
        if len(all_starts) > 0:
            sorted_indices = np.argsort(all_starts)
            all_starts = [all_starts[i] for i in sorted_indices]
            all_ends = [all_ends[i] for i in sorted_indices]
            all_values = [all_values[i] for i in sorted_indices]

            # Write to BigWig
            bw.addEntries([chrom] * len(all_starts), all_starts,
                         ends=all_ends, values=all_values)

        bw.close()

        if len(all_starts) > 0:
            print(f"  Created: {output_bw} ({len(all_starts)} windows)")


def write_single_bigwig_per_trial(predictions, regions_df, trial_idx, trial_name,
                                   output_file, chrom_sizes, window_size=32):
    """
    Write predictions for a specific trial to a single BigWig file (all chromosomes).

    Parameters:
    -----------
    predictions : torch.Tensor
        Predictions tensor of shape (n_samples, n_windows, n_trials)
    regions_df : pd.DataFrame
        DataFrame with columns ['chr', 'start', 'end', 'split'] containing genomic coordinates
    trial_idx : int
        Index of the trial to extract from predictions
    trial_name : str
        Name of the trial (used for output filename)
    output_file : str
        Output BigWig file path
    chrom_sizes : dict
        Dictionary mapping chromosome names to sizes
    window_size : int
        Size of genomic bins in bp (default: 32)
    """
    # Extract predictions for this trial
    trial_preds = predictions[:, :, trial_idx].numpy()  # shape: (n_samples, n_windows)

    # Collect all entries first (need to sort before writing)
    all_entries = []

    # Process each region
    for idx, row in tqdm(regions_df.iterrows(), total=len(regions_df),
                         desc=f"Processing {trial_name}", leave=False):
        # Get predictions for this region
        region_preds = trial_preds[idx]  # shape: (n_windows,)

        # Calculate genomic coordinates for each window
        chrom = row['chr']
        region_start = row['start']
        region_end = row['end']
        n_windows = len(region_preds)

        # Calculate the actual genomic range covered by predictions
        pred_length = n_windows * window_size

        # Center the predictions on the region
        pred_start = region_start + (region_end - region_start - pred_length) // 2

        # Create coordinates for each window
        for i, value in enumerate(region_preds):
            window_start = pred_start + i * window_size
            window_end = window_start + window_size

            # Only include windows that fall within reasonable chromosome bounds
            if window_start >= 0 and window_end <= chrom_sizes.get(chrom, 300000000):
                all_entries.append((chrom, int(window_start), int(window_end), float(value)))

    # Sort entries by chromosome and start position
    # Create a chromosome order based on chrom_sizes
    chrom_order = {chrom: i for i, chrom in enumerate(chrom_sizes.keys())}
    all_entries.sort(key=lambda x: (chrom_order.get(x[0], 999), x[1]))

    # Group entries by chromosome for writing
    from itertools import groupby

    # Create BigWig file
    bw = pyBigWig.open(output_file, "w")
    bw.addHeader(list(chrom_sizes.items()))

    # Write entries chromosome by chromosome
    for chrom, chrom_entries in groupby(all_entries, key=lambda x: x[0]):
        chrom_entries = list(chrom_entries)
        if len(chrom_entries) > 0:
            chroms = [e[0] for e in chrom_entries]
            starts = [e[1] for e in chrom_entries]
            ends = [e[2] for e in chrom_entries]
            values = [e[3] for e in chrom_entries]
            bw.addEntries(chroms, starts, ends=ends, values=values)

    bw.close()
    print(f"Created: {output_file}")


def export_trial(test_pred, regions_df, label_meta, trial_idx, output_dir,
                 chrom_sizes, window_size, per_chrom):
    """
    Worker function for exporting a single trial to BigWig format.

    Parameters:
    -----------
    test_pred : torch.Tensor
        Predictions tensor
    regions_df : pd.DataFrame
        DataFrame with genomic regions
    label_meta : pd.DataFrame
        Label metadata
    trial_idx : int
        Index of the trial to export
    output_dir : str
        Output directory
    chrom_sizes : dict
        Chromosome sizes
    window_size : int
        Window size in bp
    per_chrom : bool
        Whether to create separate files per chromosome

    Returns:
    --------
    str : Trial name that was exported
    """
    trial_name = label_meta.loc[trial_idx, 'trial']

    if per_chrom:
        # Check if all chromosome files already exist
        chroms = regions_df['chr'].unique()
        all_exist = True
        for chrom in chroms:
            output_bw = os.path.join(output_dir, f"{trial_name}_{chrom}.bw")
            if not os.path.exists(output_bw):
                all_exist = False
                break

        if all_exist:
            print(f"Skipping {trial_name}: all chromosome files already exist")
            return trial_name

        # Create separate BigWig files per chromosome
        write_predictions_to_bigwig(
            test_pred, regions_df, trial_idx, trial_name,
            output_dir, chrom_sizes, window_size
        )
    else:
        # Check if single BigWig file already exists
        output_file = os.path.join(output_dir, f"{trial_name}.bw")
        if os.path.exists(output_file):
            print(f"Skipping {trial_name}: file already exists")
            return trial_name

        # Create single BigWig file per trial
        write_single_bigwig_per_trial(
            test_pred, regions_df, trial_idx, trial_name,
            output_file, chrom_sizes, window_size
        )
    return trial_name


@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint number")
@click.option("-s", "--split", type=str, default="Test", help="Data split (Train/Valid/Test)")
@click.option("--res_base", required=True, default="./Res", help="Results base directory")
@click.option("--log_base", required=True, default="./logs", help="Logs base directory")
@click.option("-o", "--output_dir", required=True, type=str, help="Output directory for BigWig files")
@click.option("-f", "--fasta", "fasta_file", required=True, type=str, help="Reference genome FASTA file")
@click.option("--trials", type=str, default=None,
              help="Comma-separated list of trial names to export (default: all trials)")
@click.option("--per_chrom", is_flag=True, default=False,
              help="Create separate BigWig files per chromosome (default: single file per trial)")
@click.option("--window_size", type=int, default=32, help="Size of genomic bins in bp (default: 32)")
@click.option("--data_path", type=str, default=None,
              help="Path to data directory (default: derived from log_base)")
@click.option("--n_jobs", type=int, default=None,
              help="Number of parallel jobs (default: use all CPU cores)")
def main(exp_name, chk, split, res_base, log_base, output_dir, fasta_file,
         trials, per_chrom, window_size, data_path, n_jobs):
    """
    Convert model predictions from .pt files to BigWig format at 32bp resolution.
    """
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load label metadata
    label_meta_path = f"{LOG_BASE}/regression_label_meta.csv"
    if not os.path.exists(label_meta_path):
        raise FileNotFoundError(f"Label metadata not found: {label_meta_path}")

    label_meta = pd.read_csv(label_meta_path, index_col=None)
    print(f"Loaded label metadata: {len(label_meta)} trials")

    # Load predictions
    pred_file = f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt"
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    print(f"Loading predictions from: {pred_file}")
    test_res = torch.load(pred_file, map_location='cpu')

    # Extract predictions and reshape
    # Shape: (n_samples * n_windows, n_trials) -> (n_samples, n_windows, n_trials)
    test_pred = test_res["pred"]['regression'][:, :, label_meta['dim']].cpu()
    n_samples = len(test_res["index"])
    n_windows = test_pred.shape[1]
    n_trials = len(label_meta)

    print(f"Predictions shape: {test_pred.shape} ({n_samples} samples, {n_windows} windows, {n_trials} trials)")

    # Load genomic regions
    if data_path is None:
        # Try to find data path from config or use default
        data_path = f"Data/{exp_name}"
        if not os.path.exists(data_path):
            # Try alternative path
            data_path = f"{LOG_BASE}/../../../Data/{exp_name}"

    sequences_bed = f"{data_path}/sequences.bed"
    if not os.path.exists(sequences_bed):
        raise FileNotFoundError(f"Sequences bed file not found: {sequences_bed}")

    print(f"Loading genomic regions from: {sequences_bed}")
    df = pd.read_csv(sequences_bed, sep="\t", header=None)
    df.columns = ["chr", "start", "end", "split"]
    regions_df = df[df["split"] == split.lower()].reset_index(drop=True)

    # Filter regions based on indices in predictions
    pred_indices = test_res["index"].cpu().numpy()
    regions_df = regions_df.iloc[pred_indices].reset_index(drop=True)

    print(f"Processing {len(regions_df)} regions from {split} split")

    # Get chromosome sizes
    print(f"Reading chromosome sizes from: {fasta_file}")
    chrom_sizes = get_chrom_sizes(fasta_file)
    print(f"Found {len(chrom_sizes)} chromosomes")

    # Determine which trials to export
    if trials is not None:
        trial_list = [t.strip() for t in trials.split(',')]
        trial_indices = []
        for trial_name in trial_list:
            if trial_name in label_meta['trial'].values:
                trial_indices.append(label_meta[label_meta['trial'] == trial_name].index[0])
            else:
                print(f"Warning: Trial '{trial_name}' not found in metadata, skipping")
    else:
        trial_list = label_meta['trial'].tolist()
        trial_indices = label_meta.index.tolist()

    print(f"Exporting {len(trial_indices)} trials to BigWig format")

    # Export trials in parallel
    if n_jobs is None:
        n_jobs = os.cpu_count()
    n_jobs = min(n_jobs, len(trial_indices))
    print(f"Using {n_jobs} parallel jobs")

    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(export_trial)(
            test_pred, regions_df, label_meta, trial_idx, output_dir,
            chrom_sizes, window_size, per_chrom
        )
        for trial_idx in tqdm(trial_indices, desc="Exporting trials")
    )

    print(f"\nDone! BigWig files saved to: {output_dir}")


if __name__ == "__main__":
    main()
