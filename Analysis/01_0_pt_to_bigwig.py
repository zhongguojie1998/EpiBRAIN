#!/usr/bin/env python
"""
Convert model predictions from .pt files to BigWig format at 32bp resolution.

This script reads prediction files from the testing dataset (or train/valid) and
writes the predictions to BigWig format for visualization in genome browsers.
Optionally, it can also export the corresponding labels to BigWig format.

Usage:
    # Using checkpoint number (--chk):
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --chk 250 -s Test --res_base ./Res --log_base ./logs \
        -o ./bigwig_output --fasta /path/to/reference.fa \
        --trials "MiniAtlas-L56IT_K27Ac,MiniAtlas-MSN_ATAC"

    # Using direct .pt file path (--pt_file, split auto-detected from filename):
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --pt_file ./Res/my_model/Test_preds_epoch_250.pt \
        --res_base ./Res --log_base ./logs -o ./bigwig_output \
        --fasta /path/to/reference.fa

    # Using glob pattern to match files (--pt_file, split auto-detected):
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --pt_file "Res/*/Test_preds_epoch_250.pt" \
        --res_base ./Res --log_base ./logs -o ./bigwig_output \
        --fasta /path/to/reference.fa

    # Using recursive pattern (processes ALL matched files):
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --pt_file "Res/**/*_preds_*.pt" \
        --res_base ./Res --log_base ./logs -o ./bigwig_output \
        --fasta /path/to/reference.fa

    # Process multiple splits at once (Train, Valid, Test) - auto-merges at the end:
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --pt_file "Res/my_model/*_preds_epoch_250.pt" \
        --res_base ./Res --log_base ./logs -o ./bigwig_output \
        --fasta /path/to/reference.fa
    # This will create:
    # - bigwig_output/Train_preds_epoch_250/{trial}_pred.bw
    # - bigwig_output/Valid_preds_epoch_250/{trial}_pred.bw
    # - bigwig_output/Test_preds_epoch_250/{trial}_pred.bw
    # - bigwig_output/merged/{trial}_pred_merged.bw (merged across all splits)

    # To also export labels to BigWig:
    python 01_0_pt_to_bigwig.py -e basal_ganglia_miniatlas_drop_celltype_v1 \
        --chk 250 -s Test --res_base ./Res --log_base ./logs \
        -o ./bigwig_output --fasta /path/to/reference.fa \
        --export_labels
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

        for _, row in chrom_regions.iterrows():
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


def export_trial(test_pred, test_label, regions_df, label_meta, trial_idx, output_dir,
                 chrom_sizes, window_size, per_chrom, export_labels):
    """
    Worker function for exporting a single trial to BigWig format.

    Parameters:
    -----------
    test_pred : torch.Tensor
        Predictions tensor
    test_label : torch.Tensor or None
        Labels tensor (can be None if not available)
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
    export_labels : bool
        Whether to also export labels

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
            output_bw = os.path.join(output_dir, f"{trial_name}_pred_{chrom}.bw")
            if not os.path.exists(output_bw):
                all_exist = False
                break
            if export_labels and test_label is not None:
                output_bw_label = os.path.join(output_dir, f"{trial_name}_label_{chrom}.bw")
                if not os.path.exists(output_bw_label):
                    all_exist = False
                    break

        if all_exist:
            print(f"Skipping {trial_name}: all chromosome files already exist")
            return trial_name

        # Create separate BigWig files per chromosome for predictions
        write_predictions_to_bigwig(
            test_pred, regions_df, trial_idx, f"{trial_name}_pred",
            output_dir, chrom_sizes, window_size
        )

        # Create separate BigWig files per chromosome for labels
        if export_labels and test_label is not None:
            write_predictions_to_bigwig(
                test_label, regions_df, trial_idx, f"{trial_name}_label",
                output_dir, chrom_sizes, window_size
            )
    else:
        # Check if single BigWig file already exists
        output_file = os.path.join(output_dir, f"{trial_name}_pred.bw")
        if os.path.exists(output_file):
            pred_exists = True
        else:
            pred_exists = False

        label_exists = False
        if export_labels and test_label is not None:
            output_file_label = os.path.join(output_dir, f"{trial_name}_label.bw")
            if os.path.exists(output_file_label):
                label_exists = True

        if pred_exists and (not export_labels or label_exists or test_label is None):
            print(f"Skipping {trial_name}: file already exists")
            return trial_name

        # Create single BigWig file per trial for predictions
        if not pred_exists:
            write_single_bigwig_per_trial(
                test_pred, regions_df, trial_idx, f"{trial_name}_pred",
                output_file, chrom_sizes, window_size
            )

        # Create single BigWig file per trial for labels
        if export_labels and test_label is not None and not label_exists:
            output_file_label = os.path.join(output_dir, f"{trial_name}_label.bw")
            write_single_bigwig_per_trial(
                test_label, regions_df, trial_idx, f"{trial_name}_label",
                output_file_label, chrom_sizes, window_size
            )
    return trial_name


def merge_bigwig_files(output_dir, chrom_sizes, per_chrom):
    """
    Merge BigWig files from different splits/files into single merged files per trial.
    Uses bigWigMerge from UCSC tools. Merging is performed in parallel for efficiency.

    Parameters:
    -----------
    output_dir : str
        Base output directory containing subdirectories with BigWig files
    chrom_sizes : dict
        Dictionary mapping chromosome names to sizes
    per_chrom : bool
        Whether files are split by chromosome
    """
    import subprocess
    import tempfile
    from collections import defaultdict

    # Create merged output directory
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # Find all BigWig files in subdirectories
    all_bw_files = []
    for root, _, files in os.walk(output_dir):
        # Skip the merged directory itself
        if root == merged_dir or merged_dir in root:
            continue
        for file in files:
            if file.endswith('.bw'):
                all_bw_files.append(os.path.join(root, file))

    if len(all_bw_files) == 0:
        print("No BigWig files found to merge")
        return

    print(f"Found {len(all_bw_files)} BigWig files to merge")

    # Group files by trial name (remove directory prefix and get base trial name)
    # Files are named like: {trial_name}_pred.bw or {trial_name}_label.bw
    # or for per_chrom: {trial_name}_pred_{chr}.bw or {trial_name}_label_{chr}.bw
    trial_files = defaultdict(list)

    for bw_file in all_bw_files:
        basename = os.path.basename(bw_file)
        # Determine trial name by removing _pred/_label suffix and chromosome if present
        if per_chrom:
            # Pattern: trial_name_pred_chr1.bw or trial_name_label_chr1.bw
            # Extract trial name + type (pred/label) + chr
            parts = basename[:-3].split('_')  # Remove .bw
            if len(parts) >= 2:
                # Find where _pred or _label starts
                for i, part in enumerate(parts):
                    if part in ['pred', 'label']:
                        trial_key = '_'.join(parts[:i+1])  # Include pred/label in key
                        chrom = parts[-1] if i+1 < len(parts) else None
                        trial_files[(trial_key, chrom)].append(bw_file)
                        break
        else:
            # Pattern: trial_name_pred.bw or trial_name_label.bw
            if basename.endswith('_pred.bw'):
                trial_key = basename[:-8] + '_pred'  # Remove .bw, keep _pred
            elif basename.endswith('_label.bw'):
                trial_key = basename[:-9] + '_label'  # Remove .bw, keep _label
            else:
                continue
            trial_files[trial_key].append(bw_file)

    print(f"Merging {len(trial_files)} unique trials")

    # Filter out trials with only one file
    trials_to_merge = [(key, files) for key, files in trial_files.items() if len(files) > 1]

    if len(trials_to_merge) == 0:
        print("No trials need merging (all have only one file)")
        return

    print(f"Will merge {len(trials_to_merge)} trials in parallel")

    # Define worker function for parallel merging
    def merge_single_trial(trial_key, files):
        """Merge a single trial's BigWig files"""
        # Determine output filename
        if isinstance(trial_key, tuple):
            trial_name, chrom = trial_key
            output_bw = os.path.join(merged_dir, f"{trial_name}_{chrom}_merged.bw")
        else:
            output_bw = os.path.join(merged_dir, f"{trial_key}_merged.bw")

        # Skip if already exists
        if os.path.exists(output_bw):
            return f"Skipped {trial_key}: already exists"

        # Use temporary file for bedGraph output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bedGraph', delete=False) as tmp_bg:
            tmp_bedgraph = tmp_bg.name

        tmp_chromsizes = None
        try:
            # Run bigWigMerge
            cmd = ['bigWigMerge'] + files + [tmp_bedgraph]
            subprocess.run(cmd, check=True, capture_output=True)

            # Write chromosome sizes to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.chrom.sizes', delete=False) as tmp_cs:
                for chrom, size in chrom_sizes.items():
                    tmp_cs.write(f"{chrom}\t{size}\n")
                tmp_chromsizes = tmp_cs.name

            # Convert bedGraph to BigWig
            cmd = ['bedGraphToBigWig', tmp_bedgraph, tmp_chromsizes, output_bw]
            subprocess.run(cmd, check=True, capture_output=True)

            return f"Success: {output_bw}"

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else 'N/A'
            return f"Error {trial_key}: {error_msg}"
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_bedgraph):
                os.remove(tmp_bedgraph)
            if tmp_chromsizes and os.path.exists(tmp_chromsizes):
                os.remove(tmp_chromsizes)

    # Merge trials in parallel
    n_merge_jobs = min(os.cpu_count(), len(trials_to_merge))
    print(f"Using {n_merge_jobs} parallel jobs for merging")

    results = Parallel(n_jobs=n_merge_jobs, verbose=10)(
        delayed(merge_single_trial)(trial_key, files)
        for trial_key, files in trials_to_merge
    )

    # Print results summary
    print("\nMerge Results:")
    for result in results:
        print(f"  {result}")

    print(f"\nMerging complete! Merged files saved to: {merged_dir}")


@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", type=str, default=None, help="Checkpoint number (used to construct prediction file path)")
@click.option("--pt_file", type=str, default=None, help="Direct path to .pt prediction file or glob pattern (e.g., '*.pt', 'Res/**/*_preds_*.pt'). ALL matched files will be processed. Overrides --chk")
@click.option("-s", "--split", type=str, default="Test", help="Data split (Train/Valid/Test). Auto-detected from filename when using --pt_file")
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
@click.option("--export_labels", is_flag=True, default=False,
              help="Also export labels to BigWig format (default: False)")
def main(exp_name, chk, pt_file, split, res_base, log_base, output_dir, fasta_file,
         trials, per_chrom, window_size, data_path, n_jobs, export_labels):
    """
    Convert model predictions (and optionally labels) from .pt files to BigWig format at 32bp resolution.

    Output files will be named as {trial_name}_pred.bw for predictions and {trial_name}_label.bw for labels.

    Multi-file processing:
    - When --pt_file is a glob pattern that matches multiple files, ALL matched files will be processed
    - Each file is processed with its own split (Train/Valid/Test) auto-detected from filename
    - Each file's outputs are saved to a separate subdirectory named after the .pt filename
    - This allows processing multiple splits (Train/Valid/Test) or checkpoints in a single run
    - After processing all files, BigWig files are automatically merged using bigWigMerge
    - Merged files are saved to output_dir/merged/ with filenames like {trial_name}_pred_merged.bw

    Note: When using --pt_file, the data split (Train/Valid/Test) is automatically extracted from the filename
    if it follows the pattern '{Split}_preds_epoch_{n}.pt'. The --split parameter can be omitted in this case.

    Requirements for merging:
    - bigWigMerge and bedGraphToBigWig from UCSC tools must be in PATH
    """
    # Validate input: either pt_file or chk must be provided
    if pt_file is None and chk is None:
        raise ValueError("Either --pt_file or --chk must be provided")

    if pt_file is not None and chk is not None:
        print("Warning: Both --pt_file and --chk provided. Using --pt_file and ignoring --chk")

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

    # Determine prediction file path(s)
    pred_files_with_splits = []  # List of (pred_file, split) tuples

    if pt_file is not None:
        # Support glob patterns in pt_file
        import glob as glob_module
        import re

        # Expand glob pattern (use recursive=True to support **)
        matched_files = glob_module.glob(pt_file, recursive=True)

        if len(matched_files) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {pt_file}")

        # Sort by modification time (most recent first)
        matched_files = sorted(matched_files, key=os.path.getmtime, reverse=True)

        print(f"Pattern '{pt_file}' matched {len(matched_files)} file(s):")
        for f in matched_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(matched_files) > 10:
            print(f"  ... and {len(matched_files) - 10} more")

        # Process all matched files
        for pred_file in matched_files:
            pred_file = os.path.abspath(pred_file)

            # Extract split from filename
            basename = os.path.basename(pred_file)
            split_match = re.search(r'(Train|Valid|Test|train|valid|test)_preds', basename)

            if split_match:
                file_split = split_match.group(1).capitalize()
            else:
                # Use provided split or default
                if split is not None:
                    file_split = split
                else:
                    print(f"Warning: Could not extract split from filename '{basename}'. Using default split: Test")
                    file_split = "Test"

            pred_files_with_splits.append((pred_file, file_split))

        print(f"\nWill process {len(pred_files_with_splits)} prediction file(s)")
    else:
        # Single file from checkpoint
        pred_file = f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt"
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        pred_files_with_splits.append((pred_file, split))

    # Get chromosome sizes (only once for all files)
    print(f"\nReading chromosome sizes from: {fasta_file}")
    chrom_sizes = get_chrom_sizes(fasta_file)
    print(f"Found {len(chrom_sizes)} chromosomes")

    # Process each prediction file
    for file_idx, (pred_file, file_split) in enumerate(pred_files_with_splits):
        print(f"\n{'='*80}")
        print(f"Processing file {file_idx+1}/{len(pred_files_with_splits)}")
        print(f"File: {pred_file}")
        print(f"Split: {file_split}")
        print(f"{'='*80}\n")

        print(f"Loading predictions from: {pred_file}")
        test_res = torch.load(pred_file, map_location='cpu')

        # Extract predictions and reshape
        # Shape: (n_samples * n_windows, n_trials) -> (n_samples, n_windows, n_trials)
        test_pred = test_res["pred"]['regression'][:, :, label_meta['dim']].cpu()
        n_samples = len(test_res["index"])
        n_windows = test_pred.shape[1]
        n_trials = len(label_meta)

        print(f"Predictions shape: {test_pred.shape} ({n_samples} samples, {n_windows} windows, {n_trials} trials)")

        # Extract labels if requested
        test_label = None
        file_export_labels = export_labels  # Local copy for this file
        if export_labels:
            if 'label' in test_res and 'regression' in test_res['label']:
                test_label = test_res["label"]['regression'][:, :, label_meta.index.values].cpu()
                print(f"Labels shape: {test_label.shape} ({n_samples} samples, {n_windows} windows, {n_trials} trials)")
            else:
                print("Warning: Labels not found in prediction file. Label export will be skipped for this file.")
                file_export_labels = False

        # Load genomic regions for this file's split
        file_data_path = data_path
        if file_data_path is None:
            # Try to find data path from config or use default
            file_data_path = f"Data/{exp_name}"
            if not os.path.exists(file_data_path):
                # Try alternative path
                file_data_path = f"{LOG_BASE}/../../../Data/{exp_name}"

        sequences_bed = f"{file_data_path}/sequences.bed"
        if not os.path.exists(sequences_bed):
            raise FileNotFoundError(f"Sequences bed file not found: {sequences_bed}")

        print(f"Loading genomic regions from: {sequences_bed}")
        df = pd.read_csv(sequences_bed, sep="\t", header=None)
        df.columns = ["chr", "start", "end", "split"]
        regions_df = df[df["split"] == file_split.lower()].reset_index(drop=True)

        # Filter regions based on indices in predictions
        pred_indices = test_res["index"].cpu().numpy()
        regions_df = regions_df.iloc[pred_indices].reset_index(drop=True)

        print(f"Processing {len(regions_df)} regions from {file_split} split")

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

        # Create output subdirectory for this file/split combination
        file_basename = os.path.splitext(os.path.basename(pred_file))[0]
        file_output_dir = os.path.join(output_dir, file_basename)
        os.makedirs(file_output_dir, exist_ok=True)
        print(f"Output directory for this file: {file_output_dir}")

        # Export trials in parallel
        file_n_jobs = n_jobs
        if file_n_jobs is None:
            file_n_jobs = os.cpu_count()
        file_n_jobs = min(file_n_jobs, len(trial_indices))
        print(f"Using {file_n_jobs} parallel jobs")

        Parallel(n_jobs=file_n_jobs, verbose=10)(
            delayed(export_trial)(
                test_pred, test_label, regions_df, label_meta, trial_idx, file_output_dir,
                chrom_sizes, window_size, per_chrom, file_export_labels
            )
            for trial_idx in trial_indices
        )

        print(f"\nDone processing {pred_file}!")
        print(f"BigWig files saved to: {file_output_dir}")

    print(f"\n{'='*80}")
    print(f"ALL FILES PROCESSED SUCCESSFULLY!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    # Merge BigWig files if multiple .pt files were processed
    if len(pred_files_with_splits) > 1:
        print(f"\n{'='*80}")
        print(f"MERGING BIGWIG FILES FROM MULTIPLE SPLITS")
        print(f"{'='*80}\n")
        merge_bigwig_files(output_dir, chrom_sizes, per_chrom)
        print(f"\nMerged BigWig files saved to: {output_dir}/merged/")


if __name__ == "__main__":
    main()
