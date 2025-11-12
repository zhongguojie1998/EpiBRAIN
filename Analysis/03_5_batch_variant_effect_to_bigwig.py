#!/usr/bin/env python
"""
03_5_batch_variant_effect_to_bigwig.py

Batch convert variant effect h5 files to BigWig format for visualization.
Processes all variants in a VCF file.

Usage:
    python 03_5_batch_variant_effect_to_bigwig.py --vcf variants.vcf -e exp_name --chk 20
"""

import os
import sys
import warnings

import click
import h5py
import numpy as np
import pandas as pd
import pyBigWig
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


def create_bigwig_track(bw_file, chrom, start, end, values, pool_width):
    """
    Create a BigWig file from a 1D array of values.

    Args:
        bw_file: Output BigWig file path
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position
        values: 1D numpy array of values
        pool_width: Width of each bin in bp
    """
    # Create BigWig file
    bw = pyBigWig.open(bw_file, "w")

    # Add header with chromosome sizes
    # We only need the chromosome that contains this variant
    bw.addHeader([(chrom, end + 1000000)])  # Add some buffer

    # Create entries for each bin
    n_bins = len(values)
    chroms = [chrom] * n_bins
    starts = [start + i * pool_width for i in range(n_bins)]
    ends = [start + (i + 1) * pool_width for i in range(n_bins)]

    # Convert to list for pyBigWig
    values_list = values.tolist()

    # Add entries
    bw.addEntries(chroms, starts, ends=ends, values=values_list)

    bw.close()


def process_single_variant(variant_info, variant_dir, output_dir, pool_width, track_indices, track_names_list):
    """
    Process a single variant h5 file to BigWig.

    Args:
        variant_info: Tuple of (chr_name, pos, ref, alt)
        variant_dir: Directory containing variant effect h5 files
        output_dir: Output directory for BigWig files
        pool_width: Width of prediction bins in bp
        track_indices: List of track indices to process
        track_names_list: List of track names corresponding to indices

    Returns:
        True if successful, False otherwise
    """
    chr_name, pos, ref, alt = variant_info

    variant_name = f"{chr_name}_{ref}{pos}{alt}"
    variant_h5 = f"{variant_dir}/{variant_name}.h5"

    if not os.path.exists(variant_h5):
        print(f"Warning: Variant file not found: {variant_h5}")
        return False

    try:
        # Load h5 file
        with h5py.File(variant_h5, "r") as f:
            label = f["data"]["label"][:]  # (n_bins, n_trials)
            pred_ref = f["data"]["pred_wt"][:]  # (n_bins, n_trials)
            pred_alt = f["data"]["pred_alt"][:]  # (n_bins, n_trials)

            context_start = f.attrs["context_start"]
            context_end = f.attrs["context_end"]

        # Create output directory for this variant
        variant_output_dir = f"{output_dir}/{variant_name}"
        os.makedirs(variant_output_dir, exist_ok=True)

        # Process each track
        for idx, track_name_str in zip(track_indices, track_names_list):
            label_data = label[:, idx]
            pred_ref_data = pred_ref[:, idx]
            pred_alt_data = pred_alt[:, idx]
            # do log fold change
            diff_data = np.log2(pred_alt_data + 1) - np.log2(pred_ref_data + 1)

            # Create BigWig files for this track
            tracks = {
                "label": label_data,
                "pred_ref": pred_ref_data,
                "pred_alt": pred_alt_data,
                "diff": diff_data
            }

            for data_type, track_data in tracks.items():
                # Sanitize track name for filename
                safe_track_name = track_name_str.replace('/', '_').replace(':', '_')
                bw_file = f"{variant_output_dir}/{variant_name}_{safe_track_name}_{data_type}.bw"
                create_bigwig_track(bw_file, chr_name, context_start, context_end, track_data, pool_width)

        return True

    except Exception as e:
        print(f"Error processing {variant_name}: {str(e)}")
        return False


@click.command()
@click.option("--vcf", "-f", required=True, type=str, help="Path to the VCF file")
@click.option("--exp_name", "-e", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint name")
@click.option("--res_base", required=True, type=str, default="./Res", help="Results base directory")
@click.option("--pool_width", type=int, default=32, help="Prediction bin width in bp")
@click.option("--track_index", type=int, default=None, help="Track/trial index to use (default: output all tracks separately)")
@click.option("--track_name", type=str, default=None, help="Track/trial name to use (requires --log_base)")
@click.option("--log_base", type=str, default=None, help="Logs base directory (for track_name lookup)")
@click.option("--n_jobs", type=int, default=1, help="Number of parallel jobs")
@click.option("--output_dir", "-o", type=str, default=None, help="Output directory (default: <res_base>/<exp_name>/analysis_<chk>/bigwig)")
def main(vcf, exp_name, chk, res_base, pool_width, track_index, track_name, log_base, n_jobs, output_dir):
    """Batch convert variant effect h5 files to BigWig format."""

    RES_BASE = os.path.abspath(res_base)

    # Set default output path
    if output_dir is None:
        output_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/bigwig"

    os.makedirs(output_dir, exist_ok=True)

    # Load label metadata if needed
    meta_df = None
    if log_base is not None:
        LOG_BASE = os.path.abspath(log_base)
        label_meta_path = f"{LOG_BASE}/{exp_name}/regression_label_meta.csv"
        print(f"Loading label metadata from {label_meta_path}...")
        meta_df = pd.read_csv(label_meta_path)

    # Determine which track(s) to use
    track_indices = []
    track_names_list = []

    if track_name is not None:
        if meta_df is None:
            print("Error: --log_base required when using --track_name")
            sys.exit(1)

        matching_rows = meta_df[meta_df['trial'] == track_name]
        if len(matching_rows) == 0:
            print(f"Error: Track name '{track_name}' not found in label metadata")
            sys.exit(1)

        track_index = matching_rows.iloc[0]['dim']
        track_indices = [track_index]
        track_names_list = [track_name]
        print(f"Using track '{track_name}' (index {track_index})")

    elif track_index is not None:
        track_indices = [track_index]
        track_names_list = [f"track{track_index}"]
        print(f"Using track index: {track_index}")

    else:
        # Default: process all tracks separately
        # Need to peek at first h5 file to get number of tracks
        print("Determining number of tracks from first variant...")
        variant_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff"

        # Read VCF to get first variant
        vcf_df = pd.read_csv(vcf, sep="\t", comment='#', header=None)
        first_variant = f"{vcf_df.iloc[0, 0]}_{vcf_df.iloc[0, 3]}{vcf_df.iloc[0, 1]}{vcf_df.iloc[0, 4]}"
        first_h5 = f"{variant_dir}/{first_variant}.h5"

        if not os.path.exists(first_h5):
            print(f"Error: Cannot find first variant h5 file: {first_h5}")
            sys.exit(1)

        with h5py.File(first_h5, "r") as f:
            n_tracks = f["data"]["label"].shape[1]

        track_indices = list(range(n_tracks))
        track_names_list = [f"track{i}" for i in track_indices]
        print(f"Processing all {n_tracks} tracks separately")

        # If metadata is available, use actual track names
        if meta_df is not None:
            track_names_list = [meta_df.iloc[i]['trial'] for i in track_indices]

    # Read VCF file
    print(f"Reading VCF file: {vcf}")
    vcf_df = pd.read_csv(vcf, sep="\t", comment='#', header=None)

    # Process each variant
    variant_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff"

    # Prepare variant information list
    variant_infos = [
        (vcf_df.iloc[i, 0], vcf_df.iloc[i, 1], vcf_df.iloc[i, 3], vcf_df.iloc[i, 4])
        for i in range(len(vcf_df))
    ]

    print(f"\nProcessing {len(variant_infos)} variants...")
    print(f"Input directory: {variant_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tracks per variant: {len(track_indices)}")

    if n_jobs == 1:
        # Sequential processing with progress bar
        results = []
        for variant_info in tqdm(variant_infos, desc="Converting to BigWig"):
            result = process_single_variant(variant_info, variant_dir, output_dir, pool_width, track_indices, track_names_list)
            results.append(result)
    else:
        # Parallel processing
        print(f"Using {n_jobs} parallel jobs...")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single_variant)(variant_info, variant_dir, output_dir, pool_width, track_indices, track_names_list)
            for variant_info in variant_infos
        )

    # Summary
    n_success = sum(results)
    n_failed = len(results) - n_success

    print(f"\nDone!")
    print(f"  Successfully converted: {n_success} variants")
    print(f"  Failed: {n_failed} variants")
    print(f"  BigWig files per variant: {len(track_indices)} tracks × 4 types = {len(track_indices) * 4} files")
    print(f"  Total BigWig files created: ~{n_success * len(track_indices) * 4}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
