#!/usr/bin/env python
"""
03_5_variant_effect_to_bigwig.py

Convert variant effect h5 files to BigWig format for visualization.
Creates 4 tracks: label, pred_ref, pred_alt, and diff (alt - ref).

Usage:
    python 03_5_variant_effect_to_bigwig.py -i variant.h5 -o output_prefix --chrom chr1
"""

import os
import sys
import warnings

import click
import h5py
import numpy as np
import pyBigWig

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


@click.command()
@click.option("-i", "--input_h5", required=True, type=str, help="Input h5 file path")
@click.option("-o", "--output_prefix", type=str, default=None, help="Output prefix for BigWig files (default: use h5 filename as folder)")
@click.option("--chrom", type=str, default=None, help="Chromosome name (default: auto-detect from filename)")
@click.option("--pool_width", type=int, default=32, help="Width of each prediction bin in bp (default: 32)")
@click.option("--track_index", type=int, default=None, help="Track/trial index to use (default: output all tracks separately)")
@click.option("--track_name", type=str, default=None, help="Track/trial name to use (requires --label_meta)")
@click.option("--label_meta", type=str, default=None, help="Path to label metadata CSV (for track_name lookup)")
def main(input_h5, output_prefix, chrom, pool_width, track_index, track_name, label_meta):
    """
    Convert variant effect h5 file to BigWig format.

    Auto-detects chromosome from filename (e.g., chr1_A12345G.h5 -> chr1).
    If output_prefix is not specified, automatically creates a folder named after
    the h5 file (without .h5 extension) in the same directory as the input file.

    Example:
        Input:  Res/exp/var_eff/chr1_A12345G.h5
        Output: Res/exp/var_eff/chr1_A12345G/chr1_A12345G_track0_label.bw
                Res/exp/var_eff/chr1_A12345G/chr1_A12345G_track0_pred_ref.bw
                ...

    Filename format expected: {chrom}_{ref}{pos}{alt}.h5
    """

    if not os.path.exists(input_h5):
        print(f"Error: Input file not found: {input_h5}")
        sys.exit(1)

    # Get basename for auto-detection
    h5_basename = os.path.basename(input_h5)
    if h5_basename.endswith('.h5'):
        variant_name = h5_basename[:-3]  # Remove .h5
    else:
        variant_name = h5_basename

    # Auto-detect chromosome from filename if not provided
    if chrom is None:
        # Filename format: chr1_A12345G or chr1_A12345G.h5
        # Extract chromosome (everything before first underscore)
        if '_' in variant_name:
            chrom = variant_name.split('_')[0]
            print(f"Auto-detected chromosome: {chrom}")
        else:
            print("Error: Cannot auto-detect chromosome from filename. Please specify --chrom")
            print(f"Filename: {h5_basename}")
            sys.exit(1)

    # If output_prefix is None, use h5 filename as folder
    if output_prefix is None:
        # Use the same directory as input file or current directory
        input_dir = os.path.dirname(input_h5)
        if input_dir:
            output_prefix = os.path.join(input_dir, variant_name, variant_name)
        else:
            output_prefix = os.path.join(variant_name, variant_name)

        print(f"Using auto-generated output folder: {os.path.dirname(output_prefix)}")

    # Load h5 file
    print(f"Loading h5 file: {input_h5}")
    with h5py.File(input_h5, "r") as f:
        label = f["data"]["label"][:]  # (n_bins, n_trials)
        pred_ref = f["data"]["pred_wt"][:]  # (n_bins, n_trials)
        pred_alt = f["data"]["pred_alt"][:]  # (n_bins, n_trials)

        context_start = f.attrs["context_start"]
        context_end = f.attrs["context_end"]
        pos = f.attrs["pos"]
        ref = f.attrs["ref"]
        alt = f.attrs["alt"]

    print(f"  Variant: {chrom}:{pos} {ref}>{alt}")
    print(f"  Context: {chrom}:{context_start}-{context_end}")
    print(f"  Shape: {label.shape} (n_bins={label.shape[0]}, n_trials={label.shape[1]})")

    # Determine which track(s) to use
    track_indices = []
    track_names_list = []

    if track_name is not None:
        if label_meta is None:
            print("Error: --label_meta required when using --track_name")
            sys.exit(1)

        import pandas as pd
        meta_df = pd.read_csv(label_meta)
        matching_rows = meta_df[meta_df['trial'] == track_name]

        if len(matching_rows) == 0:
            print(f"Error: Track name '{track_name}' not found in label metadata")
            sys.exit(1)

        track_index = matching_rows.iloc[0]['dim']
        track_indices = [track_index]
        track_names_list = [track_name]
        print(f"  Using track '{track_name}' (index {track_index})")

    elif track_index is not None:
        track_indices = [track_index]
        track_names_list = [f"track{track_index}"]
        print(f"  Using track index: {track_index}")

    else:
        # Default: output all tracks separately
        track_indices = list(range(label.shape[1]))
        track_names_list = [f"track{i}" for i in track_indices]
        print(f"  Processing all {label.shape[1]} tracks separately")

        # If label_meta is provided, use actual track names
        if label_meta is not None:
            import pandas as pd
            meta_df = pd.read_csv(label_meta)
            track_names_list = [meta_df.iloc[i]['trial'] for i in track_indices]

    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Process each track
    total_files = 0
    for idx, track_name_str in zip(track_indices, track_names_list):
        print(f"\nProcessing {track_name_str} (index {idx})...")

        label_data = label[:, idx]
        pred_ref_data = pred_ref[:, idx]
        pred_alt_data = pred_alt[:, idx]
        diff_data = pred_alt_data - pred_ref_data

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
            bw_file = f"{output_prefix}_{safe_track_name}_{data_type}.bw"
            create_bigwig_track(bw_file, chrom, context_start, context_end, track_data, pool_width)
            total_files += 1

            # Print statistics
            print(f"  {data_type}: min={track_data.min():.4f}, max={track_data.max():.4f}, mean={track_data.mean():.4f}")

    print(f"\nDone! Created {total_files} BigWig files (4 per track × {len(track_indices)} tracks)")


if __name__ == "__main__":
    main()
