#!/usr/bin/env python3
"""
Split GWAS Data by Experiment

This script reads an HDF5 file containing multiple experiments and splits
the GWAS data for each experiment into separate output files.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from pyliftover import LiftOver


def get_available_experiments(h5_file):
    """
    Get list of all available experiments in the HDF5 file.

    Args:
        h5_file: Open HDF5 file object

    Returns:
        List of experiment names (keys under 'experiments/')
    """
    experiments = []
    if 'experiments' in h5_file:
        experiments = list(h5_file['experiments'].keys())
    return experiments


def extract_experiment_data(input_file, exp_key, output_dir):
    """
    Extract GWAS data for a specific experiment.

    Args:
        input_file: Path to input HDF5 file
        exp_key: Experiment key to extract
        output_dir: Output directory for this experiment

    Returns:
        Tuple of (variants, sum_stats, variant_effects) DataFrames
    """
    print(f"\n{'='*80}")
    print(f"Processing experiment: {exp_key}")
    print(f"{'='*80}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read HDF5 file
    res = h5py.File(input_file, mode='r')

    # Check if experiment exists
    if f"experiments/{exp_key}" not in res:
        print(f"Warning: Experiment '{exp_key}' not found in file. Skipping.")
        res.close()
        return None, None, None

    # Extract data
    print("Extracting data from HDF5...")
    tracks = res["model_meta/trial_names"][:]
    exp_index = res[f"experiments/{exp_key}/index_key"][:]
    all_variant_index = res["variants/index_key"][:]
    exp_reverse_map = res[f"experiments/{exp_key}/reverse_map"][:]
    exp_score = res[f"experiments/{exp_key}/z_score"][:]
    exp_n = res[f"experiments/{exp_key}/n_sample"][:]

    # Get variant index
    all_variant_to_pos = {val: idx for idx, val in enumerate(all_variant_index)}
    positions = [all_variant_to_pos[val] for val in exp_index]

    # Sort positions to ensure order
    sort_idx = np.argsort(positions)
    positions = np.array(positions)[sort_idx]
    exp_reverse_map = exp_reverse_map[sort_idx]
    exp_score = exp_score[sort_idx]
    exp_n = exp_n[sort_idx]

    # Calculate variant effects
    print("Calculating variant effects...")
    exp_var_score = res["results/local_log_square"][positions, :]
    final_score = exp_var_score * (1 - 2 * exp_reverse_map).reshape(-1, 1)
    variant_effects = pd.DataFrame(final_score, columns=tracks)

    # Create variant and summary statistics DataFrames
    print("Creating variant and summary statistics DataFrames...")
    variants = pd.DataFrame({
        'CHR': res['variants/chr'][positions],
        'BP': res['variants/pos'][positions],
        'SNP': res['variants/rsid'][positions],
        'A1': res['variants/alt'][positions],
        'A2': res['variants/ref'][positions]
    })

    sum_stats = pd.DataFrame({
        'SNP': res['variants/rsid'][positions],
        'A1': res['variants/alt'][positions],
        'A2': res['variants/ref'][positions],
        'Z': exp_score,
        'N': exp_n
    })

    # Close the HDF5 file
    res.close()

    # Reverse map alleles if needed
    print("Applying reverse mapping where needed...")
    for idx, reverse in zip(variants.index, exp_reverse_map):
        if reverse:
            variants.at[idx, 'A1'], variants.at[idx, 'A2'] = variants.at[idx, 'A2'], variants.at[idx, 'A1']
            sum_stats.at[idx, 'A1'], sum_stats.at[idx, 'A2'] = sum_stats.at[idx, 'A2'], sum_stats.at[idx, 'A1']

    # Convert to string types
    variants['SNP'] = variants['SNP'].astype(str)
    variants['A1'] = variants['A1'].astype(str)
    variants['A2'] = variants['A2'].astype(str)
    sum_stats['SNP'] = sum_stats['SNP'].astype(str)
    sum_stats['A1'] = sum_stats['A1'].astype(str)
    sum_stats['A2'] = sum_stats['A2'].astype(str)
    variants.index = variants['SNP'].values
    sum_stats.index = sum_stats['SNP'].values
    variants['CHR'] = variants['CHR'].astype(str)

    # Coordinate conversion from hg38 to hg19
    print("Converting coordinates from hg38 to hg19...")
    lo = LiftOver('../Data/Ref/hg38ToHg19.over.chain.gz')
    conversion_failures = 0
    for rsid in variants.index:
        try:
            hg19_pos = lo.convert_coordinate(variants['CHR'][rsid], variants['BP'][rsid]-1, strand='+')[0][1]
            variants.loc[rsid, 'BP'] = hg19_pos + 1  # Convert to 1-based indexing
        except IndexError:
            variants.loc[rsid, 'BP'] = np.nan  # If conversion fails, set to NaN
            conversion_failures += 1

    if conversion_failures > 0:
        print(f"Warning: Failed to convert {conversion_failures} variants to hg19")

    variant_effects.index = variants.index.copy()

    print(f"Extracted {len(variants)} variants for experiment '{exp_key}'")
    print(f"Number of tracks: {len(tracks)}")

    return variants, sum_stats, variant_effects


def save_experiment_data(variants, sum_stats, variant_effects, output_dir, exp_key):
    """
    Save experiment data to output files.

    Args:
        variants: Variants DataFrame
        sum_stats: Summary statistics DataFrame
        variant_effects: Variant effects DataFrame
        output_dir: Output directory
        exp_key: Experiment key (for naming files)
    """
    print(f"Saving data for experiment '{exp_key}'...")

    # Save variants
    variants_file = os.path.join(output_dir, f"{exp_key}_variants.tsv.gz")
    variants.to_csv(variants_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved variants: {variants_file}")

    # Save summary statistics
    sumstats_file = os.path.join(output_dir, f"{exp_key}_sumstats.tsv.gz")
    sum_stats.to_csv(sumstats_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved summary statistics: {sumstats_file}")

    # Save variant effects
    effects_file = os.path.join(output_dir, f"{exp_key}_variant_effects.tsv.gz")
    variant_effects.to_csv(effects_file, sep='\t', index=True, compression='gzip')
    print(f"  Saved variant effects: {effects_file}")

    # Save a combined file with all information
    combined = variants.copy()
    combined['Z'] = sum_stats['Z']
    combined['N'] = sum_stats['N']
    combined = pd.concat([combined, variant_effects], axis=1)
    combined_file = os.path.join(output_dir, f"{exp_key}_combined.tsv.gz")
    combined.to_csv(combined_file, sep='\t', index=False, compression='gzip')
    print(f"  Saved combined data: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Split GWAS data by experiment from HDF5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all experiments
  python 03_3_split_GWAS_by_experiment.py --input-file results.h5 --output-dir output/

  # Process specific experiments only
  python 03_3_split_GWAS_by_experiment.py --input-file results.h5 --output-dir output/ \\
      --experiments schizophrenia alzheimers

  # List available experiments without processing
  python 03_3_split_GWAS_by_experiment.py --input-file results.h5 --list-only
        """
    )

    parser.add_argument('--input-file', required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--output-dir', default='./gwas_by_experiment',
                        help='Output directory for split data (default: ./gwas_by_experiment)')
    parser.add_argument('--experiments', nargs='+',
                        help='Specific experiments to process (default: all experiments)')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list available experiments without processing')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)

    print(f"Reading experiments from {args.input_file}")

    # Open HDF5 file and get available experiments
    with h5py.File(args.input_file, mode='r') as f:
        available_experiments = get_available_experiments(f)

    if not available_experiments:
        print("Error: No experiments found in HDF5 file under 'experiments/' group")
        sys.exit(1)

    print(f"\nFound {len(available_experiments)} experiments:")
    for exp in available_experiments:
        print(f"  - {exp}")

    # If list-only mode, exit here
    if args.list_only:
        print("\nList-only mode enabled. Exiting without processing.")
        sys.exit(0)

    # Determine which experiments to process
    if args.experiments:
        experiments_to_process = args.experiments
        # Validate that requested experiments exist
        invalid_experiments = [exp for exp in experiments_to_process if exp not in available_experiments]
        if invalid_experiments:
            print(f"\nError: The following experiments were not found in the file:")
            for exp in invalid_experiments:
                print(f"  - {exp}")
            print(f"\nAvailable experiments: {', '.join(available_experiments)}")
            sys.exit(1)
    else:
        experiments_to_process = available_experiments

    print(f"\nProcessing {len(experiments_to_process)} experiment(s)...")

    # Track successes and failures
    successful = []
    failed = []

    # Process each experiment
    for exp_key in experiments_to_process:
        try:
            # Create experiment-specific output directory
            exp_output_dir = os.path.join(args.output_dir, exp_key)

            # Extract data
            variants, sum_stats, variant_effects = extract_experiment_data(
                args.input_file,
                exp_key,
                exp_output_dir
            )

            if variants is None:
                failed.append((exp_key, "Experiment not found"))
                continue

            # Save data
            save_experiment_data(variants, sum_stats, variant_effects, exp_output_dir, exp_key)

            successful.append(exp_key)
            print(f"Successfully processed experiment: {exp_key}")

        except Exception as e:
            print(f"\nError processing experiment '{exp_key}': {e}")
            import traceback
            traceback.print_exc()
            failed.append((exp_key, str(e)))
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("Processing Summary")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments_to_process)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSuccessfully processed experiments:")
        for exp in successful:
            print(f"  - {exp}")

    if failed:
        print(f"\nFailed experiments:")
        for exp, error in failed:
            print(f"  - {exp}: {error[:100]}")

    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
