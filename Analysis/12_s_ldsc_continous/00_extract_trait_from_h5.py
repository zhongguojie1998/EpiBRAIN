#!/usr/bin/env python3
"""
Extract GWAS trait data from HDF5 file for S-LDSC analysis.

This script extracts variant effects (predictions), variants, and summary statistics
for a specific GWAS trait from an HDF5 file and saves them in the format needed
for the S-LDSC pipeline.

Usage:
    python 00_extract_trait_from_h5.py --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
                                        --trait schizophrenia \
                                        --output-dir Data/source/schizophrenia_extracted
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from pyliftover import LiftOver


def list_available_traits(h5_file):
    """
    List all available traits in the HDF5 file.

    Args:
        h5_file: Path to HDF5 file

    Returns:
        List of available trait names
    """
    with h5py.File(h5_file, mode='r') as f:
        if 'experiments' in f:
            return list(f['experiments'].keys())
        else:
            print("Error: No 'experiments' group found in HDF5 file")
            return []


def extract_trait_data(h5_file, trait_name, output_dir, chain_file=None):
    """
    Extract data for a specific GWAS trait from HDF5 file.

    Args:
        h5_file: Path to input HDF5 file
        trait_name: Name of the trait/experiment to extract
        output_dir: Output directory for extracted data
        chain_file: Path to hg38ToHg19 liftover chain file (optional)

    Returns:
        Dictionary with paths to generated files
    """
    print(f"\n{'='*80}")
    print(f"Extracting data for trait: {trait_name}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read HDF5 file
    print("Opening HDF5 file...")
    res = h5py.File(h5_file, mode='r')

    # Check if trait exists
    if f"experiments/{trait_name}" not in res:
        available_traits = list(res['experiments'].keys()) if 'experiments' in res else []
        print(f"\nError: Trait '{trait_name}' not found in file.")
        if available_traits:
            print(f"\nAvailable traits:")
            for trait in available_traits:
                print(f"  - {trait}")
        res.close()
        sys.exit(1)

    # Extract data
    print("Extracting variant effects and metadata from HDF5...")
    tracks = res["model_meta/trial_names"][:]
    exp_index = res[f"experiments/{trait_name}/index_key"][:]
    all_variant_index = res["variants/index_key"][:]
    exp_reverse_map = res[f"experiments/{trait_name}/reverse_map"][:]
    exp_score = res[f"experiments/{trait_name}/z_score"][:]
    exp_n = res[f"experiments/{trait_name}/n_sample"][:]

    print(f"  Number of tracks: {len(tracks)}")
    print(f"  Number of variants: {len(exp_index)}")

    # Get variant index mapping
    all_variant_to_pos = {val: idx for idx, val in enumerate(all_variant_index)}
    positions = [all_variant_to_pos[val] for val in exp_index]

    # Sort positions to ensure order
    sort_idx = np.argsort(positions)
    positions = np.array(positions)[sort_idx]
    exp_reverse_map = exp_reverse_map[sort_idx]
    exp_score = exp_score[sort_idx]
    exp_n = exp_n[sort_idx]

    # Calculate variant effects (predictions)
    print("Calculating variant effects...")
    exp_var_score = res["results/local_log_square"][positions, :]

    # Create variants DataFrame
    print("Creating variants DataFrame...")
    variants = pd.DataFrame({
        'CHR': res['variants/chr'][positions],
        'BP': res['variants/pos'][positions],
        'SNP': res['variants/rsid'][positions],
        'A1': res['variants/alt'][positions],
        'A2': res['variants/ref'][positions]
    })

    # Create summary statistics DataFrame
    print("Creating summary statistics DataFrame...")
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
    reverse_count = 0
    for idx, reverse in zip(variants.index, exp_reverse_map):
        if reverse:
            variants.at[idx, 'A1'], variants.at[idx, 'A2'] = variants.at[idx, 'A2'], variants.at[idx, 'A1']
            sum_stats.at[idx, 'A1'], sum_stats.at[idx, 'A2'] = sum_stats.at[idx, 'A2'], sum_stats.at[idx, 'A1']
            reverse_count += 1
    print(f"  Reversed {reverse_count} variants")

    # Convert to string types
    variants['SNP'] = variants['SNP'].astype(str)
    variants['A1'] = variants['A1'].astype(str)
    variants['A2'] = variants['A2'].astype(str)
    sum_stats['SNP'] = sum_stats['SNP'].astype(str)
    sum_stats['A1'] = sum_stats['A1'].astype(str)
    sum_stats['A2'] = sum_stats['A2'].astype(str)
    variants['CHR'] = variants['CHR'].astype(str)

    # Coordinate conversion from hg38 to hg19
    if chain_file is None:
        chain_file = '../Data/Ref/hg38ToHg19.over.chain.gz'

    if os.path.exists(chain_file):
        print(f"Converting coordinates from hg38 to hg19 using {chain_file}...")
        lo = LiftOver(chain_file)
        conversion_failures = 0
        for idx in variants.index:
            try:
                hg19_pos = lo.convert_coordinate(
                    variants.loc[idx, 'CHR'],
                    variants.loc[idx, 'BP'] - 1,
                    strand='+'
                )[0][1]
                variants.loc[idx, 'BP'] = hg19_pos + 1  # Convert to 1-based indexing
            except (IndexError, TypeError):
                variants.loc[idx, 'BP'] = np.nan  # If conversion fails, set to NaN
                conversion_failures += 1

        if conversion_failures > 0:
            print(f"  Warning: Failed to convert {conversion_failures} variants to hg19")
            # Remove variants that failed to convert
            print(f"  Removing variants that failed to convert...")
            variants = variants.dropna(subset=['BP'])
            sum_stats = sum_stats.loc[variants.index]
            exp_var_score = exp_var_score[variants.index]
            print(f"  Remaining variants: {len(variants)}")
    else:
        print(f"Warning: Chain file not found at {chain_file}, skipping coordinate conversion")

    # Convert BP to int
    variants['BP'] = variants['BP'].astype(int)

    # Create variant effects DataFrame
    print("Creating variant effects DataFrame...")
    variant_effects = pd.DataFrame(
        exp_var_score,
        columns=[t.decode('utf-8') if isinstance(t, bytes) else t for t in tracks],
        index=variants['SNP'].values
    )

    # Save data in the format needed for S-LDSC annotation creation
    print(f"\nSaving data to {output_dir}...")

    # Save variant effects matrix as .npy
    matrix_file = output_dir / f"{trait_name}.npy"
    np.save(matrix_file, variant_effects.values)
    print(f"  ✓ Saved annotation matrix: {matrix_file}")

    # Save tracks metadata
    tracks_file = output_dir / f"{trait_name}.tracks.csv"
    tracks_df = pd.DataFrame({
        'index': range(len(variant_effects.columns)),
        'track_name': variant_effects.columns
    })
    tracks_df.to_csv(tracks_file, index=False)
    print(f"  ✓ Saved tracks metadata: {tracks_file}")

    # Save variants metadata
    variants_file = output_dir / f"{trait_name}.variants.csv"
    variants_with_index = variants.copy()
    variants_with_index.insert(0, 'index', range(len(variants)))
    variants_with_index.to_csv(variants_file, index=False)
    print(f"  ✓ Saved variants metadata: {variants_file}")

    # Also save summary statistics (useful for reference)
    sumstats_file = output_dir / f"{trait_name}.sumstats.tsv.gz"
    sum_stats.to_csv(sumstats_file, sep='\t', index=False, compression='gzip')
    print(f"  ✓ Saved summary statistics: {sumstats_file}")

    # Save combined file for reference
    combined = variants.copy()
    combined['Z'] = sum_stats['Z'].values
    combined['N'] = sum_stats['N'].values
    combined_file = output_dir / f"{trait_name}_combined.tsv.gz"
    combined.to_csv(combined_file, sep='\t', index=False, compression='gzip')
    print(f"  ✓ Saved combined data: {combined_file}")

    print(f"\n{'='*80}")
    print("Extraction complete!")
    print(f"{'='*80}")
    print(f"\nExtracted data for trait: {trait_name}")
    print(f"  Variants: {len(variants):,}")
    print(f"  Tracks: {len(variant_effects.columns):,}")
    print(f"  Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - Annotation matrix: {matrix_file.name}")
    print(f"  - Tracks metadata: {tracks_file.name}")
    print(f"  - Variants metadata: {variants_file.name}")
    print(f"  - Summary statistics: {sumstats_file.name}")
    print(f"  - Combined data: {combined_file.name}")

    return {
        'matrix_file': str(matrix_file),
        'tracks_file': str(tracks_file),
        'variants_file': str(variants_file),
        'sumstats_file': str(sumstats_file),
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract GWAS trait data from HDF5 file for S-LDSC analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available traits in the HDF5 file
  python 00_extract_trait_from_h5.py --h5-file Data/source/GWAS/full_finetune.h5 --list-traits

  # Extract data for a specific trait
  python 00_extract_trait_from_h5.py --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \\
      --trait schizophrenia \\
      --output-dir Data/source/schizophrenia_extracted

  # Extract with custom chain file for coordinate conversion
  python 00_extract_trait_from_h5.py --h5-file Data/source/GWAS/full_finetune.h5 \\
      --trait alzheimers \\
      --output-dir Data/source/alzheimers_extracted \\
      --chain-file /path/to/hg38ToHg19.over.chain.gz
        """
    )

    parser.add_argument('--h5-file', required=True,
                        help='Path to input HDF5 file')
    parser.add_argument('--trait',
                        help='Name of the trait/experiment to extract')
    parser.add_argument('--output-dir',
                        help='Output directory for extracted data')
    parser.add_argument('--chain-file',
                        help='Path to hg38ToHg19 liftover chain file (default: ../Data/Ref/hg38ToHg19.over.chain.gz)')
    parser.add_argument('--list-traits', action='store_true',
                        help='List available traits in the HDF5 file and exit')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.h5_file):
        print(f"Error: Input file {args.h5_file} does not exist")
        sys.exit(1)

    # List traits if requested
    if args.list_traits:
        print(f"Reading traits from {args.h5_file}...")
        traits = list_available_traits(args.h5_file)
        if traits:
            print(f"\nFound {len(traits)} trait(s):")
            for trait in traits:
                print(f"  - {trait}")
        else:
            print("No traits found in HDF5 file")
        sys.exit(0)

    # Validate required arguments
    if not args.trait:
        print("Error: --trait is required (or use --list-traits to see available traits)")
        sys.exit(1)

    if not args.output_dir:
        print("Error: --output-dir is required")
        sys.exit(1)

    # Extract data
    try:
        extract_trait_data(
            args.h5_file,
            args.trait,
            args.output_dir,
            args.chain_file
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
