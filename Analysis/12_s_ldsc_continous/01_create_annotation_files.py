#!/usr/bin/env python3
"""
Create LDSC annotation files from numpy matrix of continuous annotations.

This script:
1. Loads the annotation matrix (variants x tracks)
2. Splits by chromosome
3. Matches with reference bim files to get proper ordering and CM values
4. Creates one annotation file per chromosome per track (parallelized with joblib)

Usage:
    # Using specific data directory
    python 01_create_annotation_files.py --data-dir Data/source/Schizophrenia --trait-name Schizophrenia

    # With custom number of jobs
    python 01_create_annotation_files.py --data-dir Data/source/trait_data --trait-name my_trait --n-jobs 8
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import sys

# Default configuration
BASE_DIR = Path("/gpfs/commons/groups/ren_lab/guojiezhong/BICAN")
DEFAULT_REF_DIR = BASE_DIR / "Analysis/12_ldsc/reference/1000G_EUR_Phase3_plink"
DEFAULT_OUTPUT_DIR = BASE_DIR / "Analysis/12_s_ldsc_continous/annotations"

def load_data(data_dir, trait_name):
    """Load annotation matrix and metadata.

    Args:
        data_dir: Path to directory containing trait data files
        trait_name: Name of the trait (used for file naming)

    Returns:
        Tuple of (annot_matrix, track_names, variants_df)
    """
    data_dir = Path(data_dir)

    # Construct file paths
    annot_matrix_file = data_dir / f"{trait_name}.npy"
    tracks_file = data_dir / f"{trait_name}.tracks.csv"
    variants_file = data_dir / f"{trait_name}.variants.csv"

    # Validate files exist
    for file_path in [annot_matrix_file, tracks_file, variants_file]:
        if not file_path.exists():
            print(f"Error: Required file not found: {file_path}")
            sys.exit(1)

    print("Loading annotation matrix...")
    annot_matrix = np.load(annot_matrix_file)
    print(f"  Shape: {annot_matrix.shape}")

    print("Loading tracks metadata...")
    tracks_df = pd.read_csv(tracks_file)
    # Remove the first column (index column)
    track_names = tracks_df.iloc[:, 1].values
    print(f"  Number of tracks: {len(track_names)}")

    print("Loading variants metadata...")
    variants_df = pd.read_csv(variants_file)
    # Remove the first column (index column) and get CHR, BP, SNP, A1, A2
    variants_df = variants_df.iloc[:, 1:]  # Skip index column
    print(f"  Number of variants: {len(variants_df)}")

    # Compute L2 norm across all tracks for each variant
    print("Computing L2 norm across all tracks...")
    l2_norm = np.linalg.norm(annot_matrix, axis=1, keepdims=True)
    print(f"  L2 norm shape: {l2_norm.shape}")
    print(f"  L2 norm range: [{l2_norm.min():.4f}, {l2_norm.max():.4f}]")

    # Append L2 norm as a new column to the annotation matrix
    annot_matrix = np.hstack([annot_matrix, l2_norm])
    print(f"  Updated matrix shape: {annot_matrix.shape}")

    # Append 'all' to track names
    track_names = np.append(track_names, 'all')
    print(f"  Total tracks (including 'all'): {len(track_names)}")

    return annot_matrix, track_names, variants_df

def load_reference_bim(chrom, ref_dir):
    """Load reference bim file for a chromosome.

    Args:
        chrom: Chromosome number
        ref_dir: Path to reference directory

    Returns:
        DataFrame with reference bim data
    """
    ref_dir = Path(ref_dir)
    bim_file = ref_dir / f"1000G.EUR.QC.{chrom}.bim"
    bim_df = pd.read_csv(
        bim_file,
        sep="\t",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
    )
    return bim_df

def create_annotation_file(chrom, track_idx, track_name, annot_data, variants_df, ref_dir, output_dir):
    """Create annotation file for one chromosome and one track.

    Args:
        chrom: Chromosome number
        track_idx: Index of the track in the annotation matrix
        track_name: Name of the track
        annot_data: Full annotation matrix
        variants_df: DataFrame with variant metadata
        ref_dir: Path to reference directory
        output_dir: Path to output directory

    Returns:
        Dictionary with result information
    """
    output_dir = Path(output_dir)
    # Filter variants for this chromosome
    chrom_str = f"chr{chrom}"
    mask = variants_df["CHR"] == chrom_str
    chrom_variants = variants_df[mask].copy()
    chrom_annot = annot_data[mask, track_idx]

    if len(chrom_variants) == 0:
        return {
            'track': track_name,
            'chrom': chrom,
            'status': 'skipped',
            'reason': 'no_variants'
        }

    # Load reference bim file
    ref_bim = load_reference_bim(chrom, ref_dir)

    # Merge with reference to get CM values and proper ordering
    # Create merge dataframe
    merge_df = pd.DataFrame({
        "SNP": chrom_variants["SNP"].values,
        "BP": chrom_variants["BP"].values,
        track_name: chrom_annot
    })

    # Merge with reference (left join on reference to keep all ref SNPs)
    result = ref_bim.merge(merge_df, on="SNP", how="left", suffixes=("", "_annot"))

    # Fill missing annotation values with 0
    result[track_name] = result[track_name].fillna(0)

    # Select and order columns: CHR, BP, SNP, CM, [annotation]
    output_df = result[["CHR", "BP", "SNP", "CM", track_name]]

    # Create output directory for this track
    track_output_dir = output_dir / track_name
    track_output_dir.mkdir(parents=True, exist_ok=True)

    # Write annotation file
    output_file = track_output_dir / f"{track_name}.{chrom}.annot.gz"
    output_df.to_csv(output_file, sep="\t", index=False, compression="gzip")

    return {
        'track': track_name,
        'chrom': chrom,
        'status': 'success',
        'output_file': str(output_file.name),
        'total_snps': len(output_df),
        'nonzero_annot': int((result[track_name] > 0).sum())
    }

def main(data_dir, trait_name, ref_dir=None, output_dir=None, n_jobs=-1):
    """Main processing function.

    Args:
        data_dir: Path to directory containing trait data files
        trait_name: Name of the trait
        ref_dir: Path to reference directory (default: use DEFAULT_REF_DIR)
        output_dir: Path to output directory (default: use DEFAULT_OUTPUT_DIR)
        n_jobs: Number of parallel jobs (-1 uses all available cores)
    """
    # Set defaults if not provided
    if ref_dir is None:
        ref_dir = DEFAULT_REF_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)

    print("="*70)
    print("Creating LDSC annotation files from continuous annotations")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Trait name: {trait_name}")
    print(f"Reference directory: {ref_dir}")
    print(f"Output directory: {output_dir}")
    print("")

    # Load data
    annot_matrix, track_names, variants_df = load_data(data_dir, trait_name)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create list of all tasks (track_idx, track_name, chrom)
    tasks = []
    for track_idx, track_name in enumerate(track_names):
        for chrom in range(1, 23):
            tasks.append((chrom, track_idx, track_name))

    print(f"\nProcessing {len(tasks)} tasks ({len(track_names)} tracks × 22 chromosomes)")
    print(f"Using joblib with n_jobs={n_jobs}")

    # Process all tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(create_annotation_file)(
            chrom, track_idx, track_name, annot_matrix, variants_df, ref_dir, output_dir
        )
        for chrom, track_idx, track_name in tasks
    )

    # Print summary
    print("\n" + "="*70)
    print("Summary:")
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(results)}")

    # Print details for successful tasks
    print("\nSuccessful annotations:")
    for result in results:
        if result['status'] == 'success':
            print(f"  {result['track']} chr{result['chrom']}: "
                  f"{result['total_snps']} SNPs, "
                  f"{result['nonzero_annot']} non-zero")

    print("\n" + "="*70)
    print("Annotation file creation complete!")
    print(f"Output directory: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create LDSC annotation files from continuous annotations (parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default schizophrenia data
  python 01_create_annotation_files.py --data-dir Data/source/Schizophrenia --trait-name Schizophrenia

  # Using custom trait data extracted from HDF5
  python 01_create_annotation_files.py --data-dir Data/source/my_trait --trait-name my_trait

  # With custom number of parallel jobs
  python 01_create_annotation_files.py --data-dir Data/source/trait --trait-name trait --n-jobs 8
        """
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to directory containing trait data files (.npy, .tracks.csv, .variants.csv)"
    )
    parser.add_argument(
        "--trait-name",
        required=True,
        help="Name of the trait (used to find input files: <trait-name>.npy, etc.)"
    )
    parser.add_argument(
        "--ref-dir",
        help=f"Path to reference directory (default: {DEFAULT_REF_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        help=f"Path to output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all available cores, default: -1)"
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        trait_name=args.trait_name,
        ref_dir=args.ref_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs
    )
