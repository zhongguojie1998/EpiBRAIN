#!/usr/bin/env python3
"""
Visualize saturation mutagenesis results as heatmaps.

This script creates heatmaps showing the effect of all possible single nucleotide
variants at each position in a region. The heatmap shows:
- Rows: Alternative alleles (A, C, G, T)
- Columns: Genomic positions
- Values: Variant effect scores (e.g., diff_gene_sum)

Usage examples:
    # Visualize all tracks
    python 03_6_visualize_mutagenesis.py --input Res/bigwig/rs356182_SNCA_sat_muta/

    # Visualize specific track
    python 03_6_visualize_mutagenesis.py \
        --input Res/bigwig/rs356182_SNCA_sat_muta/ \
        --track "BasalGanglia-STR-D1-MSN_RNAminus"

    # Use different value column
    python 03_6_visualize_mutagenesis.py \
        --input Res/bigwig/rs356182_SNCA_sat_muta/ \
        --value diff_local_mean

    # Filter to specific cell types
    python 03_6_visualize_mutagenesis.py \
        --input Res/bigwig/rs356182_SNCA_sat_muta/ \
        --filter-celltype "Microglia"
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_saturation_mutagenesis_results(results_file):
    """
    Load saturation mutagenesis results from TSV file.

    Args:
        results_file: Path to saturation_mutagenesis_results.tsv

    Returns:
        DataFrame with results
    """
    df = pd.read_csv(results_file, sep='\t')
    print(f"Loaded {len(df)} results from {results_file}")
    print(f"  Unique positions: {df['position'].nunique()}")
    print(f"  Unique tracks: {df['track_name'].nunique()}")
    print(f"  Unique alleles: {sorted(df['alt'].unique())}")

    return df


def create_heatmap(data, track_name, value_column, output_file, figsize=(12, 4),
                   cmap='coolwarm', center=0, vmin=None, vmax=None):
    """
    Create heatmap for saturation mutagenesis results.

    Args:
        data: DataFrame with columns [position, alt, value_column]
        track_name: Name of the track for the title
        value_column: Column name to use for heatmap values
        output_file: Path to save the figure
        figsize: Figure size (width, height)
        cmap: Colormap name
        center: Center value for diverging colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    # Pivot data: rows = alternative alleles, columns = positions
    pivot_data = data.pivot(index='alt', columns='position', values=value_column)

    # Sort alleles alphabetically
    pivot_data = pivot_data.sort_index()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': value_column},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Set labels
    ax.set_xlabel('Genomic Position', fontsize=12)
    ax.set_ylabel('Alternative Allele', fontsize=12)
    ax.set_title(f'Saturation Mutagenesis: {track_name}', fontsize=14, fontweight='bold')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved heatmap to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saturation mutagenesis results as heatmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all tracks
  python 03_6_visualize_mutagenesis.py --input Res/bigwig/rs356182_SNCA_sat_muta/

  # Visualize specific track
  python 03_6_visualize_mutagenesis.py --input Res/bigwig/rs356182_SNCA_sat_muta/ --track "BasalGanglia-STR-D1-MSN_RNAminus"

  # Use different value column
  python 03_6_visualize_mutagenesis.py --input Res/bigwig/rs356182_SNCA_sat_muta/ --value diff_local_mean

  # Filter to specific modalities
  python 03_6_visualize_mutagenesis.py --input Res/bigwig/rs356182_SNCA_sat_muta/ --filter-modality "RNAminus"
        """
    )

    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing saturation_mutagenesis_results.tsv')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for heatmaps (default: same as input)')

    # Track selection
    parser.add_argument('--track', '-t', type=str, default=None,
                       help='Specific track name to visualize (default: all tracks)')
    parser.add_argument('--filter-celltype', type=str, default=None,
                       help='Filter tracks by cell type substring (e.g., "Microglia")')
    parser.add_argument('--filter-modality', type=str, default=None,
                       help='Filter tracks by modality substring (e.g., "ATAC", "RNAminus")')

    # Value column
    parser.add_argument('--value', '-v', type=str, default='diff_gene_sum',
                       choices=['diff_gene_sum', 'diff_gene_mean', 'diff_local_mean',
                               'diff_local_max', 'diff_mean', 'diff_max'],
                       help='Value column to use for heatmap (default: diff_gene_sum)')

    # Visualization options
    parser.add_argument('--figsize', type=str, default='12,4',
                       help='Figure size as "width,height" (default: 12,4)')
    parser.add_argument('--cmap', type=str, default='coolwarm',
                       help='Colormap name (default: coolwarm)')
    parser.add_argument('--center', type=float, default=0,
                       help='Center value for diverging colormap (default: 0)')
    parser.add_argument('--vmin', type=float, default=None,
                       help='Minimum value for colormap (default: auto)')
    parser.add_argument('--vmax', type=float, default=None,
                       help='Maximum value for colormap (default: auto)')

    args = parser.parse_args()

    # Resolve input/output paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    results_file = input_dir / "saturation_mutagenesis_results.tsv"
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Make sure to run saturation mutagenesis first with:")
        print("  python Analysis/01_0_quick_inference_bigwig.py --sat_mutagenesis ...")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        print(f"Error: Invalid figsize format: {args.figsize}")
        print("Expected format: 'width,height' (e.g., '12,4')")
        sys.exit(1)

    # Load results
    print("="*80)
    print("Saturation Mutagenesis Visualization")
    print("="*80)
    print(f"Input: {results_file}")
    print(f"Output: {output_dir}")
    print(f"Value column: {args.value}")
    print()

    df = load_saturation_mutagenesis_results(results_file)

    # Check if value column exists
    if args.value not in df.columns:
        print(f"Error: Value column '{args.value}' not found in results")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Filter tracks
    tracks_to_plot = df['track_name'].unique()

    if args.track:
        # Specific track
        if args.track not in tracks_to_plot:
            print(f"Error: Track '{args.track}' not found in results")
            print(f"Available tracks: {sorted(tracks_to_plot)}")
            sys.exit(1)
        tracks_to_plot = [args.track]
    else:
        # Apply filters
        if args.filter_celltype:
            tracks_to_plot = [t for t in tracks_to_plot if args.filter_celltype in t]
            print(f"Filtered to {len(tracks_to_plot)} tracks containing '{args.filter_celltype}'")

        if args.filter_modality:
            tracks_to_plot = [t for t in tracks_to_plot if args.filter_modality in t]
            print(f"Filtered to {len(tracks_to_plot)} tracks containing '{args.filter_modality}'")

    if len(tracks_to_plot) == 0:
        print("Error: No tracks matched the filters")
        sys.exit(1)

    print(f"\nGenerating heatmaps for {len(tracks_to_plot)} tracks...")
    print()

    # Generate heatmaps
    for track in sorted(tracks_to_plot):
        print(f"Processing: {track}")

        # Filter data for this track
        track_data = df[df['track_name'] == track].copy()

        # Skip if no data
        if len(track_data) == 0:
            print(f"  Warning: No data found for track {track}")
            continue

        # Check if value column has non-null values
        if track_data[args.value].isna().all():
            print(f"  Warning: All values are null for {args.value}, skipping")
            continue

        # Create output filename
        safe_track_name = track.replace('/', '_').replace('\\', '_')
        output_file = output_dir / f"heatmap_{safe_track_name}.pdf"

        # Create heatmap
        create_heatmap(
            track_data[['position', 'alt', args.value]],
            track,
            args.value,
            output_file,
            figsize=figsize,
            cmap=args.cmap,
            center=args.center,
            vmin=args.vmin,
            vmax=args.vmax
        )

    print()
    print("="*80)
    print("Complete!")
    print(f"Generated {len(tracks_to_plot)} heatmaps in {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
