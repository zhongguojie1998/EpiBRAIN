#!/usr/bin/env python3
"""
Visualize bigwig files using pyGenomeTracks

This script:
1. Takes a folder containing bigwig files
2. Selects tracks based on pattern matching
3. Creates pyGenomeTracks configuration files
4. Generates genome browser-style plots for specified regions
"""

import subprocess
from pathlib import Path
import argparse
import sys
import numpy as np
from fnmatch import fnmatch

try:
    import pyBigWig
    PYBIGWIG_AVAILABLE = True
except ImportError:
    PYBIGWIG_AVAILABLE = False


def find_bigwig_files(bigwig_dir, track_patterns=None):
    """
    Find bigwig files in directory and filter by patterns.

    Args:
        bigwig_dir: Directory containing bigwig files
        track_patterns: Comma-separated string patterns to match
                       Supports wildcards: "BasalGanglia*ATAC,MiniAtlas*K27Ac"
                       Or simple substring: "ATAC,K27Ac"

    Returns:
        List of bigwig file paths
    """
    bigwig_path = Path(bigwig_dir)

    if not bigwig_path.exists():
        print(f"Error: Directory not found: {bigwig_dir}")
        sys.exit(1)

    # Find all bigwig files and sort them for consistent ordering
    all_bigwigs = sorted(list(bigwig_path.glob("*.bw")) + list(bigwig_path.glob("*.bigwig")))

    if len(all_bigwigs) == 0:
        print(f"Error: No bigwig files found in {bigwig_dir}")
        sys.exit(1)

    print(f"Found {len(all_bigwigs)} bigwig files in {bigwig_dir}")

    # Filter by patterns if provided
    if track_patterns:
        patterns = [p.strip() for p in track_patterns.split(',')]
        selected_bigwigs = []

        # Process each pattern sequentially to maintain pattern order
        for pattern in patterns:
            # Find files matching this pattern
            pattern_matches = []

            # Check if pattern contains wildcards
            has_wildcard = '*' in pattern or '?' in pattern

            for bw_file in all_bigwigs:
                if bw_file in selected_bigwigs:
                    continue

                # Use fnmatch for wildcard patterns, substring matching otherwise
                if has_wildcard:
                    if fnmatch(bw_file.name, pattern):
                        pattern_matches.append(bw_file)
                else:
                    if pattern in bw_file.name:
                        pattern_matches.append(bw_file)

            # Sort files within this pattern group
            pattern_matches.sort()

            # Append to final list (maintains pattern order)
            selected_bigwigs.extend(pattern_matches)

        print(f"Selected {len(selected_bigwigs)} bigwig files matching patterns: {patterns}")

        if len(selected_bigwigs) == 0:
            print("Warning: No bigwig files matched the specified patterns")
            print(f"Available files: {[f.name for f in all_bigwigs[:10]]}")
            sys.exit(1)

        return selected_bigwigs
    else:
        print("No track patterns specified, using all bigwig files")
        return all_bigwigs


def compute_global_minmax(bigwig_files, chrom, start, end):
    """
    Compute global min/max values for label and pred tracks.

    Args:
        bigwig_files: List of bigwig file paths
        chrom: Chromosome (e.g., 'chr1')
        start: Start position
        end: End position

    Returns:
        dict: {'label': (min, max), 'pred': (min, max)}
    """
    if not PYBIGWIG_AVAILABLE:
        print("Warning: pyBigWig not available. Using auto scaling.")
        print("Install with: pip install pyBigWig")
        return None

    # Ensure chromosome has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    print(f"\nComputing min/max values for region {chrom}:{start}-{end}...")

    label_values = []
    pred_values = []

    for bw_file in bigwig_files:
        track_lower = bw_file.stem.lower()

        try:
            bw = pyBigWig.open(str(bw_file))

            # Check if chromosome exists in bigwig
            if chrom not in bw.chroms():
                # Try without 'chr' prefix
                chrom_alt = chrom.replace('chr', '')
                if chrom_alt in bw.chroms():
                    chrom_query = chrom_alt
                else:
                    print(f"  Warning: Chromosome {chrom} not found in {bw_file.name}")
                    bw.close()
                    continue
            else:
                chrom_query = chrom

            # Get values for the region
            values = bw.values(chrom_query, start, end)
            bw.close()

            # Filter out None/NaN values
            values = [v for v in values if v is not None and not np.isnan(v)]

            if len(values) > 0:
                if "label" in track_lower:
                    label_values.extend(values)
                elif "pred" in track_lower:
                    pred_values.extend(values)

        except Exception as e:
            print(f"  Warning: Could not read {bw_file.name}: {e}")
            continue

    # Compute global min/max
    result = {}

    if label_values:
        label_min = 0  # Ensure min is at least 0
        label_max = int(np.ceil(np.max(label_values)))
        result['label'] = (label_min, label_max)
        print(f"  Label tracks: min={label_min:.4f}, max={label_max:.4f}")
    else:
        result['label'] = None
        print("  Label tracks: No data found")

    if pred_values:
        pred_min = 0  # Ensure min is at least 0
        pred_max = int(np.ceil(np.max(pred_values)))
        result['pred'] = (pred_min, pred_max)
        print(f"  Pred tracks: min={pred_min:.4f}, max={pred_max:.4f}")
    else:
        result['pred'] = None
        print("  Pred tracks: No data found")

    return result


def get_track_color(track_name, default_color="#1f77b4"):
    """
    Determine track color based on track name.

    Args:
        track_name: Name of the track
        default_color: Default color if no pattern matches

    Returns:
        Color string in hex format
    """
    track_lower = track_name.lower()

    if "label" in track_lower:
        return "#1f77b4"  # Blue
    elif "pred" in track_lower:
        return "#d62728"  # Red
    else:
        return default_color


def create_tracks_config(bigwig_files, output_file, height=2, default_color="#1f77b4",
                         label_min=None, label_max=None, pred_min=None, pred_max=None,
                         gtf_file=None, gtf_height=5, fontsize=10, gtf_fontsize=10,
                         axis_fontsize=12, spacer_height=0.2):
    """
    Create pyGenomeTracks configuration file from bigwig files.

    Args:
        bigwig_files: List of bigwig file paths
        output_file: Output configuration file path
        height: Track height (default: 2)
        default_color: Default track color (default: blue)
        label_min: Minimum value for label tracks
        label_max: Maximum value for label tracks
        pred_min: Minimum value for pred tracks
        pred_max: Maximum value for pred tracks
        gtf_file: Path to GTF file for gene annotation track (optional)
        gtf_height: Height of GTF track (default: 5)
        fontsize: Font size for track titles (default: 10)
        gtf_fontsize: Font size for GTF gene labels (default: 10)
        axis_fontsize: Font size for x-axis (default: 12)
        spacer_height: Height of spacer between tracks (default: 0.2)
    """
    print(f"\nCreating tracks configuration: {output_file}")

    config_lines = []

    for bw_file in bigwig_files:
        # Extract track name from filename (remove .bw or .bigwig extension)
        track_name = bw_file.stem

        # Remove prefixes if present
        if track_name.startswith("BasalGanglia-"):
            track_name = track_name[len("BasalGanglia-"):]
        elif track_name.startswith("MiniAtlas-"):
            track_name = track_name[len("MiniAtlas-"):]

        track_lower = track_name.lower()

        # FIRST: Determine color based on original track name (before removing suffixes)
        track_color = get_track_color(track_name, default_color)

        # FIRST: Determine min/max values based on track type (before removing suffixes)
        if "label" in track_lower:
            min_val = label_min
            max_val = label_max
        elif "pred" in track_lower:
            min_val = pred_min
            max_val = pred_max
        else:
            min_val = None
            max_val = None

        # THEN: Remove pred/label suffixes for cleaner display (color already assigned)
        for suffix in ["_pred", "_label", "-pred", "-label", "_Pred", "_Label", "-Pred", "-Label"]:
            if track_name.endswith(suffix):
                track_name = track_name[:-len(suffix)]
                break

        # THEN: Remove common modality suffixes (case-insensitive check)
        track_lower_temp = track_name.lower()
        for modality in ["_atac", "_rna", "_k27ac", "_k27me3", "_k9me3", "_rnaminus", "_rnaplus"]:
            if track_lower_temp.endswith(modality):
                track_name = track_name[:-len(modality)]
                break

        # Add track section
        config_lines.append(f"[{track_name}]")
        config_lines.append(f"file = {bw_file.absolute()}")
        config_lines.append(f"title = {track_name}")
        config_lines.append(f"height = {height}")
        config_lines.append(f"color = {track_color}")
        config_lines.append(f"fontsize = {fontsize}")

        # Only add min/max if they are specified (not None)
        if min_val is not None:
            config_lines.append(f"min_value = {min_val}")
        if max_val is not None:
            config_lines.append(f"max_value = {max_val}")

        config_lines.append("file_type = bigwig")
        config_lines.append("number_of_bins = 700")
        config_lines.append("")

        # Add small spacer between tracks
        config_lines.append("[spacer]")
        config_lines.append(f"height = {spacer_height}")
        config_lines.append("")

    # Add GTF track if provided
    if gtf_file:
        gtf_path = Path(gtf_file)
        if gtf_path.exists():
            config_lines.append("[genes]")
            config_lines.append(f"file = {gtf_path.absolute()}")
            config_lines.append("title = Genes")
            config_lines.append(f"height = {gtf_height}")
            config_lines.append("file_type = gtf")
            config_lines.append("prefered_name = gene_name")
            config_lines.append("merge_transcripts = true")
            config_lines.append("style = flybase")
            config_lines.append(f"fontsize = {gtf_fontsize}")
            config_lines.append("")

            config_lines.append("[spacer]")
            config_lines.append(f"height = {spacer_height}")
            config_lines.append("")
        else:
            print(f"  Warning: GTF file not found: {gtf_file}")

    # Add x-axis at the bottom
    config_lines.append("[x-axis]")
    config_lines.append("where = bottom")
    config_lines.append(f"fontsize = {axis_fontsize}")

    # Write configuration file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(config_lines))

    print(f"  Configuration saved: {output_file}")
    print(f"  Number of tracks: {len(bigwig_files)}")

    return output_file


def generate_plot(config_file, chrom, start, end, output_image, width=40, dpi=300):
    """
    Generate plot using pyGenomeTracks.

    Args:
        config_file: Path to tracks configuration file
        chrom: Chromosome (e.g., 'chr1' or '1')
        start: Start position (bp)
        end: End position (bp)
        output_image: Output image file path
        width: Figure width in cm (default: 40)
        dpi: Image resolution (default: 300)
    """
    # Ensure chromosome has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    region = f"{chrom}:{start}-{end}"
    print(f"\nGenerating plot for region {region}...")

    cmd = [
        "pyGenomeTracks",
        "--tracks", str(config_file),
        "--region", region,
        "--outFileName", str(output_image),
        "--width", str(width),
        "--dpi", str(dpi)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Plot saved: {output_image}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error generating plot:")
        print(f"    {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ✗ Error: pyGenomeTracks not found")
        print("    Install with: conda install -c bioconda pygenometracks")
        return False


def parse_region(region_str):
    """
    Parse region string in format 'chr11:113325659-113545400'

    Returns:
        tuple: (chrom, start, end)
    """
    try:
        chrom, coords = region_str.split(':')
        start, end = coords.split('-')
        return chrom, int(start), int(end)
    except (ValueError, AttributeError):
        print(f"Error: Invalid region format '{region_str}'")
        print("Expected format: chr11:113325659-113545400")
        sys.exit(1)


def find_inference_folders(base_dir):
    """
    Detect folder structure from 01_0_quick_inference_bigwig.py output.

    Args:
        base_dir: Base directory containing pred/, label/, alt/, etc.

    Returns:
        dict: {folder_name: Path} for detected folders
    """
    base_path = Path(base_dir)
    folders = {}

    # Check for common folders
    for folder_name in ['pred', 'label', 'alt', 'diff']:
        folder_path = base_path / folder_name
        if folder_path.exists() and folder_path.is_dir():
            # Count bigwig files
            bw_files = list(folder_path.glob("*.bw")) + list(folder_path.glob("*.bigwig"))
            if len(bw_files) > 0:
                folders[folder_name] = folder_path
                print(f"  Found {folder_name}/ folder with {len(bw_files)} tracks")

    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bigwig files using pyGenomeTracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot using region format (default: PDF output)
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      --region chr11:113325659-113545400 \\
      --output plot.pdf

  # Plot all bigwig files using separate coordinates
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      --chr chr1 --start 10000000 --end 10100000 \\
      --output plot.pdf

  # Select specific tracks by pattern matching (comma-separated patterns)
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      -t "ATAC,K27Ac" \\
      --region chr1:10000000-10100000 \\
      --output atac_k27ac_plot.pdf

  # Select tracks using wildcard patterns
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      -t "BasalGanglia*ATAC,MiniAtlas*ATAC" \\
      --region chr1:10000000-10100000 \\
      --output basal_mini_atac.pdf

  # Custom track heights and font sizes
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      --tracks "Astrocyte,Microglia,PVALB" \\
      --chr chr6 --start 25000000 --end 35000000 \\
      --output cell_types.pdf \\
      --height 3 --gtf-height 7 \\
      --fontsize 12 --gtf-fontsize 11 --axis-fontsize 14 \\
      --width 50 --dpi 600

  # Set consistent scale for label and pred tracks
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir predictions/ \\
      --region chr11:113325659-113545400 \\
      --label-min 0 --label-max 1 \\
      --pred-min 0 --pred-max 1 \\
      --output comparison.pdf

  # Include gene annotation track (GTF is included by default)
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir predictions/ \\
      --region chr11:113325659-113545400 \\
      --output with_genes.pdf

  # Disable gene annotation track
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir predictions/ \\
      --region chr11:113325659-113545400 \\
      --no-gtf \\
      --output without_genes.pdf

  # Quick inference visualization (auto-detects pred/, label/, alt/, diff/)
  python 00_visualize_data_pygenometrack.py \\
      --inference-dir Analysis/inference_outputs/gene_GRIN2A \\
      --region chr16:9849470-9849870 \\
      --output GRIN2A_visualization.pdf

  # Variant effect visualization (creates 3 plots: ref, alt, diff)
  python 00_visualize_data_pygenometrack.py \\
      --inference-dir Analysis/inference_outputs/variant_rs1234 \\
      --region chr1:154421970-154431970 \\
      --output variant_effect.pdf
        """
    )

    # Input directory arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--bigwig-dir",
                        help="Directory containing bigwig files")
    input_group.add_argument("--inference-dir",
                        help="Quick inference output directory (auto-detects pred/, label/, alt/, diff/ folders)")

    # Region specification (either --region OR --chr/--start/--end)
    parser.add_argument("--region", type=str,
                        help="Genomic region (e.g., 'chr11:113325659-113545400'). Overrides --chr/--start/--end")
    parser.add_argument("--chr", type=str,
                        help="Chromosome (e.g., 'chr1' or '1')")
    parser.add_argument("--start", type=int,
                        help="Start position (bp)")
    parser.add_argument("--end", type=int,
                        help="End position (bp)")

    # Optional arguments
    parser.add_argument("-t", "--tracks", type=str, default=None,
                        help="Comma-separated patterns to match track names. "
                             "Supports wildcards: 'BasalGanglia*ATAC,MiniAtlas*K27Ac' or "
                             "simple substring matching: 'ATAC,K27Ac'. "
                             "Use * to match any characters, ? to match single character.")
    parser.add_argument("--output", type=str, default="pygenometrack_plot.pdf",
                        help="Output image file (default: pygenometrack_plot.pdf)")
    parser.add_argument("--config", type=str, default=None,
                        help="Output config file (default: auto-generated)")
    parser.add_argument("--width", type=float, default=40,
                        help="Figure width in cm (default: 40)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Image resolution (default: 300)")
    parser.add_argument("--height", type=float, default=2,
                        help="BigWig track height (default: 2)")
    parser.add_argument("--spacer-height", type=float, default=0.2,
                        help="Height of spacer between tracks (default: 0.2, smaller = tighter)")
    parser.add_argument("--color", type=str, default="#1f77b4",
                        help="Default track color (default: #1f77b4). Note: tracks with 'label' → blue, 'pred' → red")

    # Font size settings
    parser.add_argument("--fontsize", type=int, default=10,
                        help="Font size for track titles (default: 10)")
    parser.add_argument("--gtf-fontsize", type=int, default=10,
                        help="Font size for GTF gene labels (default: 10)")
    parser.add_argument("--axis-fontsize", type=int, default=12,
                        help="Font size for x-axis labels (default: 12)")

    # Scale settings (optional overrides)
    parser.add_argument("--label-min", type=float, default=None,
                        help="Override minimum value for label tracks (default: auto-computed from data)")
    parser.add_argument("--label-max", type=float, default=None,
                        help="Override maximum value for label tracks (default: auto-computed from data)")
    parser.add_argument("--pred-min", type=float, default=None,
                        help="Override minimum value for pred tracks (default: auto-computed from data)")
    parser.add_argument("--pred-max", type=float, default=None,
                        help="Override maximum value for pred tracks (default: auto-computed from data)")
    parser.add_argument("--no-auto-scale", action="store_true",
                        help="Disable automatic min/max computation (use pyGenomeTracks auto)")

    # GTF annotation track
    parser.add_argument("--gtf", type=str, default="Data/source/gencode.v48.annotation.gtf.gz",
                        help="Path to GTF file for gene annotation track (default: Data/source/gencode.v48.annotation.gtf.gz)")
    parser.add_argument("--gtf-height", type=float, default=5,
                        help="Height of GTF annotation track (default: 5)")
    parser.add_argument("--no-gtf", action="store_true",
                        help="Disable GTF annotation track")

    args = parser.parse_args()

    # Parse region coordinates
    if args.region:
        # Region format overrides individual coordinates
        chrom, start, end = parse_region(args.region)
        print(f"Using region: {args.region}")
    elif args.chr and args.start is not None and args.end is not None:
        # Use individual coordinates
        chrom = args.chr
        start = args.start
        end = args.end
    else:
        print("Error: Must specify either --region or all of --chr/--start/--end")
        parser.print_help()
        sys.exit(1)

    print("="*70)
    print("pyGenomeTracks Visualization for BigWig Files")
    print("="*70)
    print(f"Region: {chrom}:{start}-{end}")

    # Handle inference directory mode
    if args.inference_dir:
        print(f"\nDetecting inference folder structure in: {args.inference_dir}")
        inference_folders = find_inference_folders(args.inference_dir)

        if not inference_folders:
            print(f"Error: No valid folders (pred/, label/, alt/, diff/) found in {args.inference_dir}")
            sys.exit(1)

        # Determine which folders to visualize
        output_path = Path(args.output)
        output_base = output_path.stem
        output_ext = output_path.suffix
        output_dir = output_path.parent

        # Create separate plots for each folder type
        plot_configs = []

        # Priority order: pred (reference), alt (variant), diff (variant), label (ground truth)
        for folder_type in ['pred', 'alt', 'diff', 'label']:
            if folder_type in inference_folders:
                folder_path = inference_folders[folder_type]

                # Determine output filename
                if folder_type == 'pred':
                    # For pred, use the main output name (could be ref or just pred)
                    folder_output = output_path
                    folder_label = "predictions" if 'alt' not in inference_folders else "reference allele"
                else:
                    folder_output = output_dir / f"{output_base}_{folder_type}{output_ext}"
                    folder_label = {
                        'alt': 'alternative allele',
                        'diff': 'difference (alt - ref)',
                        'label': 'ground truth labels'
                    }.get(folder_type, folder_type)

                plot_configs.append({
                    'folder_type': folder_type,
                    'folder_path': folder_path,
                    'output_path': folder_output,
                    'label': folder_label
                })

        print(f"\nWill create {len(plot_configs)} plot(s):")
        for config in plot_configs:
            print(f"  - {config['label']}: {config['output_path']}")

        # Generate plots for each folder
        all_success = True
        for config in plot_configs:
            print(f"\n{'='*70}")
            print(f"Processing {config['label']} ({config['folder_type']}/)...")
            print(f"{'='*70}")

            # Find bigwig files in this folder
            bigwig_files = find_bigwig_files(str(config['folder_path']), args.tracks)

            # Compute min/max for this folder
            label_min = args.label_min
            label_max = args.label_max
            pred_min = args.pred_min
            pred_max = args.pred_max

            if not args.no_auto_scale:
                minmax_data = compute_global_minmax(bigwig_files, chrom, start, end)

                if minmax_data:
                    if minmax_data.get('label'):
                        computed_label_min, computed_label_max = minmax_data['label']
                        if label_min is None:
                            label_min = computed_label_min
                        if label_max is None:
                            label_max = computed_label_max

                    if minmax_data.get('pred'):
                        computed_pred_min, computed_pred_max = minmax_data['pred']
                        if pred_min is None:
                            pred_min = computed_pred_min
                        if pred_max is None:
                            pred_max = computed_pred_max

            # Print scale settings
            print("\nScale settings:")
            if label_min is not None or label_max is not None:
                print(f"  Label tracks: [{label_min}, {label_max}]")
            else:
                print(f"  Label tracks: auto")
            if pred_min is not None or pred_max is not None:
                print(f"  Pred tracks: [{pred_min}, {pred_max}]")
            else:
                print(f"  Pred tracks: auto")

            print("\nSelected tracks:")
            for i, bw in enumerate(bigwig_files, 1):
                color_label = "blue" if "label" in bw.stem.lower() else ("red" if "pred" in bw.stem.lower() else "default")
                print(f"  {i}. {bw.name} → {color_label}")

            # Create configuration file
            config_file = config['output_path'].parent / f"{config['output_path'].stem}_config.ini"

            # Determine GTF file to use
            gtf_file = None if args.no_gtf else args.gtf

            create_tracks_config(bigwig_files, config_file,
                                height=args.height, default_color=args.color,
                                label_min=label_min, label_max=label_max,
                                pred_min=pred_min, pred_max=pred_max,
                                gtf_file=gtf_file, gtf_height=args.gtf_height,
                                fontsize=args.fontsize, gtf_fontsize=args.gtf_fontsize,
                                axis_fontsize=args.axis_fontsize,
                                spacer_height=args.spacer_height)

            # Generate plot
            success = generate_plot(
                config_file,
                chrom,
                start,
                end,
                config['output_path'],
                width=args.width,
                dpi=args.dpi
            )

            if success:
                print(f"  ✓ {config['label'].capitalize()} plot saved: {config['output_path']}")
            else:
                print(f"  ✗ Failed to generate {config['label']} plot")
                all_success = False

        # Final summary
        if all_success:
            print("\n" + "="*70)
            print("All visualizations complete!")
            print("="*70)
            print(f"\nGenerated {len(plot_configs)} plot(s):")
            for config in plot_configs:
                print(f"  - {config['output_path']}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("Some visualizations failed!")
            print("="*70)
            sys.exit(1)

        return

    # Standard mode: single bigwig directory
    # Find and filter bigwig files
    bigwig_files = find_bigwig_files(args.bigwig_dir, args.tracks)

    # Compute global min/max for label and pred tracks
    label_min = args.label_min
    label_max = args.label_max
    pred_min = args.pred_min
    pred_max = args.pred_max

    if not args.no_auto_scale:
        # Compute min/max from data
        minmax_data = compute_global_minmax(bigwig_files, chrom, start, end)

        if minmax_data:
            # Use computed values if not overridden by user
            if minmax_data.get('label'):
                computed_label_min, computed_label_max = minmax_data['label']
                if label_min is None:
                    label_min = computed_label_min
                if label_max is None:
                    label_max = computed_label_max

            if minmax_data.get('pred'):
                computed_pred_min, computed_pred_max = minmax_data['pred']
                if pred_min is None:
                    pred_min = computed_pred_min
                if pred_max is None:
                    pred_max = computed_pred_max

    # Print scale settings
    print("\nScale settings:")
    if label_min is not None or label_max is not None:
        print(f"  Label tracks: [{label_min}, {label_max}]")
    else:
        print(f"  Label tracks: auto")
    if pred_min is not None or pred_max is not None:
        print(f"  Pred tracks: [{pred_min}, {pred_max}]")
    else:
        print(f"  Pred tracks: auto")

    print("\nSelected tracks:")
    for i, bw in enumerate(bigwig_files, 1):
        color_label = "blue" if "label" in bw.stem.lower() else ("red" if "pred" in bw.stem.lower() else "default")
        print(f"  {i}. {bw.name} → {color_label}")

    # Create configuration file
    if args.config:
        config_file = Path(args.config)
    else:
        output_path = Path(args.output)
        config_file = output_path.parent / f"{output_path.stem}_config.ini"

    # Determine GTF file to use
    gtf_file = None if args.no_gtf else args.gtf

    create_tracks_config(bigwig_files, config_file,
                        height=args.height, default_color=args.color,
                        label_min=label_min, label_max=label_max,
                        pred_min=pred_min, pred_max=pred_max,
                        gtf_file=gtf_file, gtf_height=args.gtf_height,
                        fontsize=args.fontsize, gtf_fontsize=args.gtf_fontsize,
                        axis_fontsize=args.axis_fontsize,
                        spacer_height=args.spacer_height)

    # Generate plot
    output_image = Path(args.output)
    success = generate_plot(
        config_file,
        chrom,
        start,
        end,
        output_image,
        width=args.width,
        dpi=args.dpi
    )

    if success:
        print("\n" + "="*70)
        print("Visualization complete!")
        print(f"  Plot: {output_image}")
        print(f"  Config: {config_file}")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Visualization failed!")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()
