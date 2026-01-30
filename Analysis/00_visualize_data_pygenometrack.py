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


def compute_global_minmax(bigwig_files, chrom, start, end, track_type=None, num_bins=700, allow_negative=False):
    """
    Compute global min/max values for label and pred tracks with binning.

    Args:
        bigwig_files: List of bigwig file paths
        chrom: Chromosome (e.g., 'chr1')
        start: Start position
        end: End position
        track_type: Optional track type ('label' or 'pred') to override filename detection
        num_bins: Number of bins for smoothing (default: 700, same as pyGenomeTracks)
        allow_negative: If True, don't force min to 0 (for diff tracks)

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

            # Get binned statistics (same as pyGenomeTracks does internally)
            # pyGenomeTracks uses stats() which computes mean over bins
            binned_values = bw.stats(chrom_query, start, end, type="mean", nBins=num_bins)
            bw.close()

            # Filter out None/NaN values
            binned_values = [v for v in binned_values if v is not None and not np.isnan(v)]

            if len(binned_values) > 0:
                # Use track_type if provided, otherwise detect from filename
                if track_type == 'label':
                    label_values.extend(binned_values)
                elif track_type == 'pred':
                    pred_values.extend(binned_values)
                elif "label" in track_lower:
                    label_values.extend(binned_values)
                elif "pred" in track_lower:
                    pred_values.extend(binned_values)

        except Exception as e:
            print(f"  Warning: Could not read {bw_file.name}: {e}")
            continue

    # Compute global min/max
    result = {}

    if label_values:
        if allow_negative:
            label_min = round(np.min(label_values), 3)
            label_max = round(np.max(label_values), 3)
        else:
            label_min = 0  # Ensure min is at least 0
            label_max = round(np.max(label_values), 3)
        result['label'] = (label_min, label_max)
        print(f"  Label tracks: min={label_min:.4f}, max={label_max:.4f} (computed from {num_bins} bins)")
    else:
        result['label'] = None
        print("  Label tracks: No data found")

    if pred_values:
        if allow_negative:
            pred_min = round(np.min(pred_values), 3)
            pred_max = round(np.max(pred_values), 3)
        else:
            pred_min = 0  # Ensure min is at least 0
            pred_max = round(np.max(pred_values), 3)
        result['pred'] = (pred_min, pred_max)
        print(f"  Pred tracks: min={pred_min:.4f}, max={pred_max:.4f} (computed from {num_bins} bins)")
    else:
        result['pred'] = None
        print("  Pred tracks: No data found")

    return result


def compute_minmax_per_modality(bigwig_files, chrom, start, end, track_type_map=None, num_bins=700, allow_negative=False):
    """
    Compute min/max values grouped by track type (label/pred) and modality suffix.

    Args:
        bigwig_files: List of bigwig file paths
        chrom: Chromosome (e.g., 'chr1')
        start: Start position
        end: End position
        track_type_map: Optional dict mapping file paths to 'label' or 'pred'
        num_bins: Number of bins for smoothing (default: 700)
        allow_negative: If True, don't force min to 0

    Returns:
        dict: {(track_type, modality): (min, max)}
              e.g., {('label', '_atac'): (0, 10), ('pred', '_atac'): (0, 12), ...}
    """
    if not PYBIGWIG_AVAILABLE:
        print("Warning: pyBigWig not available. Using auto scaling.")
        print("Install with: pip install pyBigWig")
        return {}

    # Ensure chromosome has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    print(f"\nComputing per-modality min/max values for region {chrom}:{start}-{end}...")

    # Group values by (track_type, modality)
    grouped_values = {}

    for bw_file in bigwig_files:
        track_name = bw_file.stem
        track_lower = track_name.lower()

        # Determine track type (label/pred)
        if track_type_map and bw_file in track_type_map:
            track_type = track_type_map[bw_file]
        elif "label" in track_lower:
            track_type = 'label'
        elif "pred" in track_lower:
            track_type = 'pred'
        else:
            track_type = 'other'

        # Extract modality suffix
        modality = get_modality_suffix(track_name)
        if modality is None:
            modality = 'none'  # For tracks without modality suffix

        key = (track_type, modality)

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

            # Get binned statistics
            binned_values = bw.stats(chrom_query, start, end, type="mean", nBins=num_bins)
            bw.close()

            # Filter out None/NaN values
            binned_values = [v for v in binned_values if v is not None and not np.isnan(v)]

            if len(binned_values) > 0:
                if key not in grouped_values:
                    grouped_values[key] = []
                grouped_values[key].extend(binned_values)

        except Exception as e:
            print(f"  Warning: Could not read {bw_file.name}: {e}")
            continue

    # Compute min/max for each group
    result = {}
    for key, values in grouped_values.items():
        track_type, modality = key
        if values:
            if allow_negative:
                min_val = round(np.min(values), 3)
                max_val = round(np.max(values), 3)
            else:
                min_val = 0
                max_val = round(np.max(values), 3)
            result[key] = (min_val, max_val)
            print(f"  {track_type}/{modality}: min={min_val:.4f}, max={max_val:.4f} (n={len(values)} bins)")

    return result


def get_modality_suffix(track_name, modality_suffixes=None):
    """
    Extract the modality suffix from a track name.

    Args:
        track_name: Track name string
        modality_suffixes: List of modality suffixes to check (default: common modalities)

    Returns:
        Modality suffix string if found, otherwise None
    """
    if modality_suffixes is None:
        modality_suffixes = ["_atac", "_rna", "_k27ac", "_k27me3", "_k9me3", "_rnaminus", "_rnaplus"]

    track_lower = track_name.lower()
    for modality in modality_suffixes:
        if track_lower.endswith(modality):
            return modality

    return None


def get_common_suffix(track_names, modality_suffixes=None):
    """
    Determine if all track names share a common modality suffix.

    Args:
        track_names: List of track name strings
        modality_suffixes: List of modality suffixes to check (default: common modalities)

    Returns:
        Common suffix string if all tracks share it, otherwise None
    """
    if modality_suffixes is None:
        modality_suffixes = ["_atac", "_rna", "_k27ac", "_k27me3", "_k9me3", "_rnaminus", "_rnaplus"]

    if not track_names:
        return None

    # Check each modality suffix
    for modality in modality_suffixes:
        # Check if all tracks end with this suffix (case-insensitive)
        if all(name.lower().endswith(modality) for name in track_names):
            return modality

    return None


def get_track_color(track_name, default_color="#1f77b4", label_color="#1f77b4", ref_color="#d62728"):
    """
    Determine track color based on track name.

    Args:
        track_name: Name of the track
        default_color: Default color if no pattern matches
        label_color: Color for label tracks
        ref_color: Color for ref/pred tracks

    Returns:
        Color string in hex format
    """
    track_lower = track_name.lower()

    if "label" in track_lower:
        return label_color
    elif "pred" in track_lower:
        return ref_color
    else:
        return default_color


def create_tracks_config(bigwig_files, output_file, height=2, default_color="#1f77b4",
                         label_min=None, label_max=None, pred_min=None, pred_max=None,
                         minmax_per_modality=None,
                         gtf_file=None, gtf_height=5, fontsize=10, gtf_fontsize=10,
                         axis_fontsize=12, spacer_height=0.2,
                         label_color="#1f77b4", ref_color="#d62728"):
    """
    Create pyGenomeTracks configuration file from bigwig files.

    Args:
        bigwig_files: List of bigwig file paths
        output_file: Output configuration file path
        height: Track height (default: 2)
        default_color: Default track color (default: blue)
        label_min: Minimum value for label tracks (deprecated, use minmax_per_modality)
        label_max: Maximum value for label tracks (deprecated, use minmax_per_modality)
        pred_min: Minimum value for pred tracks (deprecated, use minmax_per_modality)
        pred_max: Maximum value for pred tracks (deprecated, use minmax_per_modality)
        minmax_per_modality: Dict of {(track_type, modality): (min, max)} for per-modality scaling
        gtf_file: Path to GTF file for gene annotation track (optional)
        gtf_height: Height of GTF track (default: 5)
        fontsize: Font size for track titles (default: 10)
        gtf_fontsize: Font size for GTF gene labels (default: 10)
        axis_fontsize: Font size for x-axis (default: 12)
        spacer_height: Height of spacer between tracks (default: 0.2)
    """
    print(f"\nCreating tracks configuration: {output_file}")

    # FIRST PASS: Collect all track names after prefix and pred/label suffix removal
    # to determine if they share a common modality suffix
    track_names_for_suffix_check = []
    for bw_file in bigwig_files:
        track_name = bw_file.stem

        # Remove prefixes if present
        if track_name.startswith("BasalGanglia-"):
            track_name = track_name[len("BasalGanglia-"):]
        elif track_name.startswith("MiniAtlas-"):
            track_name = track_name[len("MiniAtlas-"):]

        # Remove pred/label suffixes
        for suffix in ["_pred", "_label", "-pred", "-label", "_Pred", "_Label", "-Pred", "-Label"]:
            if track_name.endswith(suffix):
                track_name = track_name[:-len(suffix)]
                break

        track_names_for_suffix_check.append(track_name)

    # Check if all tracks share a common modality suffix
    common_suffix = get_common_suffix(track_names_for_suffix_check)
    if common_suffix:
        print(f"  All tracks share common suffix '{common_suffix}' - will remove it from display names")
    else:
        print(f"  Tracks have different suffixes - keeping them for distinction")

    # SECOND PASS: Create config
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
        track_color = get_track_color(track_name, default_color, label_color, ref_color)

        # FIRST: Determine min/max values based on track type and modality (before removing suffixes)
        # Determine track type
        if "label" in track_lower:
            track_type = 'label'
        elif "pred" in track_lower:
            track_type = 'pred'
        else:
            track_type = 'other'

        # Extract modality from original track name
        modality = get_modality_suffix(track_name)
        if modality is None:
            modality = 'none'

        # Look up min/max from per-modality dict, fallback to legacy label/pred min/max
        min_val = None
        max_val = None
        if minmax_per_modality and (track_type, modality) in minmax_per_modality:
            min_val, max_val = minmax_per_modality[(track_type, modality)]
        else:
            # Fallback to legacy parameters
            if track_type == 'label':
                min_val = label_min
                max_val = label_max
            elif track_type == 'pred':
                min_val = pred_min
                max_val = pred_max

        # THEN: Remove pred/label suffixes for cleaner display (color already assigned)
        for suffix in ["_pred", "_label", "-pred", "-label", "_Pred", "_Label", "-Pred", "-Label"]:
            if track_name.endswith(suffix):
                track_name = track_name[:-len(suffix)]
                break

        # THEN: Remove common modality suffix ONLY if all tracks share it
        if common_suffix:
            track_lower_temp = track_name.lower()
            if track_lower_temp.endswith(common_suffix):
                track_name = track_name[:-len(common_suffix)]

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


def get_gene_coordinates(gene_name, gtf_file):
    """
    Look up gene coordinates from GTF file and return center +/- 262144bp region.

    Args:
        gene_name: Name of the gene to look up
        gtf_file: Path to GTF file

    Returns:
        tuple: (chrom, start, end) for the region
    """
    import gzip

    gtf_path = Path(gtf_file)
    if not gtf_path.exists():
        print(f"Error: GTF file not found: {gtf_file}")
        sys.exit(1)

    print(f"Looking up gene '{gene_name}' in {gtf_file}...")

    # Track gene boundaries
    gene_chrom = None
    gene_start = None
    gene_end = None

    # Open GTF file (handle gzipped files)
    open_func = gzip.open if gtf_file.endswith('.gz') else open

    with open_func(gtf_path, 'rt') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            attributes = fields[8]

            # Only look at gene features
            if feature_type != 'gene':
                continue

            # Parse attributes to find gene_name
            attr_dict = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                parts = attr.split(' ', 1)
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1].strip('"')
                    attr_dict[key] = value

            # Check if this is our gene
            if attr_dict.get('gene_name') == gene_name:
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])

                # Initialize or expand gene boundaries
                if gene_chrom is None:
                    gene_chrom = chrom
                    gene_start = start
                    gene_end = end
                else:
                    gene_start = min(gene_start, start)
                    gene_end = max(gene_end, end)

    if gene_chrom is None:
        print(f"Error: Gene '{gene_name}' not found in GTF file")
        sys.exit(1)

    # Calculate center point
    gene_center = (gene_start + gene_end) // 2

    # Create region: center +/- 524288/2 bp
    window_size = 524288 // 2  # 262144 bp
    region_start = max(0, gene_center - window_size)
    region_end = gene_center + window_size

    print(f"  Found gene: {gene_chrom}:{gene_start}-{gene_end}")
    print(f"  Gene center: {gene_center}")
    print(f"  Visualization region: {gene_chrom}:{region_start}-{region_end} (center +/- {window_size}bp)")

    return gene_chrom, region_start, region_end


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


def parse_highlight_regions(highlight_str):
    """
    Parse highlight regions string.

    Args:
        highlight_str: Comma-separated regions or positions
                      Single position: 'chr4:89753280'
                      Region: 'chr4:89753280-89753285'
                      Multiple: 'chr4:100,chr4:200-300'

    Returns:
        tuple: (regions_list, is_single_positions)
               regions_list: List of (chrom, start, end) tuples
               is_single_positions: True if all are single positions (use vlines), False if regions (use vhighlight)
    """
    if not highlight_str:
        return [], False

    regions = []
    is_single_positions = True

    for region_str in highlight_str.split(','):
        region_str = region_str.strip()
        try:
            chrom, coords = region_str.split(':')
            if '-' in coords:
                # Region format: chr4:start-end
                start, end = coords.split('-')
                regions.append((chrom, int(start), int(end)))
                if int(start) != int(end):
                    is_single_positions = False
            else:
                # Single position format: chr4:pos
                pos = int(coords)
                regions.append((chrom, pos, pos))
        except (ValueError, AttributeError):
            print(f"Warning: Invalid highlight region format '{region_str}', skipping")
            continue

    return regions, is_single_positions


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

  # Plot region around a gene (automatically centers on gene +/- 262144bp)
  python 00_visualize_data_pygenometrack.py \\
      --bigwig-dir bigwig_files/ \\
      --gene LRRK2 \\
      --output LRRK2_plot.pdf

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

  # Quick inference with gene name (automatically centers on gene)
  python 00_visualize_data_pygenometrack.py \\
      --inference-dir Analysis/figures/genes/LRRK2/ \\
      --gene LRRK2 \\
      --tracks "BasalGanglia-*K27Ac*" \\
      --output Analysis/figures/genes/LRRK2/visualization.pdf

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

    # Region specification (either --region OR --chr/--start/--end OR --gene)
    parser.add_argument("--region", type=str,
                        help="Genomic region (e.g., 'chr11:113325659-113545400'). Overrides --chr/--start/--end and --gene")
    parser.add_argument("--gene", type=str,
                        help="Gene name to visualize (e.g., 'LRRK2'). Will automatically center on gene +/- 262144bp. Requires --gtf to be available.")
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

    # Track-specific color settings
    parser.add_argument("--label-color", type=str, default="#1f77b4",
                        help="Color for label tracks (default: #1f77b4, blue)")
    parser.add_argument("--ref-color", type=str, default="#d62728",
                        help="Color for reference/pred tracks (default: #d62728, red)")
    parser.add_argument("--alt-color", type=str, default="#2ca02c",
                        help="Color for alternative tracks (default: #2ca02c, green)")
    parser.add_argument("--diff-color", type=str, default="#9467bd",
                        help="Color for diff tracks (default: #9467bd, purple)")

    # Track transparency settings
    parser.add_argument("--ref-alpha", type=float, default=1.0,
                        help="Transparency for reference tracks, 0.0-1.0 (default: 1.0, opaque)")
    parser.add_argument("--alt-alpha", type=float, default=1.0,
                        help="Transparency for alternative tracks, 0.0-1.0 (default: 1.0, opaque)")

    # Track order settings
    parser.add_argument("--alt-first", action="store_true",
                        help="Display alt track first, then overlay ref (default: ref first, then alt)")

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

    # Highlight regions
    parser.add_argument("--highlight", type=str, default=None,
                        help="Regions to highlight (e.g., 'chr4:89700000-89850000' or multiple: 'chr4:100-200,chr4:300-400')")
    parser.add_argument("--highlight-color", type=str, default="#808080",
                        help="Color for highlighted regions (default: #808080, grey)")

    args = parser.parse_args()

    # Parse region coordinates
    if args.region:
        # Region format overrides individual coordinates and gene
        chrom, start, end = parse_region(args.region)
        print(f"Using region: {args.region}")
    elif args.gene:
        # Gene name - look up coordinates from GTF
        gtf_file = args.gtf if not args.no_gtf else "Data/source/gencode.v48.annotation.gtf.gz"
        chrom, start, end = get_gene_coordinates(args.gene, gtf_file)
    elif args.chr and args.start is not None and args.end is not None:
        # Use individual coordinates
        chrom = args.chr
        start = args.start
        end = args.end
    else:
        print("Error: Must specify either --region, --gene, or all of --chr/--start/--end")
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

        # Check if this is a variant analysis (has alt/ or diff/)
        is_variant_mode = 'alt' in inference_folders or 'diff' in inference_folders

        if is_variant_mode:
            # Variant mode: Create single plot with overlaid ref/alt tracks
            print(f"\nVariant mode detected. Creating plot with overlaid ref/alt tracks")

            # Get files from all folders
            label_files = []
            pred_files = []
            alt_files = []
            diff_files = []

            if 'label' in inference_folders:
                label_files = find_bigwig_files(str(inference_folders['label']), args.tracks)
                print(f"  Found {len(label_files)} label tracks")

            if 'pred' in inference_folders:
                pred_files = find_bigwig_files(str(inference_folders['pred']), args.tracks)
                print(f"  Found {len(pred_files)} pred (reference) tracks")

            if 'alt' in inference_folders:
                alt_files = find_bigwig_files(str(inference_folders['alt']), args.tracks)
                print(f"  Found {len(alt_files)} alt (alternative) tracks")

            if 'diff' in inference_folders:
                diff_files = find_bigwig_files(str(inference_folders['diff']), args.tracks)
                print(f"  Found {len(diff_files)} diff (alt - ref) tracks")

            # Create mappings from base name to files
            label_map = {f.stem: f for f in label_files}
            pred_map = {f.stem: f for f in pred_files}
            alt_map = {f.stem: f for f in alt_files}
            diff_map = {f.stem: f for f in diff_files}

            # Get all unique base names in the order they appear (preserve pattern order)
            # Don't sort - maintain the order from the --tracks argument
            all_base_names = []
            seen = set()
            # Add files in the order they appear in pred, alt, diff, then label
            for f in pred_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)
            for f in alt_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)
            for f in diff_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)
            for f in label_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)

            if len(all_base_names) == 0:
                print("Error: No bigwig files found matching the track patterns")
                sys.exit(1)

            print(f"\nTotal cell types to visualize: {len(all_base_names)}")

            # Compute global min/max for label, pred, alt, and diff tracks
            label_min = args.label_min
            label_max = args.label_max
            pred_min = args.pred_min
            pred_max = args.pred_max
            alt_min = pred_min  # Alt uses same scale as pred
            alt_max = pred_max
            diff_min = None  # Diff can be negative, will be computed separately
            diff_max = None

            # Compute per-modality min/max
            minmax_per_modality = {}

            if not args.no_auto_scale:
                # Create track type map for label files
                track_type_map_label = {f: 'label' for f in label_files}
                if label_files:
                    label_minmax_per_mod = compute_minmax_per_modality(label_files, chrom, start, end, track_type_map=track_type_map_label)
                    minmax_per_modality.update(label_minmax_per_mod)

                # Create track type map for pred and alt files (treat both as 'pred' to share same scale)
                pred_alt_files = pred_files + alt_files
                track_type_map_pred_alt = {f: 'pred' for f in pred_alt_files}
                if pred_alt_files:
                    pred_alt_minmax_per_mod = compute_minmax_per_modality(pred_alt_files, chrom, start, end, track_type_map=track_type_map_pred_alt)
                    minmax_per_modality.update(pred_alt_minmax_per_mod)
                    # Also add as 'alt' type (same values as pred)
                    for key, value in pred_alt_minmax_per_mod.items():
                        if key[0] == 'pred':
                            alt_key = ('alt', key[1])
                            minmax_per_modality[alt_key] = value

                # Create track type map for diff files (can be negative)
                track_type_map_diff = {f: 'diff' for f in diff_files}
                if diff_files:
                    diff_minmax_per_mod = compute_minmax_per_modality(diff_files, chrom, start, end, track_type_map=track_type_map_diff, allow_negative=True)
                    minmax_per_modality.update(diff_minmax_per_mod)

            # Build configuration with overlay
            output_path = Path(args.output)
            config_file = output_path.parent / f"{output_path.stem}_config.ini"
            gtf_file = None if args.no_gtf else args.gtf

            # Check if all tracks share a common modality suffix
            track_names_for_suffix_check = []
            for base_name in all_base_names:
                track_name = base_name
                # Remove prefixes
                if track_name.startswith("BasalGanglia-"):
                    track_name = track_name[len("BasalGanglia-"):]
                elif track_name.startswith("MiniAtlas-"):
                    track_name = track_name[len("MiniAtlas-"):]
                track_names_for_suffix_check.append(track_name)

            common_suffix = get_common_suffix(track_names_for_suffix_check)
            if common_suffix:
                print(f"\nAll tracks share common suffix '{common_suffix}' - will remove it from display names")
            else:
                print(f"\nTracks have different suffixes - keeping them for distinction")

            config_lines = []

            print("\nSelected tracks (order: label, ref/alt overlay, diff per cell type):")
            track_num = 0

            for base_name in all_base_names:
                # Add label track first
                if base_name in label_map:
                    bw_file = label_map[base_name]
                    track_name_orig = bw_file.stem
                    track_name = track_name_orig

                    # Remove prefixes
                    if track_name.startswith("BasalGanglia-"):
                        track_name = track_name[len("BasalGanglia-"):]
                    elif track_name.startswith("MiniAtlas-"):
                        track_name = track_name[len("MiniAtlas-"):]

                    # Get modality for min/max lookup
                    modality = get_modality_suffix(track_name_orig)
                    if modality is None:
                        modality = 'none'

                    # Look up min/max from per-modality dict
                    min_val = None
                    max_val = None
                    if ('label', modality) in minmax_per_modality:
                        min_val, max_val = minmax_per_modality[('label', modality)]
                    elif args.label_min is not None or args.label_max is not None:
                        min_val = args.label_min
                        max_val = args.label_max

                    # Remove modality suffix ONLY if all tracks share it
                    if common_suffix:
                        track_lower_temp = track_name.lower()
                        if track_lower_temp.endswith(common_suffix):
                            track_name = track_name[:-len(common_suffix)]

                    display_name = f"{track_name} (label)"
                    track_num += 1
                    print(f"  {track_num}. {display_name} → blue")

                    # Add label track section
                    config_lines.append(f"[{display_name}]")
                    config_lines.append(f"file = {bw_file.absolute()}")
                    config_lines.append(f"title = {display_name}")
                    config_lines.append(f"height = {args.height}")
                    config_lines.append(f"color = {args.label_color}")
                    config_lines.append(f"fontsize = {args.fontsize}")
                    if min_val is not None:
                        config_lines.append(f"min_value = {min_val}")
                    if max_val is not None:
                        config_lines.append(f"max_value = {max_val}")
                    config_lines.append("file_type = bigwig")
                    config_lines.append("number_of_bins = 700")
                    config_lines.append("")
                    config_lines.append("[spacer]")
                    config_lines.append(f"height = {args.spacer_height}")
                    config_lines.append("")

                # Add overlaid ref/alt tracks
                if base_name in pred_map or base_name in alt_map:
                    # Get track name
                    track_name_orig = base_name
                    track_name = base_name
                    if track_name.startswith("BasalGanglia-"):
                        track_name = track_name[len("BasalGanglia-"):]
                    elif track_name.startswith("MiniAtlas-"):
                        track_name = track_name[len("MiniAtlas-"):]

                    # Get modality for min/max lookup
                    modality = get_modality_suffix(track_name_orig)
                    if modality is None:
                        modality = 'none'

                    # Look up min/max from per-modality dict (pred and alt share same scale)
                    pred_min_val = None
                    pred_max_val = None
                    if ('pred', modality) in minmax_per_modality:
                        pred_min_val, pred_max_val = minmax_per_modality[('pred', modality)]
                    elif args.pred_min is not None or args.pred_max is not None:
                        pred_min_val = args.pred_min
                        pred_max_val = args.pred_max

                    alt_min_val = pred_min_val  # Alt uses same scale as pred
                    alt_max_val = pred_max_val

                    # Remove modality suffix ONLY if all tracks share it
                    if common_suffix:
                        track_lower_temp = track_name.lower()
                        if track_lower_temp.endswith(common_suffix):
                            track_name = track_name[:-len(common_suffix)]

                    # Determine track order based on --alt-first flag
                    if args.alt_first:
                        # Alt first, then ref overlaid
                        first_track = ('alt', alt_map, args.alt_color, args.alt_alpha, alt_min_val, alt_max_val)
                        second_track = ('ref', pred_map, args.ref_color, args.ref_alpha, pred_min_val, pred_max_val)
                    else:
                        # Ref first, then alt overlaid (default)
                        first_track = ('ref', pred_map, args.ref_color, args.ref_alpha, pred_min_val, pred_max_val)
                        second_track = ('alt', alt_map, args.alt_color, args.alt_alpha, alt_min_val, alt_max_val)

                    # Add first track (no overlay)
                    track_type, track_map, track_color, track_alpha, min_val, max_val = first_track
                    if base_name in track_map:
                        bw_file = track_map[base_name]
                        display_name = f"{track_name} ({track_type})"
                        track_num += 1
                        alpha_desc = f"alpha={track_alpha}" if track_alpha < 1.0 else ""
                        print(f"  {track_num}. {display_name} → {track_color} {alpha_desc}".strip())

                        config_lines.append(f"[{display_name}]")
                        config_lines.append(f"file = {bw_file.absolute()}")
                        config_lines.append(f"title = {track_name} (ref/alt)")
                        config_lines.append(f"height = {args.height}")
                        config_lines.append(f"color = {track_color}")
                        if track_alpha < 1.0:
                            config_lines.append(f"alpha = {track_alpha}")
                        config_lines.append(f"fontsize = {args.fontsize}")
                        if min_val is not None:
                            config_lines.append(f"min_value = {min_val}")
                        if max_val is not None:
                            config_lines.append(f"max_value = {max_val}")
                        config_lines.append("file_type = bigwig")
                        config_lines.append("number_of_bins = 700")
                        config_lines.append("")

                    # Add second track (overlaid on first)
                    track_type, track_map, track_color, track_alpha, min_val, max_val = second_track
                    if base_name in track_map:
                        bw_file = track_map[base_name]
                        display_name = f"{track_name} ({track_type})"
                        track_num += 1
                        alpha_desc = f"alpha={track_alpha}" if track_alpha < 1.0 else ""
                        print(f"  {track_num}. {display_name} → {track_color} {alpha_desc} (overlaid)".strip())

                        config_lines.append(f"[{display_name}]")
                        config_lines.append(f"file = {bw_file.absolute()}")
                        config_lines.append(f"title =")  # Empty title for overlay
                        config_lines.append(f"height = {args.height}")
                        config_lines.append(f"color = {track_color}")
                        if track_alpha < 1.0:
                            config_lines.append(f"alpha = {track_alpha}")
                        config_lines.append(f"fontsize = {args.fontsize}")
                        if min_val is not None:
                            config_lines.append(f"min_value = {min_val}")
                        if max_val is not None:
                            config_lines.append(f"max_value = {max_val}")
                        config_lines.append("file_type = bigwig")
                        config_lines.append("overlay_previous = share-y")
                        config_lines.append("number_of_bins = 700")
                        config_lines.append("")

                    # Add spacer after ref/alt overlay
                    config_lines.append("[spacer]")
                    config_lines.append(f"height = {args.spacer_height}")
                    config_lines.append("")

                # Add diff track (alt - ref)
                if base_name in diff_map:
                    bw_file = diff_map[base_name]
                    track_name_orig = base_name
                    track_name_diff = base_name
                    if track_name_diff.startswith("BasalGanglia-"):
                        track_name_diff = track_name_diff[len("BasalGanglia-"):]
                    elif track_name_diff.startswith("MiniAtlas-"):
                        track_name_diff = track_name_diff[len("MiniAtlas-"):]

                    # Get modality for min/max lookup
                    modality = get_modality_suffix(track_name_orig)
                    if modality is None:
                        modality = 'none'

                    # Look up min/max from per-modality dict
                    diff_min_val = None
                    diff_max_val = None
                    if ('diff', modality) in minmax_per_modality:
                        diff_min_val, diff_max_val = minmax_per_modality[('diff', modality)]

                    # Remove modality suffix ONLY if all tracks share it
                    if common_suffix:
                        track_lower_temp = track_name_diff.lower()
                        if track_lower_temp.endswith(common_suffix):
                            track_name_diff = track_name_diff[:-len(common_suffix)]

                    display_name = f"{track_name_diff} (diff)"
                    track_num += 1
                    print(f"  {track_num}. {display_name} → purple")

                    config_lines.append(f"[{display_name}]")
                    config_lines.append(f"file = {bw_file.absolute()}")
                    config_lines.append(f"title = {display_name}")
                    config_lines.append(f"height = {args.height}")
                    config_lines.append(f"color = {args.diff_color}")
                    config_lines.append(f"fontsize = {args.fontsize}")
                    if diff_min_val is not None:
                        config_lines.append(f"min_value = {diff_min_val}")
                    if diff_max_val is not None:
                        config_lines.append(f"max_value = {diff_max_val}")
                    config_lines.append("file_type = bigwig")
                    config_lines.append("number_of_bins = 700")
                    config_lines.append("")

                    # Add spacer after diff
                    config_lines.append("[spacer]")
                    config_lines.append(f"height = {args.spacer_height}")
                    config_lines.append("")

            # Add GTF track if provided
            if gtf_file:
                gtf_path = Path(gtf_file)
                if gtf_path.exists():
                    config_lines.append("[genes]")
                    config_lines.append(f"file = {gtf_path.absolute()}")
                    config_lines.append("title = Genes")
                    config_lines.append(f"height = {args.gtf_height}")
                    config_lines.append("file_type = gtf")
                    config_lines.append("prefered_name = gene_name")
                    config_lines.append("merge_transcripts = true")
                    config_lines.append("style = flybase")
                    config_lines.append(f"fontsize = {args.gtf_fontsize}")
                    config_lines.append("")
                    config_lines.append("[spacer]")
                    config_lines.append(f"height = {args.spacer_height}")
                    config_lines.append("")

            # Add highlight regions if specified
            if args.highlight:
                highlight_regions, is_single_positions = parse_highlight_regions(args.highlight)
                if highlight_regions:
                    # Use vlines for single positions, vhighlight for regions
                    if is_single_positions:
                        # Create BED file for vlines (single positions)
                        highlight_file = output_path.parent / f"{output_path.stem}_vlines.bed"
                        with open(highlight_file, 'w') as f:
                            for hl_chrom, hl_start, hl_end in highlight_regions:
                                # BED format requires end > start
                                f.write(f"{hl_chrom}\t{hl_start}\t{hl_start + 1}\n")

                        config_lines.append(f"[vlines]")
                        config_lines.append(f"file = {highlight_file.absolute()}")
                        config_lines.append("type = vlines")
                        config_lines.append(f"color = {args.highlight_color}")
                        config_lines.append("")
                        print(f"\nAdded {len(highlight_regions)} vertical line(s)")
                        print(f"  BED file: {highlight_file}")
                    else:
                        # Create narrowPeak file for vhighlight (regions)
                        highlight_file = output_path.parent / f"{output_path.stem}_vhighlight.narrowPeak"
                        with open(highlight_file, 'w') as f:
                            for i, (hl_chrom, hl_start, hl_end) in enumerate(highlight_regions):
                                # narrowPeak format: chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak
                                name = f"region_{i}"
                                score = "1000"
                                strand = "."
                                signalValue = "1.0"
                                pValue = "-1"
                                qValue = "-1"
                                peak = int((hl_end - hl_start) / 2)  # Peak at center
                                f.write(f"{hl_chrom}\t{hl_start}\t{hl_end}\t{name}\t{score}\t{strand}\t{signalValue}\t{pValue}\t{qValue}\t{peak}\n")

                        config_lines.append(f"[vhighlight]")
                        config_lines.append(f"file = {highlight_file.absolute()}")
                        config_lines.append("type = vhighlight")
                        config_lines.append(f"color = {args.highlight_color}")
                        config_lines.append("")
                        print(f"\nAdded {len(highlight_regions)} highlight region(s)")
                        print(f"  narrowPeak file: {highlight_file}")

            # Add x-axis at the bottom
            config_lines.append("[x-axis]")
            config_lines.append("where = bottom")
            config_lines.append(f"fontsize = {args.axis_fontsize}")

            # Write configuration file
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                f.write('\n'.join(config_lines))

            print(f"\nConfiguration saved: {config_file}")

            # Generate plot
            success = generate_plot(
                config_file,
                chrom,
                start,
                end,
                output_path,
                width=args.width,
                dpi=args.dpi
            )

            if success:
                print("\n" + "="*70)
                print("Visualization complete!")
                print(f"  Plot: {output_path}")
                print(f"  Config: {config_file}")
                print("="*70)
            else:
                print("\n" + "="*70)
                print("Visualization failed!")
                print("="*70)
                sys.exit(1)

            return

        else:
            # Normal mode: Combine pred and label into single plot
            print(f"\nCombining pred/ and label/ tracks into single plot")

            # Collect all bigwig files from both folders with appropriate renaming
            all_bigwig_files = []
            folder_type_map = {}  # Track which folder each file came from

            # Get files from both folders
            label_files = []
            pred_files = []

            if 'label' in inference_folders:
                label_files = find_bigwig_files(str(inference_folders['label']), args.tracks)
                print(f"  Found {len(label_files)} label tracks")

            if 'pred' in inference_folders:
                pred_files = find_bigwig_files(str(inference_folders['pred']), args.tracks)
                print(f"  Found {len(pred_files)} pred tracks")

            # Interleave pred and label tracks for each cell type
            # Create mapping from base name to files
            label_map = {f.stem: f for f in label_files}
            pred_map = {f.stem: f for f in pred_files}

            # Get all unique base names in the order they appear (preserve pattern order)
            # Don't sort - maintain the order from the --tracks argument
            all_base_names = []
            seen = set()
            # First add pred files in order
            for f in pred_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)
            # Then add any label-only files
            for f in label_files:
                if f.stem not in seen:
                    all_base_names.append(f.stem)
                    seen.add(f.stem)

            # Add files in order: celltype_pred, celltype_label for each celltype
            for base_name in all_base_names:
                # Add pred first
                if base_name in pred_map:
                    bw_file = pred_map[base_name]
                    folder_type_map[bw_file] = 'pred'
                    all_bigwig_files.append(bw_file)

                # Then add label
                if base_name in label_map:
                    bw_file = label_map[base_name]
                    folder_type_map[bw_file] = 'label'
                    all_bigwig_files.append(bw_file)

            if len(all_bigwig_files) == 0:
                print("Error: No bigwig files found matching the track patterns")
                sys.exit(1)

            print(f"\nTotal tracks to visualize: {len(all_bigwig_files)}")

            # Compute per-modality min/max
            minmax_per_modality = {}

            if not args.no_auto_scale:
                # Compute min/max per modality
                minmax_per_modality = compute_minmax_per_modality(all_bigwig_files, chrom, start, end, track_type_map=folder_type_map)

            print("\nSelected tracks:")
            for i, bw in enumerate(all_bigwig_files, 1):
                folder_type = folder_type_map[bw]
                color_label = "blue" if folder_type == 'label' else "red"
                print(f"  {i}. {bw.name} ({folder_type}) → {color_label}")

            # Create a modified create_tracks_config that respects folder types
            output_path = Path(args.output)
            config_file = output_path.parent / f"{output_path.stem}_config.ini"
            gtf_file = None if args.no_gtf else args.gtf

            # Check if all tracks share a common modality suffix
            track_names_for_suffix_check = []
            for bw_file in all_bigwig_files:
                track_name = bw_file.stem
                # Remove prefixes
                if track_name.startswith("BasalGanglia-"):
                    track_name = track_name[len("BasalGanglia-"):]
                elif track_name.startswith("MiniAtlas-"):
                    track_name = track_name[len("MiniAtlas-"):]
                track_names_for_suffix_check.append(track_name)

            common_suffix = get_common_suffix(track_names_for_suffix_check)
            if common_suffix:
                print(f"\nAll tracks share common suffix '{common_suffix}' - will remove it from display names")
            else:
                print(f"\nTracks have different suffixes - keeping them for distinction")

            # Build config manually to handle folder-based coloring
            config_lines = []

            for bw_file in all_bigwig_files:
                folder_type = folder_type_map[bw_file]

                # Extract track name from filename
                track_name_orig = bw_file.stem
                track_name = track_name_orig

                # Remove prefixes if present
                if track_name.startswith("BasalGanglia-"):
                    track_name = track_name[len("BasalGanglia-"):]
                elif track_name.startswith("MiniAtlas-"):
                    track_name = track_name[len("MiniAtlas-"):]

                # Set color based on folder type
                track_color = args.label_color if folder_type == 'label' else args.ref_color

                # Get modality for min/max lookup
                modality = get_modality_suffix(track_name_orig)
                if modality is None:
                    modality = 'none'

                # Look up min/max from per-modality dict
                min_val = None
                max_val = None
                if (folder_type, modality) in minmax_per_modality:
                    min_val, max_val = minmax_per_modality[(folder_type, modality)]
                else:
                    # Fallback to user-specified values
                    if folder_type == 'label' and (args.label_min is not None or args.label_max is not None):
                        min_val = args.label_min
                        max_val = args.label_max
                    elif folder_type == 'pred' and (args.pred_min is not None or args.pred_max is not None):
                        min_val = args.pred_min
                        max_val = args.pred_max

                # Remove modality suffix ONLY if all tracks share it
                if common_suffix:
                    track_lower_temp = track_name.lower()
                    if track_lower_temp.endswith(common_suffix):
                        track_name = track_name[:-len(common_suffix)]

                # Add suffix to indicate pred vs label
                display_name = f"{track_name} ({folder_type})"

                # Add track section
                config_lines.append(f"[{display_name}]")
                config_lines.append(f"file = {bw_file.absolute()}")
                config_lines.append(f"title = {display_name}")
                config_lines.append(f"height = {args.height}")
                config_lines.append(f"color = {track_color}")
                config_lines.append(f"fontsize = {args.fontsize}")

                # Only add min/max if they are specified
                if min_val is not None:
                    config_lines.append(f"min_value = {min_val}")
                if max_val is not None:
                    config_lines.append(f"max_value = {max_val}")

                config_lines.append("file_type = bigwig")
                config_lines.append("number_of_bins = 700")
                config_lines.append("")

                # Add spacer
                config_lines.append("[spacer]")
                config_lines.append(f"height = {args.spacer_height}")
                config_lines.append("")

            # Add GTF track if provided
            if gtf_file:
                gtf_path = Path(gtf_file)
                if gtf_path.exists():
                    config_lines.append("[genes]")
                    config_lines.append(f"file = {gtf_path.absolute()}")
                    config_lines.append("title = Genes")
                    config_lines.append(f"height = {args.gtf_height}")
                    config_lines.append("file_type = gtf")
                    config_lines.append("prefered_name = gene_name")
                    config_lines.append("merge_transcripts = true")
                    config_lines.append("style = flybase")
                    config_lines.append(f"fontsize = {args.gtf_fontsize}")
                    config_lines.append("")

                    config_lines.append("[spacer]")
                    config_lines.append(f"height = {args.spacer_height}")
                    config_lines.append("")

            # Add highlight regions if specified
            if args.highlight:
                highlight_regions, is_single_positions = parse_highlight_regions(args.highlight)
                if highlight_regions:
                    # Use vlines for single positions, vhighlight for regions
                    if is_single_positions:
                        # Create BED file for vlines (single positions)
                        highlight_file = output_path.parent / f"{output_path.stem}_vlines.bed"
                        with open(highlight_file, 'w') as f:
                            for hl_chrom, hl_start, hl_end in highlight_regions:
                                # BED format requires end > start
                                f.write(f"{hl_chrom}\t{hl_start}\t{hl_start + 1}\n")

                        config_lines.append(f"[vlines]")
                        config_lines.append(f"file = {highlight_file.absolute()}")
                        config_lines.append("type = vlines")
                        config_lines.append(f"color = {args.highlight_color}")
                        config_lines.append("")
                        print(f"\nAdded {len(highlight_regions)} vertical line(s)")
                        print(f"  BED file: {highlight_file}")
                    else:
                        # Create narrowPeak file for vhighlight (regions)
                        highlight_file = output_path.parent / f"{output_path.stem}_vhighlight.narrowPeak"
                        with open(highlight_file, 'w') as f:
                            for i, (hl_chrom, hl_start, hl_end) in enumerate(highlight_regions):
                                # narrowPeak format: chrom, start, end, name, score, strand, signalValue, pValue, qValue, peak
                                name = f"region_{i}"
                                score = "1000"
                                strand = "."
                                signalValue = "1.0"
                                pValue = "-1"
                                qValue = "-1"
                                peak = int((hl_end - hl_start) / 2)  # Peak at center
                                f.write(f"{hl_chrom}\t{hl_start}\t{hl_end}\t{name}\t{score}\t{strand}\t{signalValue}\t{pValue}\t{qValue}\t{peak}\n")

                        config_lines.append(f"[vhighlight]")
                        config_lines.append(f"file = {highlight_file.absolute()}")
                        config_lines.append("type = vhighlight")
                        config_lines.append(f"color = {args.highlight_color}")
                        config_lines.append("")
                        print(f"\nAdded {len(highlight_regions)} highlight region(s)")
                        print(f"  narrowPeak file: {highlight_file}")

            # Add x-axis at the bottom
            config_lines.append("[x-axis]")
            config_lines.append("where = bottom")
            config_lines.append(f"fontsize = {args.axis_fontsize}")

            # Write configuration file
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                f.write('\n'.join(config_lines))

            print(f"\nConfiguration saved: {config_file}")
            print(f"  Number of tracks: {len(all_bigwig_files)}")

            # Generate plot
            success = generate_plot(
                config_file,
                chrom,
                start,
                end,
                output_path,
                width=args.width,
                dpi=args.dpi
            )

            if success:
                print("\n" + "="*70)
                print("Visualization complete!")
                print(f"  Plot: {output_path}")
                print(f"  Config: {config_file}")
                print("="*70)
            else:
                print("\n" + "="*70)
                print("Visualization failed!")
                print("="*70)
                sys.exit(1)

            return

    # Standard mode: single bigwig directory
    # Find and filter bigwig files
    bigwig_files = find_bigwig_files(args.bigwig_dir, args.tracks)

    # Compute per-modality min/max for label and pred tracks
    minmax_per_modality = {}

    if not args.no_auto_scale:
        # Compute min/max per modality from data
        minmax_per_modality = compute_minmax_per_modality(bigwig_files, chrom, start, end)

    # Override with user-specified values if provided
    if args.label_min is not None or args.label_max is not None or args.pred_min is not None or args.pred_max is not None:
        print("\nUser-specified min/max values:")
        if args.label_min is not None or args.label_max is not None:
            print(f"  Label tracks: [{args.label_min}, {args.label_max}]")
        if args.pred_min is not None or args.pred_max is not None:
            print(f"  Pred tracks: [{args.pred_min}, {args.pred_max}]")

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
                        label_min=args.label_min, label_max=args.label_max,
                        pred_min=args.pred_min, pred_max=args.pred_max,
                        minmax_per_modality=minmax_per_modality,
                        gtf_file=gtf_file, gtf_height=args.gtf_height,
                        fontsize=args.fontsize, gtf_fontsize=args.gtf_fontsize,
                        axis_fontsize=args.axis_fontsize,
                        spacer_height=args.spacer_height,
                        label_color=args.label_color, ref_color=args.ref_color)

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
