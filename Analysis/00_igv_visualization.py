#!/usr/bin/env python3
"""
IGV Genome Browser Visualization Script
Uses genomespy to create interactive genome browser views from data_config files
"""

import pandas as pd
import os
import sys
import argparse
import signal
import time
from genomespy import igv

# Set working directory
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')


def load_tracks_from_config(config_file, celltypes=None, modalities=None, atlases=None,
                           celltype_n_min=None, celltype_n_max=None, track_height=40,
                           use_config_scale=True, group_scale_by=None, manual_scale=None):
    """
    Load tracks from a data_config CSV file

    Parameters:
    -----------
    config_file : str
        Path to the data_config CSV file
    celltypes : list or None
        List of celltypes to include. If None, include all.
    modalities : list or None
        List of modalities to include (e.g., ['ATAC', 'K27Ac']). If None, include all.
    atlases : list or None
        List of atlas names to include. If None, include all.
    celltype_n_min : int or None
        Minimum celltype_n value to include
    celltype_n_max : int or None
        Maximum celltype_n value to include
    track_height : int
        Height of each track in pixels
    use_config_scale : bool
        Whether to use scale/clip values from config file (default: True)
    group_scale_by : str or None
        Column name to group tracks by for shared scaling (e.g., 'modality')
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}

    Returns:
    --------
    tuple: (tracks_dict, scale_groups_dict)
        - tracks_dict: Dictionary of tracks in format: {track_name: {url, height, type, ...}}
        - scale_groups_dict: Dictionary of scale groups if group_scale_by is specified
    """
    # Read the data config
    df = pd.read_csv(config_file)

    print(f"Loaded {len(df)} tracks from {config_file}")
    print(f"Available columns: {df.columns.tolist()}")

    # Check required columns
    if 'file' not in df.columns or 'exp' not in df.columns:
        raise ValueError(f"Config file must have 'file' and 'exp' columns. Found: {df.columns.tolist()}")

    # Filter by celltype if specified
    if celltypes is not None:
        if 'celltype' not in df.columns:
            print("Warning: 'celltype' column not found, cannot filter by celltype")
        else:
            df = df[df['celltype'].isin(celltypes)]
            print(f"Filtered to {len(df)} tracks for celltypes: {celltypes}")

    # Filter by modality if specified
    if modalities is not None:
        if 'modality' not in df.columns:
            print("Warning: 'modality' column not found, cannot filter by modality")
        else:
            df = df[df['modality'].isin(modalities)]
            print(f"Filtered to {len(df)} tracks for modalities: {modalities}")

    # Filter by atlas if specified
    if atlases is not None:
        if 'atlas_name' not in df.columns:
            print("Warning: 'atlas_name' column not found, cannot filter by atlas")
        else:
            df = df[df['atlas_name'].isin(atlases)]
            print(f"Filtered to {len(df)} tracks for atlases: {atlases}")

    # Filter by celltype_n if specified
    if celltype_n_min is not None or celltype_n_max is not None:
        if 'celltype_n' not in df.columns:
            print("Warning: 'celltype_n' column not found, cannot filter by celltype_n")
        else:
            if celltype_n_min is not None:
                df = df[df['celltype_n'] >= celltype_n_min]
                print(f"Filtered to {len(df)} tracks with celltype_n >= {celltype_n_min}")
            if celltype_n_max is not None:
                df = df[df['celltype_n'] <= celltype_n_max]
                print(f"Filtered to {len(df)} tracks with celltype_n <= {celltype_n_max}")

    # Build tracks dictionary
    tracks = {}
    scale_groups = {}

    for _, row in df.iterrows():
        track_name = row['exp']
        track_path = row['file']

        # Check if file exists
        if not os.path.exists(track_path):
            print(f"Warning: Track file not found: {track_path}")
            continue

        # Use relative path from current working directory
        track_config = {
            "path": track_path,
            "height": track_height,
            "type": "bigwig"
        }

        # Add scale information if available and requested
        if use_config_scale:
            if 'scale' in row and pd.notna(row['scale']):
                track_config['scale'] = float(row['scale'])
            if 'clip' in row and pd.notna(row['clip']):
                track_config['max'] = float(row['clip'])
            if 'clip_soft' in row and pd.notna(row['clip_soft']):
                track_config['min'] = float(row['clip_soft'])

        # Apply manual scale override if provided
        if manual_scale:
            if 'min' in manual_scale:
                track_config['min'] = manual_scale['min']
            if 'max' in manual_scale:
                track_config['max'] = manual_scale['max']

        tracks[track_name] = track_config

        # Group tracks if requested
        if group_scale_by and group_scale_by in row:
            group_key = row[group_scale_by]
            if group_key not in scale_groups:
                scale_groups[group_key] = []
            scale_groups[group_key].append(track_name)

    print(f"Created {len(tracks)} valid tracks")

    if group_scale_by and scale_groups:
        print(f"Grouped tracks by '{group_scale_by}':")
        for group, track_list in scale_groups.items():
            print(f"  {group}: {len(track_list)} tracks")

            # Calculate shared scale for each group
            group_tracks_data = [tracks[t] for t in track_list]

            # Collect min/max values, with defaults if none exist
            min_values = [t.get('min', 0) for t in group_tracks_data if 'min' in t]
            max_values = [t.get('max', 100) for t in group_tracks_data if 'max' in t]

            # Use collected values or defaults if empty
            group_min = min(min_values) if min_values else 0
            group_max = max(max_values) if max_values else 100

            # Apply shared scale to all tracks in group
            for track_name in track_list:
                tracks[track_name]['min'] = group_min
                tracks[track_name]['max'] = group_max

            print(f"    Shared scale: min={group_min}, max={group_max}")

    return tracks, scale_groups


def plot_region(config_file, chrom, start, end, celltypes=None, modalities=None, atlases=None,
                celltype_n_min=None, celltype_n_max=None, track_height=40,
                viz_height=600, server_port=18089, use_config_scale=True,
                group_scale_by=None, manual_scale=None):
    """
    Create an IGV genome browser view for a specific region

    Parameters:
    -----------
    config_file : str
        Path to the data_config CSV file
    chrom : str
        Chromosome name (e.g., 'chr1')
    start : int
        Start position
    end : int
        End position
    celltypes : list or None
        List of celltypes to include
    modalities : list or None
        List of modalities to include
    atlases : list or None
        List of atlas names to include
    celltype_n_min : int or None
        Minimum celltype_n value
    celltype_n_max : int or None
        Maximum celltype_n value
    track_height : int
        Height of each track in pixels (default: 40)
    viz_height : int
        Overall visualization height in pixels (default: 600)
    server_port : int
        Port for the IGV server (default: 18089)
    use_config_scale : bool
        Whether to use scale/clip values from config file (default: True)
    group_scale_by : str or None
        Column to group tracks by for shared scaling (e.g., 'modality')
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}

    Returns:
    --------
    IGV plot object
    """
    # Load tracks
    tracks, scale_groups = load_tracks_from_config(
        config_file, celltypes=celltypes, modalities=modalities,
        atlases=atlases, celltype_n_min=celltype_n_min,
        celltype_n_max=celltype_n_max, track_height=track_height,
        use_config_scale=use_config_scale, group_scale_by=group_scale_by,
        manual_scale=manual_scale
    )

    if len(tracks) == 0:
        print("Error: No valid tracks found!")
        return None

    # Create region
    region = {
        "chrom": chrom,
        "start": int(start),
        "end": int(end)
    }

    print(f"\nCreating IGV view for region: {chrom}:{start}-{end}")
    print(f"Number of tracks: {len(tracks)}")
    print(f"Visualization height: {viz_height}px")

    # Create IGV plot with all parameters
    plot = igv(tracks, region=region, height=viz_height, server_port=server_port)

    return plot


def plot_gene(config_file, gene_name, gtf_file=None, padding=5000,
              celltypes=None, modalities=None, atlases=None,
              celltype_n_min=None, celltype_n_max=None, track_height=40,
              viz_height=600, server_port=18089, use_config_scale=True,
              group_scale_by=None, manual_scale=None):
    """
    Create an IGV genome browser view for a gene region

    Parameters:
    -----------
    config_file : str
        Path to the data_config CSV file
    gene_name : str
        Gene name (e.g., 'GRIN2B')
    gtf_file : str or None
        Path to GTF file. If None, uses default GENCODE v48
    padding : int
        Padding around gene in base pairs
    celltypes : list or None
        List of celltypes to include
    modalities : list or None
        List of modalities to include
    atlases : list or None
        List of atlas names to include
    celltype_n_min : int or None
        Minimum celltype_n value
    celltype_n_max : int or None
        Maximum celltype_n value
    track_height : int
        Height of each track in pixels (default: 40)
    viz_height : int
        Overall visualization height in pixels (default: 600)
    server_port : int
        Port for the IGV server (default: 18089)
    use_config_scale : bool
        Whether to use scale/clip values from config file (default: True)
    group_scale_by : str or None
        Column to group tracks by for shared scaling (e.g., 'modality')
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}

    Returns:
    --------
    IGV plot object
    """
    import gzip

    if gtf_file is None:
        gtf_file = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v48/gencode.v48.annotation.gtf.gz'

    print(f"Searching for gene {gene_name} in {gtf_file}...")

    # Parse GTF to find gene
    gene_found = False
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            if fields[2] != 'gene':
                continue

            attributes = fields[8]

            # Parse gene_name from attributes
            attr_dict = {}
            for attr in attributes.strip().rstrip(';').split(';'):
                attr = attr.strip()
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    attr_dict[key] = value.strip('"')

            if attr_dict.get('gene_name') == gene_name:
                chrom = fields[0]
                start = int(fields[3]) - padding
                end = int(fields[4]) + padding
                gene_found = True
                print(f"Found gene {gene_name}: {chrom}:{start}-{end}")
                break

    if not gene_found:
        print(f"Error: Gene {gene_name} not found in GTF file")
        return None

    # Create plot for this region
    return plot_region(config_file, chrom, start, end, celltypes=celltypes,
                      modalities=modalities, atlases=atlases,
                      celltype_n_min=celltype_n_min, celltype_n_max=celltype_n_max,
                      track_height=track_height, viz_height=viz_height,
                      server_port=server_port, use_config_scale=use_config_scale,
                      group_scale_by=group_scale_by, manual_scale=manual_scale)


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='IGV Genome Browser Visualization using genomespy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot a specific region
  python 00_igv_visualization.py --config config.csv --region chr1:1000000-2000000

  # Plot a gene
  python 00_igv_visualization.py --config config.csv --gene GRIN2B --padding 10000

  # Plot with filters
  python 00_igv_visualization.py --config config.csv --gene GRIN2B \\
      --modality_filter ATAC K27Ac --celltype_filter BasalGanglia-STR-D1-MSN

  # Plot with celltype_n filter
  python 00_igv_visualization.py --config config.csv --region chr1:1000000-2000000 \\
      --celltype_n_filter 10000 200000

  # Group tracks by modality for shared scaling
  python 00_igv_visualization.py --config config.csv --gene GRIN2B \\
      --group_scale_by modality

  # Manual scale override
  python 00_igv_visualization.py --config config.csv --gene GRIN2B \\
      --manual_scale 0 100

  # Disable config scale and use manual scale
  python 00_igv_visualization.py --config config.csv --gene GRIN2B \\
      --no_config_scale --manual_scale 0 50
        """
    )

    parser.add_argument('--config', '-c', required=True,
                       help='Path to data_config CSV file')

    # Region or gene
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--region', '-r',
                      help='Genomic region (format: chr1:1000000-2000000)')
    group.add_argument('--gene', '-g',
                      help='Gene name (e.g., GRIN2B)')

    # Filters
    parser.add_argument('--modality_filter', '-m', nargs='+',
                       help='Filter by modality (e.g., ATAC K27Ac)')
    parser.add_argument('--celltype_filter', '-ct', nargs='+',
                       help='Filter by celltype')
    parser.add_argument('--atlas_filter', '-a', nargs='+',
                       help='Filter by atlas name')
    parser.add_argument('--celltype_n_filter', '-cn', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       help='Filter by celltype_n range (min max)')

    # Other options
    parser.add_argument('--padding', '-p', type=int, default=5000,
                       help='Padding around gene in bp (default: 5000)')
    parser.add_argument('--gtf', help='GTF file path (default: GENCODE v48)')
    parser.add_argument('--track_height', type=int, default=40,
                       help='Height of each track in pixels (default: 40)')
    parser.add_argument('--viz_height', type=int, default=600,
                       help='Overall visualization height in pixels (default: 600)')
    parser.add_argument('--port', type=int, default=18089,
                       help='Server port (default: 18089)')

    # Scaling options
    parser.add_argument('--no_config_scale', action='store_true',
                       help='Do not use scale/clip values from config file')
    parser.add_argument('--group_scale_by', choices=['modality', 'celltype', 'atlas_name'],
                       help='Group tracks by column for shared scaling')
    parser.add_argument('--manual_scale', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Manual scale override (min max values)')

    args = parser.parse_args()

    # Parse celltype_n filter
    celltype_n_min = args.celltype_n_filter[0] if args.celltype_n_filter else None
    celltype_n_max = args.celltype_n_filter[1] if args.celltype_n_filter else None

    # Parse scaling options
    use_config_scale = not args.no_config_scale
    manual_scale = None
    if args.manual_scale:
        manual_scale = {'min': args.manual_scale[0], 'max': args.manual_scale[1]}

    # Create plot
    if args.region:
        # Parse region
        try:
            chrom, coords = args.region.split(':')
            start, end = coords.split('-')
            start, end = int(start), int(end)
        except:
            print(f"Error: Invalid region format '{args.region}'. Expected format: chr1:1000000-2000000")
            sys.exit(1)

        plot = plot_region(
            args.config, chrom, start, end,
            celltypes=args.celltype_filter,
            modalities=args.modality_filter,
            atlases=args.atlas_filter,
            celltype_n_min=celltype_n_min,
            celltype_n_max=celltype_n_max,
            track_height=args.track_height,
            viz_height=args.viz_height,
            server_port=args.port,
            use_config_scale=use_config_scale,
            group_scale_by=args.group_scale_by,
            manual_scale=manual_scale
        )
    else:
        # Plot gene
        plot = plot_gene(
            args.config, args.gene,
            gtf_file=args.gtf,
            padding=args.padding,
            celltypes=args.celltype_filter,
            modalities=args.modality_filter,
            atlases=args.atlas_filter,
            celltype_n_min=celltype_n_min,
            celltype_n_max=celltype_n_max,
            track_height=args.track_height,
            viz_height=args.viz_height,
            server_port=args.port,
            use_config_scale=use_config_scale,
            group_scale_by=args.group_scale_by,
            manual_scale=manual_scale
        )

    if plot is None:
        print("Failed to create plot")
        sys.exit(1)

    print("\nIGV plot created successfully!")
    print("Displaying plot...")

    # Show the plot with specified filename
    plot.show(filename=".genomespy_temp_.html")

    # Print the URL to access the visualization
    print(f"\nVisualization available at: http://localhost:{args.port}/.genomespy_temp_.html")

    # Set up signal handling to exit gracefully
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        try:
            plot.close()
            plot.cleanup()
            print("IGV server closed.")
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\nIGV server running on port {args.port}")
    print("Press Ctrl+C to stop the server")

    # Keep the script running until signal is received
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        try:
            plot.close()
            plot.cleanup()
            print("IGV server closed.")
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        sys.exit(0)
