#!/usr/bin/env python3
"""
IGV Genome Browser Visualization Script for BigWig Folder
Uses genomespy to create interactive genome browser views from a folder of .bw files
"""

import os
import sys
import argparse
import signal
import time
import glob
from genomespy import igv

# Set working directory
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')


def load_tracks_from_folder(bigwig_folder, track_height=40, manual_scale=None,
                            pattern="*.bw", name_filter=None):
    """
    Load tracks from a folder containing BigWig files

    Parameters:
    -----------
    bigwig_folder : str
        Path to folder containing .bw files
    track_height : int
        Height of each track in pixels
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}
    pattern : str
        File pattern to match (default: "*.bw")
    name_filter : list or None
        List of substrings to filter track names (keeps tracks containing any of these)

    Returns:
    --------
    dict: Dictionary of tracks in format: {track_name: {path, height, type, ...}}

    Note:
    -----
    Tracks are automatically colored based on filename:
    - Files ending with '_pred.bw': steelblue (predictions)
    - Files ending with '_label.bw': darkorange (labels/ground truth)
    - Other files: default genomespy color
    """
    # Find all bigwig files in folder
    search_pattern = os.path.join(bigwig_folder, pattern)
    bw_files = glob.glob(search_pattern)

    # Also try with ** for subdirectories if no files found
    if len(bw_files) == 0:
        search_pattern = os.path.join(bigwig_folder, "**", pattern)
        bw_files = glob.glob(search_pattern, recursive=True)

    print(f"Found {len(bw_files)} BigWig files in {bigwig_folder}")

    if len(bw_files) == 0:
        print(f"Warning: No BigWig files found matching pattern '{pattern}' in {bigwig_folder}")
        return {}

    # Build tracks dictionary
    tracks = {}

    for bw_file in sorted(bw_files):
        # Get track name from filename (without extension)
        track_name = os.path.splitext(os.path.basename(bw_file))[0]

        # Apply name filter if specified
        if name_filter is not None:
            if not any(f in track_name for f in name_filter):
                continue

        # Check if file exists
        if not os.path.exists(bw_file):
            print(f"Warning: Track file not found: {bw_file}")
            continue

        # Create track config
        track_config = {
            "path": bw_file,
            "height": track_height,
            "type": "bigwig"
        }

        # Assign colors based on filename suffix
        if bw_file.endswith('_pred.bw'):
            track_config['color'] = 'steelblue'  # Blue for predictions
        elif bw_file.endswith('_label.bw'):
            track_config['color'] = 'darkorange'  # Orange for labels/ground truth
        # Default: no color specified (uses genomespy default)

        # Apply manual scale if provided
        if manual_scale:
            if 'min' in manual_scale:
                track_config['min'] = manual_scale['min']
            if 'max' in manual_scale:
                track_config['max'] = manual_scale['max']

        tracks[track_name] = track_config

    print(f"Created {len(tracks)} valid tracks")

    return tracks


def plot_region(bigwig_folder, chrom, start, end, track_height=40,
                viz_height=600, server_port=18089, manual_scale=None,
                pattern="*.bw", name_filter=None):
    """
    Create an IGV genome browser view for a specific region

    Parameters:
    -----------
    bigwig_folder : str
        Path to folder containing .bw files
    chrom : str
        Chromosome name (e.g., 'chr1')
    start : int
        Start position
    end : int
        End position
    track_height : int
        Height of each track in pixels (default: 40)
    viz_height : int
        Overall visualization height in pixels (default: 600)
    server_port : int
        Port for the IGV server (default: 18089)
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}
    pattern : str
        File pattern to match (default: "*.bw")
    name_filter : list or None
        List of substrings to filter track names

    Returns:
    --------
    IGV plot object
    """
    # Load tracks
    tracks = load_tracks_from_folder(
        bigwig_folder, track_height=track_height,
        manual_scale=manual_scale, pattern=pattern,
        name_filter=name_filter
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


def plot_gene(bigwig_folder, gene_name, gtf_file=None, padding=5000,
              track_height=40, viz_height=600, server_port=18089,
              manual_scale=None, pattern="*.bw", name_filter=None):
    """
    Create an IGV genome browser view for a gene region

    Parameters:
    -----------
    bigwig_folder : str
        Path to folder containing .bw files
    gene_name : str
        Gene name (e.g., 'GRIN2B')
    gtf_file : str or None
        Path to GTF file. If None, uses default GENCODE v48
    padding : int
        Padding around gene in base pairs
    track_height : int
        Height of each track in pixels (default: 40)
    viz_height : int
        Overall visualization height in pixels (default: 600)
    server_port : int
        Port for the IGV server (default: 18089)
    manual_scale : dict or None
        Manual scale override: {'min': 0, 'max': 100}
    pattern : str
        File pattern to match (default: "*.bw")
    name_filter : list or None
        List of substrings to filter track names

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
    return plot_region(bigwig_folder, chrom, start, end,
                      track_height=track_height, viz_height=viz_height,
                      server_port=server_port, manual_scale=manual_scale,
                      pattern=pattern, name_filter=name_filter)


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='IGV Genome Browser Visualization for BigWig Folder using genomespy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot a specific region from a folder of bigwig files
  python 01_0_igv_visualization.py --folder Res/basal_ganglia_miniatlas_drop_celltype_v1_no_celltype_head/bigwig/ \\
      --region chr1:1000000-2000000

  # Plot a gene
  python 01_0_igv_visualization.py --folder Res/basal_ganglia_miniatlas_drop_celltype_v1_no_celltype_head/bigwig/ \\
      --gene GRIN2B --padding 10000

  # Plot with manual scale
  python 01_0_igv_visualization.py --folder path/to/bigwigs/ --gene GRIN2B \\
      --manual_scale 0 100

  # Filter tracks by name (keeps tracks containing any of the filter strings)
  python 01_0_igv_visualization.py --folder path/to/bigwigs/ --gene GRIN2B \\
      --name_filter MSN D1 D2

  # Use custom file pattern (e.g., for .bigwig extension)
  python 01_0_igv_visualization.py --folder path/to/bigwigs/ --gene GRIN2B \\
      --pattern "*.bigwig"
        """
    )

    parser.add_argument('--folder', '-f', required=True,
                       help='Path to folder containing BigWig (.bw) files')

    # Region or gene
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--region', '-r',
                      help='Genomic region (format: chr1:1000000-2000000)')
    group.add_argument('--gene', '-g',
                      help='Gene name (e.g., GRIN2B)')

    # Filter options
    parser.add_argument('--name_filter', '-n', nargs='+',
                       help='Filter tracks by name (keeps tracks containing any of these strings)')
    parser.add_argument('--pattern', '-pt', default="*.bw",
                       help='File pattern to match (default: *.bw)')

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
    parser.add_argument('--manual_scale', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Manual scale override (min max values)')

    args = parser.parse_args()

    # Parse scaling options
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
            args.folder, chrom, start, end,
            track_height=args.track_height,
            viz_height=args.viz_height,
            server_port=args.port,
            manual_scale=manual_scale,
            pattern=args.pattern,
            name_filter=args.name_filter
        )
    else:
        # Plot gene
        plot = plot_gene(
            args.folder, args.gene,
            gtf_file=args.gtf,
            padding=args.padding,
            track_height=args.track_height,
            viz_height=args.viz_height,
            server_port=args.port,
            manual_scale=manual_scale,
            pattern=args.pattern,
            name_filter=args.name_filter
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
    def signal_handler(_signum, _frame):
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
