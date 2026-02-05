#!/usr/bin/env python3
"""
Plot DeepLift interpretation results.

This script generates plots from the data files saved by 02_motif_gene_diff_interpretation_DeepLift.py.

Usage:
    python 02_motif_interpretation_plot.py \
        --data_dir DATA_DIR \
        --name_base NAME_BASE \
        --output OUTPUT_FILE \
        [other options]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.text import TextPath
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import matplotlib.transforms
from vizsequence.viz_sequence import plot_weights_given_ax

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)

from utils.logging import BaseLogger

# Set font type to TrueType (Type 42) for editable text in Adobe Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# Helper function to draw a letter at a given position
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """Draw a DNA letter at a given position."""
    # Define letter heights and colors
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
    }

    text = LETTERS[letter]

    # Choose color
    chosen_color = COLOR_SCHEME[letter]
    if color is not None:
        chosen_color = color

    # Draw letter onto axis
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)

    return p


def create_genomic_position_formatter(x_min, x_max):
    """
    Create a formatter function for genomic positions that dynamically chooses
    the appropriate unit (bp, kb, or Mb) based on the range.

    Args:
        x_min: Minimum position value in the range
        x_max: Maximum position value in the range

    Returns:
        Formatter function
    """
    range_size = x_max - x_min

    # Choose unit based on range size
    if range_size >= 1e6:
        # Use Mb for ranges >= 1 million bp
        def formatter(x, _pos):
            return f'{x/1e6:.2f}M'
    elif range_size >= 1e3:
        # Use kb for ranges >= 1 thousand bp
        def formatter(x, _pos):
            return f'{x/1e3:.1f}k'
    else:
        # Use bp for small ranges
        def formatter(x, _pos):
            return f'{int(x)}'

    return formatter


def load_interpretation_data(data_dir, name_base, baseline_types):
    """
    Load interpretation data from saved files.

    Args:
        data_dir: Directory containing the saved data files
        name_base: Base name for the files
        baseline_types: List of baseline types used

    Returns:
        Dictionary with loaded data
    """
    logger = BaseLogger(name="DataLoader", level=logging.INFO)

    # Load metadata
    metadata_file = f"{data_dir}/{name_base}_metadata.npy"
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    metadata = np.load(metadata_file, allow_pickle=True).item()
    logger.info(f"Loaded metadata from: {metadata_file}")

    # Load individual trial label data
    label_trials_file = f"{data_dir}/{name_base}_label_trials.npy"
    if os.path.exists(label_trials_file):
        label_trials = np.load(label_trials_file, allow_pickle=True).item()
        logger.info(f"Loaded {len(label_trials)} label trials from: {label_trials_file}")
    else:
        label_trials = {}
        logger.warning(f"Label trials file not found: {label_trials_file}, proceeding without labels")

    # Load individual trial prediction data
    pred_trials_file = f"{data_dir}/{name_base}_pred_trials.npy"
    if not os.path.exists(pred_trials_file):
        raise FileNotFoundError(f"Prediction trials file not found: {pred_trials_file}")
    pred_trials = np.load(pred_trials_file, allow_pickle=True).item()
    logger.info(f"Loaded {len(pred_trials)} prediction trials from: {pred_trials_file}")

    # Load importance scores and sequence attribution for each baseline type
    importance_scores = []
    sequence_attributions = []

    for baseline_type in baseline_types:
        # Construct identifier using the same logic as interpretation script
        chr_name = metadata['chr_name']
        start = metadata['start']
        end = metadata['end']
        region_name = metadata['region_name']

        # Use original trial patterns from metadata for correct filename reconstruction
        trial_pos = metadata.get('trial_pos', ';'.join(metadata['trial_pos_list']))
        trial_neg_list = metadata['trial_neg_list']

        # Clean trial names for filenames (same logic as in the original script)
        trial_pos_clean = clean_trial_name(trial_pos, keep_suffix=True)
        neg_trial_count = len(trial_neg_list)

        # Only add trial_neg_clean if there are negative trials
        if neg_trial_count > 0:
            trial_neg_clean = f"other-{neg_trial_count}"
            identifier = f"{chr_name}_{start}_{end}_{region_name}_{trial_pos_clean}_{trial_neg_clean}_{baseline_type}"
        else:
            identifier = f"{chr_name}_{start}_{end}_{region_name}_{trial_pos_clean}_{baseline_type}"

        logger.info(f"Looking for files with identifier: {identifier}")

        importance_file = f"{data_dir}/{identifier}_importance.npy"
        if not os.path.exists(importance_file):
            logger.warning(f"Importance file not found: {importance_file}, skipping")
            continue

        importance_data = np.load(importance_file)
        importance_scores.append((baseline_type, importance_data))
        logger.info(f"Loaded importance scores from: {importance_file}")

        # Load sequence and attribution data for visualization
        seq_file = f"{data_dir}/{identifier}_sequence.npy"
        attr_file = f"{data_dir}/{identifier}_attribution.npy"

        if os.path.exists(seq_file) and os.path.exists(attr_file):
            seq_onehot = np.load(seq_file)
            attribution = np.load(attr_file)
            sequence_attributions.append((baseline_type, seq_onehot, attribution))
            logger.info(f"Loaded sequence and attribution from: {seq_file}, {attr_file}")
        else:
            logger.warning(f"Sequence/attribution files not found for {baseline_type}, skipping sequence plot")

    return {
        'metadata': metadata,
        'label_trials': label_trials,
        'pred_trials': pred_trials,
        'importance_scores': importance_scores,
        'sequence_attributions': sequence_attributions,
    }


def clean_trial_name(trial_str, keep_suffix=True):
    """
    Clean trial name(s) for shorter filenames.
    Removes 'MiniAtlas-' prefix and 'RNA' substring.
    """
    trials = trial_str.split(';')
    cleaned_trials = []

    for trial in trials:
        cleaned = trial.replace('MiniAtlas-', '')
        cleaned = cleaned.replace('RNA', '')

        if not keep_suffix:
            cleaned = cleaned.replace('_plus', '').replace('_minus', '')

        cleaned_trials.append(cleaned)

    return '-'.join(cleaned_trials)


def plot_sequence_attribution(ax, seq_onehot, attribution, x_coords):
    """
    Plot DNA sequence with attribution scores in a logo-style visualization.

    Args:
        ax: Matplotlib axis to plot on
        seq_onehot: One-hot encoded sequence [L, 4] where columns are [A, C, G, T]
        attribution: Attribution scores [L, 4] for each nucleotide
        x_coords: X-axis coordinates for genomic positions (bp resolution)
    """
    # Multiply attribution by one-hot to get actual sequence attribution
    # Shape: [L, 4] for A, C, G, T
    importance_scores = attribution * seq_onehot

    # Transpose to [4, L] for plotting (ACGT x positions)
    importance_scores = importance_scores.T

    # Extract reference sequence
    ref_seq = ""
    for j in range(importance_scores.shape[1]):
        argmax_nt = np.argmax(np.abs(importance_scores[:, j]))

        if argmax_nt == 0:
            ref_seq += "A"
        elif argmax_nt == 1:
            ref_seq += "C"
        elif argmax_nt == 2:
            ref_seq += "G"
        elif argmax_nt == 3:
            ref_seq += "T"

    # Draw letters
    for i in range(len(ref_seq)):
        mutability_score = np.sum(importance_scores[:, i])
        dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=None)

    # Set plot properties
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlim((0, len(ref_seq)))

    # Set y-axis limits
    y_min = np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores))
    y_max = np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
    ax.set_ylim(y_min, y_max)

    # Add horizontal line at y=0
    ax.axhline(y=0., color='black', linestyle='-', linewidth=1)

    # Set font sizes
    ax.tick_params(axis='both', labelsize=7)
    ax.set_ylabel("Attribution", fontsize=9)

    return None


def plot_interpretation(data, output_file, dpi=300, start_idx=None, end_idx=None, show_sequence=True, plot_width=8.0, plot_height=1.5):
    """
    Generate interpretation plot from loaded data.

    Args:
        data: Dictionary containing loaded data
        output_file: Path to save the output plot
        dpi: DPI for the saved figure
        start_idx: Start index for plot region (relative to bin indices, None means start from beginning)
        end_idx: End index for plot region (relative to bin indices, None means plot to end)
        show_sequence: Whether to show sequence attribution plots (default: True)
    """
    logger = BaseLogger(name="Plotter", level=logging.INFO)

    metadata = data['metadata']
    label_trials = data['label_trials']
    pred_trials = data['pred_trials']
    importance_scores = data['importance_scores']
    sequence_attributions = data['sequence_attributions']

    # Get the genomic coordinate range
    window_size = metadata['window_size']
    real_start = metadata['real_start']
    real_end = metadata['real_end']

    # Calculate trim for the stored data
    trim = (
        metadata['context_length'] // window_size
        - metadata['n_window']
    ) // 2

    # Total bp after trimming
    total_bp = metadata['n_window'] * window_size
    region_start_bp = real_start + trim * window_size
    region_end_bp = region_start_bp + total_bp

    # Determine genomic range to plot
    if start_idx is None:
        start_genomic = region_start_bp
    else:
        start_genomic = max(start_idx, region_start_bp)

    if end_idx is None:
        end_genomic = region_end_bp
    else:
        end_genomic = min(end_idx, region_end_bp)

    # Validate range
    if start_genomic >= end_genomic:
        raise ValueError(f"Invalid range: start position {start_genomic} must be less than end position {end_genomic}")
    if start_genomic < region_start_bp or end_genomic > region_end_bp:
        logger.warning(f"Requested range [{start_genomic}, {end_genomic}) partially outside available region [{region_start_bp}, {region_end_bp})")

    # Calculate bp indices into the data arrays (0-indexed from region_start_bp)
    start_bp_idx = start_genomic - region_start_bp
    end_bp_idx = end_genomic - region_start_bp

    logger.info(f"Plotting region: genomic coords {start_genomic} to {end_genomic} ({end_genomic - start_genomic} bp)")

    # Create bp-resolution x coordinates
    # Add 1 to convert from 0-based to 1-based genomic coordinates for display
    x_bp = np.arange(start_genomic, end_genomic) + 1

    # Prepare plot data - only importance and sequence attribution
    plot_data = []
    plot_title = []
    plot_type = []  # 'line' or 'sequence'
    plot_x = []  # Store x-coordinates for each plot

    # Add importance scores at bp resolution (skip if showing sequence)
    if not show_sequence:
        for baseline_type, importance_data in importance_scores:
            # Importance data is already at bp resolution
            # Slice to requested range
            importance_bp = importance_data[start_bp_idx:end_bp_idx]

            plot_data.append(importance_bp)
            plot_x.append(x_bp)
            plot_title.append(f"Importance Score ({baseline_type} baseline)")
            plot_type.append('line')

    # Add sequence attribution plots (if requested)
    if show_sequence:
        for baseline_type, seq_onehot, attribution in sequence_attributions:
            # Slice sequence and attribution at bp resolution
            seq_slice = seq_onehot[start_bp_idx:end_bp_idx]
            attr_slice = attribution[start_bp_idx:end_bp_idx]

            plot_data.append((seq_slice, attr_slice))
            plot_x.append(x_bp)
            plot_title.append(f"Sequence Attribution ({baseline_type} baseline)")
            plot_type.append('sequence')

    # Create figure
    n = len(plot_data) + 1
    height_ratios = [1] * len(plot_data) + [0.1]
    _, axes = plt.subplots(  # fig is accessed implicitly by plt.savefig
        nrows=n, ncols=1, figsize=(plot_width, n * plot_height), sharex=False,  # Don't share x-axis since some may be smoothed
        gridspec_kw={"height_ratios": height_ratios}
    )

    # Ensure axes is always iterable (handles single subplot case)
    if n == 2:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    # Plot data
    # Calculate x-axis limits at bp resolution
    x_min = x_bp[0]
    x_max = x_bp[-1]

    for i, ax in enumerate(axes[:-1]):
        if plot_type[i] == 'line':
            ax.plot(plot_x[i], plot_data[i])
            ax.set_title(plot_title[i])
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            # Set x-axis limits to bp resolution
            ax.set_xlim(x_min, x_max)
            # Format x-axis for genomic positions dynamically based on range
            formatter = create_genomic_position_formatter(x_min, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        elif plot_type[i] == 'sequence':
            seq_onehot, attribution = plot_data[i]
            plot_sequence_attribution(ax, seq_onehot, attribution, plot_x[i])
            ax.set_title(plot_title[i])

    # Plot chromosome bar with region highlight
    ax = axes[-1]
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.set_xlabel('Genomic Position')

    # Set x-axis limits to bp resolution
    ax.set_xlim(x_min, x_max)

    # Format x-axis for genomic positions dynamically based on range
    formatter = create_genomic_position_formatter(x_min, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(formatter))

    linewidth = 1.5
    rec_width = 1
    chr_name = metadata['chr_name']
    bin_range = metadata['bin_range']

    ax.hlines(0.5, x_min, x_max, color="black", linewidth=linewidth)
    ax.set_title(f"Chromosome {chr_name[3:]}")

    # Convert bin_range to genomic coordinates and check if they overlap with plotted region
    # bin_range contains 0-indexed bin positions relative to the trimmed region
    bin_range_genomic = [(region_start_bp + b * window_size, region_start_bp + (b + 1) * window_size)
                          for b in bin_range]

    # Find overlapping ranges with the plotted region
    for bin_start, bin_end in bin_range_genomic:
        if bin_end > start_genomic and bin_start < end_genomic:
            # This bin overlaps with the plotted region
            overlap_start = max(bin_start, start_genomic)
            overlap_end = min(bin_end, end_genomic)
            rect = Rectangle(
                (overlap_start, 0.5 - 0.5 * rec_width),
                overlap_end - overlap_start,
                rec_width,
                facecolor="lightblue",
                edgecolor="black",
                linewidth=linewidth,
            )
            ax.add_patch(rect)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot DeepLift interpretation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data_dir", "-d",
        required=True,
        type=str,
        help="Directory containing the saved data files (e.g., RES_BASE/exp_name/analysis_chk/raw_data/interp_diff)"
    )

    parser.add_argument(
        "--name_base", "-n",
        required=True,
        type=str,
        help="Base name for the files (e.g., chr1_100000_200000_GENE_trial1_other-5)"
    )

    parser.add_argument(
        "--baseline",
        "-b",
        required=False,
        nargs='+',
        type=str,
        default=["random"],
        help="Baseline types to plot (must match those used in interpretation)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output file path for the plot"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure (default: 300)"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start genomic coordinate for plot region (default: start of region)"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End genomic coordinate for plot region (default: end of region)"
    )

    parser.add_argument(
        "--show_sequence",
        action="store_true",
        default=False,
        help="Show sequence attribution plots (can be slow for large regions)"
    )

    parser.add_argument(
        "--motif-viz-plot-width",
        type=float,
        default=8.0,
        help="Width of the motif visualization plot in inches (default: 8.0)"
    )

    parser.add_argument(
        "--motif-viz-plot-height",
        type=float,
        default=1.5,
        help="Height per subplot in inches (default: 1.5)"
    )

    args = parser.parse_args()

    logger = BaseLogger(name="PlotScript", level=logging.INFO)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Load data
    logger.info(f"Loading interpretation data from: {args.data_dir}")
    logger.info(f"Name base: {args.name_base}")

    baseline_types = args.baseline if isinstance(args.baseline, list) else [args.baseline]
    data = load_interpretation_data(args.data_dir, args.name_base, baseline_types)

    # Generate plot
    logger.info("Generating plot...")
    if args.show_sequence:
        logger.info("Sequence attribution plots enabled (importance line plot hidden)")
    plot_interpretation(data, args.output, dpi=args.dpi, start_idx=args.start, end_idx=args.end, show_sequence=args.show_sequence, plot_width=args.motif_viz_plot_width, plot_height=args.motif_viz_plot_height)

    logger.info("Done!")


if __name__ == "__main__":
    main()
