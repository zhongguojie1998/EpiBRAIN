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
from matplotlib.patches import Rectangle, Polygon, Ellipse
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)

from utils.logging import BaseLogger


def add_letter_to_axis(ax, letter, x_pos, y_pos, height, width, color):
    """
    Add a nucleotide letter to the axis at the specified position.

    Args:
        ax: Matplotlib axis
        letter: One of 'A', 'C', 'G', 'T'
        x_pos: X position (center)
        y_pos: Y position (bottom)
        height: Height of the letter
        width: Width of the letter
        color: Color for the letter
    """
    if height == 0:
        return

    # Create transform for positioning and scaling
    trans = Affine2D().scale(width, height).translate(x_pos - width/2, y_pos) + ax.transData

    if letter == 'A':
        # Draw A as three polygons (two triangles and a crossbar)
        # Left side
        left_triangle = Polygon([[0.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.2, 0.0]],
                               facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(left_triangle)

        # Right side
        right_triangle = Polygon([[1.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.8, 0.0]],
                                facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(right_triangle)

        # Crossbar
        crossbar = Polygon([[0.225, 0.45], [0.775, 0.45], [0.775, 0.3], [0.225, 0.3]],
                          facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(crossbar)

    elif letter == 'C':
        # Draw C as an ellipse with a cutout
        # Outer ellipse
        outer = Ellipse((0.5, 0.5), 0.8, 1.0, facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(outer)

        # Inner ellipse (white cutout)
        inner = Ellipse((0.5, 0.5), 0.4, 0.65, facecolor='white', edgecolor='none', transform=trans, zorder=3)
        ax.add_patch(inner)

        # Right cutout rectangle
        cutout = Rectangle((0.5, 0.0), 0.5, 1.0, facecolor='white', edgecolor='none', transform=trans, zorder=3)
        ax.add_patch(cutout)

    elif letter == 'G':
        # Draw G similar to C but with additional bar
        # Outer ellipse
        outer = Ellipse((0.5, 0.5), 0.8, 1.0, facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(outer)

        # Inner ellipse (white cutout)
        inner = Ellipse((0.5, 0.5), 0.4, 0.65, facecolor='white', edgecolor='none', transform=trans, zorder=3)
        ax.add_patch(inner)

        # Right cutout rectangle
        cutout = Rectangle((0.5, 0.0), 0.5, 1.0, facecolor='white', edgecolor='none', transform=trans, zorder=3)
        ax.add_patch(cutout)

        # Horizontal bar for G
        g_bar = Rectangle((0.5, 0.35), 0.35, 0.15, facecolor=color, edgecolor='none', transform=trans, zorder=4)
        ax.add_patch(g_bar)

    elif letter == 'T':
        # Draw T as two rectangles (vertical stem and horizontal top)
        # Vertical stem
        stem = Rectangle((0.4, 0.0), 0.2, 1.0, facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(stem)

        # Horizontal top
        top = Rectangle((0.0, 0.8), 1.0, 0.2, facecolor=color, edgecolor='none', transform=trans)
        ax.add_patch(top)


def format_genomic_position(x, _pos):
    """
    Format genomic position for axis labels.
    Converts to kb or Mb as appropriate.

    Args:
        x: Position value
        _pos: Tick position (unused but required by FuncFormatter)

    Returns:
        Formatted string
    """
    if abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.0f}k'
    else:
        return f'{int(x)}'


def smooth_data(data, x_coords, target_bins=1024):
    """
    Smooth data by binning/averaging when there are too many bins.

    Args:
        data: 1D array of values to smooth
        x_coords: Corresponding x-coordinates
        target_bins: Target number of bins after smoothing (default: 1024)

    Returns:
        Tuple of (smoothed_data, smoothed_x_coords)
    """
    n_bins = len(data)

    if n_bins <= target_bins:
        # No smoothing needed
        return data, x_coords

    # Calculate bin size for downsampling
    bin_size = n_bins / target_bins

    # Create new bins
    smoothed_data = []
    smoothed_x = []

    for i in range(target_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)

        # Average the data in this bin
        bin_mean = np.mean(data[start_idx:end_idx])
        smoothed_data.append(bin_mean)

        # Use the center position of the bin
        bin_x_mean = np.mean(x_coords[start_idx:end_idx])
        smoothed_x.append(bin_x_mean)

    return np.array(smoothed_data), np.array(smoothed_x)


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
    if not os.path.exists(label_trials_file):
        raise FileNotFoundError(f"Label trials file not found: {label_trials_file}")
    label_trials = np.load(label_trials_file, allow_pickle=True).item()
    logger.info(f"Loaded {len(label_trials)} label trials from: {label_trials_file}")

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
        trial_neg_clean = f"other-{neg_trial_count}"

        identifier = f"{chr_name}_{start}_{end}_{region_name}_{trial_pos_clean}_{trial_neg_clean}_{baseline_type}"
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


def plot_sequence_attribution(ax, seq_onehot, attribution, x_coords, bin_highlights=None, start_bin_idx=0):
    """
    Plot DNA sequence with attribution scores in a logo-style visualization.

    Args:
        ax: Matplotlib axis to plot on
        seq_onehot: One-hot encoded sequence [L, 4] where columns are [A, C, G, T]
        attribution: Attribution scores [L, 4] for each nucleotide
        x_coords: X-axis coordinates for genomic positions
        bin_highlights: List of bin indices to highlight (e.g., exon positions)
        start_bin_idx: Starting bin index for the secondary x-axis (default: 0)
    """
    # Define colors for each nucleotide (A=green, C=blue, G=orange, T=red)
    colors = ['#00B050', '#0070C0', '#FFC000', '#C00000']
    nucleotides = ['A', 'C', 'G', 'T']

    # Multiply attribution by one-hot to get actual sequence attribution
    seq_attr = attribution * seq_onehot

    # Calculate letter width based on spacing
    if len(x_coords) > 1:
        letter_width = (x_coords[1] - x_coords[0]) * 0.9  # 90% of bin width
    else:
        letter_width = 1.0

    # Track min/max for axis limits
    max_pos = 0
    min_neg = 0

    # For each position, plot nucleotide letters with heights proportional to attribution
    for i, x in enumerate(x_coords):
        # Get attribution values at this position for all nucleotides
        attr_vals = seq_attr[i]  # [4] values for A, C, G, T

        # Sort by absolute value to stack letters
        sorted_indices = np.argsort(np.abs(attr_vals))

        # Separate positive and negative contributions
        pos_height = 0
        neg_height = 0

        for nuc_idx in sorted_indices:
            attr_val = attr_vals[nuc_idx]

            if attr_val > 0:
                # Draw letter above previous positive letters
                add_letter_to_axis(ax, nucleotides[nuc_idx], x, pos_height,
                                 attr_val, letter_width, colors[nuc_idx])
                pos_height += attr_val
                max_pos = max(max_pos, pos_height)
            elif attr_val < 0:
                # Draw letter below previous negative letters
                add_letter_to_axis(ax, nucleotides[nuc_idx], x, neg_height + attr_val,
                                 abs(attr_val), letter_width, colors[nuc_idx])
                neg_height += attr_val
                min_neg = min(min_neg, neg_height)

    # Set axis limits with padding
    padding = max(max_pos - min_neg, 0.1) * 0.1  # 10% padding
    ax.set_ylim(min_neg - padding, max_pos + padding)
    ax.set_xlim(x_coords[0] - letter_width, x_coords[-1] + letter_width)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5, zorder=1)

    # Highlight specified bins (e.g., exons) - draw behind letters
    if bin_highlights is not None and len(bin_highlights) > 0:
        for bin_idx in bin_highlights:
            if 0 <= bin_idx < len(x_coords):
                x = x_coords[bin_idx]
                width = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1
                rect = Rectangle((x - width/2, min_neg - padding), width,
                                (max_pos + padding) - (min_neg - padding),
                                facecolor='lightblue', alpha=0.2, edgecolor='none', zorder=0)
                ax.add_patch(rect)

    # Create legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=nucleotides[i])
                      for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4,
             framealpha=0.9, fontsize=8)

    # Set up dual x-axes
    ax.set_ylabel('Attribution Score')
    ax.set_xlabel('Genomic Position', color='black')
    ax.grid(True, alpha=0.3, axis='y', zorder=0)

    # Format x-axis to show kb/Mb
    ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_position))

    # Add secondary x-axis for bin indices
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Calculate bin positions
    n_bins = len(x_coords)
    bin_indices = np.arange(start_bin_idx, start_bin_idx + n_bins)

    # Set up secondary axis ticks
    # Show ticks at regular intervals
    tick_interval = max(1, n_bins // 10)  # Show ~10 ticks
    tick_positions_idx = np.arange(0, n_bins, tick_interval)
    tick_positions_genomic = x_coords[tick_positions_idx]
    tick_labels = bin_indices[tick_positions_idx]

    ax2.set_xticks(tick_positions_genomic)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Bin Index (Relative)', color='navy')
    ax2.tick_params(axis='x', labelcolor='navy')

    return ax2


def plot_interpretation(data, output_file, dpi=300, start_idx=None, end_idx=None, show_sequence=True):
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

    # Calculate x coordinates first to determine data length
    trim = (
        metadata['context_length'] // metadata['window_size']
        - metadata['n_window']
    ) // 2
    real_start = metadata['real_start']
    real_end = metadata['real_end']
    x_full = np.arange(real_start, real_end).reshape(-1, metadata['window_size'])[trim:-trim, 0]

    # Determine slice indices
    data_length = len(x_full)
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = data_length

    # Validate indices
    if start_idx < 0:
        start_idx = 0
    if end_idx > data_length:
        end_idx = data_length
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx ({start_idx}) must be less than end_idx ({end_idx})")

    logger.info(f"Plotting region: bins {start_idx} to {end_idx} (out of {data_length} total bins)")

    # Slice x coordinates
    x = x_full[start_idx:end_idx]
    n_bins_to_plot = len(x)

    # Determine if smoothing is needed
    needs_smoothing = n_bins_to_plot > 1024
    if needs_smoothing:
        logger.info(f"Smoothing data from {n_bins_to_plot} bins to 1024 bins for better visualization")

    # Prepare plot data - plot each trial separately
    plot_data = []
    plot_title = []
    plot_type = []  # 'line' or 'sequence'
    plot_x = []  # Store x-coordinates for each plot (may differ if smoothed)

    # Add individual label trials (sliced and optionally smoothed)
    for trial_name, trial_data in label_trials.items():
        data_slice = trial_data[start_idx:end_idx]
        if needs_smoothing:
            smoothed_data, smoothed_x = smooth_data(data_slice, x, target_bins=1024)
            plot_data.append(smoothed_data)
            plot_x.append(smoothed_x)
        else:
            plot_data.append(data_slice)
            plot_x.append(x)
        plot_title.append(f"{trial_name} Target")
        plot_type.append('line')

    # Add individual prediction trials (sliced and optionally smoothed)
    for trial_name, trial_data in pred_trials.items():
        data_slice = trial_data[start_idx:end_idx]
        if needs_smoothing:
            smoothed_data, smoothed_x = smooth_data(data_slice, x, target_bins=1024)
            plot_data.append(smoothed_data)
            plot_x.append(smoothed_x)
        else:
            plot_data.append(data_slice)
            plot_x.append(x)
        plot_title.append(f"{trial_name} Pred")
        plot_type.append('line')

    # Add importance scores (sliced, not smoothed to preserve sharp features)
    for baseline_type, importance_data in importance_scores:
        plot_data.append(importance_data[start_idx:end_idx])
        plot_x.append(x)
        plot_title.append(f"Importance Score ({baseline_type} baseline)")
        plot_type.append('line')

    # Add sequence attribution plots (if requested, never smoothed)
    if show_sequence:
        for baseline_type, seq_onehot, attribution in sequence_attributions:
            plot_data.append((seq_onehot[start_idx:end_idx], attribution[start_idx:end_idx]))
            plot_x.append(x)  # Use original x for sequence plots
            plot_title.append(f"Sequence Attribution ({baseline_type} baseline)")
            plot_type.append('sequence')

    # Create figure
    n = len(plot_data) + 1
    height_ratios = [1] * len(plot_data) + [0.1]
    _, axes = plt.subplots(  # fig is accessed implicitly by plt.savefig
        nrows=n, ncols=1, figsize=(8, n * 1.5), sharex=False,  # Don't share x-axis since some may be smoothed
        gridspec_kw={"height_ratios": height_ratios}
    )

    # Ensure axes is always iterable (handles single subplot case)
    if n == 2:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    # Plot data
    bin_range = metadata['bin_range']
    for i, ax in enumerate(axes[:-1]):
        if plot_type[i] == 'line':
            ax.plot(plot_x[i], plot_data[i])
            ax.set_title(plot_title[i])
            ax.set_ylabel(None)
            ax.set_xlabel(None)
            # Set x-axis limits to match the original range
            ax.set_xlim(x[0], x[-1])
            # Format x-axis for genomic positions
            ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_position))
        elif plot_type[i] == 'sequence':
            seq_onehot, attribution = plot_data[i]
            # Adjust bin_range to be relative to the plotted region
            bin_highlights_adjusted = []
            for bin_idx in bin_range:
                if start_idx <= bin_idx < end_idx:
                    bin_highlights_adjusted.append(bin_idx - start_idx)

            plot_sequence_attribution(ax, seq_onehot, attribution, plot_x[i],
                                    bin_highlights=bin_highlights_adjusted,
                                    start_bin_idx=start_idx)
            ax.set_title(plot_title[i])

    # Plot chromosome bar with region highlight
    ax = axes[-1]
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.set_xlabel('Genomic Position')

    # Set x-axis limits to match the original range
    ax.set_xlim(x[0], x[-1])

    # Format x-axis for genomic positions
    ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_position))

    linewidth = 1.5
    rec_width = 1
    chr_name = metadata['chr_name']
    bin_range = metadata['bin_range']

    ax.hlines(0.5, x[0], x[-1], color="black", linewidth=linewidth)
    ax.set_title(f"Chromosome {chr_name[3:]}")

    # Adjust bin_range to be relative to the plotted region
    # Find which bins in the original bin_range are within the plotted region
    bin_range_in_plot = []
    for bin_idx in bin_range:
        if start_idx <= bin_idx < end_idx:
            bin_range_in_plot.append(bin_idx - start_idx)

    # Only draw rectangle if there are bins to highlight in the plotted region
    if len(bin_range_in_plot) > 0:
        rect = Rectangle(
            (x[bin_range_in_plot[0]], 0.5 - 0.5 * rec_width),
            x[bin_range_in_plot[-1]] - x[bin_range_in_plot[0]],
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
        help="Start index for plot region (relative bin index, default: 0)"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for plot region (relative bin index, default: plot to end)"
    )

    parser.add_argument(
        "--show_sequence",
        action="store_true",
        default=False,
        help="Show sequence attribution plots (can be slow for large regions)"
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
        logger.info("Sequence attribution plots enabled")
    plot_interpretation(data, args.output, dpi=args.dpi, start_idx=args.start, end_idx=args.end, show_sequence=args.show_sequence)

    logger.info("Done!")


if __name__ == "__main__":
    main()
