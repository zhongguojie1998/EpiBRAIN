import os
import warnings

warnings.filterwarnings("ignore")

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import pyranges as pr

base_cmap = plt.get_cmap("tab20")


def calculate_modality_metrics_batch(mod_name, label_data, pred_data):
    """
    Calculate Pearson correlation for all samples of one modality using vectorized operations
    Calculates both raw scale and log scale correlations
    label_data, pred_data: [n_regions, n_celltypes] arrays
    """
    # RAW SCALE CORRELATION
    # Center the data (subtract mean along celltype axis)
    label_centered = label_data - label_data.mean(axis=1, keepdims=True)
    pred_centered = pred_data - pred_data.mean(axis=1, keepdims=True)

    # Calculate standard deviations
    label_std = label_data.std(axis=1)
    pred_std = pred_data.std(axis=1)

    # Vectorized Pearson correlation: sum(centered_x * centered_y) / (n * std_x * std_y)
    n_celltypes = label_data.shape[1]
    numerator = (label_centered * pred_centered).sum(axis=1)
    denominator = n_celltypes * label_std * pred_std

    # Handle division by zero (constant arrays)
    with np.errstate(divide='ignore', invalid='ignore'):
        pearsonr_vals_raw = numerator / denominator
        pearsonr_vals_raw = np.where(np.isfinite(pearsonr_vals_raw), pearsonr_vals_raw, np.nan)

    # LOG SCALE CORRELATION
    # Use log1p (log(1+x)) to handle zeros and ensure all values are positive
    label_log = np.log1p(np.abs(label_data))
    pred_log = np.log1p(np.abs(pred_data))

    # Center the log-transformed data
    label_log_centered = label_log - label_log.mean(axis=1, keepdims=True)
    pred_log_centered = pred_log - pred_log.mean(axis=1, keepdims=True)

    # Calculate standard deviations for log-transformed data
    label_log_std = label_log.std(axis=1)
    pred_log_std = pred_log.std(axis=1)

    # Vectorized Pearson correlation for log scale
    numerator_log = (label_log_centered * pred_log_centered).sum(axis=1)
    denominator_log = n_celltypes * label_log_std * pred_log_std

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pearsonr_vals_log = numerator_log / denominator_log
        pearsonr_vals_log = np.where(np.isfinite(pearsonr_vals_log), pearsonr_vals_log, np.nan)

    # Calculate label statistics (raw scale)
    label_vars_raw = label_data.var(axis=1)
    label_means_raw = label_data.mean(axis=1)

    # Calculate label statistics (log scale)
    label_vars_log = label_log.var(axis=1)
    label_means_log = label_log.mean(axis=1)

    return mod_name, (pearsonr_vals_raw, pearsonr_vals_log, label_vars_raw, label_means_raw, label_vars_log, label_means_log)


def load_genomic_coordinates(exp_name, split, test_res, bin_size=32):
    """
    Load genomic coordinates for predictions.
    Loads from Data/{exp_name}/sequences.bed, filters by split, indexes by test_res['index'],
    and chunks sequences into bin_size bp windows.

    Parameters:
    -----------
    exp_name: str, experiment name
    split: str, data split (e.g., 'Test', 'Train', 'Valid')
    test_res: dict, loaded predictions containing 'index' key
    bin_size: int, size of bins in bp (default: 32)

    Returns:
    --------
    coords: pd.DataFrame with columns ['chr', 'start', 'end'] for each bin
    """
    # Load the full sequences bed file
    sequences_bed = f"Data/{exp_name}/sequences.bed"
    if not os.path.exists(sequences_bed):
        raise FileNotFoundError(f"Could not find sequences bed file: {sequences_bed}")

    print(f"Loading sequences from: {sequences_bed}")
    # Read bed file - typically has chr, start, end, and split column as last column
    sequences = pd.read_csv(sequences_bed, sep='\t', header=None)

    # Assume last column is the split identifier
    n_cols = len(sequences.columns)
    col_names = ['chr', 'start', 'end'] + [f'col{i}' for i in range(3, n_cols-1)] + ['split']
    sequences.columns = col_names

    # Filter to the corresponding split (convert Test -> test)
    split_lower = split.lower()
    sequences_filtered = sequences[sequences['split'] == split_lower].reset_index(drop=True)
    print(f"Filtered to {len(sequences_filtered)} sequences for split '{split_lower}'")

    # Get indices from test_res
    if 'index' not in test_res:
        raise KeyError("test_res does not contain 'index' key")

    indices = test_res['index']
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()

    print(f"Using {len(indices)} sequence indices from test_res['index']")

    # Select sequences based on indices
    selected_sequences = sequences_filtered.iloc[indices].reset_index(drop=True)

    # Chunk each sequence into bin_size bp windows
    # Note: Drop first 32 and last 32 bins (model context requirement)
    drop_bins_start = 32
    drop_bins_end = 32
    print(f"Chunking sequences into {bin_size}bp bins (dropping first {drop_bins_start} and last {drop_bins_end} bins)...")
    all_bins = []
    for _, seq in tqdm(selected_sequences.iterrows(), total=len(selected_sequences), desc="Chunking sequences"):
        chr_name = str(seq['chr'])
        seq_start = int(seq['start'])
        seq_end = int(seq['end'])

        # Generate all bins for this sequence
        seq_bins = []
        for bin_start in range(seq_start, seq_end, bin_size):
            bin_end = min(bin_start + bin_size, seq_end)
            seq_bins.append({
                'chr': chr_name,
                'start': bin_start,
                'end': bin_end
            })

        # Drop first 32 and last 32 bins
        if len(seq_bins) > drop_bins_start + drop_bins_end:
            seq_bins = seq_bins[drop_bins_start:-drop_bins_end]
            all_bins.extend(seq_bins)
        else:
            print(f"Warning: Sequence too short ({len(seq_bins)} bins), skipping")

    coords = pd.DataFrame(all_bins)
    print(f"Generated {len(coords)} bins from {len(selected_sequences)} sequences")

    return coords


def aggregate_predictions_in_regions(bed_regions, coords, labels, preds, aggregation='mean', n_jobs=-1):
    """
    Aggregate predictions and labels within each bed region using PyRanges for efficient overlap detection.

    Parameters:
    -----------
    bed_regions: pd.DataFrame with columns ['chr', 'start', 'end', ...]
    coords: pd.DataFrame with columns ['chr', 'start', 'end'] for each prediction bin
    labels: np.array of shape [n_bins, n_celltypes]
    preds: np.array of shape [n_bins, n_celltypes]
    aggregation: str, method to aggregate ('mean', 'sum', 'max')
    n_jobs: int, number of parallel jobs (not used with PyRanges, kept for compatibility)

    Returns:
    --------
    aggregated_labels: np.array of shape [n_regions, n_celltypes]
    aggregated_preds: np.array of shape [n_regions, n_celltypes]
    """
    n_regions = len(bed_regions)
    n_celltypes = labels.shape[1]

    print(f"Aggregating predictions for {n_regions} regions using PyRanges...")

    # Prepare bed_regions DataFrame with required columns for PyRanges
    regions_df = bed_regions[['chr', 'start', 'end']].copy()
    regions_df['region_idx'] = np.arange(len(regions_df))

    # Ensure correct column names for PyRanges (Chromosome, Start, End)
    regions_df = regions_df.rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})

    # Prepare coords DataFrame with required columns
    coords_df = coords.copy()
    coords_df['bin_idx'] = np.arange(len(coords_df))
    coords_df = coords_df.rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})

    # Create PyRanges objects
    regions_pr = pr.PyRanges(regions_df)
    coords_pr = pr.PyRanges(coords_df)

    # Find overlaps between regions and bins
    print("Finding overlaps between regions and bins...")
    overlaps = regions_pr.join(coords_pr, how='left')

    # Convert back to DataFrame for processing
    if len(overlaps) > 0:
        overlaps_df = overlaps.df
    else:
        overlaps_df = pd.DataFrame(columns=['region_idx', 'bin_idx'])

    # Initialize result arrays
    aggregated_labels = np.full((n_regions, n_celltypes), np.nan)
    aggregated_preds = np.full((n_regions, n_celltypes), np.nan)

    print("Aggregating values for each region...")
    # Group by region_idx and aggregate
    if len(overlaps_df) > 0:
        for region_idx, group in tqdm(overlaps_df.groupby('region_idx'), desc="Aggregating regions"):
            bin_indices = group['bin_idx'].values.astype(int)

            if len(bin_indices) > 0:
                if aggregation == 'mean':
                    aggregated_labels[region_idx, :] = labels[bin_indices, :].mean(axis=0)
                    aggregated_preds[region_idx, :] = preds[bin_indices, :].mean(axis=0)
                elif aggregation == 'sum':
                    aggregated_labels[region_idx, :] = labels[bin_indices, :].sum(axis=0)
                    aggregated_preds[region_idx, :] = preds[bin_indices, :].sum(axis=0)
                elif aggregation == 'max':
                    aggregated_labels[region_idx, :] = labels[bin_indices, :].max(axis=0)
                    aggregated_preds[region_idx, :] = preds[bin_indices, :].max(axis=0)

    return aggregated_labels, aggregated_preds


@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint name")
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"], help="Data splits to process")
@click.option("--res_base", required=True, default="./Res", help="Results base directory")
@click.option("--log_base", required=True, default="./logs", help="Logs base directory")
@click.option("--bed_file", required=True, type=str, help="BED file with regions to analyze")
@click.option("--coord_file", type=str, default=None, help="Coordinate file for predictions (optional, defaults to Data/{exp_name}/sequences.bed)")
@click.option("--bin_size", type=int, default=32, help="Bin size in bp (default: 32)")
@click.option("--aggregation", type=str, default="mean", help="Aggregation method: mean, sum, or max")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
def main(exp_name, chk, splits, res_base, log_base, bed_file, coord_file, bin_size, aggregation, n_processes):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    bed_basename = os.path.splitext(os.path.basename(bed_file))[0]
    output_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}_bed_{bed_basename}"
    os.makedirs(f"{output_dir}/plot", exist_ok=True)
    os.makedirs(f"{output_dir}/raw_data", exist_ok=True)

    # Load bed file
    print(f"Loading BED file: {bed_file}")
    bed_regions = pd.read_csv(bed_file, sep='\t', header=None)
    # Standard BED format: chr, start, end, [name, score, strand, ...]
    bed_columns = ['chr', 'start', 'end']
    if len(bed_regions.columns) > 3:
        bed_columns.extend([f'col{i}' for i in range(3, len(bed_regions.columns))])
    bed_regions.columns = bed_columns[:len(bed_regions.columns)]
    # Ensure chr is string type for comparison
    bed_regions['chr'] = bed_regions['chr'].astype(str)
    print(f"Loaded {len(bed_regions)} regions")

    # Load label meta info
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    # Calculate correlation and other metrics
    for split in splits:
        output_file = f"{output_dir}/raw_data/{split}_metric_across_celltypes_bed.csv"
        if not os.path.exists(output_file):
            print(f"\nCalculate metric for {split}")

            # Load predictions
            test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
            # Convert to numpy arrays to avoid deprecation warnings
            test_label_full = test_res["label"]['regression'].reshape(-1, test_res["label"]['regression'].shape[-1]).cpu().numpy()
            test_pred_full = test_res["pred"]['regression'].reshape(-1, test_res["pred"]['regression'].shape[-1]).cpu().numpy()

            print(f"Loaded predictions: {test_label_full.shape[0]} bins, {test_label_full.shape[1]} cell types (raw)")

            # Filter predictions to only include cell types defined in label_meta
            pred_indices = label_meta['dim'].values
            label_indices = label_meta.index.values
            test_label = test_label_full[:, label_indices]
            test_pred = test_pred_full[:, pred_indices]

            print(f"Filtered to {test_label.shape[1]} cell types defined in label_meta")

            # Load genomic coordinates
            if coord_file:
                print(f"Loading coordinates from: {coord_file}")
                coords = pd.read_csv(coord_file, sep='\t', header=None, names=['chr', 'start', 'end'])
                # Ensure chr is string type
                coords['chr'] = coords['chr'].astype(str)
            else:
                print("Loading coordinates from sequences.bed...")
                coords = load_genomic_coordinates(exp_name, split, test_res, bin_size=bin_size)

            print(f"Loaded {len(coords)} coordinate entries")

            # Verify dimensions match
            if len(coords) != test_label.shape[0]:
                raise ValueError(f"Coordinate entries ({len(coords)}) don't match prediction bins ({test_label.shape[0]})")

            # Set number of processes
            n_proc = n_processes if n_processes else cpu_count()
            print(f"Using {n_proc} processes for parallel processing")

            # Aggregate predictions within bed regions
            aggregated_label, aggregated_pred = aggregate_predictions_in_regions(
                bed_regions, coords, test_label, test_pred, aggregation=aggregation, n_jobs=n_proc
            )

            print(f"Aggregated to {aggregated_label.shape[0]} regions")

            # Initialize metrics dataframe
            metric = pd.DataFrame(index=label_meta["modality"].unique(),
                                  columns=["PearsonR_raw:mean", "PearsonR_raw:std", "PearsonR_raw:median", "PearsonR_raw:25%", "PearsonR_raw:75%",
                                           "PearsonR_log:mean", "PearsonR_log:std", "PearsonR_log:median", "PearsonR_log:25%", "PearsonR_log:75%"])

            # Precompute modality indices and prepare batched data
            modality_data = {}
            for mod in metric.index:
                mod_celltypes = label_meta[label_meta["modality"] == mod]
                label_indices = mod_celltypes.index.values
                pred_indices = mod_celltypes.index.values

                # Prepare batched data for this modality (all regions at once)
                modality_data[mod] = (
                    aggregated_label[:, label_indices],
                    aggregated_pred[:, pred_indices]
                )
            print(f"Calculating metrics for {len(metric.index)} modalities with {aggregated_label.shape[0]} regions each")

            # Calculate metrics in parallel, one task per modality (batched)
            results = Parallel(n_jobs=n_proc, backend='loky', verbose=10)(
                delayed(calculate_modality_metrics_batch)(
                    mod,
                    modality_data[mod][0],
                    modality_data[mod][1]
                )
                for mod in metric.index
            )

            # Store results
            metric_dict = {mod: {"pearsonr_raw": [], "pearsonr_log": [],
                                 "label_mean_raw": [], "label_var_raw": [],
                                 "label_mean_log": [], "label_var_log": []} for mod in metric.index}
            for mod_name, (pearsonr_vals_raw, pearsonr_vals_log, label_vars_raw, label_means_raw, label_vars_log, label_means_log) in results:
                metric_dict[mod_name]["pearsonr_raw"] = pearsonr_vals_raw.tolist()
                metric_dict[mod_name]["pearsonr_log"] = pearsonr_vals_log.tolist()
                metric_dict[mod_name]["label_mean_raw"] = label_means_raw.tolist()
                metric_dict[mod_name]["label_var_raw"] = label_vars_raw.tolist()
                metric_dict[mod_name]["label_mean_log"] = label_means_log.tolist()
                metric_dict[mod_name]["label_var_log"] = label_vars_log.tolist()

            with open(f"{output_dir}/raw_data/{split}_metric_across_celltypes_bed.pkl", "wb") as f:
                pickle.dump(metric_dict, f)

            for mod in metric.index:
                # Raw scale metrics
                pearsonr_vals_raw = np.array(metric_dict[mod]["pearsonr_raw"])
                # remove nan values
                pearsonr_vals_raw = pearsonr_vals_raw[~np.isnan(pearsonr_vals_raw)]
                if pearsonr_vals_raw.size > 0:
                    metric.loc[mod, "PearsonR_raw:mean"] = np.mean(pearsonr_vals_raw)
                    metric.loc[mod, "PearsonR_raw:std"] = np.std(pearsonr_vals_raw)
                    metric.loc[mod, "PearsonR_raw:median"] = np.median(pearsonr_vals_raw)
                    metric.loc[mod, "PearsonR_raw:25%"] = np.percentile(pearsonr_vals_raw, 25)
                    metric.loc[mod, "PearsonR_raw:75%"] = np.percentile(pearsonr_vals_raw, 75)

                # Log scale metrics
                pearsonr_vals_log = np.array(metric_dict[mod]["pearsonr_log"])
                # remove nan values
                pearsonr_vals_log = pearsonr_vals_log[~np.isnan(pearsonr_vals_log)]
                if pearsonr_vals_log.size > 0:
                    metric.loc[mod, "PearsonR_log:mean"] = np.mean(pearsonr_vals_log)
                    metric.loc[mod, "PearsonR_log:std"] = np.std(pearsonr_vals_log)
                    metric.loc[mod, "PearsonR_log:median"] = np.median(pearsonr_vals_log)
                    metric.loc[mod, "PearsonR_log:25%"] = np.percentile(pearsonr_vals_log, 25)
                    metric.loc[mod, "PearsonR_log:75%"] = np.percentile(pearsonr_vals_log, 75)

            metric.to_csv(output_file)
            print(f"Saved metrics to {output_file}")

    # Plot (modality level)
    for split in splits:
        print(f"\nPlot metric (modality level) for {split}")

        with open(f"{output_dir}/raw_data/{split}_metric_across_celltypes_bed.pkl", "rb") as f:
            metric_dict = pickle.load(f)

        metric = pd.read_csv(f"{output_dir}/raw_data/{split}_metric_across_celltypes_bed.csv", index_col=0)
        metric["modality"] = metric.index.str.rsplit("_", n=1).str[-1]

        # Plot for both raw and log scale
        for scale in ["raw", "log"]:
            # Prepare data for plotting
            plot_df = pd.DataFrame(columns=["modality", "PearsonR"])
            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod][f"pearsonr_{scale}"])
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                temp_df = pd.DataFrame({"modality": [mod]*len(pearsonr_vals), "PearsonR": pearsonr_vals})
                plot_df = pd.concat([plot_df, temp_df], ignore_index=True)

            # Create figure with adjusted dimensions
            fig, ax = plt.subplots(figsize=(10, 20))

            # Create violin plot
            sns.violinplot(
                y="PearsonR", x="modality", data=plot_df, palette="tab20", inner="quartile", cut=0, ax=ax
            )

            # Add grid line
            ax.grid(axis="x", linestyle="--", alpha=0.6, color="gray")
            sns.despine(left=True, bottom=True)

            # Adjust legend position
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")
            ax.set_title(f"Pearson Correlation - {scale.capitalize()} Scale\n{bed_basename} regions ({split} Set)")
            fig.tight_layout()
            fig.savefig(
                f"{output_dir}/plot/{split}_pearsonr_{scale}_across_cell_types_bed.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Plot mean-variance colored by pearsonr (separate plots for each modality)
        for scale in ["raw", "log"]:
            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod][f"pearsonr_{scale}"])
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                if len(pearsonr_vals) == 0:
                    continue
                label_means = np.array(metric_dict[mod][f"label_mean_{scale}"])
                label_means = label_means[~np.isnan(label_means)]
                if len(label_means) == 0:
                    continue
                label_vars = np.array(metric_dict[mod][f"label_var_{scale}"])
                label_vars = label_vars[~np.isnan(label_vars)]
                if len(label_vars) == 0:
                    continue
                # Match lengths by truncating to the shortest array
                min_len = min(len(pearsonr_vals), len(label_means), len(label_vars))
                pearsonr_vals = pearsonr_vals[:min_len]
                label_means = label_means[:min_len]
                label_vars = label_vars[:min_len]

                # Create separate plot for this modality
                fig, ax = plt.subplots(figsize=(10, 8))
                sc = ax.scatter(
                    label_means,
                    label_vars,
                    c=pearsonr_vals,
                    cmap="viridis",
                    alpha=0.6,
                    s=20,
                    vmin=-1,
                    vmax=1,
                )
                ax.set_xlabel(f"Label Mean ({scale.capitalize()} Scale)")
                ax.set_ylabel(f"Label Variance ({scale.capitalize()} Scale)")
                ax.set_title(f"Label Mean-Variance Colored by PearsonR ({scale.capitalize()})\n{mod} - {bed_basename} regions ({split} Set)")
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("PearsonR")
                fig.tight_layout()

                # Create safe filename
                safe_mod = mod.replace('/', '_').replace(' ', '_')
                fig.savefig(
                    f"{output_dir}/plot/{split}_{safe_mod}_label_mean_var_colored_by_pearsonr_{scale}_bed.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
