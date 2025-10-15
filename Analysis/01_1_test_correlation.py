import multiprocessing as mp
import os
import warnings

warnings.filterwarnings("ignore")

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import quantile_transform
from tqdm import tqdm
import numpy as np

base_cmap = plt.get_cmap("tab20")


def apply_transform(data, transform_type="none", label_meta=None):
    """
    Apply transformation to data.

    Args:
        data: numpy array of shape (n_samples, n_features)
        transform_type: str, one of "none", "log", "quantile", "log_quantile", "log_quantile_substract_mean"
        label_meta: pandas DataFrame with trial metadata, required for "log_quantile_substract_mean"

    Returns:
        Transformed data
    """
    if transform_type == "none":
        return data

    data = data.copy()

    if transform_type == "log":
        # Apply log1p transformation (log(1 + x))
        data = np.log1p(np.maximum(data, 0))

    elif transform_type == "quantile":
        # Apply quantile normalization
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))

    elif transform_type == "log_quantile":
        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))

    elif transform_type == "log_quantile_substract_mean":
        if label_meta is None:
            raise ValueError("label_meta is required for log_quantile_substract_mean transform")

        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))

        # Extract modality information from trial names
        # Assuming trial names are in format: celltype_modality
        modalities = label_meta["trial"].str.rsplit("_", n=1).str[-1]

        # Subtract mean for each modality separately (bin-wise within each modality)
        for modality in modalities.unique():
            modality_mask = (modalities == modality).values
            modality_indices = np.where(modality_mask)[0]
            # Subtract mean across cell types within this modality for each bin
            data[:, modality_indices] = data[:, modality_indices] - data[:, modality_indices].mean(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return data


def calculate_trial_metrics(trial_data):
    """
    Calculate metrics for a single trial.

    Args:
        trial_data: Tuple of (trial_name, trial_label, trial_pred)

    Returns:
        Tuple of (trial_name, (mse, mae, pearsonr_val))
    """
    trial_name, trial_label, trial_pred = trial_data

    mse = mean_squared_error(trial_label, trial_pred)
    mae = mean_absolute_error(trial_label, trial_pred)
    pearsonr_val = pearsonr(trial_label, trial_pred)[0]

    return trial_name, (mse, mae, pearsonr_val)


@click.command()
@click.option("-e", "--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"])
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--data_base", required=True, default="./Data")
@click.option("--use_mp", is_flag=True, default=False, help="Enable multiprocessing for metric calculation")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
@click.option("--transform", multiple=True, type=click.Choice(['none', 'log', 'quantile', 'log_quantile', 'log_quantile_substract_mean']),
              default=['none'], help="Data transformation(s) to apply before calculating correlation (can specify multiple)")
def main(exp_name, chk, splits, res_base, log_base, data_base, use_mp, n_processes, transform):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)
    DATA_BASE = os.path.abspath(f"{data_base}/")

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data", exist_ok=True)

    # Convert transform tuple to list
    transform_list = list(transform)
    print(f"Using transformations: {transform_list}")

    # load label meta info
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    # splits = ["Train", "Valid", "Test"]

    # calculate correlation and other metrics
    for split in splits:
        metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv"

        print(f"Calculate metric for {split}")

        # Load predictions once
        test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
        # Get original shapes for sequence-level correlation
        test_pred_3d = test_res["pred"]['regression'][:, :, label_meta['dim']]  # (n_samples, n_windows, n_trials)
        n_samples = test_pred_3d.shape[0]
        n_windows = test_pred_3d.shape[1]
        n_trials = test_pred_3d.shape[2]

        test_label_orig = test_res["label"]['regression'].reshape(-1, len(label_meta))
        test_pred_orig = test_pred_3d.reshape(-1, len(label_meta))

        # Initialize dataframe to store all metrics for all transforms
        all_metrics = pd.DataFrame(index=label_meta["trial"])

        # Calculate metrics for each transformation
        for trans in transform_list:
            print(f"  Processing transform: {trans}")

            # Apply transformation
            if trans != "none":
                test_label_np = test_label_orig.numpy()
                test_pred_np = test_pred_orig.numpy()

                test_label_np = apply_transform(test_label_np, trans, label_meta=label_meta)
                test_pred_np = apply_transform(test_pred_np, trans, label_meta=label_meta)

                test_label = torch.from_numpy(test_label_np)
                test_pred = torch.from_numpy(test_pred_np)
            else:
                test_label = test_label_orig
                test_pred = test_pred_orig

            # Create column names with transform suffix
            col_suffix = f"_{trans}" if trans != "none" else ""
            metric_cols = {
                "MSE": f"MSE{col_suffix}",
                "MAE": f"MAE{col_suffix}",
                "PearsonR": f"PearsonR{col_suffix}",
                "AvgSeqPearsonR": f"AvgSeqPearsonR{col_suffix}"
            }

            # Calculate sequence-level correlation metric
            # Reshape to (n_samples, n_windows, n_trials) for sequence-level correlation
            test_label_3d = test_label.reshape(n_samples, n_windows, n_trials)
            test_pred_3d = test_pred.reshape(n_samples, n_windows, n_trials)

            # Calculate correlation for each sequence and trial using vectorized operations
            # Convert to numpy if needed
            if isinstance(test_label_3d, torch.Tensor):
                label_np = test_label_3d.numpy()
            else:
                label_np = test_label_3d
            if isinstance(test_pred_3d, torch.Tensor):
                pred_np = test_pred_3d.numpy()
            else:
                pred_np = test_pred_3d

            # Center the data (subtract mean along window dimension)
            label_centered = label_np - label_np.mean(axis=1, keepdims=True)
            pred_centered = pred_np - pred_np.mean(axis=1, keepdims=True)

            # Compute Pearson correlation using vectorized operations
            numerator = (label_centered * pred_centered).sum(axis=1)  # (n_samples, n_trials)
            denominator = np.sqrt((label_centered ** 2).sum(axis=1) * (pred_centered ** 2).sum(axis=1))
            seq_corrs = numerator / denominator  # (n_samples, n_trials)

            # Find best sequence (averaged across all trials)
            avg_corr_per_seq = np.nanmean(seq_corrs, axis=1)  # Average across trials for each sequence
            best_seq_idx = test_res["index"].cpu().numpy()[np.argmax(avg_corr_per_seq)]
            best_seq_corr = avg_corr_per_seq[np.argmax(avg_corr_per_seq)]
            seq_info = pd.read_csv(f"{DATA_BASE}/{exp_name}/sequences.bed", sep="\t", header=None,
                                   names=["chrom", "start", "end", "split"])
            seq_info = seq_info[seq_info["split"] == split.lower()].reset_index(drop=True)
            best_seq = seq_info.iloc[best_seq_idx]
            seq_corrs_df = pd.DataFrame(seq_corrs, columns=label_meta["trial"])
            seq_corrs_df['chrom'] = seq_info['chrom'].iloc[test_res["index"].cpu().numpy()].values
            seq_corrs_df['start'] = seq_info['start'].iloc[test_res["index"].cpu().numpy()].values
            seq_corrs_df['end'] = seq_info['end'].iloc[test_res["index"].cpu().numpy()].values
            seq_corrs_df.to_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_avg_sequence_correlation{col_suffix}.csv", index=False)
            print(f"    Best sequence: index={best_seq_idx}, avg_correlation={best_seq_corr:.4f}")
            print(f"    Best sequence info: {best_seq['chrom']}:{best_seq['start']}-{best_seq['end']}")

            # Average across trials for each sequence, then average across sequences
            avg_seq_pearsonr_per_trial = np.mean(seq_corrs, axis=0)  # Average across sequences for each trial

            if use_mp:
                # Prepare data for multiprocessing
                trial_data_list = []
                for i in label_meta.index:
                    trial_name = label_meta.loc[i, "trial"]
                    trial_label = test_label[:, i]
                    trial_pred = test_pred[:, i]
                    trial_data_list.append((trial_name, trial_label, trial_pred))

                # Set number of processes
                n_proc = n_processes if n_processes else mp.cpu_count()
                print(f"    Using {n_proc} processes for parallel calculation")

                # Calculate metrics in parallel
                with mp.Pool(n_proc) as pool:
                    results = list(
                        tqdm(
                            pool.imap(calculate_trial_metrics, trial_data_list),
                            total=len(trial_data_list),
                            desc=f"    Calculating metrics ({trans})",
                        )
                    )

                # Store results
                for trial_name, (mse, mae, pearsonr_val) in results:
                    all_metrics.loc[trial_name, metric_cols["MSE"]] = mse
                    all_metrics.loc[trial_name, metric_cols["MAE"]] = mae
                    all_metrics.loc[trial_name, metric_cols["PearsonR"]] = pearsonr_val

                # Store sequence-level correlation for each trial
                for i, trial_name in enumerate(label_meta["trial"]):
                    all_metrics.loc[trial_name, metric_cols["AvgSeqPearsonR"]] = avg_seq_pearsonr_per_trial[i]
            else:
                # Original sequential approach
                for i in tqdm(label_meta.index, desc=f"    Calculating metrics ({trans})"):
                    trail_label = test_label[:, i]
                    trail_pred = test_pred[:, i]
                    trial_name = label_meta.loc[i, "trial"]

                    all_metrics.loc[trial_name, metric_cols["MSE"]] = mean_squared_error(trail_label, trail_pred)
                    all_metrics.loc[trial_name, metric_cols["MAE"]] = mean_absolute_error(trail_label, trail_pred)
                    all_metrics.loc[trial_name, metric_cols["PearsonR"]] = pearsonr(trail_label, trail_pred)[0]
                    all_metrics.loc[trial_name, metric_cols["AvgSeqPearsonR"]] = avg_seq_pearsonr_per_trial[i]

        # Save all metrics to single CSV file
        all_metrics.to_csv(metric_file)
        print(f"  Saved metrics to: {metric_file}")

    # plot (cell type level)
    for split in splits:
        metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv"
        metric = pd.read_csv(metric_file, index_col=0)
        metric["cell_type"] = metric.index.str.rsplit("_", n=1).str[0]
        metric["modality"] = metric.index.str.rsplit("_", n=1).str[-1]

        # Create plots for each transformation
        for trans in transform_list:
            print(f"Plot metric (cell type level) for {split} with transform: {trans}")

            # Select the appropriate PearsonR column
            pearsonr_col = "PearsonR" if trans == "none" else f"PearsonR_{trans}"

            if pearsonr_col not in metric.columns:
                print(f"  Warning: Column {pearsonr_col} not found, skipping")
                continue

            # Calculate median values and create x-positions as before
            median_values = (
                metric.groupby("cell_type")[pearsonr_col].median().sort_values(ascending=False)
            )
            celltype_to_x = {celltype: i for i, celltype in enumerate(median_values.index)}
            plot_df = metric.copy()
            plot_df["y_pos"] = plot_df["cell_type"].map(celltype_to_x)

            n_modalities = len(plot_df["modality"].unique())
            selected_colors = [base_cmap(i) for i in range(0, base_cmap.N, 2)][:n_modalities]

            # Create figure with adjusted dimensions
            plt.figure(figsize=(10, 10))

            # Create horizontal boxplot and stripplot
            sns.boxplot(
                y="y_pos", x=pearsonr_col, data=plot_df, color="white", width=0.6, orient="h"
            )

            stripplot = sns.stripplot(
                y="y_pos",
                x=pearsonr_col,
                hue="modality",
                data=plot_df,
                palette=selected_colors,
                size=6,
                edgecolor="w",
                linewidth=0.5,
                alpha=0.8,
                jitter=0.25,
                orient="h",
            )

            # Customize y-axis to show cell type names
            plt.yticks(range(len(median_values)), median_values.index)
            plt.ylabel(None)
            plt.xlabel("PearsonR")

            # add grid line
            plt.grid(axis="x", linestyle="--", alpha=0.6, color="gray")
            sns.despine(left=True, bottom=True)

            # Adjust legend position
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")

            title = f"Pearson Correlation by Cell Type ({split} Set)"
            if trans != "none":
                title += f" [{trans}]"
            plt.title(title)
            plt.tight_layout()

            # Create filename suffix
            transform_suffix = "" if trans == "none" else f"_{trans}"
            plt.savefig(
                f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_by_cell_type{transform_suffix}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # plot (trial level)
    for split in splits:
        metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv"
        metric = pd.read_csv(metric_file, index_col=0)
        metric["cell_type"] = metric.index.str.rsplit("_", n=0).str[0]
        metric["modality"] = metric.index.str.rsplit("_", n=0).str[-1]

        # Create plots for each transformation
        for trans in transform_list:
            print(f"Plot metric (modality level) for {split} with transform: {trans}")

            # Select the appropriate PearsonR column
            pearsonr_col = "PearsonR" if trans == "none" else f"PearsonR_{trans}"

            if pearsonr_col not in metric.columns:
                print(f"  Warning: Column {pearsonr_col} not found, skipping")
                continue

            # Calculate median values and create x-positions as before
            median_values = (
                metric.groupby("modality")[pearsonr_col].median().sort_values(ascending=False)
            )
            modality_to_x = {modality: i for i, modality in enumerate(median_values.index)}
            plot_df = metric.copy()
            plot_df["x_pos"] = plot_df["modality"].map(modality_to_x)

            # Create figure with adjusted dimensions
            plt.figure(figsize=(10, 10))

            # Create boxplot and stripplot
            sns.boxplot(x="x_pos", y=pearsonr_col, data=plot_df, color="white", width=0.6)

            stripplot = sns.stripplot(
                x="x_pos",
                y=pearsonr_col,
                hue="modality",
                data=plot_df,
                palette="tab20",
                size=6,
                edgecolor="w",
                linewidth=0.5,
                alpha=0.8,
                jitter=0.25,
            )

            # Customize x-axis to show modality names
            plt.xticks(range(len(median_values)), median_values.index)
            plt.xlabel(None)
            plt.ylabel("PearsonR")

            # add grid line
            plt.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
            sns.despine(left=True, bottom=True)

            # Adjust legend position
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")

            title = f"Pearson Correlation by Modality ({split} Set)"
            if trans != "none":
                title += f" [{trans}]"
            plt.title(title)
            plt.tight_layout()

            # Create filename suffix
            transform_suffix = "" if trans == "none" else f"_{trans}"
            plt.savefig(
                f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_by_modality{transform_suffix}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    main()
