# %%
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

base_cmap = plt.get_cmap("tab20")


def calculate_modality_metrics_batch(mod_name, label_data, pred_data):
    """
    Calculate Pearson correlation for all samples of one modality using vectorized operations
    Calculates both raw scale and log scale correlations
    label_data, pred_data: [n_samples, n_celltypes] arrays
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

# %%
@click.command()
@click.option("-e", "--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"])
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
def main(exp_name, chk, splits, res_base, log_base, n_processes):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data", exist_ok=True)

    # load label meta info
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    # splits = ["Train", "Valid", "Test"]

    # calculate correlation and other metrics
    for split in splits:
        if not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.csv"):
            print(f"Calculate metric for {split}")

            test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
            # Convert to numpy arrays to avoid deprecation warnings
            test_label = test_res["label"]['regression'].reshape(-1, test_res["label"]['regression'].shape[-1]).cpu().numpy()
            test_pred = test_res["pred"]['regression'].reshape(-1, test_res["pred"]['regression'].shape[-1]).cpu().numpy()

            metric = pd.DataFrame(index=label_meta["modality"].unique(),
                                  columns=["PearsonR_raw:mean", "PearsonR_raw:std", "PearsonR_raw:median", "PearsonR_raw:25%", "PearsonR_raw:75%",
                                           "PearsonR_log:mean", "PearsonR_log:std", "PearsonR_log:median", "PearsonR_log:25%", "PearsonR_log:75%"])

            # Precompute modality indices and prepare batched data
            modality_data = {}
            for mod in metric.index:
                mod_celltypes = label_meta[label_meta["modality"] == mod]
                label_indices = mod_celltypes.index.values
                pred_indices = mod_celltypes['dim'].values

                # Prepare batched data for this modality (all samples at once)
                modality_data[mod] = (
                    test_label[:, label_indices],
                    test_pred[:, pred_indices]
                )

            # Set number of processes
            n_proc = n_processes if n_processes else cpu_count()
            print(f"Using {n_proc} processes for parallel calculation")
            print(f"Processing {len(metric.index)} modalities with {test_label.shape[0]} samples each")

            # Calculate metrics in parallel, one task per modality (batched)
            results = Parallel(n_jobs=n_proc, backend='loky')(
                delayed(calculate_modality_metrics_batch)(
                    mod,
                    modality_data[mod][0],
                    modality_data[mod][1]
                )
                for mod in tqdm(metric.index, desc="Calculating metrics")
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
            with open(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.pkl", "wb") as f:
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

            metric.to_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.csv")

    # %% plot (modality level)
    for split in splits:
        print(f"Plot metric (modality level) for {split}")

        with open(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.pkl", "rb") as f:
            metric_dict = pickle.load(f)
            
        metric = pd.read_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.csv", index_col=0)

        # Plot for both raw and log scale
        for scale in ["raw", "log"]:
            # prepare data for plotting
            plot_df = pd.DataFrame(columns=["modality", "PearsonR"])
            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod][f"pearsonr_{scale}"])
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                temp_df = pd.DataFrame({"modality": [mod]*len(pearsonr_vals), "PearsonR": pearsonr_vals})
                plot_df = pd.concat([plot_df, temp_df], ignore_index=True)

            # Create figure with adjusted dimensions
            fig, ax = plt.subplots(figsize=(6, 6))

            # Create violin plot
            sns.violinplot(
                y="PearsonR", x="modality", data=plot_df, palette="tab20", inner="quartile", cut=0, ax=ax
            )

            # add grid line
            ax.grid(axis="x", linestyle="--", alpha=0.6, color="gray")
            sns.despine(left=True, bottom=True)

            # Adjust legend position (optional - may not be needed with y-axis labels)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")
            ax.set_title(f"Pearson Correlation - {scale.capitalize()} Scale ({split} Set)")
            fig.tight_layout()
            fig.savefig(
                f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_{scale}_across_cell_types.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        # %% plot correlation between mean/var and pearsonr
        for stat in ["mean", "var"]:
            for scale in ["raw", "log"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, mod in enumerate(metric.index):
                    pearsonr_vals = np.array(metric_dict[mod][f"pearsonr_{scale}"])
                    pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                    if len(pearsonr_vals) == 0:
                        continue
                    # Use label stats that match the scale
                    label_stats = np.array(metric_dict[mod][f"label_{stat}_{scale}"])
                    label_stats = label_stats[~np.isnan(label_stats)]
                    if len(label_stats) == 0:
                        continue
                    # Match lengths by truncating to the shorter array
                    min_len = min(len(pearsonr_vals), len(label_stats))
                    pearsonr_vals = pearsonr_vals[:min_len]
                    label_stats = label_stats[:min_len]

                    ax.scatter(
                        label_stats,
                        pearsonr_vals,
                        color=base_cmap(i),
                        alpha=0.6,
                        label=mod,
                        s=10,
                    )
                ax.set_xlabel(f"Label {stat.capitalize()} ({scale.capitalize()} Scale)")
                ax.set_ylabel(f"PearsonR ({scale.capitalize()} Scale)")
                ax.set_title(f"PearsonR ({scale.capitalize()}) vs Label {stat.capitalize()} ({split} Set)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")
                fig.tight_layout()
                fig.savefig(
                    f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_{scale}_vs_label_{stat}_{scale}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
        # %% plot by mean-var and color by pearsonr (separate plots for each modality)
        for scale in ["raw", "log"]:
            for i, mod in enumerate(metric.index):
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
                ax.set_title(f"Label Mean-Variance Colored by PearsonR ({scale.capitalize()})\n{mod} ({split} Set)")
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("PearsonR")
                fig.tight_layout()

                # Create safe filename
                safe_mod = mod.replace('/', '_').replace(' ', '_')
                fig.savefig(
                    f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_{safe_mod}_label_mean_var_colored_by_pearsonr_{scale}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

# %%
if __name__ == "__main__":
    main()
