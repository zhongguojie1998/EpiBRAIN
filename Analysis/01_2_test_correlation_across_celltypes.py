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
from sklearn.preprocessing import quantile_transform

base_cmap = plt.get_cmap("tab20")


def apply_transform(data, transform_type="none"):
    """
    Apply transformation to data.

    Args:
        data: numpy array of shape (n_samples, n_features)
        transform_type: str, one of "none", "log", "log_quantile"

    Returns:
        Transformed data
    """
    if transform_type == "none":
        return data

    data = data.copy()

    if transform_type == "log":
        # Apply log1p transformation (log(1 + x))
        data = np.log1p(np.maximum(data, 0))

    elif transform_type == "log_quantile":
        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return data


def calculate_modality_metrics_batch(mod_name, label_data, pred_data):
    """
    Calculate Pearson correlation for all samples of one modality using vectorized operations
    label_data, pred_data: [n_samples, n_celltypes] arrays
    """
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
        pearsonr_vals = numerator / denominator
        pearsonr_vals = np.where(np.isfinite(pearsonr_vals), pearsonr_vals, np.nan)

    # Calculate label statistics
    label_vars = label_data.var(axis=1)
    label_means = label_data.mean(axis=1)

    return mod_name, (pearsonr_vals, label_vars, label_means)

# %%
@click.command()
@click.option("-e", "--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"])
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
@click.option("--transform", multiple=True, type=click.Choice(['none', 'log', 'log_quantile']),
              default=['none'], help="Data transformation(s) to apply before calculating correlation (can specify multiple)")
def main(exp_name, chk, splits, res_base, log_base, n_processes, transform):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

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
        for trans in transform_list:
            metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.csv"
            if os.path.exists(metric_file):
                print(f"Skipping {split} with transform {trans}: file already exists")
                continue
            print(f"Calculate metric for {split}")

            test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
            # Convert to numpy arrays to avoid deprecation warnings
            test_label_orig = test_res["label"]['regression'].reshape(-1, test_res["label"]['regression'].shape[-1]).cpu().numpy()
            test_pred_orig = test_res["pred"]['regression'].reshape(-1, test_res["pred"]['regression'].shape[-1]).cpu().numpy()

            # Apply transformation
            print(f"  Applying transform: {trans}")
            test_label = apply_transform(test_label_orig, trans)
            test_pred = apply_transform(test_pred_orig, trans)

            metric = pd.DataFrame(index=label_meta["modality"].unique(),
                                  columns=["PearsonR:mean", "PearsonR:std", "PearsonR:median", "PearsonR:25%", "PearsonR:75%"])

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
            metric_dict = {mod: {"pearsonr": [], "label_mean": [], "label_var": []} for mod in metric.index}
            for mod_name, (pearsonr_vals, label_vars, label_means) in results:
                metric_dict[mod_name]["pearsonr"] = pearsonr_vals.tolist()
                metric_dict[mod_name]["label_mean"] = label_means.tolist()
                metric_dict[mod_name]["label_var"] = label_vars.tolist()

            pkl_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(metric_dict, f)

            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
                # remove nan values
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                if pearsonr_vals.size > 0:
                    metric.loc[mod, "PearsonR:mean"] = np.mean(pearsonr_vals)
                    metric.loc[mod, "PearsonR:std"] = np.std(pearsonr_vals)
                    metric.loc[mod, "PearsonR:median"] = np.median(pearsonr_vals)
                    metric.loc[mod, "PearsonR:25%"] = np.percentile(pearsonr_vals, 25)
                    metric.loc[mod, "PearsonR:75%"] = np.percentile(pearsonr_vals, 75)

            metric.to_csv(metric_file)

    # %% plot (modality level)
    for split in splits:
        for trans in transform_list:
            print(f"Plot metric (modality level) for {split} with transform {trans}")

            pkl_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.pkl"
            if not os.path.exists(pkl_file):
                print(f"Skipping plotting for {split} {trans}: file not found")
                continue

            with open(pkl_file, "rb") as f:
                metric_dict = pickle.load(f)

            metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.csv"
            metric = pd.read_csv(metric_file, index_col=0)

            # prepare data for plotting
            plot_df = pd.DataFrame(columns=["modality", "PearsonR"])
            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
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
            transform_label = "Raw" if trans == "none" else trans.replace("_", " ").title()
            ax.set_title(f"Pearson Correlation - {transform_label} ({split} Set)")
            fig.tight_layout()
            fig.savefig(
                f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_{trans}_across_cell_types.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            # %% plot correlation between mean/var and pearsonr
            for stat in ["mean", "var"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, mod in enumerate(metric.index):
                    pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
                    pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                    if len(pearsonr_vals) == 0:
                        continue
                    # Use label stats
                    label_stats = np.array(metric_dict[mod][f"label_{stat}"])
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
                transform_label = "Raw" if trans == "none" else trans.replace("_", " ").title()
                ax.set_xlabel(f"Label {stat.capitalize()} ({transform_label})")
                ax.set_ylabel(f"PearsonR ({transform_label})")
                ax.set_title(f"PearsonR ({transform_label}) vs Label {stat.capitalize()} ({split} Set)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")
                fig.tight_layout()
                fig.savefig(
                    f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_{trans}_vs_label_{stat}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

            # %% plot by mean-var and color by pearsonr (separate plots for each modality)
            for i, mod in enumerate(metric.index):
                pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                if len(pearsonr_vals) == 0:
                    continue
                label_means = np.array(metric_dict[mod]["label_mean"])
                label_means = label_means[~np.isnan(label_means)]
                if len(label_means) == 0:
                    continue
                label_vars = np.array(metric_dict[mod]["label_var"])
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
                transform_label = "Raw" if trans == "none" else trans.replace("_", " ").title()
                ax.set_xlabel(f"Label Mean ({transform_label})")
                ax.set_ylabel(f"Label Variance ({transform_label})")
                ax.set_title(f"Label Mean-Variance Colored by PearsonR ({transform_label})\n{mod} ({split} Set)")
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("PearsonR")
                fig.tight_layout()

                # Create safe filename
                safe_mod = mod.replace('/', '_').replace(' ', '_')
                fig.savefig(
                    f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_{safe_mod}_label_mean_var_colored_by_pearsonr_{trans}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

# %%
if __name__ == "__main__":
    main()
