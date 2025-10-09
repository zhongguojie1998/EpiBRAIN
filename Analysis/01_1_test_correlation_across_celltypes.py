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
            test_label = test_res["label"]['regression'].reshape(-1, test_res["label"]['regression'].shape[-1])
            test_pred = test_res["pred"]['regression'].reshape(-1, test_res["pred"]['regression'].shape[-1])

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
            with open(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.pkl", "wb") as f:
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

            metric.to_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.csv")

    # plot (modality level)
    for split in splits:
        print(f"Plot metric (modality level) for {split}")

        with open(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes.pkl", "rb") as f:
            metric_dict = pickle.load(f)
        
        metric["modality"] = metric.index.str.rsplit("_", n=1).str[-1]

        # prepare data for plotting
        plot_df = pd.DataFrame(columns=["modality", "PearsonR"])
        for mod in metric.index:
            pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
            pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
            temp_df = pd.DataFrame({"modality": [mod]*len(pearsonr_vals), "PearsonR": pearsonr_vals})
            plot_df = pd.concat([plot_df, temp_df], ignore_index=True)

        # Create figure with adjusted dimensions
        fig, ax = plt.subplots(figsize=(10, 20))

        # Create violin plot
        sns.violinplot(
            y="PearsonR", x="modality", data=plot_df, palette="tab20", inner="quartile", cut=0, ax=ax
        )

        # add grid line
        ax.grid(axis="x", linestyle="--", alpha=0.6, color="gray")
        sns.despine(left=True, bottom=True)

        # Adjust legend position (optional - may not be needed with y-axis labels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")
        ax.set_title(f"Pearson Correlation ({split} Set)")
        fig.tight_layout()
        fig.savefig(
            f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_across_cell_types.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

if __name__ == "__main__":
    main()
