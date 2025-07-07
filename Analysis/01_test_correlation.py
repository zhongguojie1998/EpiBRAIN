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
from tqdm import tqdm

base_cmap = plt.get_cmap("tab20")


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
@click.option("--data_base", required=True, default="./Data/enformer_style_split_data_v1")
@click.option("--use_mp", is_flag=True, default=False, help="Enable multiprocessing for metric calculation")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
def main(exp_name, chk, splits, res_base, data_base, use_mp, n_processes):
    DATA_BASE = os.path.abspath(data_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data", exist_ok=True)

    # load label meta info
    label_meta = pd.read_csv(f"{DATA_BASE}/label_meta.csv", index_col=0)

    # splits = ["Train", "Valid", "Test"]

    # calculate correlation and other metrics
    for split in splits:
        if not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv"):
            print(f"Calculate metric for {split}")

            test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
            test_label = test_res["label"].reshape(-1, len(label_meta))
            test_pred = test_res["pred"].reshape(-1, len(label_meta))

            metric = pd.DataFrame(index=label_meta["trial"], columns=["MSE", "MAE", "PearsonR"])

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
                print(f"Using {n_proc} processes for parallel calculation")

                # Calculate metrics in parallel
                with mp.Pool(n_proc) as pool:
                    results = list(
                        tqdm(
                            pool.imap(calculate_trial_metrics, trial_data_list),
                            total=len(trial_data_list),
                            desc="Calculating metrics",
                        )
                    )

                # Store results
                for trial_name, (mse, mae, pearsonr_val) in results:
                    metric.loc[trial_name] = (mse, mae, pearsonr_val)
            else:
                # Original sequential approach
                for i in tqdm(label_meta.index):
                    trail_label = test_label[:, i]
                    trail_pred = test_pred[:, i]

                    metric.loc[label_meta.loc[i, "trial"]] = (
                        mean_squared_error(trail_label, trail_pred),
                        mean_absolute_error(trail_label, trail_pred),
                        pearsonr(trail_label, trail_pred)[0],
                    )

            metric.to_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv")

    # plot (cell type level)
    for split in splits:
        print(f"Plot metric (cell type level) for {split}")

        metric = pd.read_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv", index_col=0)
        metric["cell_type"] = metric.index.str.rsplit("_", n=0).str[0]
        metric["modality"] = metric.index.str.rsplit("_", n=0).str[-1]

        # Calculate median values and create x-positions as before
        median_values = (
            metric.groupby("cell_type")["PearsonR"].median().sort_values(ascending=False)
        )  # Note: ascending=True for horizontal plot
        celltype_to_x = {celltype: i for i, celltype in enumerate(median_values.index)}
        plot_df = metric.copy()
        plot_df["y_pos"] = plot_df["cell_type"].map(celltype_to_x)  # Using y_pos instead of x_pos

        n_modalities = len(plot_df["modality"].unique())
        selected_colors = [base_cmap(i) for i in range(0, base_cmap.N, 2)][:n_modalities]

        # Create figure with adjusted dimensions
        plt.figure(figsize=(10, 10))  # Wider and taller to accommodate horizontal layout

        # Create horizontal boxplot and stripplot
        sns.boxplot(
            y="y_pos", x="PearsonR", data=plot_df, color="white", width=0.6, orient="h"
        )  # Horizontal orientation

        stripplot = sns.stripplot(
            y="y_pos",
            x="PearsonR",
            hue="modality",
            data=plot_df,
            palette=selected_colors,
            size=6,
            edgecolor="w",
            linewidth=0.5,
            alpha=0.8,
            jitter=0.25,
            orient="h",  # Horizontal orientation
        )

        # Customize y-axis to show cell type names
        plt.yticks(range(len(median_values)), median_values.index)
        plt.ylabel(None)
        plt.xlabel("PearsonR")

        # add grid line
        plt.grid(axis="x", linestyle="--", alpha=0.6, color="gray")
        sns.despine(left=True, bottom=True)

        # Adjust legend position (optional - may not be needed with y-axis labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")

        plt.title(f"Pearson Correlation by Cell Type ({split} Set)")
        plt.tight_layout()
        plt.savefig(
            f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_by_cell_type.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # plot (trial level)
    for split in splits:
        print(f"Plot metric (modality level) for {split}")

        metric = pd.read_csv(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric.csv", index_col=0)
        metric["cell_type"] = metric.index.str.rsplit("_", n=0).str[0]
        metric["modality"] = metric.index.str.rsplit("_", n=0).str[-1]

        # Calculate median values and create x-positions as before
        median_values = (
            metric.groupby("modality")["PearsonR"].median().sort_values(ascending=False)
        )  # Note: ascending=True for horizontal plot
        modality_to_x = {modality: i for i, modality in enumerate(median_values.index)}
        plot_df = metric.copy()
        plot_df["x_pos"] = plot_df["modality"].map(modality_to_x)

        # Create figure with adjusted dimensions
        plt.figure(figsize=(10, 10))  # Wider and taller to accommodate horizontal layout

        # Create horizontal boxplot and stripplot
        sns.boxplot(x="x_pos", y="PearsonR", data=plot_df, color="white", width=0.6)

        stripplot = sns.stripplot(
            x="x_pos",
            y="PearsonR",
            hue="modality",
            data=plot_df,
            palette="tab20",
            size=6,
            edgecolor="w",
            linewidth=0.5,
            alpha=0.8,
            jitter=0.25,
        )

        # Customize y-axis to show cell type names
        plt.xticks(range(len(median_values)), median_values.index)
        plt.xlabel(None)
        plt.ylabel("PearsonR")

        # add grid line
        plt.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
        sns.despine(left=True, bottom=True)

        # Adjust legend position (optional - may not be needed with y-axis labels)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, ncol=1, title="Modality")

        plt.title(f"Pearson Correlation by Modality ({split} Set)")
        plt.tight_layout()
        plt.savefig(
            f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/{split}_pearsonr_by_modality.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
