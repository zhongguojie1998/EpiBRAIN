import os
import warnings

warnings.filterwarnings("ignore")

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from omegaconf import OmegaConf
from tqdm import tqdm

from Analysis.01_1_test_correlation import apply_transform


@click.command()
@click.option("-e", "--exp_name", required=True, type=str)
@click.option("--epoch_start", default=1, type=int, help="Start epoch (inclusive)")
@click.option("--epoch_end", default=20, type=int, help="End epoch (inclusive)")
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"])
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--data_base", required=True, default="./Data")
@click.option("-t", "--transform", multiple=True,
              type=click.Choice(['none', 'log', 'quantile', 'log_quantile', 'log_quantile_substract_mean']),
              default=['none'], help="Data transformation(s) to apply before calculating correlation")
def main(exp_name, epoch_start, epoch_end, splits, res_base, log_base, data_base, transform):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_over_time/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_over_time/raw_data", exist_ok=True)

    transform_list = list(transform)
    print(f"Using transformations: {transform_list}")

    # Load config and label meta
    config = OmegaConf.load(f"{LOG_BASE}/overall_setting.yaml")
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    epochs = list(range(epoch_start, epoch_end + 1))

    for split in splits:
        for trans in transform_list:
            col_suffix = f"_{trans}" if trans != "none" else ""
            print(f"Processing split={split}, transform={trans}")

            # Collect per-epoch results: {epoch: {celltype: avg_pearsonr}}
            records = []

            for epoch in tqdm(epochs, desc=f"  Epochs ({split}, {trans})"):
                pred_file = f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{epoch}.pt"
                if not os.path.exists(pred_file):
                    print(f"    Skipping epoch {epoch}: {pred_file} not found")
                    continue

                test_res = torch.load(pred_file, map_location="cpu")
                test_pred_3d = test_res["pred"]['regression'][:, :, label_meta['dim']]
                test_label_orig = test_res["label"]['regression'].reshape(-1, len(label_meta))
                test_pred_orig = test_pred_3d.reshape(-1, len(label_meta))

                # Apply transformation
                if trans != "none":
                    test_label_np = apply_transform(test_label_orig.numpy(), trans, label_meta=label_meta)
                    test_pred_np = apply_transform(test_pred_orig.numpy(), trans, label_meta=label_meta)
                    test_label = torch.from_numpy(test_label_np)
                    test_pred = torch.from_numpy(test_pred_np)
                else:
                    test_label = test_label_orig
                    test_pred = test_pred_orig

                # Compute pearson correlation per trial
                trial_pearsonr = {}
                for i in label_meta.index:
                    trial_name = label_meta.loc[i, "trial"]
                    r = pearsonr(test_label[:, i], test_pred[:, i])[0]
                    trial_pearsonr[trial_name] = r

                # Aggregate by cell type (average across modalities within each cell type)
                trial_df = pd.DataFrame({
                    "trial": list(trial_pearsonr.keys()),
                    "PearsonR": list(trial_pearsonr.values()),
                })
                trial_df["cell_type"] = trial_df["trial"].str.rsplit("_", n=1).str[0]
                celltype_avg = trial_df.groupby("cell_type")["PearsonR"].mean()

                for ct, avg_r in celltype_avg.items():
                    records.append({"epoch": epoch, "cell_type": ct, "PearsonR": avg_r})

            if not records:
                print(f"  No data found for split={split}, transform={trans}")
                continue

            df = pd.DataFrame(records)

            # Save per-celltype dataframe
            df_pivot = df.pivot(index="epoch", columns="cell_type", values="PearsonR")
            df_pivot.to_csv(
                f"{RES_BASE}/{exp_name}/analysis_over_time/raw_data/"
                f"{split}_pearsonr_over_time{col_suffix}.csv"
            )

            # Compute overall average across cell types per epoch
            avg_over_celltypes = df.groupby("epoch")["PearsonR"].mean().reset_index()
            avg_over_celltypes.columns = ["epoch", "avg_PearsonR"]
            avg_over_celltypes.to_csv(
                f"{RES_BASE}/{exp_name}/analysis_over_time/raw_data/"
                f"{split}_avg_pearsonr_over_time{col_suffix}.csv",
                index=False,
            )

            # Plot 1: Average across all cell types
            plt.figure(figsize=(8, 5))
            plt.plot(avg_over_celltypes["epoch"], avg_over_celltypes["avg_PearsonR"],
                     marker="o", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Average Pearson R")
            plt.xticks(avg_over_celltypes["epoch"])
            plt.grid(axis="both", linestyle="--", alpha=0.6, color="gray")
            title = f"Average Pearson R Over Epochs ({split})"
            if trans != "none":
                title += f" [{trans}]"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(
                f"{RES_BASE}/{exp_name}/analysis_over_time/plot/"
                f"{split}_avg_pearsonr_over_time{col_suffix}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

            # Plot 2: Per cell type lines
            plt.figure(figsize=(12, 7))
            for ct in sorted(df_pivot.columns):
                plt.plot(df_pivot.index, df_pivot[ct], marker=".", alpha=0.7, label=ct)
            plt.xlabel("Epoch")
            plt.ylabel("Pearson R (avg over modalities)")
            plt.xticks(df_pivot.index)
            plt.grid(axis="both", linestyle="--", alpha=0.6, color="gray")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0,
                       ncol=1, fontsize="small", title="Cell Type")
            title = f"Pearson R by Cell Type Over Epochs ({split})"
            if trans != "none":
                title += f" [{trans}]"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(
                f"{RES_BASE}/{exp_name}/analysis_over_time/plot/"
                f"{split}_pearsonr_by_celltype_over_time{col_suffix}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

            print(f"  Saved results for split={split}, transform={trans}")


if __name__ == "__main__":
    main()
