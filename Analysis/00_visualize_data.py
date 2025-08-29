import multiprocessing as mp
import os
import warnings
from functools import partial

warnings.filterwarnings("ignore")

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_manhattan(args, outdir):
    celltype, df = args
    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    chrom_to_int = {chrom: i + 1 for i, chrom in enumerate(chrom_order)}
    colors = ["#1f77b4", "#ff7f0e"]
    # copy so we don’t clobber the original
    df = df.copy()
    # numeric chromosome index
    df["chrom_idx"] = df["chrom"].str.replace("chr", "").map(lambda x: chrom_to_int.get(x, np.nan)).astype(int)
    df = df.dropna(subset=["chrom_idx"])

    # per-chr max midpoint & offsets
    max_mid = df.groupby("chrom_idx")["midpoint"].max().sort_index()
    offsets = max_mid.cumsum().shift(fill_value=0)

    # vectorized “cumulative midpoint”
    df["cum_midpoint"] = df["midpoint"] + df["chrom_idx"].map(offsets)

    # alternating colors
    df["color"] = df["chrom_idx"].map(lambda idx: colors[idx % 2])

    # plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(df["cum_midpoint"], df["value"], c=df["color"], s=2, edgecolor="none", alpha=0.6)

    # xticks at chromosome centers
    chrom_centers = (offsets + max_mid / 2).to_dict()
    ticks = list(chrom_centers.values())
    labels = [chrom_order[idx - 1] for idx in chrom_centers.keys()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)

    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Value")
    ax.set_title(f"Manhattan Plot — {celltype}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    # ensure output dir exists
    fig.savefig(f"{outdir}/{celltype}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_dfs(label_path, dataset_path, metadata_path):
    # if label_path is a directory, load all '.pt' files in it
    if os.path.isdir(label_path):
        all = []
        info = []
        for file in os.listdir(label_path):
            if file.endswith(".pt"):
                all.append(torch.load(os.path.join(label_path, file)))
                info.append(file.replace(".pt", ""))
        all = torch.cat([i['label']['regression'].unsqueeze(0) for i in all], dim=0)
    else:
        all = torch.load(label_path)
        info = None
    sequences = pd.read_csv(dataset_path, sep="\t", header=None)
    sequences.columns = ["chrom", "start", "end", "split"]
    # reorder all by sequences
    if info is not None:
        sequences.index = sequences['chrom'] + "_" + sequences['start'].astype(str) + "_" + sequences['end'].astype(str)
        sequences = sequences.loc[info]
    sequences.shape, sequences[sequences["split"] == "train"].shape, sequences[
        sequences["split"] == "valid"
    ].shape, sequences[sequences["split"] == "test"].shape
    metadata = pd.read_csv(metadata_path)
    dfs = {}
    for i, celltype in enumerate(metadata["trial"]):
        print(f"Preparing dataframe for {celltype}")
        df = pd.DataFrame(all[:, :, i].flatten().numpy(), columns=["value"])
        # Add chromosome information to the DataFrame, should be sequences['chrom'] repeated for each row in df
        df["chrom"] = sequences["chrom"].repeat(all.shape[1]).reset_index(drop=True)
        offset_start = np.arange(320 * 128, 320 * 128 + all.shape[1] * 128, 128)
        df["start"] = sequences["start"].repeat(all.shape[1]).values + np.tile(offset_start, all.shape[0])
        df["end"] = df["start"] + 128
        df["midpoint"] = (df["start"] + df["end"]) / 2
        dfs[celltype] = df
    return dfs


@click.command()
@click.option("-o", "--output_dir", default="./Data/other/data_quality/manhattan_plot")
@click.option("-e", "--exp_name", default="enformer_style_split_data_v1")
@click.option("--base_dir", default="./Data")
@click.option("--num_worker", default=48)
def main(output_dir, exp_name, base_dir, num_worker):

    os.makedirs(output_dir, exist_ok=True)

    # load in data
    dfs = prepare_dfs(
        label_path=f"{base_dir}/{exp_name}/data/all_label.pt",
        dataset_path=f"{base_dir}/{exp_name}/sequences.bed",
        metadata_path=f"{base_dir}/{exp_name}/label_meta.csv",
    )

    plot_func = partial(plot_manhattan, outdir=output_dir)

    print("Start ploting")
    with mp.Pool(processes=num_worker) as pool:
        pool.map(plot_func, dfs.items())


if __name__ == "__main__":
    main()
