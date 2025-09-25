import os
import warnings

warnings.filterwarnings("ignore")

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed


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


def process_celltype(args):
    label_path, celltype, sequences_data, outdir = args
    print(f"Preparing dataframe and plotting for {celltype}")

    # Read data from h5py file
    h5_path = f'{label_path}/regression_{celltype}.h5'
    with h5py.File(h5_path, 'r') as f:
        data = f['targets'][:]

    df = pd.DataFrame(data.flatten(), columns=["value"])
    # Add chromosome information to the DataFrame, should be sequences['chrom'] repeated for each row in df
    df["chrom"] = sequences_data["chrom"].repeat(data.shape[1]).reset_index(drop=True)
    offset_start = np.arange(320 * 128, 320 * 128 + data.shape[1] * 128, 128)
    df["start"] = sequences_data["start"].repeat(data.shape[1]).values + np.tile(offset_start, data.shape[0])
    df["end"] = df["start"] + 128
    df["midpoint"] = (df["start"] + df["end"]) / 2

    # Directly plot using the plot_manhattan logic
    plot_manhattan((celltype, df), outdir)

    return celltype


def process_celltype_stats(args):
    label_path, celltype = args
    print(f"Processing statistics for {celltype}")

    # Read data from h5py file
    h5_path = f'{label_path}/regression_{celltype}.h5'
    with h5py.File(h5_path, 'r') as f:
        data = f['targets'][:]

    # Calculate statistics
    flattened_data = data.flatten()
    stats = {
        'celltype': celltype,
        'min': np.min(flattened_data),
        'max': np.max(flattened_data),
        'mean': np.mean(flattened_data),
        'std': np.std(flattened_data),
        '5%': np.percentile(flattened_data, 5),
        '10%': np.percentile(flattened_data, 10),
        '25%': np.percentile(flattened_data, 25),
        '50%': np.percentile(flattened_data, 50),
        '75%': np.percentile(flattened_data, 75),
        '90%': np.percentile(flattened_data, 90),
        '95%': np.percentile(flattened_data, 95),
        '99%': np.percentile(flattened_data, 99),
        '99.9%': np.percentile(flattened_data, 99.9),
        '99.99%': np.percentile(flattened_data, 99.99),
        '99.999%': np.percentile(flattened_data, 99.999),
    }

    return stats


def load_pt_file(args):
    file, label_path = args
    if file.endswith(".pt") and file.startswith("chr") and "transcript" not in file:
        return torch.load(os.path.join(label_path, file)), file.replace(".pt", "")
    return None, None


def prepare_dfs(label_path, dataset_path, output_dir, n_jobs=-1, celltype_patterns=None):
    sequences = pd.read_csv(dataset_path, sep="\t", header=None)
    sequences.columns = ["chrom", "start", "end", "split"]
    sequences.shape, sequences[sequences["split"] == "train"].shape, sequences[
        sequences["split"] == "valid"
    ].shape, sequences[sequences["split"] == "test"].shape

    # Get celltypes from files under label_path
    celltypes = []
    for filename in os.listdir(label_path):
        if filename.startswith("regression_") and filename.endswith(".h5"):
            celltype = filename.replace("regression_", "").replace(".h5", "")
            celltypes.append(celltype)

    # Filter cell types based on patterns if specified
    if celltype_patterns:
        filtered_celltypes = []
        for celltype in celltypes:
            if any(pattern in celltype for pattern in celltype_patterns):
                filtered_celltypes.append(celltype)
        celltypes = filtered_celltypes

    processed_celltypes = Parallel(n_jobs=n_jobs)(
        delayed(process_celltype)((label_path, celltype, sequences, output_dir))
        for celltype in celltypes
    )

    return processed_celltypes


def compute_stats(label_path, celltype_patterns=None, n_jobs=-1):
    # Get celltypes from files under label_path
    celltypes = []
    for filename in os.listdir(label_path):
        if filename.startswith("regression_") and filename.endswith(".h5"):
            celltype = filename.replace("regression_", "").replace(".h5", "")
            celltypes.append(celltype)

    # Filter cell types based on patterns if specified
    if celltype_patterns:
        filtered_celltypes = []
        for celltype in celltypes:
            if any(pattern in celltype for pattern in celltype_patterns):
                filtered_celltypes.append(celltype)
        celltypes = filtered_celltypes

    stats_list = Parallel(n_jobs=n_jobs)(
        delayed(process_celltype_stats)((label_path, celltype))
        for celltype in celltypes
    )

    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats_list)

    # Sort by celltype name
    stats_df = stats_df.sort_values('celltype').reset_index(drop=True)

    return stats_df


@click.command()
@click.option("-o", "--output_dir", default="./Data/other/data_quality/manhattan_plot")
@click.option("-e", "--exp_name", default="enformer_style_split_data_v1")
@click.option("--base_dir", default="./Data")
@click.option("--num_worker", default=48)
@click.option("--celltype_patterns", default=None, help="Comma-separated patterns to filter cell types (e.g., 'BasalGanglia-Astrocyte,MiniAtlas-AST')")
@click.option("--stats_only", is_flag=True, help="Only compute statistics, skip visualization")
def main(output_dir, exp_name, base_dir, num_worker, celltype_patterns, stats_only):

    os.makedirs(output_dir, exist_ok=True)

    # Parse celltype patterns if provided
    patterns = None
    if celltype_patterns:
        patterns = [p.strip() for p in celltype_patterns.split(',')]

    label_path = f"{base_dir}/{exp_name}/labels/"

    if stats_only:
        # Compute and display statistics only
        stats_df = compute_stats(
            label_path=label_path,
            celltype_patterns=patterns,
            n_jobs=num_worker
        )
        stats_output_path = os.path.join(output_dir, "stats.csv")
        stats_df.to_csv(stats_output_path, index=False)
        print(f"Statistics saved to: {stats_output_path}")
        print("\nData Statistics Summary:")
        print(stats_df.to_string(index=False))
    else:
        # load in data and generate plots in parallel
        processed_celltypes = prepare_dfs(
            label_path=label_path,
            dataset_path=f"{base_dir}/{exp_name}/sequences.bed",
            output_dir=output_dir,
            n_jobs=num_worker,
            celltype_patterns=patterns,
        )

        print(f"Completed processing and plotting for {len(processed_celltypes)} celltypes")


if __name__ == "__main__":
    main()
