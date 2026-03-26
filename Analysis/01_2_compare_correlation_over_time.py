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

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_gene_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_5_test_correlation_by_gene.py"),
)
_gene_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gene_module)
untransform_predictions = _gene_module.untransform_predictions
aggregate_genes_from_predictions = _gene_module.aggregate_genes_from_predictions
make_genes_exon = _gene_module.make_genes_exon


def compute_correlations_one_epoch(
    pred_file, label_meta, sequences_bed_path, genes_bed_file, pool_width, gene_scale="length"
):
    """
    Compute per-trial Pearson R for one epoch.

    For non-RNA modalities: bin-level correlation on untransformed predictions.
    For RNA modalities: gene-level aggregation then correlation.

    Returns:
        dict mapping trial_name -> PearsonR
    """
    test_res = torch.load(pred_file, map_location="cpu")

    # 3D predictions: (n_sequences, n_bins, n_trials)
    predictions = test_res["pred"]["regression"][:, :, label_meta["dim"]].cpu().numpy()
    targets = test_res["label"]["regression"].cpu().numpy()

    # Reorder by index
    index_order = np.argsort(test_res["index"])
    predictions = predictions[index_order]
    targets = targets[index_order]

    # Untransform to original scale
    predictions = untransform_predictions(predictions, label_meta=label_meta)
    targets = untransform_predictions(targets, label_meta=label_meta)

    trial_pearsonr = {}

    # --- Non-RNA modalities: bin-level correlation ---
    non_rna_mask = ~label_meta["modality"].isin(["RNAplus", "RNAminus"])
    for i in label_meta.index[non_rna_mask]:
        trial_name = label_meta.loc[i, "trial"]
        pred_flat = predictions[:, :, i].ravel()
        tgt_flat = targets[:, :, i].ravel()
        r = pearsonr(tgt_flat, pred_flat)[0]
        trial_pearsonr[trial_name] = r

    # --- RNA modalities: gene-level aggregation ---
    rna_mask = label_meta["modality"].isin(["RNAplus", "RNAminus"])
    if rna_mask.any():
        split = "test"  # used for filtering sequences.bed
        gene_targets, gene_preds, gene_ids, _, label_meta_rna = (
            aggregate_genes_from_predictions(
                predictions, targets, label_meta,
                sequences_bed_path, genes_bed_file, split,
                pool_width, gene_scale=gene_scale,
            )
        )
        # label_meta_rna has RNA modality unified; compute correlation per trial
        for ti in range(len(label_meta_rna)):
            trial_name = label_meta_rna.iloc[ti]["trial"]
            r = pearsonr(gene_targets[:, ti], gene_preds[:, ti])[0]
            # Only keep the RNA trials (skip non-RNA which are duplicated)
            if label_meta_rna.iloc[ti]["modality"] == "RNA":
                trial_pearsonr[trial_name] = r

    return trial_pearsonr


@click.command()
@click.option("-e", "--exp_names", required=True, multiple=True, type=str,
              help="Experiment names to compare (specify multiple with -e name1 -e name2)")
@click.option("--epoch_start", default=1, type=int, help="Start epoch (inclusive)")
@click.option("--epoch_end", default=20, type=int, help="End epoch (inclusive)")
@click.option("-s", "--split", type=str, default="Test")
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--genes_gtf", type=str, default="Data/source/gencode.v48.annotation.gtf.gz")
@click.option("--pool_width", type=int, default=32)
@click.option("--gene_scale", type=click.Choice(["length", "rpkm", "none"]), default="length")
@click.option("-o", "--output_dir", required=True, default="./Res/comparison")
def main(exp_names, epoch_start, epoch_end, split, res_base, log_base, genes_gtf, pool_width, gene_scale, output_dir):
    RES_BASE = os.path.abspath(res_base)
    LOG_BASE = os.path.abspath(log_base)

    os.makedirs(f"{output_dir}/plot", exist_ok=True)
    os.makedirs(f"{output_dir}/raw_data", exist_ok=True)

    epochs = list(range(epoch_start, epoch_end + 1))

    # Create gene BED file (shared across experiments)
    genes_bed_file = f"{output_dir}/genes.bed"
    if not os.path.exists(genes_bed_file):
        print("Creating gene BED file...")
        make_genes_exon(genes_bed_file, genes_gtf, output_dir)

    # Use first experiment to get sequences.bed path and label_meta
    first_config = OmegaConf.load(f"{LOG_BASE}/{exp_names[0]}/overall_setting.yaml")
    sequences_bed_path = f"{first_config.data.preprocess.storage_path}/sequences.bed"

    # Collect data per experiment
    all_exp_data = {}

    for exp_name in exp_names:
        print(f"\n{'='*60}")
        print(f"Processing experiment: {exp_name}")
        print(f"{'='*60}")

        label_meta = pd.read_csv(f"{LOG_BASE}/{exp_name}/regression_label_meta.csv", index_col=None)

        records = []
        for epoch in epochs:
            pred_file = f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{epoch}.pt"
            if not os.path.exists(pred_file):
                print(f"  Skipping epoch {epoch}: not found")
                continue

            print(f"  Epoch {epoch}...")
            trial_pearsonr = compute_correlations_one_epoch(
                pred_file, label_meta, sequences_bed_path, genes_bed_file, pool_width, gene_scale
            )

            for trial_name, r in trial_pearsonr.items():
                # Parse modality from trial name
                modality = trial_name.rsplit("_", 1)[-1]
                records.append({
                    "epoch": epoch,
                    "trial": trial_name,
                    "modality": modality,
                    "PearsonR": r,
                })

        df = pd.DataFrame(records)
        # Average across cell types within each modality per epoch
        modality_avg = df.groupby(["epoch", "modality"])["PearsonR"].mean().reset_index()
        all_exp_data[exp_name] = modality_avg

        # Save raw data
        modality_avg.to_csv(
            f"{output_dir}/raw_data/{exp_name}_{split}_modality_avg_pearsonr.csv",
            index=False,
        )
        # Also save per-trial data
        df.to_csv(
            f"{output_dir}/raw_data/{exp_name}_{split}_per_trial_pearsonr.csv",
            index=False,
        )

    # Get all modalities (sorted)
    all_modalities = sorted(
        set().union(*(d["modality"].unique() for d in all_exp_data.values()))
    )
    n_modalities = len(all_modalities)

    # Create subplots: one per modality
    n_cols = 3
    n_rows = (n_modalities + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    colors = plt.cm.tab10.colors

    for idx, modality in enumerate(all_modalities):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for exp_idx, exp_name in enumerate(exp_names):
            df = all_exp_data[exp_name]
            mod_data = df[df["modality"] == modality].sort_values("epoch")
            if mod_data.empty:
                continue
            ax.plot(
                mod_data["epoch"], mod_data["PearsonR"],
                marker="o", markersize=4, linewidth=1.5,
                color=colors[exp_idx % len(colors)],
                label=exp_name,
            )

        ax.set_title(modality, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Pearson R")
        ax.grid(axis="both", linestyle="--", alpha=0.5, color="gray")
        ax.set_xticks(epochs)

    # Hide unused subplots
    for idx in range(n_modalities, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Single legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"Avg Pearson R Over Epochs by Modality ({split})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save
    exp_tag = "__vs__".join(exp_names)
    out_path = f"{output_dir}/plot/{split}_compare_modality_{exp_tag}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
