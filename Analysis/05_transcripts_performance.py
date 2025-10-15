#!/usr/bin/env python
"""
Transcript-level RNA expression performance analysis.

This script evaluates model predictions at the gene/transcript level by:
1. Loading model predictions for RNA tracks
2. Extracting gene/transcript coordinates from GTF file
3. Computing average expression for each gene/transcript
4. Calculating correlation metrics

Modes:
- use_exon_only = False: Uses full transcript ranges (including introns), one entry per transcript
- use_exon_only = True: Aggregates by gene (combining all isoforms) and uses only exon bins,
                        excluding introns. This gives more biologically meaningful gene expression.

Transform options:
- none: No transformation (applies log1p during correlation calculation)
- log: Apply log1p transformation before correlation
- quantile: Apply quantile normalization before correlation
- log_quantile: Apply log1p + quantile normalization before correlation
- log_quantile_substract_mean: Apply log1p + quantile normalization, then subtract mean across tracks for each gene

Usage:
    python 05_transcripts_performance.py -e exp_name --chk 150 -s Test --res_base ./Res --log_base ./logs --use_exon_only --transform log
    python 05_transcripts_performance.py -e exp_name --chk 150 -s Test --res_base ./Res --log_base ./logs --use_exon_only --filter_protein_coding
"""

import sys
import os
from pathlib import Path
import click

# Add Model directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))

from utils.get_transcripts import get_transcripts_in_region
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import quantile_transform
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def apply_transform(data, transform_type="none"):
    """
    Apply transformation to data.

    Args:
        data: numpy array of shape (n_samples, n_features)
        transform_type: str, one of "none", "log", "quantile", "log_quantile", "log_quantile_substract_mean"

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
        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))
        # For each gene (row), subtract the mean across tracks (columns)
        data = data - data.mean(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return data


def process_sequence(args):
    """Process a single sequence - extract transcripts and compute predictions."""
    i, row, test_preds, test_labels, pred_rna_idxes, label_rna_idxes, use_exon_only, filter_protein_coding = args
    chr = row['chr']
    start = row['start'] + 1023
    end = row['end'] - 1024

    # Get gene/transcript info
    # When use_exon_only=True, returns gene-aggregated data with exon bins
    # When use_exon_only=False, returns transcript-level data
    df = get_transcripts_in_region(f"{chr}:{start}-{end}", return_exon_bins_only=use_exon_only,
                                   filter_protein_coding=filter_protein_coding)

    # Get predictions for this sequence (i-th test sequence)
    pred = test_preds[i]  # Shape: (16320, num_tracks)
    label = test_labels[i]  # Shape: (16320, num_tracks)

    # Process each gene/transcript in this sequence
    seq_predictions = []
    seq_labels = []

    for _, r in df.iterrows():
        if use_exon_only:
            # Use only exon bins (already aggregated by gene)
            if 'exon_bins' in r and len(r['exon_bins']) > 0:
                exon_bins = r['exon_bins']
                gene_pred = pred[exon_bins, :][:, pred_rna_idxes]
                gene_label = label[exon_bins, :][:, label_rna_idxes]
            else:
                # Skip genes with no exon bins
                continue
        else:
            # Use all bins from start to end (original behavior for transcripts)
            gene_start_bin = r['start_bin_idx']
            gene_end_bin = r['end_bin_idx']
            gene_pred = pred[gene_start_bin:gene_end_bin+1, pred_rna_idxes]
            gene_label = label[gene_start_bin:gene_end_bin+1, label_rna_idxes]

        # average over bins, transform back -1 + np.sqrt(1 + seq_cov)
        gene_pred_mean = ((gene_pred + 1)**2-1).mean(axis=0, keepdims=True)
        gene_label_mean = ((gene_label + 1)**2-1).mean(axis=0, keepdims=True)
        seq_predictions.append(gene_pred_mean)
        seq_labels.append(gene_label_mean)

    # also output raw values of predictions and labels
    return df, seq_predictions, seq_labels, \
        (pred[:, pred_rna_idxes]+1)**2-1, (label[:, label_rna_idxes]+1)**2-1

@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint number")
@click.option("-s", "--split", type=str, default="Test", help="Data split (Train/Valid/Test)")
@click.option("--res_base", type=str, default="./Res", help="Results base directory")
@click.option("--log_base", type=str, default="./logs", help="Logs base directory")
@click.option("--data_path", type=str, default=None, help="Path to data directory (default: Data/{exp_name})")
@click.option("--use_exon_only", is_flag=True, default=False,
              help="Use only exon bins for gene expression (aggregates by gene)")
@click.option("--filter_protein_coding", is_flag=True, default=False,
              help="Only include protein_coding genes")
@click.option("--n_jobs", type=int, default=36, help="Number of parallel jobs (default: 36)")
@click.option("--gtf_file", type=str, default="Data/source/gencode.v48.annotation.gtf.gz",
              help="Path to GTF annotation file")
@click.option("--transform", type=click.Choice(['none', 'log', 'quantile', 'log_quantile', 'log_quantile_substract_mean']),
              default='none', help="Data transformation to apply before calculating correlation")
def main(exp_name, chk, split, res_base, log_base, data_path, use_exon_only, filter_protein_coding, n_jobs, gtf_file, transform):
    """
    Analyze transcript/gene-level RNA expression performance.
    """
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    # Determine data path
    if data_path is None:
        data_path = f"Data/{exp_name}"

    # Create output directories
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data", exist_ok=True)

    # Determine suffix for output files
    suffix = "_exon" if use_exon_only else ""
    if filter_protein_coding:
        suffix += "_protein_coding"
    if transform != "none":
        suffix += f"_{transform}"
    mode_str = "exon-only" if use_exon_only else "full transcript"
    if filter_protein_coding:
        mode_str += " (protein_coding only)"
    transform_str = f" with {transform} transform" if transform != "none" else ""

    print(f"="*60)
    print(f"Experiment: {exp_name}")
    print(f"Checkpoint: {chk}")
    print(f"Split: {split}")
    print(f"Mode: {mode_str}{transform_str}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"="*60)

    # Load predictions
    print("\nLoading predictions...")
    pred_file = f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt"
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    preds_data = torch.load(pred_file, map_location='cpu', weights_only=False)
    test_preds = preds_data['pred']['regression'].numpy()
    test_labels = preds_data['label']['regression'].numpy()
    print(f"Predictions shape: {test_preds.shape}")
    print(f"Labels shape: {test_labels.shape}")

    # Load sequence info
    sequences_bed = f"{data_path}/sequences.bed"
    if not os.path.exists(sequences_bed):
        raise FileNotFoundError(f"Sequences bed file not found: {sequences_bed}")

    info = pd.read_csv(sequences_bed, sep='\t', header=None)
    info.columns = ['chr', 'start', 'end', 'split']
    info = info[info['split'] == split.lower()].reset_index(drop=True)
    print(f"{split} sequences: {len(info)}")

    # Get track info
    pred_track_info = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv")
    label_track_info = pd.read_csv(f"{data_path}/raw_label_meta.csv")
    pred_rna_idxes = pred_track_info[pred_track_info['modality'].str.contains('RNA')]['dim'].values
    label_rna_idxes = label_track_info[label_track_info['modality'].str.contains('RNA')].index.values
    pred_rna_track_info = pred_track_info[pred_track_info['modality'].str.contains('RNA')].reset_index(drop=True)
    label_rna_track_info = label_track_info[label_track_info['modality'].str.contains('RNA')].reset_index(drop=True)
    print(f"RNA tracks: {len(pred_rna_idxes)}")

    # Prepare arguments for parallel processing
    args_list = [(i, row, test_preds, test_labels, pred_rna_idxes, label_rna_idxes, use_exon_only, filter_protein_coding)
                 for i, row in info.iterrows()]

    # Process sequences in parallel
    print(f"\nProcessing sequences in parallel (mode: {mode_str})...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_sequence)(args) for args in args_list
    )

    # Collect results
    gene_info = pd.DataFrame()
    predictions = []
    labels = []
    raw_predictions = []
    raw_labels = []

    for df, seq_predictions, seq_labels, rna_raw_predictions, rna_raw_labels in results:
        gene_info = pd.concat([gene_info, df], ignore_index=True)
        predictions.extend(seq_predictions)
        labels.extend(seq_labels)
        raw_predictions.append(rna_raw_predictions)
        raw_labels.append(rna_raw_labels)

    # Save results
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    raw_predictions = np.concatenate(raw_predictions, axis=0)
    raw_labels = np.concatenate(raw_labels, axis=0)

    # Apply transformation if requested
    if transform != "none":
        print(f"\nApplying {transform} transformation...")
        predictions = apply_transform(predictions, transform)
        labels = apply_transform(labels, transform)
        raw_predictions_flat = raw_predictions.reshape(-1, raw_predictions.shape[-1])
        raw_labels_flat = raw_labels.reshape(-1, raw_labels.shape[-1])
        raw_predictions_flat = apply_transform(raw_predictions_flat, transform)
        raw_labels_flat = apply_transform(raw_labels_flat, transform)
        raw_predictions = raw_predictions_flat.reshape(raw_predictions.shape)
        raw_labels = raw_labels_flat.reshape(raw_labels.shape)

    print("\nSaving results...")
    output_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data"
    np.save(f'{output_dir}/RNA_raw_predictions{suffix}.npy', raw_predictions)
    np.save(f'{output_dir}/RNA_raw_labels{suffix}.npy', raw_labels)
    np.save(f'{output_dir}/RNA_gene_predictions{suffix}.npy', predictions)
    np.save(f'{output_dir}/RNA_gene_labels{suffix}.npy', labels)
    gene_info.to_csv(f'{output_dir}/RNA_gene_info{suffix}.csv', index=False)
    print(f"Saved results to {output_dir}")

    # Compute correlations
    print("\nComputing correlations...")
    corr_pearsons, corr_spearmans, raw_corr_pearsons, raw_corr_spearmans = [], [], [], []
    for i in range(predictions.shape[1]):
        # Apply log1p only if no transform was applied (transform already handles normalization)
        if transform == "none":
            filter = labels[:, i] > -0.5*32/1000
            corr_pearson = pearsonr(np.log1p(predictions[:, i][filter]), np.log1p(labels[:, i][filter]))[0]
            corr_spearman = spearmanr(np.log1p(predictions[:, i][filter]), np.log1p(labels[:, i][filter]))[0]
            raw_corr_pearson = pearsonr(np.log1p(raw_predictions[:, i].flatten()), np.log1p(raw_labels[:, i].flatten()))[0]
            raw_corr_spearman = spearmanr(np.log1p(raw_predictions[:, i].flatten()), np.log1p(raw_labels[:, i].flatten()))[0]
        else:
            # Use transformed data directly without additional log1p
            # For filtering, use a small threshold appropriate for transformed data
            filter = np.abs(labels[:, i]) > -1e-6
            corr_pearson = pearsonr(predictions[:, i][filter], labels[:, i][filter])[0]
            corr_spearman = spearmanr(predictions[:, i][filter], labels[:, i][filter])[0]
            raw_corr_pearson = pearsonr(raw_predictions[:, i].flatten(), raw_labels[:, i].flatten())[0]
            raw_corr_spearman = spearmanr(raw_predictions[:, i].flatten(), raw_labels[:, i].flatten())[0]
        corr_pearsons.append(corr_pearson)
        corr_spearmans.append(corr_spearman)
        raw_corr_pearsons.append(raw_corr_pearson)
        raw_corr_spearmans.append(raw_corr_spearman)
    pred_rna_track_info['corr_pearson'] = corr_pearsons
    pred_rna_track_info['corr_spearman'] = corr_spearmans
    pred_rna_track_info['raw_corr_pearson'] = raw_corr_pearsons
    pred_rna_track_info['raw_corr_spearman'] = raw_corr_spearmans

    # Compute the correlations for canonical genes
    # Note: When use_exon_only=True, gene_info is already aggregated by gene (all isoforms combined)
    # When use_exon_only=False, gene_info contains individual transcripts
    if use_exon_only:
        # Already aggregated by gene, no need to filter
        gene_info_canonical = gene_info.copy()
    else:
        # For transcript-level data, keep only the longest transcript per gene
        gene_info['length'] = gene_info['end_bin_idx'] - gene_info['start_bin_idx']
        gene_info_canonical = gene_info.sort_values('length', ascending=False).drop_duplicates('geneName')

    corr_pearsons, corr_spearmans = [], []
    for i in range(predictions.shape[1]):
        if transform == "none":
            filter = labels[gene_info_canonical.index, i] > 0.5*32/1000
            corr_pearson = pearsonr(np.log1p(predictions[gene_info_canonical.index, i][filter]), np.log1p(labels[gene_info_canonical.index, i][filter]))[0]
            corr_spearman = spearmanr(np.log1p(predictions[gene_info_canonical.index, i][filter]), np.log1p(labels[gene_info_canonical.index, i][filter]))[0]
        else:
            filter = np.abs(labels[gene_info_canonical.index, i]) > 1e-6
            corr_pearson = pearsonr(predictions[gene_info_canonical.index, i][filter], labels[gene_info_canonical.index, i][filter])[0]
            corr_spearman = spearmanr(predictions[gene_info_canonical.index, i][filter], labels[gene_info_canonical.index, i][filter])[0]
        corr_pearsons.append(corr_pearson)
        corr_spearmans.append(corr_spearman)
    pred_rna_track_info['corr_canonical_pearson'] = corr_pearsons
    pred_rna_track_info['corr_canonical_spearman'] = corr_spearmans

    # Save correlation results
    pred_rna_track_info.to_csv(f'{output_dir}/RNA_correlations{suffix}.csv', index=False)
    print(f"Saved correlation results to {output_dir}/RNA_correlations{suffix}.csv")

    # Compute per-gene across-cell-type correlations
    print("Computing per-gene across-cell-type correlations...")
    gene_correlations = []

    for gene_idx in tqdm(range(predictions.shape[0]), desc="Computing per-gene correlations"):
        # Get predictions and labels for this gene across all cell types
        gene_pred = predictions[gene_idx, :]
        gene_label = labels[gene_idx, :]
        # Compute correlation across cell types for this gene
        if len(gene_pred) > 1 and gene_pred.std() > 0 and gene_label.std() > 0:
            corr = pearsonr(gene_pred, gene_label)[0]
        else:
            corr = np.nan

        gene_correlations.append(corr)

    # Add to gene_info
    gene_info_canonical['across_celltype_corr'] = np.array(gene_correlations)[gene_info_canonical.index]

    # Save gene-level correlations
    gene_info_canonical.to_csv(f'{output_dir}/RNA_gene_correlations{suffix}.csv', index=False)
    print(f"Saved per-gene correlations to {output_dir}/RNA_gene_correlations{suffix}.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/plot"

    # Try to load cell type info for plotting
    cell_type_info_file = f"{data_path}/../data_config/{exp_name}.csv"
    if os.path.exists(cell_type_info_file):
        cell_type_info = pd.read_csv(cell_type_info_file, index_col=0)
        pred_rna_track_info['celltype_n'] = cell_type_info['celltype_n'].groupby(cell_type_info['exp']).first().loc[pred_rna_track_info['trial']].values
    else:
        print(f"Warning: Cell type info file not found: {cell_type_info_file}")
        pred_rna_track_info['celltype_n'] = 1  # Default value

    fig, ax = plt.subplots(nrows=2, figsize=(4, 4))
    # do two histogram plots up and down sharing x axis
    sns.histplot(pred_rna_track_info['raw_corr_pearson'], bins=20, ax=ax[0])
    xlabel_raw = 'RNA bin Pearson Correlation' + (f' [{transform}]' if transform != 'none' else '')
    ax[0].set_xlabel(xlabel_raw)
    ax[0].set_ylabel('Tracks')
    sns.histplot(pred_rna_track_info['corr_canonical_pearson'], bins=20, ax=ax[1], color='orange')
    xlabel_gene = ('RNA gene (exon-only) Pearson Correlation' if use_exon_only else 'RNA canonical transcript Pearson Correlation')
    xlabel_gene += (f' [{transform}]' if transform != 'none' else '')
    ax[1].set_xlabel(xlabel_gene)
    ax[1].set_ylabel('Tracks')
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/RNA_correlation_hist{suffix}.png', dpi=300)
    plt.close(fig)
    print(f"Saved correlation histogram to RNA_correlation_hist{suffix}.png")

    # Draw comparison scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=pred_rna_track_info['raw_corr_pearson'],
                    y=pred_rna_track_info['corr_canonical_pearson'],
                    hue=np.log(pred_rna_track_info['celltype_n']),
                    ax=ax)
    # add diagnal line using sns
    sns.lineplot(x=[0, 1], y=[0, 1], color='k', linestyle='--', ax=ax)
    xlabel_scatter = 'RNA 32bp Pearson Correlation' + (f' [{transform}]' if transform != 'none' else '')
    ax.set_xlabel(xlabel_scatter)
    ylabel = ('RNA gene (exon-only) Pearson Correlation' if use_exon_only else 'RNA canonical transcript Pearson Correlation')
    ylabel += (f' [{transform}]' if transform != 'none' else '')
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/RNA_correlation_scatter{suffix}.png', dpi=300)
    plt.close(fig)
    print(f"Saved correlation scatter plot to RNA_correlation_scatter{suffix}.png")

    # Plot per-gene across-cell-type correlations
    print("Plotting per-gene across-cell-type correlations...")

    # Histogram of per-gene correlations
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_corrs = gene_info_canonical['across_celltype_corr'].dropna()
    sns.histplot(valid_corrs, bins=50, ax=ax, kde=True)
    ax.axvline(valid_corrs.median(), color='red', linestyle='--', label=f'Median: {valid_corrs.median():.3f}')
    xlabel_gene_corr = 'Across Cell Type Pearson Correlation' + (f' [{transform}]' if transform != 'none' else '')
    ax.set_xlabel(xlabel_gene_corr)
    ax.set_ylabel('Number of Genes')
    title = 'Per-Gene Across Cell Type Correlation\n(Gene-level exon-only)' if use_exon_only else 'Per-Gene Across Cell Type Correlation'
    if transform != "none":
        title += f' [{transform}]'
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/RNA_per_gene_correlation_hist{suffix}.png', dpi=300)
    plt.close(fig)
    print(f"Saved per-gene correlation histogram to RNA_per_gene_correlation_hist{suffix}.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("Per-Gene Across Cell Type Correlation Summary:")
    print("="*60)
    print(f"Total genes: {len(valid_corrs)}")
    print(f"Mean correlation: {valid_corrs.mean():.3f}")
    print(f"Median correlation: {valid_corrs.median():.3f}")
    print(f"Std correlation: {valid_corrs.std():.3f}")
    print(f"Min correlation: {valid_corrs.min():.3f}")
    print(f"Max correlation: {valid_corrs.max():.3f}")
    print(f"Genes with corr > 0.5: {(valid_corrs > 0.5).sum()} ({100*(valid_corrs > 0.5).sum()/len(valid_corrs):.1f}%)")
    print(f"Genes with corr > 0.7: {(valid_corrs > 0.7).sum()} ({100*(valid_corrs > 0.7).sum()/len(valid_corrs):.1f}%)")
    print("="*60)

    # Print top and bottom genes by correlation
    print("\nTop 10 genes by across cell type correlation:")
    gene_id_col = 'geneID' if use_exon_only else 'transcriptID'
    if gene_id_col in gene_info_canonical.columns:
        top_genes = gene_info_canonical.nlargest(10, 'across_celltype_corr')[['geneName', gene_id_col, 'across_celltype_corr']]
    else:
        top_genes = gene_info_canonical.nlargest(10, 'across_celltype_corr')[['geneName', 'across_celltype_corr']]
    print(top_genes.to_string(index=False))

    print("\nBottom 10 genes by across cell type correlation:")
    if gene_id_col in gene_info_canonical.columns:
        bottom_genes = gene_info_canonical.nsmallest(10, 'across_celltype_corr')[['geneName', gene_id_col, 'across_celltype_corr']]
    else:
        bottom_genes = gene_info_canonical.nsmallest(10, 'across_celltype_corr')[['geneName', 'across_celltype_corr']]
    print(bottom_genes.to_string(index=False))

    print("\nDone!")
    print(f"All results saved to: {RES_BASE}/{exp_name}/analysis_{chk}/")


if __name__ == "__main__":
    main()
