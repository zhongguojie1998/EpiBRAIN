# %%
"""
Transcript-level RNA expression performance analysis.

This script evaluates model predictions at the gene/transcript level by:
1. Loading model predictions for RNA tracks
2. Extracting gene/transcript coordinates from GTF file
3. Computing average expression for each gene/transcript
4. Calculating correlation metrics

Modes:
- USE_EXON_ONLY = False: Uses full transcript ranges (including introns), one entry per transcript
- USE_EXON_ONLY = True: Aggregates by gene (combining all isoforms) and uses only exon bins,
                        excluding introns. This gives more biologically meaningful gene expression.
"""

import sys
import os
PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
from Model.utils.get_transcripts import get_transcripts_in_region
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

def process_sequence(args):
    """Process a single sequence - extract transcripts and compute predictions."""
    i, row, test_preds, test_labels, pred_rna_idxes, label_rna_idxes, use_exon_only = args
    chr = row['chr']
    start = row['start'] + 1023
    end = row['end'] - 1024

    # Get gene/transcript info
    # When use_exon_only=True, returns gene-aggregated data with exon bins
    # When use_exon_only=False, returns transcript-level data
    df = get_transcripts_in_region(f"{chr}:{start}-{end}", return_exon_bins_only=use_exon_only)

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

# %% for each sequence in info, get the gene info and corresponding RNA predictions (parallel)
if __name__ == '__main__':
    # %% Load predictions once
    print("Loading predictions...")
    preds_data = torch.load('Res/basal_ganglia_miniatlas_drop_celltype_v1/Test_preds_epoch_150.pt',
                            map_location='cpu', weights_only=False)
    test_preds = preds_data['pred']['regression'].numpy()  # Shape: (436, 16320, 222)
    test_labels = preds_data['label']['regression'].numpy()  # Shape: (436, 16320, 199)
    print(f"Predictions shape: {test_preds.shape}")
    print(f"Labels shape: {test_labels.shape}")

    # %% load sequence info
    info = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/sequences.bed', sep='\t', header=None)
    info.columns = ['chr', 'start', 'end', 'split']
    info = info[info['split'] == 'test']
    # reindex
    info = info.reset_index(drop=True)
    print(f"Test sequences: {len(info)}")

    # %% get track info
    pred_track_info = pd.read_csv('logs/basal_ganglia_miniatlas_drop_celltype_v1/regression_label_meta.csv')
    label_track_info = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/raw_label_meta.csv')
    pred_rna_idxes = pred_track_info[pred_track_info['modality'].str.contains('RNA')]['dim'].values
    label_rna_idxes = label_track_info[label_track_info['modality'].str.contains('RNA')].index.values
    pred_rna_track_info = pred_track_info[pred_track_info['modality'].str.contains('RNA')].reset_index(drop=True)
    label_rna_track_info = label_track_info[label_track_info['modality'].str.contains('RNA')].reset_index(drop=True)
    print(f"RNA tracks: {len(pred_rna_idxes)}")

    # %%
    # Set to True to use only exon bins for gene expression calculation
    # Set to False to use all bins (including introns) - original behavior
    USE_EXON_ONLY = True  # Change this to True to use exon-only mode

    # Prepare arguments for parallel processing
    args_list = [(i, row, test_preds, test_labels, pred_rna_idxes, label_rna_idxes, USE_EXON_ONLY) for i, row in info.iterrows()]

    # Use joblib to process sequences in parallel
    # n_jobs=-1 uses all available CPU cores
    mode_str = "exon-only" if USE_EXON_ONLY else "full transcript"
    print(f"Processing sequences in parallel (mode: {mode_str})...")
    results = Parallel(n_jobs=36, verbose=10)(
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

    # %% save results
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    raw_predictions = np.concatenate(raw_predictions, axis=0)
    raw_labels = np.concatenate(raw_labels, axis=0)

    print("Saving results...")
    # Add suffix based on mode
    suffix = "_exon" if USE_EXON_ONLY else ""
    np.save(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_raw_predictions{suffix}.npy', raw_predictions)
    np.save(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_raw_labels{suffix}.npy', raw_labels)
    np.save(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_predictions{suffix}.npy', predictions)
    np.save(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_labels{suffix}.npy', labels)
    gene_info.to_csv(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_info{suffix}.csv', index=False)

    # %% make plots
    # Load saved results (update suffix if running separately)
    suffix = "_exon" if USE_EXON_ONLY else ""
    raw_predictions = np.load(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_raw_predictions{suffix}.npy')
    raw_labels = np.load(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_raw_labels{suffix}.npy')
    predictions = np.load(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_predictions{suffix}.npy')
    labels = np.load(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_labels{suffix}.npy')
    gene_info = pd.read_csv(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_gene_info{suffix}.csv')

    # %% compute correlations
    print("Computing correlations...")
    corr_pearsons, corr_spearmans, raw_corr_pearsons, raw_corr_spearmans = [], [], [], []
    for i in range(predictions.shape[1]):
        filter = labels[:, i] > 0.5*32/1000
        corr_pearson = pearsonr(np.log1p(predictions[:, i][filter]), np.log1p(labels[:, i][filter]))[0]
        corr_spearman = spearmanr(predictions[:, i][filter], labels[:, i][filter])[0]
        raw_corr_pearson = pearsonr(np.log1p(raw_predictions[:, i].flatten()), np.log1p(raw_labels[:, i].flatten()))[0]
        raw_corr_spearman = spearmanr(raw_predictions[:, i].flatten(), raw_labels[:, i].flatten())[0]
        corr_pearsons.append(corr_pearson)
        corr_spearmans.append(corr_spearman)
        raw_corr_pearsons.append(raw_corr_pearson)
        raw_corr_spearmans.append(raw_corr_spearman)
    pred_rna_track_info['corr_pearson'] = corr_pearsons
    pred_rna_track_info['corr_spearman'] = corr_spearmans
    pred_rna_track_info['raw_corr_pearson'] = raw_corr_pearsons
    pred_rna_track_info['raw_corr_spearman'] = raw_corr_spearmans

    # %% compute the correlations for canonical genes
    # Note: When USE_EXON_ONLY=True, gene_info is already aggregated by gene (all isoforms combined)
    # When USE_EXON_ONLY=False, gene_info contains individual transcripts
    if USE_EXON_ONLY:
        # Already aggregated by gene, no need to filter
        gene_info_canonical = gene_info.copy()
    else:
        # For transcript-level data, keep only the longest transcript per gene
        gene_info['length'] = gene_info['end_bin_idx'] - gene_info['start_bin_idx']
        gene_info_canonical = gene_info.sort_values('length', ascending=False).drop_duplicates('geneName')

    # %%
    corr_pearsons, corr_spearmans = [], []
    for i in range(predictions.shape[1]):
        filter = labels[gene_info_canonical.index, i] > 0.5*32/1000
        corr_pearson = pearsonr(np.log1p(predictions[gene_info_canonical.index, i][filter]), np.log1p(labels[gene_info_canonical.index, i][filter]))[0]
        corr_spearman = spearmanr(np.log1p(predictions[gene_info_canonical.index, i][filter]), np.log1p(labels[gene_info_canonical.index, i][filter]))[0]
        corr_pearsons.append(corr_pearson)
        corr_spearmans.append(corr_spearman)
    pred_rna_track_info['corr_canonical_pearson'] = corr_pearsons
    pred_rna_track_info['corr_canonical_spearman'] = corr_spearmans

    # Save correlation results
    pred_rna_track_info.to_csv(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_correlations{suffix}.csv', index=False)
    print(f"Saved correlation results to RNA_correlations{suffix}.csv")

    # %% plot
    cell_type_info = pd.read_csv('Data/data_config/basal_ganglia_miniatlas_drop_celltype_v1.csv', index_col=0)
    pred_rna_track_info['celltype_n'] = cell_type_info['celltype_n'].groupby(cell_type_info['exp']).first().loc[pred_rna_track_info['trial']].values

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(x=pred_rna_track_info['raw_corr_pearson'],
                    y=pred_rna_track_info['corr_canonical_pearson'],
                    hue=np.log(pred_rna_track_info['celltype_n']),
                    ax=ax)
    # add diagnal line using sns
    sns.lineplot(x=[0, 1], y=[0, 1], color='k', linestyle='--', ax=ax)
    ax.set_xlabel('RNA 32bp Pearson Correlation')
    ylabel = 'RNA gene (exon-only) Pearson Correlation' if USE_EXON_ONLY else 'RNA canonical transcript Pearson Correlation'
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_correlation_scatter{suffix}.png', dpi=300)
    print(f"Saved correlation scatter plot to RNA_correlation_scatter{suffix}.png")

    # %% visualize a certain cell type
    fig, ax = plt.subplots(figsize=(5, 5))
    # Find first RNA track for plotting
    example_rna_idx = 0
    example_trial = pred_rna_track_info.iloc[example_rna_idx]['trial']

    sns.scatterplot(x=np.log1p(predictions[gene_info_canonical.index, example_rna_idx]),
                    y=np.log1p(labels[gene_info_canonical.index, example_rna_idx]),
                    ax=ax, alpha=0.5)
    ax.set_xlabel(f'log1p(Predictions) - {example_trial}')
    ax.set_ylabel(f'log1p(Labels) - {example_trial}')
    title_prefix = 'Gene-level (exon-only)' if USE_EXON_ONLY else 'Gene-level'
    ax.set_title(f'{title_prefix} predictions vs labels\nSpearman: {pred_rna_track_info.iloc[example_rna_idx]["corr_canonical_spearman"]:.3f}')
    plt.tight_layout()
    plt.savefig(f'Res/basal_ganglia_miniatlas_drop_celltype_v1/RNA_example_scatter{suffix}.png', dpi=300)
    print(f"Saved example scatter plot to RNA_example_scatter{suffix}.png")

    print("\nDone!")

# %%
