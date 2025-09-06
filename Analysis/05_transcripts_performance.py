# %%
import sys
import os
PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
from Model.utils.get_transcripts import get_transcripts_in_region
import torch
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

def process_sequence(args):
    """Process a single sequence - extract transcripts and compute predictions."""
    i, row, rna_idxes = args
    batch_idx = i // 2
    local_idx = i % 2
    chr = row['chr']
    start = row['start'] + 1023
    end = row['end'] - 1024
    
    # Get transcript info
    df = get_transcripts_in_region(f"{chr}:{start}-{end}")
    
    # Load predictions
    pred = torch.load(f'Res/basel_ganglia_complete_v1/Test_preds_rank_cuda:0_epoch_45_batch_{batch_idx}.pt')
    
    # Process each gene in this sequence
    seq_predictions = []
    seq_labels = []
    
    for _, r in df.iterrows():
        gene_start_bin = r['start_bin_idx']
        gene_end_bin = r['end_bin_idx']
        # get the predictions for this gene
        gene_pred = pred['pred']['regression'][local_idx, gene_start_bin:gene_end_bin+1, rna_idxes]
        gene_label = pred['label']['regression'][local_idx, gene_start_bin:gene_end_bin+1, rna_idxes]
        # average over bins, transform back -1 + np.sqrt(1 + seq_cov)
        gene_pred_mean = ((gene_pred + 1)**2-1).mean(dim=0, keepdim=True).numpy()
        gene_label_mean = ((gene_label + 1)**2-1).mean(dim=0, keepdim=True).numpy()
        seq_predictions.append(gene_pred_mean)
        seq_labels.append(gene_label_mean)

    # also output raw values of predictions and labels
    return df, seq_predictions, seq_labels, \
        (pred['pred']['regression'][:, :, rna_idxes].numpy()+1)**2-1, (pred['label']['regression'][:, :, rna_idxes].numpy()+1)**2-1   

# %% for each sequence in info, get the gene info and corresponding RNA predictions (parallel)
if __name__ == '__main__':
    # %% load sequence info
    info = pd.read_csv('Data/basel_ganglia_complete_v1/sequences.bed', sep='\t', header=None)
    info.columns = ['chr', 'start', 'end', 'split']
    info = info[info['split'] == 'test']
    # reindex
    info = info.reset_index(drop=True)

    # %% get track info
    track_info = pd.read_csv('Data/basel_ganglia_complete_v1/raw_label_meta.csv')
    rna_idxes = track_info[track_info['modality'] == 'RNA'].index.values
    rna_track_info = track_info.loc[rna_idxes].reset_index(drop=True)
    # %%
    # Prepare arguments for parallel processing
    args_list = [(i, row, rna_idxes) for i, row in info.iterrows()]

    # Use multiprocessing to process sequences in parallel
    # Pool() defaults to using all available CPU cores
    with Pool() as pool:
        results = pool.map(process_sequence, args_list)

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
        raw_predictions.extend(rna_raw_predictions)
        raw_labels.extend(rna_raw_labels)

    # %% save results
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    raw_predictions = np.concatenate(raw_predictions, axis=0)
    raw_labels = np.concatenate(raw_labels, axis=0)

    np.save('Res/basel_ganglia_complete_v1/RNA_raw_predictions.npy', raw_predictions)
    np.save('Res/basel_ganglia_complete_v1/RNA_raw_labels.npy', raw_labels)
    np.save('Res/basel_ganglia_complete_v1/RNA_gene_predictions.npy', predictions)
    np.save('Res/basel_ganglia_complete_v1/RNA_gene_labels.npy', labels)
    gene_info.to_csv('Res/basel_ganglia_complete_v1/RNA_gene_info.csv', index=False)
    # %% make plots
    raw_predictions = np.load('Res/basel_ganglia_complete_v1/RNA_raw_predictions.npy')
    raw_labels = np.load('Res/basel_ganglia_complete_v1/RNA_raw_labels.npy')
    predictions = np.load('Res/basel_ganglia_complete_v1/RNA_gene_predictions.npy')
    labels = np.load('Res/basel_ganglia_complete_v1/RNA_gene_labels.npy')
    gene_info = pd.read_csv('Res/basel_ganglia_complete_v1/RNA_gene_info.csv')
    # %% compute correlations
    corr_pearsons, corr_spearmans, raw_corr_pearsons, raw_corr_spearmans = [], [], [], []
    for i in range(predictions.shape[1]):
        filter = labels[:, i] > 0.5*32/1000
        corr_pearson = pearsonr(np.log1p(predictions[:, i][filter]), np.log1p(labels[:, i][filter]))[0]
        corr_spearman = spearmanr(predictions[:, i][filter], labels[:, i][filter])[0]
        raw_corr_pearson = pearsonr(np.log1p(raw_predictions[:, i]), np.log1p(raw_labels[:, i]))[0]
        raw_corr_spearman = spearmanr(raw_predictions[:, i], raw_labels[:, i])[0]
        corr_pearsons.append(corr_pearson)
        corr_spearmans.append(corr_spearman)
        raw_corr_pearsons.append(raw_corr_pearson)
        raw_corr_spearmans.append(raw_corr_spearman)
    rna_track_info['corr_pearson'] = corr_pearsons
    rna_track_info['corr_spearman'] = corr_spearmans
    rna_track_info['raw_corr_pearson'] = raw_corr_pearsons
    rna_track_info['raw_corr_spearman'] = raw_corr_spearmans
    # %% compute the correlations for each genes with longest transcripts
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
    rna_track_info['corr_canonical_pearson'] = corr_pearsons
    rna_track_info['corr_canonical_spearman'] = corr_spearmans
    # %% plot
    cell_type_info = pd.read_csv('Data/data_config/basel_ganglia_complete_v1.csv', index_col=0)
    rna_track_info['celltype_n'] = cell_type_info['celltype_n'].groupby(cell_type_info['exp']).first().loc[rna_track_info['trial']].values
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(x=rna_track_info['raw_corr_spearman'], 
                    y=rna_track_info['corr_canonical_spearman'], 
                    hue=np.log(rna_track_info['celltype_n']),
                    ax=ax)
    # add diagnal line using sns
    sns.lineplot(x=[0, 1], y=[0, 1], color='k', linestyle='--', ax=ax)
    ax.set_xlabel('RNA 32bp Spearman Correlation')
    ax.set_ylabel('RNA canonical transcript Spearman Correlation')
    # %% visualize a certain cell type
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=np.log1p(raw_predictions[gene_info_canonical.index, rna_track_info['trial']=='Oligodendrocyte_RNA']), 
                    y=np.log1p(raw_labels[gene_info_canonical.index, rna_track_info['trial']=='Oligodendrocyte_RNA']), 
                    ax=ax)
    ax.set_xlabel('RNA 32bp predictions')
    ax.set_ylabel('RNA 32bp labels')

# %%
