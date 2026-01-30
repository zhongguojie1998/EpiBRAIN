# %% import libs
import os
import sys

import h5py
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

PWD = f'{os.environ["workingHOME"]}/BICAN'
sys.path.append(f'{PWD}')
os.chdir(f'{PWD}')
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')
# %% borzoi results
eqtl_h5 = h5py.File(f'Data/source/eQTL/borzoi_res.h5', 'r')
log_square = eqtl_h5['results/log_square'][:]
eqtl_info_df = pd.DataFrame({'chr': eqtl_h5['variants/chr'][:],
                             'pos': eqtl_h5['variants/pos'][:],
                             'ref': eqtl_h5['variants/ref'][:],
                             'alt': eqtl_h5['variants/alt'][:]})
eqtl_info_df.index = eqtl_info_df['chr'].astype(str) + '_' + eqtl_info_df['pos'].astype(str) + '_' + eqtl_info_df['ref'].astype(str) + '_' + eqtl_info_df['alt'].astype(str) + '_b38'
# %% define biotypes to keep
biotype_to_keep = ['protein_coding', 'lncRNA', 'IG_V_gene', 'TR_V_gene', 'IG_C_gene', 'snoRNA', 'snRNA', 'TR_C_gene', 'miRNA']
brain_organs = ['Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
                'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
                'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
                'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra']
basal_ganglia_organs = ['Brain_Caudate_basal_ganglia', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia']
cortex_organs = ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9']

# %% Define helper function to compute metrics
def compute_auroc_with_se(trait_values, score_values, n_bootstraps=100):
    """
    Compute AUROC with standard error using bootstrap resampling.

    Parameters:
    -----------
    trait_values : array-like
        Binary labels (0 or 1)
    score_values : array-like
        Prediction scores
    n_bootstraps : int or None
        Number of bootstrap samples (default: 100)
        If None, only compute AUROC without SE

    Returns:
    --------
    tuple: (auroc, se)
    """
    # Calculate main AUROC
    try:
        auroc = roc_auc_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # If n_bootstraps is None, skip SE calculation
    if n_bootstraps is None:
        return auroc, np.nan

    # Convert to boolean if needed, then to polars DataFrame for efficient resampling
    trait_values_bool = np.asarray(trait_values).astype(bool)
    V = pl.DataFrame({"label": trait_values_bool, "score": score_values})

    # Bootstrap resampling within each class
    def resample(V, seed):
        V_pos = V.filter(pl.col("label"))
        V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
        V_neg = V.filter(~pl.col("label"))
        V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
        return pl.concat([V_pos, V_neg])

    # Calculate AUROC on bootstrap samples
    V_bs = [resample(V, i) for i in range(n_bootstraps)]
    bootstrap_aurocs = []
    for V_b in V_bs:
        try:
            # Convert back to int for sklearn compatibility
            bootstrap_aurocs.append(roc_auc_score(V_b["label"].cast(pl.Int64), V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_aurocs).std() if bootstrap_aurocs else np.nan

    return auroc, se

def compute_auprc_with_se(trait_values, score_values, n_bootstraps=100):
    """
    Compute AUPRC with standard error using bootstrap resampling.

    Parameters:
    -----------
    trait_values : array-like
        Binary labels (0 or 1)
    score_values : array-like
        Prediction scores
    n_bootstraps : int or None
        Number of bootstrap samples (default: 100)
        If None, only compute AUPRC without SE

    Returns:
    --------
    tuple: (auprc, se)
    """
    # Calculate main AUPRC
    try:
        auprc = average_precision_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # If n_bootstraps is None, skip SE calculation
    if n_bootstraps is None:
        return auprc, np.nan

    # Convert to boolean if needed, then to polars DataFrame for efficient resampling
    trait_values_bool = np.asarray(trait_values).astype(bool)
    V = pl.DataFrame({"label": trait_values_bool, "score": score_values})

    # Bootstrap resampling within each class
    def resample(V, seed):
        V_pos = V.filter(pl.col("label"))
        V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
        V_neg = V.filter(~pl.col("label"))
        V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
        return pl.concat([V_pos, V_neg])

    # Calculate AUPRC on bootstrap samples
    V_bs = [resample(V, i) for i in range(n_bootstraps)]
    bootstrap_auprcs = []
    for V_b in V_bs:
        try:
            # Convert back to int for sklearn compatibility
            bootstrap_auprcs.append(average_precision_score(V_b["label"].cast(pl.Int64), V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_auprcs).std() if bootstrap_auprcs else np.nan

    return auprc, se

def compute_metrics(label, scores, eqtl_organ_unique, group, n_bootstraps=100):
    """
    Compute AUROC, AUPRC with SE, and positive/negative counts for a given group.

    Parameters:
    -----------
    label : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    eqtl_organ_unique : pd.DataFrame
        DataFrame containing eQTL information with 'INFO' and 'group' columns
    group : str
        Group name ('all' for all data, or specific group like '<3k', '3k-12k', etc.)
    n_bootstraps : int or None
        Number of bootstrap samples for SE calculation (default: 100)
        If None, only compute metrics without SE

    Returns:
    --------
    tuple: (auroc, auroc_se, auprc, auprc_se, positives, negatives)
    """
    if group == 'all':
        # Compute metrics on all data
        auroc, auroc_se = compute_auroc_with_se(label, scores, n_bootstraps)
        auprc, auprc_se = compute_auprc_with_se(label, scores, n_bootstraps)
        positives = (eqtl_organ_unique['INFO'] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'] == 'negative').sum()
    else:
        # Compute metrics for specific group
        group_mask = eqtl_organ_unique['group'] == group
        auroc, auroc_se = compute_auroc_with_se(label[group_mask], scores[group_mask], n_bootstraps)
        auprc, auprc_se = compute_auprc_with_se(label[group_mask], scores[group_mask], n_bootstraps)
        positives = (eqtl_organ_unique['INFO'][group_mask] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'][group_mask] == 'negative').sum()

    return auroc, auroc_se, auprc, auprc_se, positives, negatives

# %%
track_results = pd.DataFrame()
for organ in ['All', 'Brain', 'Basal_ganglia', 'Cortex']:
    # if is directory
    if organ == 'All':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
    elif os.path.isdir(f'Data/source/eQTL/{organ}'):
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/{organ}/all.vcf', sep='\t')
        # get info
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/{organ}/info.csv')
    elif organ == 'Brain':
        # all brain eQTL
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(brain_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    elif organ == 'Basal_ganglia':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(basal_ganglia_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    elif organ == 'Cortex':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(cortex_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    else:
        continue
    # filter eqtl_organ and eqtl_organ_info by biotype_to_keep
    eqtl_organ_info = eqtl_organ_info.loc[(eqtl_organ_info['biotype'].isin(biotype_to_keep)) | (eqtl_organ['INFO'] == 'negative')]
    eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    eqtl_organ['group'] = eqtl_organ_info['group'].groupby(eqtl_organ_info['variant_id']).first().loc[eqtl_organ['ID']].values
    # drop duplicates
    eqtl_organ_unique = eqtl_organ.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
    eqtl_scores = log_square[eqtl_info_df.index.get_indexer(eqtl_organ_unique['ID'])]
    # load track annotations
    track_anno = pd.read_csv('borzoi.published.targets.txt', index_col=0, sep='\t')
    track_anno['modality'] = track_anno['file'].str.split('/').str[6]
    track_anno['organ'] = organ
    # get overall precision recall
    for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
        track_anno['group'] = group
        label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
        for i, idx in enumerate(track_anno.index):
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, eqtl_scores[:, i], eqtl_organ_unique, group, n_bootstraps=None)
            track_anno.loc[idx, 'AUROC'] = auroc
            track_anno.loc[idx, 'AUROC_SE'] = auroc_se
            track_anno.loc[idx, 'AUPRC'] = auprc
            track_anno.loc[idx, 'AUPRC_SE'] = auprc_se
            track_anno.loc[idx, f'n_pos'] = positives
            track_anno.loc[idx, f'n_neg'] = negatives
        for i, modality in enumerate(track_anno['modality'].unique()):
            modality_idx = track_anno.index[track_anno['modality'] == modality].values
            # L2 norm across modality_idx
            modality_merged = np.linalg.norm(eqtl_scores[:, modality_idx], axis=1)
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
            track_results = pd.concat([track_results,
                                       pd.DataFrame({'identifier': f'borzoi_{modality}', 'modality': modality,
                                                     'organ': organ, 'group': group,
                                                     'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                     f'n_pos': positives, f'n_neg': negatives},
                                                    index=[0])], ignore_index=True)
        # get overall all modality
        modality_merged = np.linalg.norm(eqtl_scores, axis=1)
        auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
        track_results = pd.concat([track_results,
                                   pd.DataFrame({'identifier': f'borzoi_all', 'modality': 'ALL',
                                                 'organ': organ, 'group': group,
                                                 'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                 f'n_pos': positives, f'n_neg': negatives},
                                                index=[0])], ignore_index=True)
        track_results = pd.concat([track_results, track_anno])
# %% add annotation
track_results['mod'] = 'borzoi'
# %% save results
track_results.to_csv('Data/source/eQTL/borzoi_track_results.csv')
# %% plot
organs = ['All', 'Brain', 'Basal_ganglia']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = track_results[track_results['organ'] == organ].copy().dropna()
    sns.violinplot(data=to_plot, x='group', y='AUROC', ax=ax)
    ax.set_title(organ)
# %%
