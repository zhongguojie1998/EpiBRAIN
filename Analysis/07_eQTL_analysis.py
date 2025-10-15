# %% import libs
import h5py
import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')
# %% read in results
eqtl_h5 = h5py.File(f'Data/source/eQTL/basal_ganglia_miniatlas_drop_celltype_v1_res_epoch_150.h5', 'r')
log_square = eqtl_h5['results/local_log_square'][:]
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
import sklearn

def compute_metrics(label, scores, eqtl_organ_unique, group):
    """
    Compute AUROC, AUPRC, and positive/negative counts for a given group.

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

    Returns:
    --------
    tuple: (auroc, auprc, positives, negatives)
    """
    if group == 'all':
        # Compute metrics on all data
        auroc = sklearn.metrics.roc_auc_score(label, scores)
        auprc = sklearn.metrics.average_precision_score(label, scores)
        positives = (eqtl_organ_unique['INFO'] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'] == 'negative').sum()
    else:
        # Compute metrics for specific group
        group_mask = eqtl_organ_unique['group'] == group
        auroc = sklearn.metrics.roc_auc_score(label[group_mask], scores[group_mask])
        auprc = sklearn.metrics.average_precision_score(label[group_mask], scores[group_mask])
        positives = (eqtl_organ_unique['INFO'][group_mask] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'][group_mask] == 'negative').sum()

    return auroc, auprc, positives, negatives

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
    track_anno = pd.read_csv('./logs/basal_ganglia_miniatlas_drop_celltype_v1/regression_label_meta.csv')
    track_anno['organ'] = organ
    # get overall precision recall
    for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
        track_anno['group'] = group
        label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
        for i, idx in enumerate(track_anno.index):
            auroc, auprc, positives, negatives = compute_metrics(label, eqtl_scores[:, i], eqtl_organ_unique, group)
            track_anno.loc[idx, 'AUROC'] = auroc
            track_anno.loc[idx, 'AUPRC'] = auprc
            track_anno.loc[idx, f'n_pos'] = positives
            track_anno.loc[idx, f'n_neg'] = negatives
        for i, modality in enumerate(track_anno['modality'].unique()):
            modality_idx = track_anno.index[track_anno['modality'] == modality].values
            # L2 norm across modality_idx
            modality_merged = np.linalg.norm(eqtl_scores[:, modality_idx], axis=1)
            auroc, auprc, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group)
            track_results = pd.concat([track_results,
                                       pd.DataFrame({'exp': f'bican_{modality}', 'modality': modality, 'celltype': 'ALL',
                                                     'organ': organ, 'group': group,
                                                     'AUROC': auroc, 'AUPRC': auprc,
                                                     f'n_pos': positives, f'n_neg': negatives},
                                                    index=[0])], ignore_index=True)
        for i, modality in enumerate(track_anno['cell_type'].unique()):
            modality_idx = track_anno.index[(track_anno['cell_type'] == modality) & 
                                            (track_anno['modality'].str.contains('RNA') == False) &
                                            (track_anno['modality'].str.contains('K9Me3') == False)].values
            # L2 norm across modality_idx
            modality_merged = np.linalg.norm(eqtl_scores[:, modality_idx], axis=1)
            auroc, auprc, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group)
            track_results = pd.concat([track_results,
                                       pd.DataFrame({'exp': f'bican_celltype_{modality}', 'modality': 'ALL', 'celltype': modality,
                                                     'organ': organ, 'group': group,
                                                     'AUROC': auroc, 'AUPRC': auprc,
                                                     f'n_pos': positives, f'n_neg': negatives},
                                                    index=[0])], ignore_index=True)
        track_results = pd.concat([track_results, track_anno])
# %% add annotation
track_results.loc[track_results['exp'].isna(), 'exp'] = track_results['trial'][track_results['exp'].isna()].copy()
# %% save results
track_results.to_csv('Data/source/eQTL/basal_ganglia_miniatlas_drop_celltype_v1.track_results.csv')
# %%
track_results = pd.read_csv('Data/source/eQTL/basal_ganglia_miniatlas_drop_celltype_v1.track_results.csv', index_col=0)
# %% Add borzoi results
borzoi = pd.read_csv('Data/source/eQTL/borzoi_track_results.csv', index_col=0)
borzoi['exp'] = borzoi['identifier'].copy()
borzoi.loc[borzoi['identifier'].str.startswith('borzoi'), 'celltype'] = 'ALL'
track_results = pd.concat([track_results, borzoi])
# %% Add chrombpnet results
chrombpnet = pd.read_csv('Data/source/eQTL/chrombpnet_miniatlas_ATAC_results.csv', index_col=0)
chrombpnet['mod'] = 'chrombpnet'
chrombpnet.loc[chrombpnet['exp'].str.startswith('ATAC'), 'celltype'] = 'ALL'
track_results = pd.concat([track_results, chrombpnet])
# %% split bican/borzoi/ATAC and track results
track_results_model = track_results[track_results['exp'].str.startswith(('bican', 'borzoi', 'ATAC'))].copy()
track_results_track = track_results[~track_results['exp'].str.startswith(('bican', 'borzoi', 'ATAC'))].copy()
track_results_track.loc[track_results_track['mod'].isna(), 'mod'] = track_results_track['modality'][track_results_track['mod'].isna()].copy()
track_results_celltype = track_results_model[track_results_model['celltype'] != 'ALL'].copy()
track_results_overall = track_results_model[track_results_model['celltype'] == 'ALL'].copy()
# %% plot
organs = ['All', 'Brain', 'Basal_ganglia', 'Cortex']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = track_results_track[track_results_track['organ'] == organ].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
    sns.violinplot(data=to_plot, x='group', y='AUROC', hue='mod', ax=ax)
    ax.set_title(organ)
# %% plot cell type performances
organ = 'Basal_ganglia'
fig, ax = plt.subplots(figsize=(10, 6))
to_plot = track_results_celltype[(track_results_celltype['organ'] == organ) & 
                                 (track_results_celltype['group'] == 'all')].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
# rank by AUROC
to_plot = to_plot.sort_values(by='AUROC', ascending=False)
to_plot['type'] = 'borzoi'
to_plot.loc[to_plot['exp'].str.contains('bican_celltype_BasalGanglia'), 'type'] = 'bican:BasalGanglia'
to_plot.loc[to_plot['exp'].str.contains('bican_celltype_MiniAtlas'), 'type'] = 'bican:MiniAtlas'
to_plot.loc[to_plot['exp'] == 'ATAC', 'type'] = 'chrombpnet'
to_plot['exp'] = to_plot['exp'].str.replace('bican_celltype_', '')
sns.barplot(data=to_plot, x='exp', y='AUROC', hue='type', ax=ax)
# rotate x labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title(organ)
# %% plot overall model performances
organ = 'Basal_ganglia'
fig, ax = plt.subplots(figsize=(10, 6))
to_plot = track_results_overall[(track_results_overall['organ'] == organ) & 
                                (track_results_overall['group'] == 'all')].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
# rank by AUROC
to_plot = to_plot.sort_values(by='AUROC', ascending=False)
to_plot['type'] = 'borzoi'
to_plot.loc[to_plot['exp'].str.contains('bican'), 'type'] = 'bican'
to_plot.loc[to_plot['exp'] == 'ATAC', 'type'] = 'chrombpnet'
to_plot['exp'] = to_plot['exp'].str.replace('bican_celltype_', '')
sns.barplot(data=to_plot, x='exp', y='AUROC', hue='type', ax=ax)
# rotate x labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title(organ)

# %%
