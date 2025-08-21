# %% import libs
import h5py
import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from functools import partial
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
# %% get unique var names
eqtl['file'] = '/share/vault/Users/gz2294/BICAN/Res/250719_atac_rna_ft/analysis_best_valid_loss/raw_data/var_eff/' + \
    eqtl['#CHROM'].astype(str) + '_' + eqtl['REF'].astype(str) + eqtl['POS'].astype(str) + eqtl['ALT'].astype(str) + '.h5'
def variant_score_optimized(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            diff_data = f['data']['diff'][:]
            pred_alt = f['data']['pred_alt'][:]
            pred_wt = f['data']['pred_wt'][:]
            
            sum_score = diff_data.sum(axis=0)
            log_diff = np.log2(pred_alt + 1) - np.log2(pred_wt + 1)
            l2_score = np.sqrt((log_diff**2).sum(axis=0))
            
            return np.concatenate((sum_score, l2_score), axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# apply variant_score to all eqtl
eqtl_unique = eqtl.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
# %% Use multiprocessing for parallel execution
with Pool() as pool:
    results = pool.map(variant_score_optimized, eqtl_unique['file'].tolist())

# Filter out None results and stack
valid_results = [r for r in results if r is not None]
scores = np.vstack(valid_results) if valid_results else np.array([])
# %%
# save the results
np.save('Data/source/eQTL/all_scores.npy', scores)
# %%
scores = np.load('Data/source/eQTL/all_scores.npy')
eqtl_unique.index = eqtl_unique['ID']
# %%
import sklearn
track_results = pd.DataFrame()
for organ in os.listdir('Data/source/eQTL/') + ['All']:
    # if is directory
    if organ == 'All':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
    elif os.path.isdir(f'Data/source/eQTL/{organ}'):
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/{organ}/all.vcf', sep='\t')
        # get info
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/{organ}/info.csv')
    else:
        continue
    eqtl_organ['group'] = eqtl_organ_info['group'].groupby(eqtl_organ_info['variant_id']).first().loc[eqtl_organ['ID']].values
    # drop duplicates
    eqtl_organ_unique = eqtl_organ.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
    eqtl_scores = scores[eqtl_unique.index.get_indexer(eqtl_organ_unique['ID'])]
    # load track annotations
    track_anno = pd.read_csv('./Data/data_config/HM_ATAC_RNA_v1.csv', index_col=0)
    track_anno['organ'] = organ
    # get overall precision recall
    for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
        track_anno['group'] = group
        label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
        for i, idx in enumerate(track_anno.index):
            if group == 'all':
                auroc = sklearn.metrics.roc_auc_score(label, eqtl_scores[:, i+203])
                positives = (eqtl_organ_unique['INFO'] == 'positive').sum()
                negatives = (eqtl_organ_unique['INFO'] == 'negative').sum()
            else:
                auroc = sklearn.metrics.roc_auc_score(label[eqtl_organ_unique['group'] == group],
                                                        eqtl_scores[eqtl_organ_unique['group'] == group, i+203])
                positives = (eqtl_organ_unique['INFO'][eqtl_organ_unique['group'] == group] == 'positive').sum()
                negatives = (eqtl_organ_unique['INFO'][eqtl_organ_unique['group'] == group] == 'negative').sum()
            track_anno.loc[idx, 'AUROC'] = auroc
            track_anno.loc[idx, f'n_pos'] = positives
            track_anno.loc[idx, f'n_neg'] = negatives
        track_results = pd.concat([track_results, track_anno])
# %% add annotation
track_results['celltype'] = track_results['exp'].str.split('_').str[:-1].str.join('_')
track_results['mod'] = track_results['exp'].str.split('_').str[-1]
# %% save results
track_results.to_csv('Data/source/eQTL/track_results.csv')
# %% plot
import seaborn as sns
import matplotlib.pyplot as plt
# organs = ['Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
#           'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
#           'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
#           'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra']
organs = ['Brain_Caudate_basal_ganglia', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=track_results[track_results['organ'] == organ], x='group', y='AUROC', hue='mod', ax=ax)
    ax.set_title(organ)
# %%
