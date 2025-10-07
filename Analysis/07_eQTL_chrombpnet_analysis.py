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
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')
# %% chrombpnet results
cell_types = []
for cell_type in os.listdir('Data/source/chrombpnet/BICAN/ATAC/'):
    if not os.path.isfile(f'Data/source/chrombpnet/BICAN/ATAC/{cell_type}/variant_scores.tsv'):
        continue
    cell_types.append(cell_type)
# %% define biotypes to keep
biotype_to_keep = ['protein_coding', 'lncRNA', 'IG_V_gene', 'TR_V_gene', 'IG_C_gene', 'snoRNA', 'snRNA', 'TR_C_gene', 'miRNA']
brain_organs = ['Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
                'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
                'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
                'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra']
basal_ganglia_organs = ['Brain_Caudate_basal_ganglia', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia']
# %%
import sklearn
result_df = pd.DataFrame()
for organ in ['All', 'Brain', 'Basal_ganglia']:
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
    else:
        continue
    # filter eqtl_organ and eqtl_organ_info by biotype_to_keep
    eqtl_organ_info = eqtl_organ_info.loc[eqtl_organ_info['biotype'].isin(biotype_to_keep)]
    eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    eqtl_organ['group'] = eqtl_organ_info['group'].groupby(eqtl_organ_info['variant_id']).first().loc[eqtl_organ['ID']].values
    # drop duplicates
    eqtl_organ_unique = eqtl_organ.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
    # for each cell type get the chrombpnet scores
    for cell_type in cell_types:
        chrombpnet_res = pd.read_csv(f'Data/source/chrombpnet/BICAN/ATAC/{cell_type}/variant_scores.tsv', sep='\t')
        eqtl_scores = chrombpnet_res['logfc'].groupby(chrombpnet_res['variant_id']).first().loc[eqtl_organ_unique['ID']].values
        # get overall precision recall
        for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
            label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
            if group == 'all':
                # auroc
                auroc = sklearn.metrics.roc_auc_score(label, eqtl_scores)
                # auprc
                precision, recall, _ = sklearn.metrics.precision_recall_curve(label, eqtl_scores)
                auprc = sklearn.metrics.auc(recall, precision)
                positives = (eqtl_organ_unique['INFO'] == 'positive').sum()
                negatives = (eqtl_organ_unique['INFO'] == 'negative').sum()
            else:
                # auroc
                auroc = sklearn.metrics.roc_auc_score(
                    label[eqtl_organ_unique['group'] == group],
                    eqtl_scores[eqtl_organ_unique['group'] == group]
                )
                # auprc
                precision, recall, _ = sklearn.metrics.precision_recall_curve(
                    label[eqtl_organ_unique['group'] == group],
                    eqtl_scores[eqtl_organ_unique['group'] == group]
                )
                auprc = sklearn.metrics.auc(recall, precision)
                positives = (eqtl_organ_unique['INFO'][eqtl_organ_unique['group'] == group] == 'positive').sum()
                negatives = (eqtl_organ_unique['INFO'][eqtl_organ_unique['group'] == group] == 'negative').sum()
            result_df = pd.concat([result_df, 
                                   pd.DataFrame({'exp': cell_type, 'organ': organ, 'group': group,
                                                 'AUROC': auroc, 'AUPRC': auprc,
                                                 'positives': positives, 'negatives': negatives}, 
                                                index=[0])], ignore_index=True)
# %% save results
result_df.to_csv('Data/source/eQTL/chrombpnet_miniatlas_ATAC_results.csv')
# %% plot
import seaborn as sns
import matplotlib.pyplot as plt
organs = ['All', 'Brain', 'Basal_ganglia']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = result_df[result_df['organ'] == organ].copy().dropna()
    sns.violinplot(data=to_plot, x='group', y='AUROC', ax=ax)
    ax.set_title(organ)
# %%
