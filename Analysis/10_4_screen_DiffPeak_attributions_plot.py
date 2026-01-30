# %%
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %%
res = pd.read_csv('Res/basal_ganglia_miniatlas_drop_celltype_v1/DiffPeak_attributions_screened.tsv', sep='\t', index_col=None)
# %% calculate mean in each cell type in two groups
for celltype in res['celltype'].unique():
    res_celltype = res[res['celltype'] == celltype]
    mean_positive = res_celltype[res_celltype['is_diffpeak'] == True]['sum_score'].mean()
    mean_negative = res_celltype[res_celltype['is_diffpeak'] == False]['sum_score'].mean()
    print(f'Cell type: {celltype}, Mean Positive: {mean_positive}, Mean Negative: {mean_negative}, Difference: {mean_positive - mean_negative}')
# %% for each pt_file, perform z-score normalization of sum_score
res['sum_bin_score_z'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: (x - x.mean()) / x.std())
res['sum_bin_score_abs'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: x.abs())
# quantile normalization
res['sum_bin_score_quantile'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: quantile_transform(x.values.reshape(-1, 1), axis=0, copy=True).flatten())
# rank percentile
res['sum_bin_score_percentile'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: x.rank(pct=True))
# %%
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=res, x='celltype', y='sum_bin_score_percentile', hue='is_diffpeak', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Sum of attributions (Absolute)')
# in log scale
ax.set_xlabel('Cell type')
ax.set_title('Distribution of Sum of Attributions by Cell Type and DiffPeak Status')
plt.legend(title='Is DiffPeak', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('Res/basal_ganglia_miniatlas_drop_celltype_v1/DiffPeak_attributions_violinplot.png', dpi=300)
plt.show()
# %% calculate the AUROC of sum_bin_score to distinguish positive and negative
from sklearn.metrics import roc_auc_score, average_precision_score
for celltype in res['celltype'].unique():
    res_celltype = res[res['celltype'] == celltype]
    if len(res_celltype['is_diffpeak'].unique()) < 2:
        print(f'Cell type: {celltype}, only one class present, skipping AUROC/AUPRC calculation.')
        continue
    auroc = roc_auc_score(res_celltype['is_diffpeak'], res_celltype['sum_bin_score'])
    auprc = average_precision_score(res_celltype['is_diffpeak'], res_celltype['sum_bin_score'])
    print(f'Cell type: {celltype}, AUROC: {auroc}, AUPRC: {auprc}')

# %%
