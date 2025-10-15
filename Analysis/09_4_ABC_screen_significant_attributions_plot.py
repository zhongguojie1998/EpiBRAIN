# %%
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %%
res = pd.read_csv('Res/basal_ganglia_miniatlas_drop_celltype_v1/ABC_attributions_screened.tsv', sep='\t', index_col=None)
# %% calculate mean in each cell type in two groups
for celltype in res['celltype'].unique():
    res_celltype = res[res['celltype'] == celltype]
    mean_positive = res_celltype[res_celltype['is_abc'] == True]['sum_score'].mean()
    mean_negative = res_celltype[res_celltype['is_abc'] == False]['sum_score'].mean()
    print(f'Cell type: {celltype}, Mean Positive: {mean_positive}, Mean Negative: {mean_negative}, Difference: {mean_positive - mean_negative}')
# %% for each pt_file, perform z-score normalization of sum_score
res['sum_bin_score_z'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: (x.abs() - x.abs().mean()) / x.abs().std())
res['sum_bin_score_abs'] = res.groupby('pt_file')['sum_bin_score'].transform(lambda x: x.abs())
# %%
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=res, x='celltype', y='sum_bin_score_abs', hue='is_abc', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Sum of attributions (Absolute)')
# in log scale
ax.set_yscale('log')
ax.set_xlabel('Cell type')
ax.set_title('Distribution of Sum of Attributions by Cell Type and ABC Status')
plt.legend(title='Is ABC', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('Res/basal_ganglia_miniatlas_drop_celltype_v1/ABC_attributions_violinplot.png', dpi=300)
plt.show()
# %%
