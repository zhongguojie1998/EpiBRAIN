# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
PWD = f'{os.environ["workingHOME"]}/BICAN'
import sys
os.chdir(PWD)
sys.path.append(PWD)
# %%
# load the bin level results
bin_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/raw_data/Test_metric.csv', index_col=0)
bin_level_results_orig = pd.read_csv('Res/full_finetune_original_loss/analysis_20/raw_data/Test_metric.csv', index_col=0)
# get the number of cells from meta data
cell_type_meta = pd.read_csv('Data/data_config/basal_ganglia_miniatlas_drop_celltype_v1.csv', index_col=0)
# annotate the bin level results with cell type information
bin_level_results = bin_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
bin_level_results_orig = bin_level_results_orig.merge(cell_type_meta, left_on='trial', right_on='exp')
# rename modality of RNA strands
bin_level_results['modality'] = bin_level_results['modality'].replace(
    {'RNAplus': 'RNA-', 'RNAminus': 'RNA+', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
bin_level_results_orig['modality'] = bin_level_results_orig['modality'].replace(
    {'RNAplus': 'RNA-', 'RNAminus': 'RNA+', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})


# %% plot the density plots of PearsonR for each modality and atlas
for res in [bin_level_results, bin_level_results_orig]:
    fig, ax = plt.subplots(figsize=(6, 4))
    # calculate the average PearsonR for each modality and annotate on the legend
    modality_groups = res.groupby('modality')
    for modality, group in modality_groups:
        mean_value = group['PearsonR'].mean()
        modality_label = f'{modality} (avg: {mean_value:.4f})'
        # rename in the dataframe
        res.loc[res['modality'] == modality, 'modality: avg'] = modality_label
    sns.kdeplot(data=res, x='PearsonR', hue='modality: avg', ax=ax, fill=False, common_norm=False, alpha=1)
    # set x limit from 0 to 1
    ax.set_xlim(0, 1)
    if res is bin_level_results:
        ax.set_title('Density plot of PearsonR on Test dataset')
    else:
        ax.set_title('Density plot of PearsonR on Test dataset (No cell type head)')
    plt.tight_layout()


# %% plot modality metrics box plots between atlases
for res in [bin_level_results, bin_level_results_orig]:
    fig, ax = plt.subplots(figsize=(6, 4))
    # set the box thinner, drop outliers, smaller fliersize
    sns.boxplot(data=res, x='modality', y='PearsonR', hue='atlas_name', 
                ax=ax, width=0.4, showfliers=False)
    # set y limit from 0 to 1
    ax.set_ylim(0, 1)
    # annotate average values on top of each box
    modality_atlas_groups = res.groupby(['modality', 'atlas_name'])
    for (modality, atlas_name), group in modality_atlas_groups:
        mean_value = group['PearsonR'].mean()
        max_value = group['PearsonR'].max()
        ax.text(x=list(res['modality'].unique()).index(modality) + 
                (0.2 if atlas_name == 'MiniAtlas' else -0.2), 
                y=max_value * 1.05, s=f'{mean_value:.2f}', 
                ha='center', va='bottom', fontsize=8, color='black')
    plt.legend(title='Atlas Name', loc='lower left')
    if res is bin_level_results:
        ax.set_title('PearsonR by Modality and Atlas')
    else:
        ax.set_title('PearsonR by Modality and Atlas (No cell type head)')
    plt.tight_layout()


# %% load gene level metrics
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_metrics.csv', index_col=0)
gene_level_results = gene_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results['modality'] = gene_level_results['modality_x'].replace(
    {'RNAplus': 'RNA-', 'RNAminus': 'RNA+', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})

# %% plot gene level density
fig, ax = plt.subplots(figsize=(6, 4))
sns.kdeplot(data=gene_level_results, x='pearsonr', hue='modality', ax=ax, fill=False, common_norm=False, alpha=1)

# %%
