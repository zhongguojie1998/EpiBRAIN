# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()
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
    fig.tight_layout()
    save_dir = 'figures/' + ('celltype_head/' if res is bin_level_results else 'original/')
    fig.savefig(save_dir + 'PearsonR_density_by_modality.pdf')

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
    fig.tight_layout()
    fig.savefig('figures/' + ('celltype_head/' if res is bin_level_results else 'original/') + 'PearsonR_boxplot_by_modality_atlas.pdf')

# %% load gene level metrics
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_metrics.csv', index_col=0)
gene_level_results = gene_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results['modality'] = gene_level_results['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
gene_level_results_orig = pd.read_csv('Res/full_finetune_original_loss/analysis_20/gene_level/raw_data/Test_gene_metrics.csv', index_col=0)
gene_level_results_orig = gene_level_results_orig.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results_orig['modality'] = gene_level_results_orig['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})

# %% plot modality metrics box plots between atlases
for res, use_celltype_head in zip([gene_level_results, gene_level_results_orig], [True, False]):
    fig, ax = plt.subplots(figsize=(6, 4))
    res = res[res['modality'] == 'RNA']
    # calculate the average PearsonR for each modality and annotate on the legend
    modality_groups = res.groupby('modality')
    for modality, group in modality_groups:
        mean_value = group['pearsonr_log'].mean()
        modality_label = f'{modality} (avg: {mean_value:.4f})'
        # rename in the dataframe
        res.loc[res['modality'] == modality, 'modality: avg'] = modality_label
    for atlas_name, group in res.groupby('atlas_name'):
        mean_value = group['pearsonr_log'].mean()
        atlas_label = f'{atlas_name} (avg: {mean_value:.4f})'
        # rename in the dataframe
        res.loc[res['atlas_name'] == atlas_name, 'atlas_name: avg'] = atlas_label
    sns.histplot(data=res, x='pearsonr_log', hue='atlas_name: avg', ax=ax, fill=True, alpha=1, bins=20)
    if use_celltype_head:
        ax.set_title('Density plot of PearsonR on Test dataset')
    else:
        ax.set_title('Density plot of PearsonR on Test dataset (No cell type head)')
    fig.tight_layout()
    save_dir = 'figures/' + ('celltype_head/' if use_celltype_head else 'original/')
    fig.savefig(save_dir + 'Gene_level_PearsonR_density_RNA_by_atlas.pdf')


# %% get gene level raw counts
gene_pred_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_50/gene_level/raw_data/Test_gene_preds_raw.tsv', index_col=0, sep='\t')
gene_target_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_50/gene_level/raw_data/Test_gene_targets_raw.tsv', index_col=0, sep='\t')
# filter to RNA tracks
rna_tracks = gene_pred_raw.columns[gene_pred_raw.columns.str.contains('RNA')]
gene_pred_rna = gene_pred_raw[rna_tracks]
gene_target_rna = gene_target_raw[rna_tracks]
# add pseudo coverage
pseudo_qtl= 0.05
for ti in range(gene_target_rna.shape[1]):
    nonzero_index = np.nonzero(gene_target_rna.iloc[:, ti] != 0.)[0]

    pseudo_t = np.quantile(gene_target_rna.iloc[:, ti][nonzero_index], q=pseudo_qtl)
    pseudo_p = np.quantile(gene_pred_rna.iloc[:, ti][nonzero_index], q=pseudo_qtl)
    gene_target_rna.iloc[:, ti] += pseudo_t
    gene_pred_rna.iloc[:, ti] += pseudo_p
# log transform
gene_pred_rna_log = np.log1p(gene_pred_rna)
gene_target_rna_log = np.log1p(gene_target_rna)
# for each gene and cell type, calculate fold change to mean of other cell types
gene_pred_rna_log_fc = gene_pred_rna_log.copy()
gene_target_rna_log_fc = gene_target_rna_log.copy()
for celltype in rna_tracks:
    other_celltypes = [ct for ct in rna_tracks if ct != celltype]
    gene_pred_rna_log_fc[celltype] = gene_pred_rna_log[celltype] - gene_pred_rna_log[other_celltypes].mean(axis=1)
    gene_target_rna_log_fc[celltype] = gene_target_rna_log[celltype] - gene_target_rna_log[other_celltypes].mean(axis=1)
# %% pick variable genes
variable_genes = pd.read_csv('Data/source/DiffExpress/MiniAtlas_RNA_merged_dual_filt_clean_corrected_250529.var.feature', header=None)[0].tolist()
# transform variable genes to ENSG using gtf file
gtf_file = 'Data/source/gencode.v48.annotation.gtf.gz'
import gzip
variable_genes_ensg = set()
with gzip.open(gtf_file, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if fields[2] == 'gene':
            info_fields = fields[8].split('; ')
            gene_name = ''
            gene_id = ''
            for info in info_fields:
                if info.startswith('gene_name'):
                    gene_name = info.split(' ')[1].strip('"')
                elif info.startswith('gene_id'):
                    gene_id = info.split(' ')[1].strip('"')
            if gene_name in variable_genes:
                variable_genes_ensg.add(gene_id)
# filter to variable genes
gene_pred_rna_log = gene_pred_rna_log[gene_pred_rna_log.index.isin(variable_genes_ensg)]
gene_target_rna_log = gene_target_rna_log[gene_target_rna_log.index.isin(variable_genes_ensg)]
# %% quantile normalize
from qnorm import quantile_normalize
gene_pred_rna_quantile = pd.DataFrame(
    quantile_normalize(gene_pred_rna_log, ncpus=2),
    index=gene_pred_rna_log.index,
    columns=gene_pred_rna_log.columns
)
gene_target_rna_quantile = pd.DataFrame(
    quantile_normalize(gene_target_rna_log, ncpus=2),
    index=gene_target_rna_log.index,
    columns=gene_target_rna_log.columns
)
# substract mean
gene_pred_rna_quantile_centered = gene_pred_rna_quantile.sub(gene_pred_rna_quantile.mean(axis=1), axis=0)
gene_target_rna_quantile_centered = gene_target_rna_quantile.sub(gene_target_rna_quantile.mean(axis=1), axis=0)
# %% visualize some of the genes to check why
celltype = gene_pred_rna_log.columns[0]
fig, ax = plt.subplots(nrows=3, figsize=(6, 12))
sns.scatterplot(x=gene_pred_rna_log[celltype], y=gene_target_rna_log[celltype], ax=ax[0])
ax[0].set_xlabel('Predicted log1p RNA counts')
ax[0].set_ylabel('Target log1p RNA counts')
ax[0].set_title(f'Scatter plot of predicted vs target RNA counts ({celltype})')
sns.scatterplot(x=gene_pred_rna_quantile[celltype], y=gene_target_rna_quantile[celltype], ax=ax[1])
ax[1].set_xlabel('Predicted quantile normalized log1p RNA counts')
ax[1].set_ylabel('Target quantile normalized log1p RNA counts')
ax[1].set_title(f'Scatter plot of predicted vs target RNA counts ({celltype})')
sns.scatterplot(x=gene_pred_rna_log_fc[celltype], y=gene_target_rna_log_fc[celltype], ax=ax[2])
ax[2].set_xlabel('Predicted quantile normalized log1p RNA counts')
ax[2].set_ylabel('Target quantile normalized log1p RNA counts')
ax[2].set_title(f'Scatter plot of predicted vs target RNA counts ({celltype})')

# %% load differential expression meta data
diffexp = pd.read_csv('Data/source/DiffExpress/subclass_corrected_edgeR.dds')
# %% pearson correlation for each track
from scipy.stats import pearsonr
pearsonr_values = {}
for track in rna_tracks:
    pred_values = gene_pred_rna_log_fc[track]
    target_values = gene_target_rna_log_fc[track]
    corr, _ = pearsonr(pred_values, target_values)
    pearsonr_values[track] = corr
# convert to dataframe
pearsonr_df = pd.DataFrame.from_dict(pearsonr_values, orient='index', columns=['PearsonR'])
pearsonr_df = pearsonr_df.merge(cell_type_meta, left_index=True, right_on='exp')
# label the tracks with neuron or non-neuron
def label_neuron(row):
    if any([x in row['celltype'] for x in ['Astrocyte', 'Microglia', 'Oligodendrocyte', 'AST', 'MGC', 'OGC']]):
        return 'Non-Neuron'
    else:
        return 'Neuron'
pearsonr_df['cell_type_group'] = pearsonr_df.apply(label_neuron, axis=1)
# %%
# plot the distribution
fig, ax = plt.subplots(figsize=(6, 4))
# calculate the average PearsonR for each modality and annotate on the legend
modality_groups = pearsonr_df.groupby('modality')
for ct, group in pearsonr_df.groupby('cell_type_group'):
    mean_value = group['PearsonR'].mean()
    cell_type_label = f'{ct} (avg: {mean_value:.4f})'
    # rename in the dataframe
    pearsonr_df.loc[pearsonr_df['cell_type_group'] == ct, 'cell_type_group: avg'] = cell_type_label
sns.histplot(data=pearsonr_df, x='PearsonR', hue='cell_type_group: avg', ax=ax, fill=False, bins=20, alpha=1)



# %% plot the cross cell type pearsonr with regard to relative variance
import pickle
with open('Res/full_finetune_original_loss/analysis_20/raw_data/Test_metric_across_celltypes_none.pkl', 'rb') as f:
    cross_cell_type_data = pickle.load(f)
# %% plot
from scipy.stats import gaussian_kde

fill_color = {'ATAC': 'Blues', 'K27Ac': 'Oranges', 'K27Me3': 'Greens', 'K9Me3': 'Purples'}
for mod in ['ATAC', 'K27Ac', 'K27Me3', 'K9Me3']:
    data = cross_cell_type_data[mod]
    # calculate relative variance for each bin
    relative_variances = np.array(data['label_var']) / np.array(data['label_mean'])
    pearsonr_values = np.array(data['pearsonr'])

    # Remove any NaN or Inf values and filter relative variance <= 10
    valid_mask = np.isfinite(relative_variances) & np.isfinite(pearsonr_values) & (relative_variances <= 10)
    relative_variances = relative_variances[valid_mask]
    pearsonr_values = pearsonr_values[valid_mask]

    # Downsample if dataset is large (for speed)
    max_points = 10000
    if len(relative_variances) > max_points:
        sample_idx = np.random.choice(len(relative_variances), max_points, replace=False)
        relative_variances_kde = relative_variances[sample_idx]
        pearsonr_values_kde = pearsonr_values[sample_idx]
    else:
        relative_variances_kde = relative_variances
        pearsonr_values_kde = pearsonr_values

    fig, ax = plt.subplots(figsize=(6, 4))

    # Fast KDE with scipy using Scott's bandwidth (automatic and fast)
    values = np.vstack([relative_variances_kde, pearsonr_values_kde])
    kernel = gaussian_kde(values, bw_method='scott')

    # Create coarser evaluation grid for speed (50x50 instead of 100x100)
    x_min, x_max = relative_variances.min(), relative_variances.max()
    y_min, y_max = pearsonr_values.min(), pearsonr_values.max()

    # Add margins
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    xx, yy = np.mgrid[x_min-x_margin:x_max+x_margin:50j,
                      y_min-y_margin:y_max+y_margin:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Evaluate KDE on grid
    density = np.reshape(kernel(positions).T, xx.shape)

    # Plot contours (contour is faster than contourf)
    contour = ax.contour(xx, yy, density, levels=8, cmap=fill_color[mod], linewidths=1.5)

    # Add light scatter of downsampled points (much faster)
    if len(relative_variances) > 5000:
        scatter_idx = np.random.choice(len(relative_variances), 5000, replace=False)
        ax.scatter(relative_variances[scatter_idx], pearsonr_values[scatter_idx],
                   s=0.5, alpha=0.2, c='gray', rasterized=True)
    else:
        ax.scatter(relative_variances, pearsonr_values, s=0.5, alpha=0.2, c='gray', rasterized=True)

    # Add colorbar
    cb = plt.colorbar(contour, ax=ax)
    cb.set_label('Density')

    ax.set_xlabel('Relative Variance (Variance / Mean)')
    ax.set_ylabel('PearsonR across Cell Types')
    ax.set_title(f'Cross Cell Type PearsonR vs Relative Variance ({mod})')
# %%
