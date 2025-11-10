# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
from scipy.stats import pearsonr
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
cell_type_meta.reset_index(inplace=True, drop=True)
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






# %% Section 2
# ================================================================================
# load gene level metrics
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
gene_pred_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_preds_raw.tsv', index_col=0, sep='\t')
gene_target_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_targets_raw.tsv', index_col=0, sep='\t')
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
# remove genes that low expressed in all cell types (average log count in target < 0.1)
mean_log_counts = gene_target_rna_log.mean(axis=1)
genes_to_keep = mean_log_counts[mean_log_counts >= 1].index
gene_pred_rna_log = gene_pred_rna_log.loc[genes_to_keep]
gene_target_rna_log = gene_target_rna_log.loc[genes_to_keep]
# %% pick variable genes or differentially expressed genes
variable_genes = pd.read_csv('Data/source/DiffExpress/MiniAtlas_RNA_merged_dual_filt_clean_corrected_250529.var.feature', header=None)[0].tolist()
diffexp = pd.read_csv('Data/source/DiffExpress/subclass_corrected_edgeR.dds')
def transform_gene_name_to_ensg(gene_names):
    # transform variable genes to ENSG using gtf file
    gtf_file = 'Data/source/gencode.v48.annotation.gtf.gz'
    gene_ensgs = {}
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
                if gene_name in gene_names:
                    gene_ensgs[gene_name] = gene_id
                elif gene_id in gene_names:
                    gene_ensgs[gene_id] = gene_id
    return gene_ensgs
# filter to variable genes
variable_genes_ensg_dict = transform_gene_name_to_ensg(variable_genes)
diffexp_genes_ensg_dict = transform_gene_name_to_ensg(diffexp['gene'].tolist())
# %% get variable genes ensg and diffexp genes ensg
genes_to_use = 'all'  # options: 'variable' or 'diffexp' or 'all'
if genes_to_use == 'variable':
    variable_genes_ensg = list(variable_genes_ensg_dict.values())
    gene_pred_rna_log = gene_pred_rna_log[gene_pred_rna_log.index.isin(variable_genes_ensg)]
    gene_target_rna_log = gene_target_rna_log[gene_target_rna_log.index.isin(variable_genes_ensg)]
elif genes_to_use == 'diffexp':
    diffexp_genes_ensg = list(diffexp_genes_ensg_dict.values())
    gene_pred_rna_log = gene_pred_rna_log[gene_pred_rna_log.index.isin(diffexp_genes_ensg)]
    gene_target_rna_log = gene_target_rna_log[gene_target_rna_log.index.isin(diffexp_genes_ensg)]
    diffexp_filtered = diffexp[diffexp['gene'].isin(diffexp_genes_ensg_dict.keys())]
    # and we filter the tracks to MiniAtlas only
    miniatlas_tracks = [ct for ct in rna_tracks if 'MiniAtlas' in ct]
    gene_pred_rna_log = gene_pred_rna_log[miniatlas_tracks]
    gene_target_rna_log = gene_target_rna_log[miniatlas_tracks]
    rna_tracks = miniatlas_tracks
# %% calculate log fc
# for each gene and cell type, calculate fold change to mean of other cell types
gene_pred_rna_log_fc = gene_pred_rna_log.copy()
gene_target_rna_log_fc = gene_target_rna_log.copy()
for celltype in rna_tracks:
    other_celltypes = [ct for ct in rna_tracks if ct != celltype]
    gene_pred_rna_log_fc[celltype] = gene_pred_rna_log[celltype] - gene_pred_rna_log[other_celltypes].mean(axis=1)
    gene_target_rna_log_fc[celltype] = gene_target_rna_log[celltype] - gene_target_rna_log[other_celltypes].mean(axis=1)
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

# %% pearson correlation for each track
transform_to_use = 'quantile_centered'  # options: 'log_fc' or 'quantile_centered'
pearsonr_values = {}
for track in rna_tracks:
    if transform_to_use == 'quantile_centered':
        gene_pred_track = gene_pred_rna_quantile_centered[track]
        gene_target_track = gene_target_rna_quantile_centered[track]
    elif transform_to_use == 'log_fc':
        gene_pred_track = gene_pred_rna_log_fc[track]
        gene_target_track = gene_target_rna_log_fc[track]
    if genes_to_use == 'diffexp':
        # filter to diffexp genes for that cell type only
        celltype_name = track.replace('MiniAtlas-', '').split('_')[0]
        diff_genes = diffexp_filtered[diffexp_filtered['celltype'] == celltype_name]
        diff_genes_ensg = [diffexp_genes_ensg_dict[gene] for gene in diff_genes['gene'] if gene in diffexp_genes_ensg_dict]
        gene_pred_track = gene_pred_track[gene_pred_track.index.isin(diff_genes_ensg)]
        gene_target_track = gene_target_track[gene_target_track.index.isin(diff_genes_ensg)]
    corr, _ = pearsonr(gene_pred_track, gene_target_track)
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

# %% calculate the correlation for each gene across all tracks
gene_pearsonr_values = {}
for gene in gene_pred_rna_log_fc.index:
    if transform_to_use == 'quantile_centered':
        gene_pred_values = gene_pred_rna_quantile_centered.loc[gene]
        gene_target_values = gene_target_rna_quantile_centered.loc[gene]
    elif transform_to_use == 'log_fc':
        gene_pred_values = gene_pred_rna_log_fc.loc[gene]
        gene_target_values = gene_target_rna_log_fc.loc[gene]
    else:
        gene_pred_values = gene_pred_rna_log.loc[gene]
        gene_target_values = gene_target_rna_log.loc[gene]
    corr, _ = pearsonr(gene_pred_values, gene_target_values)
    # also calculate the mean expression level and variance
    gene_mean_expression = gene_target_values.mean()
    gene_variance_expression = gene_target_values.var()
    gene_pearsonr_values[gene] = {
        'pearsonr': corr,
        'mean': gene_mean_expression,
        'variance': gene_variance_expression
    }

# %% visualize one track
track_to_visualize = 'MiniAtlas-PV-CHC_RNAminus'
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=gene_pred_rna_log_fc[track_to_visualize], y=gene_target_rna_log_fc[track_to_visualize], s=1, alpha=0.5)
ax.set_xlabel('Predicted LogFC')
ax.set_ylabel('True LogFC')
ax.set_title(f'Gene LogFC Prediction for {track_to_visualize}')







# %% Section 3
# ================================================================================
# plot the cross cell type pearsonr in atac peaks with regard to relative variance
import pickle
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_merged_all_peaks/raw_data/Test_aggregated_label_bed.pkl', 'rb') as f:
    cross_cell_type_label = pickle.load(f)
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_merged_all_peaks/raw_data/Test_aggregated_pred_bed.pkl', 'rb') as f:
    cross_cell_type_pred = pickle.load(f)
# read the peak bed
peak_bed = pd.read_csv('Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed', sep='\t', header=None)
# drop nan lines
valid_indices = ~np.isnan(cross_cell_type_label).any(axis=1) & ~np.isnan(cross_cell_type_pred).any(axis=1)
cross_cell_type_label = cross_cell_type_label[valid_indices]
cross_cell_type_pred = cross_cell_type_pred[valid_indices]
peak_bed = peak_bed.iloc[valid_indices, :].reset_index(drop=True)
# %% read abc links
abc_links = pd.read_csv('Data/source/ABC/broad_abc_filtcelltype_conns.txt', sep='\t')
abc_links['peak_id'] = abc_links['chr'] + ':' + abc_links['start'].astype(str) + '-' + abc_links['end'].astype(str)
peak_bed['peak_id'] = peak_bed[0].astype(str) + ':' + peak_bed[1].astype(str) + '-' + peak_bed[2].astype(str)
peak_bed['is_abc_linked'] = peak_bed['peak_id'].isin(abc_links['peak_id'])
# %% calculate pearsonr across cell types for each modality for each peak, and filter to MiniAtlas only
cross_cell_type_data = {}
for mod in ['ATAC', 'K27Ac', 'K27Me3']:
    dims = cell_type_meta.index[(cell_type_meta['modality'] == mod) & (cell_type_meta['atlas_name'] == 'MiniAtlas')].tolist()
    labels = cross_cell_type_label[:, dims]
    preds = cross_cell_type_pred[:, dims]
    # log to get log_cpm
    labels = np.log1p(labels)
    preds = np.log1p(preds)
    # calculate mean and variance for each peak
    label_means = np.mean(labels, axis=1)
    label_vars = np.var(labels, axis=1)
    preds_means = np.mean(preds, axis=1)
    preds_vars = np.var(preds, axis=1)

    # Vectorized Pearson correlation calculation
    # Center the data
    labels_centered = labels - label_means[:, np.newaxis]
    preds_centered = preds - preds_means[:, np.newaxis]

    # Compute numerator (covariance)
    covariance = (labels_centered * preds_centered).sum(axis=1)

    # Compute denominators (standard deviations)
    labels_std = np.sqrt((labels_centered ** 2).sum(axis=1))
    preds_std = np.sqrt((preds_centered ** 2).sum(axis=1))

    # Pearson correlation
    pearsonr_array = covariance / (labels_std * preds_std)

    cross_cell_type_data[mod] = {
        'label_mean': label_means,
        'label_var': label_vars,
        'label_coeff_var': label_vars / label_means,
        'pearsonr': pearsonr_array,
        'is_abc_linked': peak_bed['is_abc_linked'].values
    }
# %% plot scatter plots
data_together = None
for mod, cmap in zip(['ATAC', 'K27Ac', 'K27Me3'], ['Blues', 'Oranges', 'Greens']):
    data = pd.DataFrame(cross_cell_type_data[mod])
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    # histogram plot
    # sns.histplot(data['pearsonr'], bins=30, kde=False, ax=ax, color='blue', alpha=0.5)
    # draw kde plot
    sns.kdeplot(x=data['label_coeff_var'][data['is_abc_linked']], y=data['pearsonr'][data['is_abc_linked']], 
                ax=ax[0], fill=True, cmap=cmap, levels=20, thresh=0.05)
    ax[0].set_xlabel('Coefficient Variance across Cell Types')
    ax[0].set_ylabel('PearsonR across Cell Types')
    ax[0].set_title(f'MiniAtlas {mod} (ABC linked peaks)')
    # abc non linked
    sns.kdeplot(x=data['label_coeff_var'][~data['is_abc_linked']], y=data['pearsonr'][~data['is_abc_linked']], 
                ax=ax[1], fill=True, cmap=cmap, levels=20, thresh=0.05)
    ax[1].set_xlabel('Coefficient Variance across Cell Types')
    ax[1].set_ylabel('PearsonR across Cell Types')
    ax[1].set_title(f'MiniAtlas {mod} (Non-ABC linked peaks)')
    fig.tight_layout()
    fig.savefig('figures/celltype_head/MiniAtlas_cross_cell_type_pearsonr_vs_coeff_var_' + mod + '.pdf')
    # add modality column for later combined plotting
    data['modality'] = mod
    if data_together is None:
        data_together = data
    else:
        data_together = pd.concat([data_together, data], axis=0)
# %% plot together
fig, ax = plt.subplots(figsize=(6, 4))
data_toplot = data_together[data_together['label_coeff_var'] > 1]
# Create palette matching the cmaps from line 304
palette = {'ATAC': sns.color_palette('Blues', n_colors=10)[6],
           'K27Ac': sns.color_palette('Oranges', n_colors=10)[6],
           'K27Me3': sns.color_palette('Greens', n_colors=10)[6]}
# plot histogram with kde
sns.kdeplot(data_toplot, x='pearsonr', hue='modality', ax=ax, palette=palette, fill=True, alpha=0.5)
ax.set_xlabel('PearsonR across Cell Types (Coeff Var > 1)')

# Calculate and display average for each modality
y_text_pos = 0.95
for i, (mod, color) in enumerate(zip(['ATAC', 'K27Ac', 'K27Me3'],
                                      [palette['ATAC'], palette['K27Ac'], palette['K27Me3']])):
    mod_data = data_toplot[data_toplot['modality'] == mod]
    mean_val = mod_data['pearsonr'].mean()
    ax.text(0.02, y_text_pos - i*0.08, f'{mod}: μ = {mean_val:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color=color, fontweight='bold')

fig.savefig('figures/celltype_head/MiniAtlas_cross_cell_type_pearsonr_kde_coeff_var_all_modalities.pdf')

# %% plot abc linked vs non linked boxplot
data_toplot['label'] = data_toplot['modality'] + ":" + data_toplot['is_abc_linked'].astype(str)
palette = {f'ATAC:False': sns.color_palette('Blues', n_colors=10)[6],
           f'ATAC:True': sns.color_palette('Blues', n_colors=10)[9],
           f'K27Ac:False': sns.color_palette('Oranges', n_colors=10)[6],
           f'K27Ac:True': sns.color_palette('Oranges', n_colors=10)[9],
           f'K27Me3:False': sns.color_palette('Greens', n_colors=10)[6],
           f'K27Me3:True': sns.color_palette('Greens', n_colors=10)[9]}
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 9), sharex=True)
y_text_pos = 0.95
for i, mod in enumerate(['ATAC', 'K27Ac', 'K27Me3']):
    mod_data = data_toplot[data_toplot['modality'] == mod]
    sns.kdeplot(data=mod_data, x='pearsonr', ax=ax[i], fill=True, alpha=0.5, hue='label',
                palette=palette)
    # print average values
    for j, (is_linked, group) in enumerate(mod_data.groupby('is_abc_linked')):
        mean_value = group['pearsonr'].mean()
        ax[i].text(0.02, y_text_pos - j*0.08, f'{is_linked}: μ = {mean_value:.3f}',
                   transform=ax[i].transAxes, fontsize=10, verticalalignment='top',
                   color=palette[f'{mod}:{is_linked}'], fontweight='bold')
    ax[i].set_title(f'MiniAtlas Cross Cell Types ({mod})')
fig.tight_layout()
fig.savefig('figures/celltype_head/MiniAtlas_cross_cell_type_pearsonr_kde_coeff_var_abc_linked_vs_nonlinked.pdf')

# %% peaks minus average then pearsonr
cross_cell_type_label_centered = cross_cell_type_label[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'MiniAtlas')].tolist()]
cross_cell_type_pred_centered = cross_cell_type_pred[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'MiniAtlas')].tolist()]
# log transform
cross_cell_type_pred_centered = np.log1p(cross_cell_type_pred_centered)
cross_cell_type_label_centered = np.log1p(cross_cell_type_label_centered)
# substract mean
cross_cell_type_pred_centered = cross_cell_type_pred_centered - cross_cell_type_pred_centered.mean(axis=1, keepdims=True)
cross_cell_type_label_centered = cross_cell_type_label_centered - cross_cell_type_label_centered.mean(axis=1, keepdims=True)
cell_type_meta_miniatlas = cell_type_meta[cell_type_meta['atlas_name'] == 'MiniAtlas'].reset_index(drop=True)
# recalculate pearsonr
valid_indices = ~np.isnan(cross_cell_type_label_centered).any(axis=1) & ~np.isnan(cross_cell_type_pred_centered).any(axis=1)
cross_cell_type_label_centered = cross_cell_type_label_centered[valid_indices]
cross_cell_type_pred_centered = cross_cell_type_pred_centered[valid_indices]
# for each track, do pearsonr
pearsonr_values = {}
for track in cell_type_meta_miniatlas['exp']:
    gene_pred_track = cross_cell_type_pred_centered[:, cell_type_meta_miniatlas.index[cell_type_meta_miniatlas['exp'] == track][0]]
    gene_target_track = cross_cell_type_label_centered[:, cell_type_meta_miniatlas.index[cell_type_meta_miniatlas['exp'] == track][0]]
    corr, _ = pearsonr(gene_pred_track, gene_target_track)
    pearsonr_values[track] = corr
# convert to dataframe
pearsonr_df = pd.DataFrame.from_dict(pearsonr_values, orient='index', columns=['PearsonR'])
pearsonr_df = pearsonr_df.merge(cell_type_meta_miniatlas, left_index=True, right_on='exp')
# drop RNA modalities
pearsonr_df = pearsonr_df[(pearsonr_df['modality'] != 'RNAminus') & (pearsonr_df['modality'] != 'RNAplus')]
# %% plot the distribution
fig, ax = plt.subplots(figsize=(6, 4))
# calculate the average PearsonR for each modality and annotate on the legend
modality_groups = pearsonr_df.groupby('modality')
for mod, group in modality_groups:
    mean_value = group['PearsonR'].mean()
    modality_label = f'{mod} (avg: {mean_value:.4f})'
    # rename in the dataframe
    pearsonr_df.loc[pearsonr_df['modality'] == mod, 'modality: avg'] = modality_label
sns.kdeplot(data=pearsonr_df, x='PearsonR', hue='modality: avg', ax=ax, fill=True, alpha=0.5)
ax.set_xlabel('PearsonR across Cell Types (Centered)')
fig.savefig('figures/celltype_head/MiniAtlas_cross_cell_type_pearsonr_kde_log_centered_all_modalities.pdf')