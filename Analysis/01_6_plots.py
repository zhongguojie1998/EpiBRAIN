# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
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
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results_orig = pd.read_csv('Res/full_finetune_original_loss/analysis_20/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
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
# bin_level_results = bin_level_results[bin_level_results['atlas_name'] == 'MiniAtlas']
# bin_level_results_orig = bin_level_results_orig[bin_level_results_orig['atlas_name'] == 'MiniAtlas']


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

# %% plot modality metrics box plots between atlases - COMBINED SUBPLOT VERSION
# This will be filled in after loading gene-level data

# %% plot comparison between cell type head model and original model
fig, ax = plt.subplots(figsize=(6, 4))
bin_level_results['model'] = 'Cell Type Head'
bin_level_results_orig['model'] = 'No Cell Type Head'
res = pd.concat([bin_level_results, bin_level_results_orig], axis=0)
sns.boxplot(data=res, x='modality', y='PearsonR', hue='model', ax=ax, width=0.4, showfliers=False)
modality_model_groups = res.groupby(['modality', 'model'])
for (modality, model), group in modality_model_groups:
    mean_value = group['PearsonR'].mean()
    max_value = group['PearsonR'].max()
    ax.text(x=list(res['modality'].unique()).index(modality) + (-0.2 if model == 'Cell Type Head' else 0.2), 
            y=max_value * 1.05, s=f'{mean_value:.2f}', 
            ha='center', va='bottom', fontsize=8, color='black')
# set y limit from 0 to 1
ax.set_ylim(0, 1)
ax.set_title('Comparison of Cell Type Head and No Cell Type Head Models')
fig.tight_layout()
fig.savefig('figures/Comparison_PearsonR_boxplot_celltype_head_vs_original.pdf')

# %% plot relative change (%) per modality: violin + jitter + box
bin_merged = bin_level_results[['exp', 'modality', 'PearsonR']].merge(
    bin_level_results_orig[['exp', 'modality', 'PearsonR']],
    on=['exp', 'modality'], suffixes=('_new', '_orig')
)
bin_merged['rel_change'] = (bin_merged['PearsonR_new'] - bin_merged['PearsonR_orig']) / bin_merged['PearsonR_orig'] * 100

fig, ax = plt.subplots(figsize=(6, 4))
modality_order = sorted(bin_merged['modality'].unique())
sns.violinplot(data=bin_merged, x='modality', y='rel_change', order=modality_order,
               ax=ax, inner=None, color='lightgray', linewidth=1)
sns.boxplot(data=bin_merged, x='modality', y='rel_change', order=modality_order,
            ax=ax, width=0.15, showfliers=False, boxprops=dict(zorder=2),
            medianprops=dict(color='black'), whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1))
sns.stripplot(data=bin_merged, x='modality', y='rel_change', order=modality_order,
              ax=ax, size=3, alpha=0.6, jitter=True, color='steelblue', zorder=3)
ax.axhline(0, color='black', lw=1, linestyle='--')
ax.set_xlabel('Modality')
ax.set_ylabel('Relative Change (%)')
ax.set_title('Relative PearsonR Change: Cell Type Head vs Original')
fig.tight_layout()
fig.savefig('figures/Comparison_PearsonR_rel_change_violin_celltype_head_vs_original.pdf')




# %% Section 2
# ================================================================================
# load gene level metrics
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results = gene_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results['modality'] = gene_level_results['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
gene_level_results_orig = pd.read_csv('Res/full_finetune_original_loss/analysis_20/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results_orig = gene_level_results_orig.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results_orig['modality'] = gene_level_results_orig['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
gene_level_results_atac = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear_atac_only/analysis_15/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results_atac = gene_level_results_atac.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results_atac['modality'] = gene_level_results_atac['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
# create barplot with reversed colors
palette_reversed = {'BasalGanglia': sns.color_palette()[0], 'MiniAtlas': sns.color_palette()[1]}

# %% plot modality metrics bar plots by cell type
for res, use_celltype_head in zip([gene_level_results, gene_level_results_orig], [True, False]):
    fig, ax = plt.subplots(figsize=(8, 4))
    res = res[res['modality'] == 'RNA']
    # calculate the average PearsonR for each cell type
    celltype_groups = res.groupby('celltype')['pearsonr_log'].mean().reset_index()
    celltype_groups = celltype_groups.sort_values('pearsonr_log', ascending=False)
    celltype_groups['celltype'] = celltype_groups['celltype'].str.replace('_RNAminus', '').str.replace('_RNAplus', '')
    celltype_groups['atlas_name'] = celltype_groups['celltype'].apply(
        lambda x: 'BasalGanglia' if 'BasalGanglia' in x else 'MiniAtlas')
    sns.barplot(data=celltype_groups, x='celltype', y='pearsonr_log', hue='atlas_name', ax=ax, palette=palette_reversed)

    # rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Mean PearsonR')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # add text showing overall average
    avg_pearsonr = celltype_groups['pearsonr_log'].mean()
    ax.text(0.02, 0.98, f'Overall Average: {avg_pearsonr:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    if use_celltype_head:
        ax.set_title('Gene-level PearsonR by Cell Type (RNA)')
    else:
        ax.set_title('Gene-level PearsonR by Cell Type (RNA, No cell type head)')
    fig.tight_layout()
    save_dir = 'figures/' + ('celltype_head/' if use_celltype_head else 'original/')
    fig.savefig(save_dir + 'Gene_level_PearsonR_barplot_RNA_by_celltype.pdf')
    fig, ax = plt.subplots(figsize=(2, 3))
    sns.boxplot(data=res, x='atlas_name', y='pearsonr_log', ax=ax, width=0.4, hue='atlas_name', palette=palette_reversed, showfliers=False)
    fig.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.savefig(save_dir + 'Gene_level_PearsonR_boxplot_RNA_by_atlas.pdf')
# %% comparison of gene level metrics between atac-only and full model
from scipy.stats import ttest_rel

res_full_model = gene_level_results[gene_level_results['modality'] == 'RNA']
res_atac_only = gene_level_results_atac[gene_level_results_atac['modality'] == 'RNA']
# merge on celltype and gene
merged_res = res_full_model.merge(res_atac_only, on=['celltype', 'atlas_name'], suffixes=('_full', '_atac_only'))

# ============================================================================
# Separate scatter plot - saved individually
# ============================================================================
fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=merged_res, y='pearsonr_log_full', x='pearsonr_log_atac_only',
                ax=ax_scatter, hue='atlas_name', palette=palette_reversed, s=50, alpha=0.7)
# add diagonal line
max_val = max(merged_res['pearsonr_log_full'].max(), merged_res['pearsonr_log_atac_only'].max())
min_val = min(merged_res['pearsonr_log_full'].min(), merged_res['pearsonr_log_atac_only'].min())
ax_scatter.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
ax_scatter.set_ylabel('Gene-level PearsonR (Full Model)')
ax_scatter.set_xlabel('Gene-level PearsonR (ATAC-only Model)')
ax_scatter.set_title('ATAC-only vs Full Model')
fig_scatter.tight_layout()
fig_scatter.savefig('figures/Gene_level_PearsonR_comparison_ATAC_vs_Full_model.pdf')

# %%
# bar plot of relative change per cell type (Full vs ATAC-only)
merged_res['rel_change'] = (merged_res['pearsonr_log_full'] - merged_res['pearsonr_log_atac_only']) / merged_res['pearsonr_log_atac_only'] * 100
merged_res_sorted = merged_res.sort_values('rel_change', ascending=False)
celltype_labels = merged_res_sorted['celltype'].str.replace('BasalGanglia-', '', regex=False)

fig, ax = plt.subplots(figsize=(10, 4))
colors = [palette_reversed.get(a, 'gray') for a in merged_res_sorted['atlas_name']]
ax.bar(range(len(merged_res_sorted)), merged_res_sorted['rel_change'], color=colors)
ax.set_xticks(range(len(merged_res_sorted)))
ax.set_xticklabels(celltype_labels, rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='black', lw=1, linestyle='--')
ax.set_xlabel('Cell Type')
ax.set_ylabel('Relative Change (%)')
ax.set_title('Gene-level PearsonR Relative Change: Full vs ATAC-only Model')
fig.tight_layout()
fig.savefig('figures/Gene_level_PearsonR_rel_change_barplot_Full_vs_ATAC_by_celltype.pdf')

# %%
# ============================================================================
# Combined figure with 2 subplots (plot 1 wider than plot 3)
# ============================================================================
merged_res = merged_res[merged_res['atlas_name'] == 'BasalGanglia']
fig, axes = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [2.5, 1]})

# ----------------------------------------------------------------------------
# Subplot 1: Half-boxplot/half-violin plot (wider)
# ----------------------------------------------------------------------------
ax = axes[0]
res = bin_level_results

modalities = res['modality'].unique()
# Add one more position for gene-level full model data
positions = np.arange(len(modalities) + 1)

# Prepare box data: bin-level by modality + gene-level full model
box_data = [res[res['modality'] == mod]['PearsonR'].values for mod in modalities]
# Add gene-level full model data from subplot 3
box_data.append(merged_res['pearsonr_log_full'].values)

# Prepare labels - simple modality names
labels = list(modalities) + ['RNA']

bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=1.5),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1))

# Modify boxplot to show only left half
for i, box in enumerate(bp['boxes']):
    center = positions[i]
    path = box.get_path()
    vertices = path.vertices
    vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, center)

# Clip whiskers and caps to left half
for line in bp['whiskers'] + bp['caps']:
    xdata = line.get_xdata()
    center = np.mean(xdata)
    line.set_xdata(np.clip(xdata, -np.inf, center))

# Keep median lines on the left half
for i, line in enumerate(bp['medians']):
    center = positions[i // 1]
    xdata = line.get_xdata()
    line.set_xdata(np.clip(xdata, -np.inf, center))

# Create violin plot
violin_parts = ax.violinplot(box_data, positions=positions,
                              widths=0.5, showmeans=False,
                              showmedians=False, showextrema=False)

# Modify violin patches to only show right half
for pc in violin_parts['bodies']:
    m = np.mean(pc.get_paths()[0].vertices[:, 0])
    vertices = pc.get_paths()[0].vertices
    vertices[:, 0] = np.clip(vertices[:, 0], m, np.inf)
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.6)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel('modality')
ax.set_ylabel('PearsonR')
ax.set_ylim(0, 1)

# annotate average values on top of each box
# For bin-level modalities
modality_atlas_groups = res.groupby(['modality', 'atlas_name'])
for (modality, atlas_name), group in modality_atlas_groups:
    mean_value = group['PearsonR'].mean()
    max_value = group['PearsonR'].max()
    ax.text(x=list(modalities).index(modality) +
            (0.2 if atlas_name == 'MiniAtlas' else 0),
            y=max_value * 1.05, s=f'{mean_value:.2f}',
            ha='center', va='bottom', fontsize=8, color='black')

# For gene-level full model
gene_mean = merged_res['pearsonr_log_full'].mean()
gene_max = merged_res['pearsonr_log_full'].max()
ax.text(x=len(modalities), y=gene_max * 1.05, s=f'{gene_mean:.2f}',
        ha='center', va='bottom', fontsize=8, color='black')

# Add shared annotation lines for grouping
y_annot = 0.3  # y position for annotation lines

# Line and annotation for 32-bp bin level (first 6 boxes)
ax.plot([positions[0] - 0.3, positions[len(modalities)-1] + 0.3],
        [y_annot, y_annot], 'k-', linewidth=1.5)
ax.text((positions[0] + positions[len(modalities)-1]) / 2, y_annot - 0.02,
        '32 bp\nbin level', ha='center', va='top', fontsize=9, fontweight='bold')

# Line and annotation for gene-level (last box)
ax.plot([positions[len(modalities)] - 0.3, positions[len(modalities)] + 0.3],
        [y_annot, y_annot], 'k-', linewidth=1.5)
ax.text(positions[len(modalities)], y_annot - 0.02,
        'Gene\nlevel', ha='center', va='top', fontsize=9, fontweight='bold')

# Adjust y-axis limit
ax.set_ylim(0, 1)

ax.set_title('PearsonR in Testing Data')

# ----------------------------------------------------------------------------
# Subplot 2: Boxplot with paired t-test (narrower)
# ----------------------------------------------------------------------------
ax = axes[1]
data_to_plot = pd.DataFrame({
    'Full Model': merged_res['pearsonr_log_full'],
    'ATAC-only Model': merged_res['pearsonr_log_atac_only']
})
sns.boxplot(data=data_to_plot, ax=ax, width=0.4, showfliers=False)
ax.set_ylabel('Gene-level PearsonR')
# perform paired t-test
t_stat, p_value = ttest_rel(merged_res['pearsonr_log_full'], merged_res['pearsonr_log_atac_only'])
ax.text(0.5, 0.15, f'Paired t-test\np-value: {p_value:.2e}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')
ax.set_title('Model Comparison')

# Save combined figure
fig.tight_layout()
fig.savefig('figures/Combined_PearsonR_analysis.pdf')

# %% get gene level raw counts
gene_pred_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_preds_raw_length.tsv', index_col=0, sep='\t')
gene_target_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_targets_raw_length.tsv', index_col=0, sep='\t')
# filter to RNA tracks
rna_tracks = gene_pred_raw.columns[gene_pred_raw.columns.str.contains('RNA')]
basalganglia_tracks = [c for c in rna_tracks if 'BasalGanglia' in c]
miniatlas_tracks = [c for c in rna_tracks if 'MiniAtlas' in c]
gene_pred_rna = gene_pred_raw[rna_tracks]
gene_target_rna = gene_target_raw[rna_tracks]
# add pseudo coverage
def add_pseudo_coverage(gene_rna, pseudo_qtl=0.05):
    for ti in range(gene_rna.shape[1]):
        nonzero_index = np.nonzero(gene_rna.iloc[:, ti] != 0.)[0]
        pseudo_t = np.quantile(gene_rna.iloc[:, ti][nonzero_index], q=pseudo_qtl)
        gene_rna.iloc[:, ti] += pseudo_t
    return gene_rna
gene_target_rna = add_pseudo_coverage(gene_target_rna, pseudo_qtl=0.05)
gene_pred_rna = add_pseudo_coverage(gene_pred_rna, pseudo_qtl=0.05)
# log transform
gene_pred_rna_log = np.log1p(gene_pred_rna)
gene_target_rna_log = np.log1p(gene_target_rna)
# # remove genes that low expressed in all cell types (average log count in target < 0.1)
# mean_log_counts = gene_target_rna_log.mean(axis=1)
# genes_to_keep = mean_log_counts[mean_log_counts >= 1].index
# gene_pred_rna_log = gene_pred_rna_log.loc[genes_to_keep]
# gene_target_rna_log = gene_target_rna_log.loc[genes_to_keep]
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
gene_pred_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_preds_raw_rpkm.tsv', index_col=0, sep='\t')
gene_target_raw = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/gene_level/raw_data/Test_gene_targets_raw_rpkm.tsv', index_col=0, sep='\t')
# filter to RNA tracks
rna_tracks = gene_pred_raw.columns[gene_pred_raw.columns.str.contains('RNA')]
basalganglia_tracks = [c for c in rna_tracks if 'BasalGanglia' in c]
miniatlas_tracks = [c for c in rna_tracks if 'MiniAtlas' in c]
gene_pred_rna = gene_pred_raw[basalganglia_tracks]
gene_target_rna = gene_target_raw[basalganglia_tracks]
# log transform
gene_pred_rna_log = np.log(gene_pred_rna)
gene_target_rna_log = np.log(gene_target_rna)
gene_pearsonr_values = {}
for gene in gene_pred_rna_log.index:
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
# plot
gene_pearsonr_df = pd.DataFrame.from_dict(gene_pearsonr_values, orient='index')
fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.4})
sns.kdeplot(data=gene_pearsonr_df, x='mean', y='pearsonr', 
            ax=ax[0], fill=True, cmap='Blues', levels=20, thresh=0.05)
ax[0].set_xlabel('Mean Expression Level (log rpkm)')
ax[0].set_ylabel('PearsonR across Cell Types')
ax[0].set_title('All Genes')
sns.kdeplot(data=gene_pearsonr_df[gene_pearsonr_df['mean'] >= 1], x='mean', y='pearsonr', 
            ax=ax[1], fill=True, cmap='Oranges', levels=20, thresh=0.05)
ax[1].set_xlabel('Mean of Expression Level (log rpkm)')
ax[1].set_ylabel('PearsonR across Cell Types')
ax[1].set_title('Genes with Mean Expression >= 1')
fig.savefig('figures/celltype_head/Gene_level_PearsonR_vs_mean_expression_logrpkm.pdf')
# kde on pearsonr only
fig, ax = plt.subplots(figsize=(6, 4))
gene_pearsonr_df['Expression Level Group'] = pd.cut(gene_pearsonr_df['mean'], bins=[-np.inf, 1, 3, 5, np.inf], labels=['Low (<1)', 'Medium (1-3)', 'High (3-5)', 'Very High (>5)'])
# Define consistent color palette for each group
group_colors = {
    'Low (<1)': '#4575b4',
    'Medium (1-3)': '#91bfdb',
    'High (3-5)': '#fc8d59',
    'Very High (>5)': '#d73027'
}
sns.kdeplot(data=gene_pearsonr_df, x='pearsonr', ax=ax, hue='Expression Level Group', fill=True, alpha=0.2, palette=group_colors, common_norm=False, legend=False)
ax.set_xlabel('PearsonR across Cell Types')
# add text on average pearsonr
mean_pearsonr = gene_pearsonr_df['pearsonr'].mean()
ax.text(0.02, 0.95, f'Average PearsonR: {mean_pearsonr:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='black')
ax.text(0.02, 0.90, 'Avg Expression Level (log rpkm):', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='gray')
# for each expression level group, add average pearsonr with matching colors
for i, (group, group_df) in enumerate(gene_pearsonr_df.groupby('Expression Level Group')):
    group_mean_pearsonr = group_df['pearsonr'].mean()
    n_genes = len(group_df)
    ax.text(0.02, 0.85 - i*0.05, f'{group}: μ = {group_mean_pearsonr:.2f} (n={n_genes})', transform=ax.transAxes, fontsize=10, verticalalignment='top', color=group_colors[group])
ax.set_title('Gene-level PearsonR Density across Cell Types')
fig.savefig('figures/celltype_head/Gene_level_PearsonR_density_across_cell_types.pdf')


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
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_Subclass.filtered.peaks/raw_data/Test_aggregated_label_bed.pkl', 'rb') as f:
    cross_cell_type_label = pickle.load(f)
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_Subclass.filtered.peaks/raw_data/Test_aggregated_pred_bed.pkl', 'rb') as f:
    cross_cell_type_pred = pickle.load(f)
# read the peak bed
peak_bed = pd.read_csv('Data/source/ATAC_peak/Subclass.filtered.peaks.bed', sep='\t', header=None)
# drop nan lines
valid_indices = ~np.isnan(cross_cell_type_label).any(axis=1) & ~np.isnan(cross_cell_type_pred).any(axis=1)
cross_cell_type_label = cross_cell_type_label[valid_indices]
cross_cell_type_pred = cross_cell_type_pred[valid_indices]
peak_bed = peak_bed.iloc[valid_indices, :].reset_index(drop=True)
# %% read abc links
# abc_links = pd.read_csv('Data/source/ABC/broad_abc_filtcelltype_conns.txt', sep='\t')
# abc_links['peak_id'] = abc_links['chr'] + ':' + abc_links['start'].astype(str) + '-' + abc_links['end'].astype(str)
peak_bed['peak_id'] = peak_bed[0].astype(str) + ':' + peak_bed[1].astype(str) + '-' + peak_bed[2].astype(str)
# peak_bed['is_abc_linked'] = peak_bed['peak_id'].isin(abc_links['peak_id'])
peak_bed['is_abc_linked'] = True # placeholder, since abc links file is not available
# %% calculate pearsonr across cell types for each modality for each peak, and filter to MiniAtlas only
cross_cell_type_data = {}
for mod in ['ATAC', 'K27Ac', 'K27Me3', 'K9Me3']:
    dims = cell_type_meta.index[(cell_type_meta['modality'] == mod) & (cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()
    labels = cross_cell_type_label[:, dims]
    preds = cross_cell_type_pred[:, dims]
    if mod == 'ATAC':
        # times 100 to get original scale (cpm)
        labels = labels * 100
        preds = preds * 100
    # log to get log_cpm
    labels = np.log10(labels + 1e-6)
    preds = np.log10(preds + 1e-6)
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
for mod, cmap in zip(['ATAC', 'K27Ac', 'K27Me3', 'K9Me3'], ['Blues', 'Oranges', 'Greens', 'Purples']):
    data = pd.DataFrame(cross_cell_type_data[mod])
    # fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    # # histogram plot
    # # sns.histplot(data['pearsonr'], bins=30, kde=False, ax=ax, color='blue', alpha=0.5)
    # # draw kde plot
    # sns.kdeplot(x=data['label_coeff_var'][data['is_abc_linked']], y=data['pearsonr'][data['is_abc_linked']], 
    #             ax=ax[0], fill=True, cmap=cmap, levels=20, thresh=0.05)
    # ax[0].set_xlabel('Coefficient Variance across Cell Types')
    # ax[0].set_ylabel('PearsonR across Cell Types')
    # ax[0].set_title(f'Basal Ganglia {mod} (ABC linked peaks)')
    # # abc non linked
    # sns.kdeplot(x=data['label_coeff_var'][~data['is_abc_linked']], y=data['pearsonr'][~data['is_abc_linked']], 
    #             ax=ax[1], fill=True, cmap=cmap, levels=20, thresh=0.05)
    # ax[1].set_xlabel('Coefficient Variance across Cell Types')
    # ax[1].set_ylabel('PearsonR across Cell Types')
    # ax[1].set_title(f'Basal Ganglia {mod} (Non-ABC linked peaks)')
    # fig.tight_layout()
    # fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_vs_coeff_var_' + mod + '.pdf')
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
           'K27Me3': sns.color_palette('Greens', n_colors=10)[6],
           'K9Me3': sns.color_palette('Purples', n_colors=10)[6]}
# plot histogram with kde
sns.kdeplot(data_toplot, x='pearsonr', hue='modality', ax=ax, palette=palette, fill=True, alpha=0.5, common_norm=False)
ax.set_xlabel('PearsonR across Cell Types (Coeff Var > 1)')

# Calculate and display average for each modality
y_text_pos = 0.95
for i, (mod, color) in enumerate(zip(['ATAC', 'K27Ac', 'K27Me3', 'K9Me3'],
                                      [palette['ATAC'], palette['K27Ac'], palette['K27Me3'], palette['K9Me3']])):
    mod_data = data_toplot[data_toplot['modality'] == mod]
    mean_val = mod_data['pearsonr'].mean()
    ax.text(0.02, y_text_pos - i*0.08, f'{mod}: μ = {mean_val:.3f} (n={len(mod_data)})',
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
                palette=palette, legend=False)
    # add modality label above non-ABC-linked annotation
    mod_color = sns.color_palette('Blues' if mod == 'ATAC' else 'Oranges' if mod == 'K27Ac' else 'Greens', n_colors=10)[7]
    ax[i].text(0.02, y_text_pos + 0.06, mod, transform=ax[i].transAxes, fontsize=12,
               verticalalignment='top', fontweight='bold', color=mod_color)
    # print average values
    for j, (is_linked, group) in enumerate(mod_data.groupby('is_abc_linked')):
        mean_value = group['pearsonr'].mean()
        label_text = 'ABC-linked' if is_linked else 'non-ABC-linked'
        ax[i].text(0.02, y_text_pos - j*0.08, f'{label_text}: μ = {mean_value:.3f} (n={len(group)})',
                   transform=ax[i].transAxes, fontsize=10, verticalalignment='top',
                   color=palette[f'{mod}:{is_linked}'], fontweight='bold')
fig.tight_layout()
fig.savefig('figures/celltype_head/MiniAtlas_cross_cell_type_pearsonr_kde_coeff_var_abc_linked_vs_nonlinked.pdf')

# %% peaks minus average then pearsonr
cross_cell_type_label_centered = cross_cell_type_label[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()]
cross_cell_type_pred_centered = cross_cell_type_pred[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()]
# log transform
cross_cell_type_pred_centered = np.log10(cross_cell_type_pred_centered + 1e-6)
cross_cell_type_label_centered = np.log10(cross_cell_type_label_centered + 1e-6)
# substract mean
cross_cell_type_pred_centered = cross_cell_type_pred_centered - cross_cell_type_pred_centered.mean(axis=1, keepdims=True)
cross_cell_type_label_centered = cross_cell_type_label_centered - cross_cell_type_label_centered.mean(axis=1, keepdims=True)
cell_type_meta_basalganglia = cell_type_meta[cell_type_meta['atlas_name'] == 'BasalGanglia'].reset_index(drop=True)
# recalculate pearsonr
valid_indices = ~np.isnan(cross_cell_type_label_centered).any(axis=1) & ~np.isnan(cross_cell_type_pred_centered).any(axis=1)
cross_cell_type_label_centered = cross_cell_type_label_centered[valid_indices]
cross_cell_type_pred_centered = cross_cell_type_pred_centered[valid_indices]
# for each track, do pearsonr
pearsonr_values = {}
for track in cell_type_meta_basalganglia['exp']:
    gene_pred_track = cross_cell_type_pred_centered[:, cell_type_meta_basalganglia.index[cell_type_meta_basalganglia['exp'] == track][0]]
    gene_target_track = cross_cell_type_label_centered[:, cell_type_meta_basalganglia.index[cell_type_meta_basalganglia['exp'] == track][0]]
    corr, _ = pearsonr(gene_pred_track, gene_target_track)
    pearsonr_values[track] = corr
# convert to dataframe
pearsonr_df = pd.DataFrame.from_dict(pearsonr_values, orient='index', columns=['PearsonR'])
pearsonr_df = pearsonr_df.merge(cell_type_meta_basalganglia, left_index=True, right_on='exp')
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
palette = {'ATAC': sns.color_palette('Blues', n_colors=10)[6],
           'K27Ac': sns.color_palette('Oranges', n_colors=10)[6],
           'K27Me3': sns.color_palette('Greens', n_colors=10)[6],
           'K9Me3': sns.color_palette('Purples', n_colors=10)[6]}
sns.kdeplot(data=pearsonr_df, x='PearsonR', hue='modality', ax=ax, fill=True, alpha=0.5, common_norm=False, palette=palette)
# print average pearsonr for each modality
y_text_pos = 0.95
for i, (mod, group) in enumerate(modality_groups):
    mean_value = group['PearsonR'].mean()
    ax.text(0.02, y_text_pos - i*0.08, f'{mod}: μ = {mean_value:.3f} (n={len(group)})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color=palette[mod], fontweight='bold')
ax.set_xlabel('PearsonR within Cell Types (Minus average across cell types)')
fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_kde_log_centered_all_modalities.pdf')




# %% Section 4 
# ================================================================================
# plot the cross cell type pearsonr in atac peaks with regard to relative variance in basal ganglia
import pickle
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_Subclass.filtered.peaks/raw_data/Test_aggregated_label_bed.pkl', 'rb') as f:
    cross_cell_type_label = pickle.load(f)
with open('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20_bed_Subclass.filtered.peaks/raw_data/Test_aggregated_pred_bed.pkl', 'rb') as f:
    cross_cell_type_pred = pickle.load(f)
# read the peak bed
peak_bed = pd.read_csv('Data/source/ATAC_peak/Subclass.filtered.peaks.bed', sep='\t', header=None)
# drop nan lines
valid_indices = ~np.isnan(cross_cell_type_label).any(axis=1) & ~np.isnan(cross_cell_type_pred).any(axis=1)
cross_cell_type_label = cross_cell_type_label[valid_indices]
cross_cell_type_pred = cross_cell_type_pred[valid_indices]
peak_bed = peak_bed.iloc[valid_indices, :].reset_index(drop=True)
peak_bed.columns = ['chr', 'start', 'end', 'peak_id']
# %% read ABC file
abc_links = pd.read_csv('Data/source/ABC/BasalGanglia/merged_abc_results.with_annotations_v1_5kb_resolution_no_powerlaw_scale.tsv', sep='\t')
abc_links['peak_id'] = abc_links['chr'] + ':' + abc_links['start'].astype(str) + '-' + abc_links['end'].astype(str)
# filter to correspoinding cell types
peak_bed['is_abc_linked'] = peak_bed['peak_id'].isin(abc_links['peak_id'])

# %% calculate pearsonr across cell types for each modality for each peak, and filter to BasalGanglia only
cross_cell_type_data = {}
for mod in ['ATAC', 'K27Ac', 'K27Me3', 'K9Me3']:
    dims = cell_type_meta.index[(cell_type_meta['modality'] == mod) & (cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()
    labels = cross_cell_type_label[:, dims]
    preds = cross_cell_type_pred[:, dims]
    # log to get log_cpm
    if mod == 'ATAC':
        labels *= 2000  # rpkm
        preds *= 2000
    else:
        labels *= 10
        preds *= 10
    labels = np.log1p(labels) # cpm
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
for mod, cmap in zip(['ATAC', 'K27Ac', 'K27Me3', 'K9Me3'], ['Blues', 'Oranges', 'Greens', 'Purples']):
    data = pd.DataFrame(cross_cell_type_data[mod])
    # fig, ax = plt.subplots(figsize=(4, 4))
    # draw kde plot
    # sns.kdeplot(x=data['label_coeff_var'], y=data['pearsonr'], 
    #             ax=ax, fill=True, cmap=cmap, levels=20, thresh=0.05)
    # ax.set_xlabel('Coefficient Variance across Cell Types')
    # ax.set_ylabel('PearsonR across Cell Types')
    # ax.set_title(f'BasalGanglia {mod}')
    # fig.tight_layout()
    # fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_vs_coeff_var_' + mod + '.pdf')
    # add modality column for later combined plotting
    data['modality'] = mod
    if data_together is None:
        data_together = data
    else:
        data_together = pd.concat([data_together, data], axis=0)
# %% plot together
fig, ax = plt.subplots(figsize=(6, 4))
data_toplot = data_together[(data_together['label_coeff_var'] > 1) | ((data_together['modality'] == 'ATAC') & (data_together['label_coeff_var'] > 0.5))]
# Create palette matching the cmaps from line 304
palette = {'ATAC': sns.color_palette('Blues', n_colors=10)[6],
           'K27Ac': sns.color_palette('Oranges', n_colors=10)[6],
           'K27Me3': sns.color_palette('Greens', n_colors=10)[6],
           'K9Me3': sns.color_palette('Purples', n_colors=10)[6]}
# plot histogram with kde
sns.kdeplot(data_toplot, x='pearsonr', hue='modality', ax=ax, palette=palette, fill=True, alpha=0.5, common_norm=False, legend=False)
ax.set_xlabel('PearsonR across Cell Types (Coeff Var > 1)')
ax.set_title('cCRE level PearsonR Density across Cell Types')

# Calculate and display average for each modality
y_text_pos = 0.95
for i, (mod, color) in enumerate(zip(['ATAC', 'K27Ac', 'K27Me3', 'K9Me3'],
                                      [palette['ATAC'], palette['K27Ac'], palette['K27Me3'], palette['K9Me3']])):
    mod_data = data_toplot[data_toplot['modality'] == mod]
    mean_val = mod_data['pearsonr'].mean()
    ax.text(0.02, y_text_pos - i*0.08, f'{mod}: μ = {mean_val:.3f} (n={len(mod_data)})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color=color, fontweight='bold')

fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_kde_coeff_var_all_modalities.pdf')

# %% plot abc linked vs non linked boxplot
data_toplot['label'] = data_toplot['modality'] + ":" + data_toplot['is_abc_linked'].astype(str)
palette = {f'ATAC:False': sns.color_palette('Blues', n_colors=10)[6],
           f'ATAC:True': sns.color_palette('Blues', n_colors=10)[9],
           f'K27Ac:False': sns.color_palette('Oranges', n_colors=10)[6],
           f'K27Ac:True': sns.color_palette('Oranges', n_colors=10)[9],
           f'K27Me3:False': sns.color_palette('Greens', n_colors=10)[6],
           f'K27Me3:True': sns.color_palette('Greens', n_colors=10)[9],
           f'K9Me3:False': sns.color_palette('Purples', n_colors=10)[6],
           f'K9Me3:True': sns.color_palette('Purples', n_colors=10)[9]}
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6, 9), sharex=True)
y_text_pos = 0.95
for i, mod in enumerate(['ATAC', 'K27Ac', 'K27Me3', 'K9Me3']):
    mod_data = data_toplot[data_toplot['modality'] == mod]
    sns.kdeplot(data=mod_data, x='pearsonr', ax=ax[i], fill=True, alpha=0.5, hue='label', common_norm=False,
                palette=palette, legend=False)
    # add modality label above non-ABC-linked annotation
    color_map = {'ATAC': 'Blues', 'K27Ac': 'Oranges', 'K27Me3': 'Greens', 'K9Me3': 'Purples'}
    mod_color = sns.color_palette(color_map[mod], n_colors=10)[7]
    ax[i].text(0.02, y_text_pos + 0.02, mod, transform=ax[i].transAxes, fontsize=12,
               verticalalignment='top', fontweight='bold', color=mod_color)
    # print average values
    for j, (is_linked, group) in enumerate(mod_data.groupby('is_abc_linked')):
        mean_value = group['pearsonr'].mean()
        label_text = 'ABC-linked' if is_linked else 'non-ABC-linked'
        ax[i].text(0.02, y_text_pos - j*0.08 - 0.06, f'{label_text}: μ = {mean_value:.3f} (n={len(group)})',
                   transform=ax[i].transAxes, fontsize=10, verticalalignment='top',
                   color=palette[f'{mod}:{is_linked}'], fontweight='bold')
fig.tight_layout()
fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_kde_coeff_var_abc_linked_vs_nonlinked.pdf')

# %% peaks minus average then pearsonr
cross_cell_type_label_centered = cross_cell_type_label[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()]
cross_cell_type_pred_centered = cross_cell_type_pred[:, cell_type_meta.index[(cell_type_meta['atlas_name'] == 'BasalGanglia')].tolist()]
# log transform
cross_cell_type_pred_centered = np.log1p(cross_cell_type_pred_centered)
cross_cell_type_label_centered = np.log1p(cross_cell_type_label_centered)
# substract mean
cross_cell_type_pred_centered = cross_cell_type_pred_centered - cross_cell_type_pred_centered.mean(axis=1, keepdims=True)
cross_cell_type_label_centered = cross_cell_type_label_centered - cross_cell_type_label_centered.mean(axis=1, keepdims=True)
cell_type_meta_miniatlas = cell_type_meta[cell_type_meta['atlas_name'] == 'BasalGanglia'].reset_index(drop=True)
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
sns.kdeplot(data=pearsonr_df, x='PearsonR', hue='modality: avg', ax=ax, fill=True, alpha=0.5, common_norm=False)
ax.set_xlabel('PearsonR across Cell Types (Centered)')
fig.savefig('figures/celltype_head/BasalGanglia_cross_cell_type_pearsonr_kde_log_centered_all_modalities.pdf')

# %% Section 5
# plot PearsonR with regard to the cell type sizes and saturation
bin_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/analysis_17/raw_data/Test_metric.csv', index_col=0)
# get the number of cells from meta data
cell_type_meta = pd.read_csv('Data/data_config/basal_ganglia_miniatlas_complete_v1.csv', index_col=0)
cell_type_meta.reset_index(inplace=True, drop=True)
# annotate the bin level results with cell type information
bin_level_results = bin_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
# rename modality of RNA strands
bin_level_results['modality'] = bin_level_results['modality'].replace(
    {'RNAplus': 'RNA-', 'RNAminus': 'RNA+', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
# calibrate the basal ganglia
basal_ganglia_calibrate = pd.read_csv('Data/source/BGC_cell_type_counts.csv', index_col=0)
basal_ganglia_calibrate.index = basal_ganglia_calibrate.index.str.replace(' ', '-')
basal_ganglia_calibrate.index = 'BasalGanglia-' + basal_ganglia_calibrate.index
# update celltype_n for ATAC modality using calibrated basal ganglia counts
atac_mask = (bin_level_results['modality'] == 'ATAC') & (bin_level_results['celltype'].isin(basal_ganglia_calibrate.index))
bin_level_results.loc[atac_mask, 'celltype_n'] = bin_level_results.loc[atac_mask, 'celltype'].map(basal_ganglia_calibrate['ATAC'])
# plot PearsonR vs number of cells
# log transform the cell type number
bin_level_results['log_celltype_n'] = np.log10(bin_level_results['celltype_n'])
# create bins for cell type sizes per modality
bin_level_results['celltype_bin'] = bin_level_results.groupby('modality')['log_celltype_n'].transform(
    lambda x: pd.cut(x, bins=10)
)
# calculate mean and standard error for each bin and modality
grouped_stats = bin_level_results.groupby(['celltype_bin', 'modality'])['PearsonR'].agg(['mean', 'sem', 'count']).reset_index()
# get bin centers for plotting
grouped_stats['bin_center'] = grouped_stats['celltype_bin'].apply(lambda x: x.mid)

fig, ax = plt.subplots(figsize=(6, 4))
modalities = ['ATAC', 'H3K27ac', 'H3K27me3', 'H3K9me3']
palette = sns.color_palette()
for i, modality in enumerate(modalities):
    modality_data = grouped_stats[grouped_stats['modality'] == modality]
    ax.errorbar(modality_data['bin_center'], modality_data['mean'],
                yerr=modality_data['sem'], label=modality,
                color=palette[i], linewidth=2.5, marker='o', markersize=6,
                capsize=4, capthick=2)
ax.set_xlabel('log10(Number of Cells)')
ax.set_ylabel('PearsonR')
ax.set_title('Bin-level PearsonR vs Number of Cells')
ax.legend()
fig.savefig('figures/celltype_head/Bin_level_PearsonR_vs_number_of_cells.pdf')

# %% # load gene level metrics
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/analysis_17/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results = gene_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results['modality'] = gene_level_results['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
# log transform the cell type number
gene_level_results['log_celltype_n'] = np.log10(gene_level_results['celltype_n'])
# create bins for cell type sizes
gene_level_results['celltype_bin'] = pd.cut(gene_level_results['log_celltype_n'], bins=10)
# calculate mean and standard error for each bin and modality
grouped_stats_gene = gene_level_results.groupby(['celltype_bin', 'modality'])['pearsonr_log'].agg(['mean', 'sem', 'count']).reset_index()
# get bin centers for plotting
grouped_stats_gene['bin_center'] = grouped_stats_gene['celltype_bin'].apply(lambda x: x.mid)

fig, ax = plt.subplots(figsize=(6, 4))
modalities = ['RNA']
palette = sns.color_palette()
for i, modality in enumerate(modalities):
    modality_data = grouped_stats_gene[grouped_stats_gene['modality'] == modality]
    ax.errorbar(modality_data['bin_center'], modality_data['mean'],
                yerr=modality_data['sem'], label=modality,
                color=palette[4], linewidth=2.5, marker='o', markersize=6,
                capsize=4, capthick=2)
ax.set_xlabel('log10(Number of Cells)')
ax.set_ylabel('PearsonR')
ax.set_title('Gene-level PearsonR vs Number of Cells')
ax.legend()
fig.savefig('figures/celltype_head/Gene_level_PearsonR_vs_number_of_cells.pdf')

# %%
