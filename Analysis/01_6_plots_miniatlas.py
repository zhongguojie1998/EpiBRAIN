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
bin_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/analysis_17/raw_data/Test_metric.csv', index_col=0)
# get the number of cells from meta data
cell_type_meta = pd.read_csv('Data/data_config/basal_ganglia_miniatlas_complete_v1.csv', index_col=0)
cell_type_meta.reset_index(inplace=True, drop=True)
# annotate the bin level results with cell type information
bin_level_results = bin_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
# drop cell types with < 1000 cells
bin_level_results = bin_level_results[bin_level_results['celltype_n'] >= 1000]

# %% Section 2
# ================================================================================
# load gene level metrics
gene_level_results = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/analysis_17/gene_level/raw_data/Test_gene_metrics_rpkm.csv', index_col=0)
gene_level_results = gene_level_results.merge(cell_type_meta, left_on='trial', right_on='exp')
gene_level_results['modality'] = gene_level_results['modality_x'].replace(
    {'RNA': 'RNA', 'RNAplus': 'RNA', 'RNAminus': 'RNA', 'K27Ac': 'H3K27ac', 'K27Me3': 'H3K27me3', 'K9Me3': 'H3K9me3'})
# create barplot with reversed colors
palette_reversed = {'BasalGanglia': sns.color_palette()[0], 'MiniAtlas': sns.color_palette()[1]}
# drop cell types with < 1000 cells
gene_level_results = gene_level_results[gene_level_results['celltype_n'] >= 1000]

# %% comparison of gene level metrics between atac-only and full model
# merge on celltype and gene
res_full_model = gene_level_results[gene_level_results['modality'] == 'RNA']
merged_res = res_full_model.copy()

# ============================================================================
# Combined figure with 2 subplots (plot 1 wider than plot 3)
# ============================================================================
merged_res = merged_res[merged_res['atlas_name'] == 'MiniAtlas']
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# ----------------------------------------------------------------------------
# Subplot 1: Half-boxplot/half-violin plot (wider)
# ----------------------------------------------------------------------------
res = bin_level_results[bin_level_results['atlas_name'] == 'MiniAtlas']

modalities = res['modality'].unique()
# Add one more position for gene-level full model data
positions = np.arange(len(modalities) + 1)

# Prepare box data: bin-level by modality + gene-level full model
box_data = [res[res['modality'] == mod]['PearsonR'].values for mod in modalities]
# Add gene-level full model data from subplot 3
box_data.append(merged_res['pearsonr_log'].values)

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
    ax.text(x=list(modalities).index(modality),
            y=max_value * 1.05, s=f'{mean_value:.2f}',
            ha='center', va='bottom', fontsize=8, color='black')

# For gene-level full model
gene_mean = merged_res['pearsonr_log'].mean()
gene_max = merged_res['pearsonr_log'].max()
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

# Save combined figure
fig.tight_layout()
fig.savefig('figures/MiniAtlas_PearsonR_analysis.pdf')


# %%
