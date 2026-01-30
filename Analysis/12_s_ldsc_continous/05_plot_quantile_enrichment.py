# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys
import importlib.util

# Load subclass colors from the parent directory
spec = importlib.util.spec_from_file_location("subclass_colors_module", "../00_basalganglia_subclass_colors.py")
subclass_colors_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subclass_colors_module)
# Replace spaces with hyphens in keys to match cell type names
subclass_colors = {k.replace(' ', '-'): v for k, v in subclass_colors_module.subclass_colors.items()}

# Make text editable in Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# %%
# load schizophrenia quantile enrichment results
# read the files in quantile_results_by_trait/quantile_results_PGC_Nature_2014_Schizophrenia_fullinfo.sumstats/
files = glob.glob("quantile_results_by_trait/quantile_results_PGC_Nature_2014_Schizophrenia_fullinfo.sumstats/*h2g.txt")
tracks = [file.split("/")[-1].replace(".quantile_h2g.txt", "") for file in files]
df_meta = pd.read_csv('/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/data_config/basal_ganglia_miniatlas_drop_celltype_v1.csv', sep=',', index_col=0)

# %% read in files
res_matrix = pd.DataFrame()
res_matrix_se = pd.DataFrame()
for idx, row in df_meta.iterrows():
    track_name = row['exp']
    if track_name in tracks:
        file_path = f"quantile_results_by_trait/quantile_results_PGC_Nature_2014_Schizophrenia_fullinfo.sumstats/{track_name}.quantile_h2g.txt"
        df_temp = pd.read_csv(file_path, sep='\t')
        # read in the prop_h2g column
        res_temp = df_temp[['prop_h2g']]
        res_temp_se = df_temp[['prop_h2g_se']]
        res_temp.columns = [track_name]
        res_temp_se.columns = [track_name]
        if res_matrix.empty:
            res_matrix = res_temp
            res_matrix_se = res_temp_se
        else:
            res_matrix = pd.concat([res_matrix, res_temp], axis=1)
            res_matrix_se = pd.concat([res_matrix_se, res_temp_se], axis=1)

# %% filter to basal ganglia cell types
basal_ganglia_celltypes = df_meta[df_meta['atlas_name'] == 'BasalGanglia'].copy()
for modality in basal_ganglia_celltypes['modality'].unique():
    basal_ganglia_celltypes_mod = basal_ganglia_celltypes[basal_ganglia_celltypes['modality'] == modality]
    # plot heatmap for each modality
    res_matrix_bg_mod = res_matrix[basal_ganglia_celltypes_mod['exp'].values]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(res_matrix_bg_mod.T, cmap='coolwarm', cbar_kws={'label': 'Proportion of Heritability (prop_h2g)'}, ax=ax)
    ax.set_title(f'Schizophrenia Quantile Enrichment in Basal Ganglia Cell Types - {modality}')
    ax.set_xlabel('Quantiles')
    ax.set_ylabel('Basal Ganglia Cell Types')
    fig.tight_layout()
    fig.show()

# %% plot barplot for the top 10% instead
# get the top 10% quantile for borzoi
borzoi_res = pd.read_csv('quantile_results_by_trait/quantile_results_borzoi_PGC_Nature_2014_Schizophrenia_fullinfo.sumstats/all.quantile_h2g.txt', sep='\t')
borzoi_top10 = borzoi_res['prop_h2g'].values[-1]
borzoi_top10_se = borzoi_res['prop_h2g_se'].values[-1]
for modality in basal_ganglia_celltypes['modality'].unique():
    basal_ganglia_celltypes_mod = basal_ganglia_celltypes[basal_ganglia_celltypes['modality'] == modality]
    res_matrix_bg_mod = res_matrix[basal_ganglia_celltypes_mod['exp'].values]
    res_matrix_se_bg_mod = res_matrix_se[basal_ganglia_celltypes_mod['exp'].values]
    # get the top 10% quantile (assuming 10 quantiles, so the last row)
    top10 = res_matrix_bg_mod.iloc[-1, :]
    top10_se = res_matrix_se_bg_mod.iloc[-1, :]
    # remove the prefix and suffix for index
    top10.index = [basal_ganglia_celltypes_mod[basal_ganglia_celltypes_mod['exp'] == idx]['celltype'].values[0].replace('BasalGanglia-', '') for idx in top10.index]
    top10_se.index = top10.index
    # rename column name
    top10.name = 'BICAN Top 5%'
    top10_se.name = 'BICAN_SE'
    # order by top10 values
    top10 = top10.sort_values(ascending=False)
    top10_se = top10_se[top10.index]
    # Create color mapping using subclass_colors
    colors = [subclass_colors.get(celltype, '#808080') for celltype in top10.index]
    # plot barplot
    fig, ax = plt.subplots(figsize=(6, 4))
    top10.plot(kind='bar', yerr=top10_se, ax=ax, capsize=4, color=colors, edgecolor='black')
    # add horizontal line for borzoi
    ax.axhline(y=borzoi_top10, color='red', linestyle='--', label='Borzoi Top 5%')
    ax.fill_between([-1, len(top10)], borzoi_top10 - borzoi_top10_se, borzoi_top10 + borzoi_top10_se, color='red', alpha=0.2)
    # move legend to upper right
    ax.legend(loc='lower right')
    # rotate x-axis
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Schizophrenia Heritability - {modality}')
    ax.set_ylabel('Proportion of Heritability')
    ax.set_xlabel('Basal Ganglia Cell Types')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/figures/12_s_ldsc_continous/schizophrenia_basal_ganglia_{modality}_top10_barplot.pdf')
    
# %%
