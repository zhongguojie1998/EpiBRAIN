# %%
import pandas as pd
import numpy as np
import os
import pyBigWig
from dotenv import load_dotenv
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
load_dotenv()
PWD = f'{os.environ["workingHOME"]}/BICAN'
import sys
os.chdir(PWD)
sys.path.append(PWD)
# make fonts AI compatible
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# %%
# read Single Brain Fine Mapped eQTL results
eqtl_res = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/var_eff/variant_effects_by_gene_single_brain.tsv', sep='\t')

# %% load eqtl_ref
eqtl_finemap = pd.read_excel('Data/source/Jang2025_SingleBrain/FineMapped.xlsx', sheet_name='Table12_Fine-mapping', skiprows=2)

# %% for each eqtl in eqtl_ref, get the pred ref and alt from eqtl_res
eqtl_res = {}
for tissue in eqtl_finemap['QTL'].unique():
    print(f'Processing tissue: {tissue}')
    eqtl_res[tissue] = pd.read_csv(f'Data/source/Jang2025_SingleBrain/{tissue}_eqtl_full_assoc.tsv.gz', sep='\t', compression='gzip')
# %%
cols_to_exclude = ['feature', 'variant_id']

# Pre-determine all columns that will be added (run once before loop)
all_new_cols = set()
for tissue_data in eqtl_res.values():
    all_new_cols.update(tissue_data.columns)
all_new_cols -= set(cols_to_exclude)
all_new_cols -= set(eqtl_finemap.columns)

# Add all new columns at once with NaN
for col in all_new_cols:
    eqtl_finemap[col] = np.nan

# Prepare a list to collect updates, then apply them in batch
updates = []
for idx, row in eqtl_finemap.iterrows():
    # get rsid represented chr+start+ref+alt
    tissue = row['QTL']
    eqtl_ref = eqtl_res[tissue][eqtl_res[tissue]['variant_id'] == row['SNP']]
    # fetch the qtl gene ID
    eqtl_ref = eqtl_ref[eqtl_ref['feature'].str.contains(row['QTL gene ID'])]
    # append to eqtl_finemap
    if eqtl_ref.empty:
        print(f'No match found for {row["chr"]}:{row["pos"]} {row["ref"]}>{row["alt"]} {row["QTL gene ID"]}')
        continue

    eqtl_ref = eqtl_ref.drop(columns=cols_to_exclude)
    # Collect updates as a dictionary
    update_dict = {'idx': idx}
    update_dict.update(eqtl_ref.iloc[0].to_dict())
    updates.append(update_dict)

# Apply all updates at once using pandas update
if updates:
    updates_df = pd.DataFrame(updates).set_index('idx')
    eqtl_finemap.update(updates_df)
    
# write to file
eqtl_finemap.to_csv('Data/source/Jang2025_SingleBrain/finemapped_variants_info.tsv', sep='\t', index=False)
# %% load the updated file
eqtl_finemap = pd.read_csv('Data/source/Jang2025_SingleBrain/finemapped_variants_info.tsv', sep='\t')

# %% filter the eqtl_res to snp-gene pairs
# drop basal gangila cell types
eqtl_res = eqtl_res[eqtl_res['track'].str.contains('BasalGanglia-')]
# remove the minus or plus from track names
eqtl_res['track'] = eqtl_res['track'].str.replace('minus', '').str.replace('plus', '')
# only keep the RNA results
eqtl_res = eqtl_res[eqtl_res['track'].str.contains('RNA')]
# add a column to indicate the eqtl cell type name
eqtl_res['cell_type'] = eqtl_res['track'].str.replace('_RNA', '').str.replace('BasalGanglia-', '')
# %% create three matrixes to explore the "direction" prediction result and "effect_size" prediction result (pearson+spearman)
single_brain_cell_types = eqtl_finemap['QTL'].unique()
bican_cell_types = eqtl_res['cell_type'].unique()
res_mats = {'direction': pd.DataFrame(index=single_brain_cell_types, columns=bican_cell_types),
            'effect_size_pearson': pd.DataFrame(index=single_brain_cell_types, columns=bican_cell_types), 
            'effect_size_spearman': pd.DataFrame(index=single_brain_cell_types, columns=bican_cell_types)}
for single_brain_cell_type in single_brain_cell_types:
    for bican_cell_type in bican_cell_types:
        # get the prediction and reference
        single_brain_eqtl = eqtl_finemap[eqtl_finemap['QTL'] == single_brain_cell_type].copy()
        bican_eqtl = eqtl_res[eqtl_res['cell_type'] == bican_cell_type].copy()
        # create snp-geneID map
        single_brain_eqtl['snp_geneID'] = single_brain_eqtl['chr'] + ":" + single_brain_eqtl['pos'].astype(int).astype(str) + ":" + single_brain_eqtl['ref'] + ":" + single_brain_eqtl['alt'] + ":" + single_brain_eqtl['QTL gene ID']
        bican_eqtl['snp_geneID'] = bican_eqtl['chr'] + ":" + bican_eqtl['pos'].astype(int).astype(str) + ":" + bican_eqtl['ref'] + ":" + bican_eqtl['alt'] + ":" + bican_eqtl['gene_id'].str.replace('\\..*', '', regex=True)
        # filter to unique snp-geneID
        single_brain_eqtl = single_brain_eqtl.drop_duplicates(subset=['snp_geneID'])
        bican_eqtl = bican_eqtl.drop_duplicates(subset=['snp_geneID'])
        # set to index for easy lookup
        single_brain_eqtl = single_brain_eqtl.set_index('snp_geneID')
        bican_eqtl = bican_eqtl.set_index('snp_geneID')
        # get common snp-geneID
        common_snp_geneID = single_brain_eqtl.index.intersection(bican_eqtl.index)
        # filter to common snp-geneID
        single_brain_eqtl = single_brain_eqtl.loc[common_snp_geneID]
        bican_eqtl = bican_eqtl.loc[common_snp_geneID]
        # calculate log fold change for bican_eqtl
        bican_eqtl['lfdc'] = np.log2(bican_eqtl['alt_value']) - np.log2(bican_eqtl['ref_value'])
        bican_eqtl['diff'] = bican_eqtl['alt_value'] / bican_eqtl['ref_value']

        # Get signs of both beta and lfdc
        single_brain_sign = (single_brain_eqtl['fixed_beta'] > 0).astype(int)
        bican_sign = (bican_eqtl['lfdc'] > 0).astype(int)

        # calculate F1 score for sign agreement between single_brain_eqtl['fixed_beta'] and bican_eqtl['lfdc']
        f1 = f1_score(single_brain_sign, bican_sign)
        res_mats['direction'].loc[single_brain_cell_type, bican_cell_type] = f1

        # calculate spearman correlation between single_brain_eqtl['fixed_beta'] and bican_eqtl['lfdc']
        spearman_corr, _ = spearmanr(single_brain_eqtl['fixed_beta'], bican_eqtl['lfdc'])
        res_mats['effect_size_spearman'].loc[single_brain_cell_type, bican_cell_type] = spearman_corr

        # calculate pearson correlation between single_brain_eqtl['fixed_beta'] and bican_eqtl['lfdc']
        pearson_corr, _ = pearsonr(single_brain_eqtl['fixed_beta'], bican_eqtl['lfdc'])
        res_mats['effect_size_pearson'].loc[single_brain_cell_type, bican_cell_type] = pearson_corr


# %% save the results
for key, df in res_mats.items():
    df.to_csv(f'Data/source/Jang2025_SingleBrain/{key}_comparison_matrix.csv')
# %% plot the results
res_mats = {}
for key in ['direction', 'effect_size_pearson', 'effect_size_spearman']:
    res_mats[key] = pd.read_csv(f'Data/source/Jang2025_SingleBrain/{key}_comparison_matrix.csv', index_col=0)

# %% filter the eqtl result to largest fold change ones
matching_celltypes = {
    'Ast': 'Astrocyte',
    'MG': 'Microglia',
    'OD': 'Oligodendrocyte',
    'OPC': 'OPC'
}
eqtl_finemap['cell_type'] = eqtl_finemap['QTL'].map(matching_celltypes)
eqtl_finemap_filtered = eqtl_finemap[eqtl_finemap['cell_type'].notnull()].copy()
eqtl_res_filtered = eqtl_res[eqtl_res['cell_type'].isin(matching_celltypes.values())].copy()
eqtl_finemap_filtered['snp_geneID'] = eqtl_finemap_filtered['cell_type'] + ":" + eqtl_finemap_filtered['chr'] + ":" + eqtl_finemap_filtered['pos'].astype(int).astype(str) + ":" + eqtl_finemap_filtered['ref'] + ":" + eqtl_finemap_filtered['alt'] + ":" + eqtl_finemap_filtered['QTL gene ID']
eqtl_res_filtered['snp_geneID'] = eqtl_res_filtered['cell_type'] + ":" + eqtl_res_filtered['chr'] + ":" + eqtl_res_filtered['pos'].astype(int).astype(str) + ":" + eqtl_res_filtered['ref'] + ":" + eqtl_res_filtered['alt'] + ":" + eqtl_res_filtered['gene_id'].str.replace('\\..*', '', regex=True)
eqtl_finemap_filtered = eqtl_finemap_filtered.drop_duplicates(subset=['snp_geneID'])
eqtl_res_filtered = eqtl_res_filtered.drop_duplicates(subset=['snp_geneID'])
eqtl_finemap_filtered = eqtl_finemap_filtered.set_index('snp_geneID')
eqtl_res_filtered = eqtl_res_filtered.set_index('snp_geneID')
common_snp_geneID = eqtl_finemap_filtered.index.intersection(eqtl_res_filtered.index)
eqtl_finemap_filtered = eqtl_finemap_filtered.loc[common_snp_geneID]
eqtl_res_filtered = eqtl_res_filtered.loc[common_snp_geneID]
# calculate lfdc
eqtl_res_filtered['lfdc'] = np.log2(eqtl_res_filtered['alt_value']) - np.log2(eqtl_res_filtered['ref_value'])
# calculate diff
eqtl_res_filtered['diff'] = eqtl_res_filtered['alt_value'] - eqtl_res_filtered['ref_value']
# calculate avg_variant_effect_logsum
eqtl_res_filtered['avg_variant_effect_logsum'] = eqtl_res_filtered['variant_effect_logsum'] / eqtl_res_filtered['n_exons']
# %% plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(eqtl_finemap_filtered['fixed_beta'], eqtl_res_filtered['lfdc'], alpha=0.5)
ax.set_xlabel('Finemap LFC')
ax.set_ylabel('BICAN LFC')
ax.set_title('Comparison of LFC between Finemap and BICAN')
# %% check sign prediction, use > 0.25 and < -0.25 as threshold
fig, ax = plt.subplots(figsize=(5, 4))
finemap_sign = (eqtl_finemap_filtered['fixed_beta'] > 0).astype(int)
# finemap_sign[eqtl_finemap_filtered['fixed_beta'] < -0.25] = 0
# finemap_sign[(eqtl_finemap_filtered['fixed_beta'] > -0.25) & (eqtl_finemap_filtered['fixed_beta'] < 0.25)] = np.nan
bican_sign = (eqtl_res_filtered['lfdc'] > 0).astype(int)
# bican_sign[eqtl_res_filtered['lfdc'] < -0.05] = 0
# bican_sign[(eqtl_res_filtered['lfdc'] > -0.05) & (eqtl_res_filtered['lfdc'] < 0.05)] = np.nan
# drop nan
valid_idx = (((eqtl_res_filtered['lfdc'] < -0.05) | (eqtl_res_filtered['lfdc'] > 0.05)))
# valid_idx = (eqtl_finemap_filtered['PIP'] > 0.9) & (~pd.isna(eqtl_res_filtered['lfdc']))
# calculate F1 score
spearman_corr, _ = spearmanr(eqtl_finemap_filtered['fixed_beta'][valid_idx], eqtl_res_filtered['lfdc'][valid_idx])
print(f'Spearman correlation: {spearman_corr}')
sns.scatterplot(x=eqtl_finemap_filtered['fixed_beta'][valid_idx], y=eqtl_res_filtered['lfdc'][valid_idx], alpha=0.5, s=75)
# add x=0 and y=0 lines
ax.axhline(0, color='black', linestyle='--')
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel('eQTL beta (Finemapped variants)')
ax.set_ylabel('Predicted LFC')
# add spearman correlation text
ax.text(0.05, 0.95, f'Spearman r={spearman_corr:.2f} (n={len(eqtl_res_filtered.loc[valid_idx])})', transform=ax.transAxes, verticalalignment='top')
fig.savefig('figures/Jang2025_SingleBrain_finemapped_variants_lfdc_comparison.pdf')


# %%
# read Meat Brain Fine Mapped eQTL results
eqtl_res = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/var_eff/variant_effects_by_gene.tsv', sep='\t')
eqtl_finemap = pd.read_csv('Data/source/MetaBrain/2021-07-23-basalganglia-EUR-30PCs-TopEffects.txt.gz', sep='\t', compression='gzip')
eqtl_res['snp_geneID'] = eqtl_res['chr'] + ":" + eqtl_res['pos'].astype(int).astype(str) + ":" + eqtl_res['ref'] + ":" + eqtl_res['alt'] + ":" + eqtl_res['gene_id']
eqtl_finemap['snp_geneID'] = 'chr' + eqtl_finemap['SNPChr'].astype(int).astype(str) + ":" + eqtl_finemap['SNPPos'].astype(int).astype(str) + ":" + eqtl_finemap['SNPAlleles'].str.replace('/', ':') + ":" + eqtl_finemap['Gene']

# %%
# aggregate the variant effect results in all cell types
# only keep RNA tracks
eqtl_res = eqtl_res[(eqtl_res['track'].str.contains('RNA')) & (eqtl_res['track'].str.contains('BasalGanglia-'))]
# for each snp_geneID, aggregate the variant effect results by summing up
eqtl_res_agg = eqtl_res.groupby('snp_geneID').agg({'ref_value': 'sum',
                                                   'alt_value': 'sum',
                                                   'n_exons': 'first'}).reset_index()
# %% plot 
eqtl_res_agg['diff'] = np.log2(eqtl_res_agg['alt_value']) - np.log2(eqtl_res_agg['ref_value'])
eqtl_res_agg = eqtl_res_agg.set_index('snp_geneID')
eqtl_finemap = eqtl_finemap.set_index('snp_geneID')
common_variants = eqtl_res_agg.index.intersection(eqtl_finemap.index)
eqtl_res_agg = eqtl_res_agg.loc[common_variants]
eqtl_finemap = eqtl_finemap.loc[common_variants]
# %% filter low effect size variants
eqtl_res_agg_plot = eqtl_res_agg[np.abs(eqtl_res_agg['diff']) > 0.05]
eqtl_finemap_plot = eqtl_finemap.loc[eqtl_res_agg_plot.index]
# %% plot
fig, ax = plt.subplots(figsize=(5, 4))
sns.scatterplot(x=eqtl_finemap_plot['MetaBeta'], y=eqtl_res_agg_plot['diff'], alpha=0.5, s=75, ax=ax)
spearman_corr, _ = spearmanr(eqtl_finemap_plot['MetaBeta'], eqtl_res_agg_plot['diff'])
ax.text(0.05, 0.95, f'Spearman r={spearman_corr:.2f} (n={len(eqtl_res_agg_plot)})', transform=ax.transAxes, verticalalignment='top')
ax.axhline(0, color='black', linestyle='--')
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel('Finemap Beta')
ax.set_ylabel('BICAN log2(Alt) - log2(Ref)')
fig.savefig('figures/MetaBrain_top_variants_lfdc_comparison.pdf')
# %%
