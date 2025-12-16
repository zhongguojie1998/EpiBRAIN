# %%
import pandas as pd
import numpy as np
import os
import pyBigWig
from dotenv import load_dotenv
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score
import seaborn as sns
load_dotenv()
PWD = f'{os.environ["workingHOME"]}/BICAN'
import sys
os.chdir(PWD)
sys.path.append(PWD)
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
eqtl_res = eqtl_res[eqtl_res['track'].str.contains('MiniAtlas-')]
# remove the minus or plus from track names
eqtl_res['track'] = eqtl_res['track'].str.replace('minus', '').str.replace('plus', '')
# only keep the RNA results
eqtl_res = eqtl_res[eqtl_res['track'].str.contains('RNA')]
# add a column to indicate the eqtl cell type name
eqtl_res['cell_type'] = eqtl_res['track'].str.replace('_RNA', '').str.replace('MiniAtlas-', '')
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

# %%
# read Meta Brain Fine Mapped eQTL results
eqtl_res = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/var_eff/variant_effects_by_gene.tsv', sep='\t')
eqtl_finemap = pd.read_csv('Data/source/MetaBrain/2021-07-23-basalganglia-EUR-30PCs-TopEffects.txt.gz', sep='\t', compression='gzip')

# %%
