# %% 
import pandas as pd
import numpy as np
import os
import pyBigWig
from dotenv import load_dotenv
load_dotenv()
PWD = f'{os.environ["workingHOME"]}/BICAN'
import sys
os.chdir(PWD)
sys.path.append(PWD)
# %%
# read Single Brain Fine Mapped eQTL results
eqtl_res = pd.read_csv('Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/var_eff/variant_effects_by_gene.tsv', sep='\t')

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

# %%
