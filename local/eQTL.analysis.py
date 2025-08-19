# %% import libs
import h5py
import numpy as np
import pandas as pd
import os
import sys
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
# %% get unique var names
eqtl['file'] = '/share/vault/Users/gz2294/BICAN/Res/250719_atac_rna_ft/analysis_best_valid_loss/raw_data/var_eff/' + \
    eqtl['#CHROM'].astype(str) + '_' + eqtl['REF'].astype(str) + eqtl['POS'].astype(str) + eqtl['ALT'].astype(str) + '.h5'
def variant_score(row):
    # sum
    with h5py.File(row['file'], 'r') as f:
        sum_score = f['data']['diff'][:].sum(axis=0)
        l2_score = np.sqrt(((np.log2(f['data']['pred_alt'][:]+1) - np.log2(f['data']['pred_wt'][:]+1))**2).sum(axis=0))
        # concat
        score = np.concatenate((sum_score, l2_score), axis=0)
    return score
# apply variant_score to all eqtl
eqtl_unique = eqtl.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
scores = np.vstack(eqtl_unique.apply(variant_score, axis=1))
# %%
