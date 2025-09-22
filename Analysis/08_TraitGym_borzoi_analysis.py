# %%
import h5py
import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from functools import partial
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% load borzoi results
borzoi_h5 = h5py.File(f'Data/source/TraitGym/basal_ganglia_v1_res/mendelian_traits_test.h5', 'r')
log_square = borzoi_h5['results/log_square'][:]
borzoi_info_df = pd.DataFrame({'chr': borzoi_h5['variants/chr'][:],
                               'pos': borzoi_h5['variants/pos'][:],
                               'ref': borzoi_h5['variants/ref'][:],
                               'alt': borzoi_h5['variants/alt'][:]})
borzoi_info_df.index = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str) + '_b38'
# %% get borzoi res
borzoi_res = pd.read_parquet('/gpfs/commons/groups/ren_lab/guojiezhong/TraitGym/results/dataset/complex_traits_matched_9/features/Borzoi_L2.parquet')

# %% read in TraitGym files
trait_info = pd.read_csv('Data/source/TraitGym/mendelian_traits_test.csv', sep=',')
# %% keep non-duplicate in borzoi_info_df and borzoi_h5
borzoi_info_df['ID'] = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str)
log_square = log_square[~borzoi_info_df['ID'].duplicated(keep='first')]
borzoi_info_df = borzoi_info_df[~borzoi_info_df['ID'].duplicated(keep='first')]
borzoi_info_df = borzoi_info_df.reset_index(drop=True)
# %% merge trait_info with borzoi_info_df
trait_info['ID'] = 'chr' + trait_info['chrom'].astype(str) + '_' + trait_info['pos'].astype(str) + '_' + trait_info['ref'].astype(str) + '_' + trait_info['alt'].astype(str)
merged_df = pd.merge(trait_info, borzoi_info_df, left_on='ID', right_on='ID', how='inner')
# %% calculate AUROC for each dimension
def compute_auprc(trait_values, log_square_values):
    from sklearn.metrics import auc, precision_recall_curve
    try:
        # get pr curve
        precision, recall, _ = precision_recall_curve(trait_values, log_square_values)
        # calculate auprc
        aupr = auc(recall, precision)
    except ValueError:
        aupr = np.nan
    return aupr
auprc = []
for dim in range(log_square.shape[1]):
    auprc.append(compute_auprc(merged_df['label'], log_square[:, dim]))

# %% get a sum of scores
total_auprc = compute_auprc(merged_df['label'], np.sum(log_square**2, axis=1))
# %%
