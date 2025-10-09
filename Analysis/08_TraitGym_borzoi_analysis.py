# %%
import h5py
import numpy as np
import pandas as pd
import polars as pl
import os
import sys
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% load borzoi results
borzoi_h5 = h5py.File(f'Data/source/TraitGym/borzoi_grelu/complex_traits_test.h5', 'r')
log_square = borzoi_h5['results/log_square'][:]
borzoi_info_df = pd.DataFrame({'chr': borzoi_h5['variants/chr'][:],
                               'pos': borzoi_h5['variants/pos'][:],
                               'ref': borzoi_h5['variants/ref'][:],
                               'alt': borzoi_h5['variants/alt'][:]})
borzoi_info_df.index = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str) + '_b38'
# %% get borzoi res
# borzoi_res = pd.read_parquet('/gpfs/commons/groups/ren_lab/guojiezhong/TraitGym/results/dataset/complex_traits_matched_9/features/Borzoi_L2.parquet')
borzoi_res = pd.read_parquet('/gpfs/commons/groups/ren_lab/guojiezhong/TraitGym/complex_traits_borzoi_scores.parquet')
# %% read in TraitGym files
trait_info = pd.read_csv('Data/source/TraitGym/complex_traits_test.csv', sep=',')
# %% keep non-duplicate in borzoi_info_df and borzoi_h5
borzoi_info_df['ID'] = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str)
log_square = log_square[~borzoi_info_df['ID'].duplicated(keep='first')]
borzoi_info_df = borzoi_info_df[~borzoi_info_df['ID'].duplicated(keep='first')]
borzoi_info_df = borzoi_info_df.reset_index(drop=True)
# %% merge trait_info with borzoi_info_df
trait_info['ID'] = 'chr' + trait_info['chrom'].astype(str) + '_' + trait_info['pos'].astype(str) + '_' + trait_info['ref'].astype(str) + '_' + trait_info['alt'].astype(str)
merged_df = pd.merge(trait_info, borzoi_info_df, left_on='ID', right_on='ID', how='inner')
# %% calculate AUPRC with SE for each dimension
def compute_auprc_with_se(trait_values, score_values, n_bootstraps=100):
    from sklearn.metrics import average_precision_score

    # Convert to polars DataFrame for efficient resampling
    V = pl.DataFrame({"label": trait_values, "score": score_values})

    # Calculate main AUPRC
    try:
        auprc = average_precision_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # Bootstrap resampling within each class
    def resample(V, seed):
        V_pos = V.filter(pl.col("label"))
        V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
        V_neg = V.filter(~pl.col("label"))
        V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
        return pl.concat([V_pos, V_neg])

    # Calculate AUPRC on bootstrap samples
    V_bs = [resample(V, i) for i in range(n_bootstraps)]
    bootstrap_auprcs = []
    for V_b in V_bs:
        try:
            bootstrap_auprcs.append(average_precision_score(V_b["label"], V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_auprcs).std() if bootstrap_auprcs else np.nan

    return auprc, se

auprc = []
auprc_se = []
for dim in tqdm(range(borzoi_res.shape[1]), desc="Calculating AUPRC per dimension"):
    ap, se = compute_auprc_with_se(merged_df['label'].values, borzoi_res.iloc[:, dim].values)
    auprc.append(ap)
    auprc_se.append(se)

# %% get a sum of scores
total_score = np.sqrt(np.sum(borzoi_res**2, axis=1))
total_auprc, total_auprc_se = compute_auprc_with_se(trait_info['label'].values, total_score)
# %%
