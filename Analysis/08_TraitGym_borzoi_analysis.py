# %%
import h5py
import numpy as np
import pandas as pd
import polars as pl
import os
import sys
from joblib import Parallel, delayed
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')

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
# %%
# Parallel computation helper functions
def compute_dim_auprc(dim, label_values, score_matrix):
    """Compute AUPRC for a single dimension (works with numpy arrays)"""
    return compute_auprc_with_se(label_values, score_matrix[:, dim])

# %%
if __name__ == "__main__":
    # %% load bican results
    bican_h5 = h5py.File(f'Data/source/TraitGym/basal_ganglia_miniatlas_drop_celltype_v1/mendelian_traits_test.h5', 'r')
    bican_res = bican_h5['results/log_square'][:]
    bican_info_df = pd.DataFrame({'chr': bican_h5['variants/chr'][:],
                                  'pos': bican_h5['variants/pos'][:],
                                  'ref': bican_h5['variants/ref'][:],
                                  'alt': bican_h5['variants/alt'][:]})
    bican_info_df.index = bican_info_df['chr'].astype(str) + '_' + bican_info_df['pos'].astype(str) + '_' + bican_info_df['ref'].astype(str) + '_' + bican_info_df['alt'].astype(str) + '_b38'
    bican_info_df['ID'] = bican_info_df['chr'].astype(str) + '_' + bican_info_df['pos'].astype(str) + '_' + bican_info_df['ref'].astype(str) + '_' + bican_info_df['alt'].astype(str)
    bican_res = bican_res[~bican_info_df['ID'].duplicated(keep='first')]
    bican_info_df = bican_info_df[~bican_info_df['ID'].duplicated(keep='first')]
    bican_info_df = bican_info_df.reset_index(drop=True)

    # %% get borzoi results
    borzoi_h5 = h5py.File(f'Data/source/TraitGym/borzoi_grelu/mendelian_traits_test.h5', 'r')
    borzoi_res = borzoi_h5['results/log_square'][:]
    borzoi_info_df = pd.DataFrame({'chr': borzoi_h5['variants/chr'][:],
                                    'pos': borzoi_h5['variants/pos'][:],
                                    'ref': borzoi_h5['variants/ref'][:],
                                    'alt': borzoi_h5['variants/alt'][:]})
    borzoi_info_df.index = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str) + '_b38'
    borzoi_info_df['ID'] = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str)
    borzoi_res = borzoi_res[~borzoi_info_df['ID'].duplicated(keep='first')]
    borzoi_info_df = borzoi_info_df[~borzoi_info_df['ID'].duplicated(keep='first')]
    borzoi_info_df = borzoi_info_df.reset_index(drop=True)

    # %% read in TraitGym files
    trait_info = pd.read_csv('Data/source/TraitGym/mendelian_traits_test.csv', sep=',')

    # %% merge trait_info with borzoi_info_df
    trait_info['ID'] = 'chr' + trait_info['chrom'].astype(str) + '_' + trait_info['pos'].astype(str) + '_' + trait_info['ref'].astype(str) + '_' + trait_info['alt'].astype(str)
    borzoi_info_df = pd.merge(borzoi_info_df, trait_info, left_on='ID', right_on='ID', how='inner')
    bican_info_df = pd.merge(bican_info_df, trait_info, left_on='ID', right_on='ID', how='inner')

    # %% read in track annotations
    borzoi_track_anno = pd.read_csv('Data/data_config/borzoi.published.targets.csv', sep=',', index_col=0)
    bican_track_anno = pd.read_csv('Data/data_config/basal_ganglia_miniatlas_drop_celltype_v1.csv', sep=',', index_col=0)

    # %% Parallel AUPRC calculation
    print(f"Calculating AUPRC per dimension for Borzoi (parallel, {borzoi_res.shape[1]} dimensions)...")
    borzoi_results = Parallel(n_jobs=36, backend='loky', verbose=10)(
        delayed(compute_dim_auprc)(dim, borzoi_info_df['label'].values, borzoi_res)
        for dim in range(borzoi_res.shape[1])
    )
    borzoi_auprc, borzoi_auprc_se = zip(*borzoi_results)

    # Append to borzoi_track_anno
    borzoi_track_anno['AUPRC'] = borzoi_auprc
    borzoi_track_anno['AUPRC_SE'] = borzoi_auprc_se

    print(f"Calculating AUPRC per dimension for BICAN (parallel, {bican_res.shape[1]} dimensions)...")
    bican_results = Parallel(n_jobs=36, backend='loky', verbose=10)(
        delayed(compute_dim_auprc)(dim, bican_info_df['label'].values, bican_res)
        for dim in range(bican_res.shape[1])
    )
    bican_auprc, bican_auprc_se = zip(*bican_results)

    # Append to bican_track_anno
    bican_track_anno['AUPRC'] = list(bican_auprc)
    bican_track_anno['AUPRC_SE'] = list(bican_auprc_se)

    # %% save results
    borzoi_track_anno.to_csv('Data/source/TraitGym/borzoi_mendelian_track_results.csv')
    bican_track_anno.to_csv('Data/source/TraitGym/bican_mendelian_track_results.csv')

    print("Done! Results saved.")