# %% import libs
import os
import sys
import argparse

import h5py
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='eQTL Borzoi track analysis')
parser.add_argument('--filter', type=str, default=None,
                    choices=['brain', 'basal_ganglia', 'cortex', 'gtex_brain'],
                    help='Filter tracks by region: brain, basal_ganglia, cortex, or gtex_brain')
args = parser.parse_args()
track_filter = args.filter

PWD = f'{os.environ["workingHOME"]}/BICAN'
sys.path.append(f'{PWD}')
os.chdir(f'{PWD}')
# %% brain-only track filter option
brain_track_pattern = (
    r'brain|cerebr|hippocamp|amygdal|'
    r'frontal.*(lobe|gyrus|cortex|area)|'
    r'neuron|astrocyte|oligodendrocyte|microglia|cerebellum|thalamus|'
    r'parietal.*(lobe|cortex)|temporal.*(lobe|gyrus)|occipital|'
    r'putamen|caudate|substantia|nucleus accumbens|cingulate|'
    r'spinal cord|neurosphere'
)
basal_ganglia_track_pattern = (
    r'putamen|caudate|nucleus accumbens|globus pallidus|striatum|basal.ganglia'
)
cortex_track_pattern = (
    r'cerebral cortex|frontal cortex|occipital cortex|parietal cortex|'
    r'frontal.*(?:lobe|gyrus)|parietal lobe|temporal lobe|occipital.*(?:lobe|pole)|'
    r'cingulate|prefrontal'
)
gtex_brain_track_pattern = r'^RNA:brain$'
embryo_fetal_track_pattern = r'embryo|fetal|fetus|embryonic|prenatal|newborn'
disease_track_pattern = r'disease|disorder|syndrome|cancer|tumor|tumour|carcinoma|leukemia|lymphoma|alzheimer|parkinson|huntington|autism|schizophrenia|epilepsy|atrophy|injury|stroke|glioma|glioblastoma'
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')
# %% borzoi results
eqtl_h5 = h5py.File(f'Data/source/eQTL/borzoi_res.h5', 'r')
log_square = eqtl_h5['results/log_square'][:]
eqtl_info_df = pd.DataFrame({'chr': eqtl_h5['variants/chr'][:],
                             'pos': eqtl_h5['variants/pos'][:],
                             'ref': eqtl_h5['variants/ref'][:],
                             'alt': eqtl_h5['variants/alt'][:]})
eqtl_info_df.index = eqtl_info_df['chr'].astype(str) + '_' + eqtl_info_df['pos'].astype(str) + '_' + eqtl_info_df['ref'].astype(str) + '_' + eqtl_info_df['alt'].astype(str) + '_b38'
# %% define biotypes to keep
biotype_to_keep = ['protein_coding', 'lncRNA', 'IG_V_gene', 'TR_V_gene', 'IG_C_gene', 'snoRNA', 'snRNA', 'TR_C_gene', 'miRNA']
brain_organs = ['Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
                'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
                'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
                'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra']
basal_ganglia_organs = ['Brain_Caudate_basal_ganglia', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia']
cortex_organs = ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9']

# %% Define helper function to compute metrics
def compute_auroc_with_se(trait_values, score_values, n_bootstraps=100):
    """
    Compute AUROC with standard error using bootstrap resampling.

    Parameters:
    -----------
    trait_values : array-like
        Binary labels (0 or 1)
    score_values : array-like
        Prediction scores
    n_bootstraps : int or None
        Number of bootstrap samples (default: 100)
        If None, only compute AUROC without SE

    Returns:
    --------
    tuple: (auroc, se)
    """
    # Calculate main AUROC
    try:
        auroc = roc_auc_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # If n_bootstraps is None, skip SE calculation
    if n_bootstraps is None:
        return auroc, np.nan

    # Convert to boolean if needed, then to polars DataFrame for efficient resampling
    trait_values_bool = np.asarray(trait_values).astype(bool)
    V = pl.DataFrame({"label": trait_values_bool, "score": score_values})

    # Bootstrap resampling within each class
    def resample(V, seed):
        V_pos = V.filter(pl.col("label"))
        V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
        V_neg = V.filter(~pl.col("label"))
        V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
        return pl.concat([V_pos, V_neg])

    # Calculate AUROC on bootstrap samples
    V_bs = [resample(V, i) for i in range(n_bootstraps)]
    bootstrap_aurocs = []
    for V_b in V_bs:
        try:
            # Convert back to int for sklearn compatibility
            bootstrap_aurocs.append(roc_auc_score(V_b["label"].cast(pl.Int64), V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_aurocs).std() if bootstrap_aurocs else np.nan

    return auroc, se

def compute_auprc_with_se(trait_values, score_values, n_bootstraps=100):
    """
    Compute AUPRC with standard error using bootstrap resampling.

    Parameters:
    -----------
    trait_values : array-like
        Binary labels (0 or 1)
    score_values : array-like
        Prediction scores
    n_bootstraps : int or None
        Number of bootstrap samples (default: 100)
        If None, only compute AUPRC without SE

    Returns:
    --------
    tuple: (auprc, se)
    """
    # Calculate main AUPRC
    try:
        auprc = average_precision_score(trait_values, score_values)
    except ValueError:
        return np.nan, np.nan

    # If n_bootstraps is None, skip SE calculation
    if n_bootstraps is None:
        return auprc, np.nan

    # Convert to boolean if needed, then to polars DataFrame for efficient resampling
    trait_values_bool = np.asarray(trait_values).astype(bool)
    V = pl.DataFrame({"label": trait_values_bool, "score": score_values})

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
            # Convert back to int for sklearn compatibility
            bootstrap_auprcs.append(average_precision_score(V_b["label"].cast(pl.Int64), V_b["score"]))
        except ValueError:
            continue

    # Calculate SE as standard deviation of bootstrap estimates
    se = pl.Series(bootstrap_auprcs).std() if bootstrap_auprcs else np.nan

    return auprc, se

def compute_metrics(label, scores, eqtl_organ_unique, group, n_bootstraps=100):
    """
    Compute AUROC, AUPRC with SE, and positive/negative counts for a given group.

    Parameters:
    -----------
    label : array-like
        Binary labels (0 or 1)
    scores : array-like
        Prediction scores
    eqtl_organ_unique : pd.DataFrame
        DataFrame containing eQTL information with 'INFO' and 'group' columns
    group : str
        Group name ('all' for all data, or specific group like '<3k', '3k-12k', etc.)
    n_bootstraps : int or None
        Number of bootstrap samples for SE calculation (default: 100)
        If None, only compute metrics without SE

    Returns:
    --------
    tuple: (auroc, auroc_se, auprc, auprc_se, positives, negatives)
    """
    if group == 'all':
        # Compute metrics on all data
        auroc, auroc_se = compute_auroc_with_se(label, scores, n_bootstraps)
        auprc, auprc_se = compute_auprc_with_se(label, scores, n_bootstraps)
        positives = (eqtl_organ_unique['INFO'] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'] == 'negative').sum()
    else:
        # Compute metrics for specific group
        group_mask = eqtl_organ_unique['group'] == group
        auroc, auroc_se = compute_auroc_with_se(label[group_mask], scores[group_mask], n_bootstraps)
        auprc, auprc_se = compute_auprc_with_se(label[group_mask], scores[group_mask], n_bootstraps)
        positives = (eqtl_organ_unique['INFO'][group_mask] == 'positive').sum()
        negatives = (eqtl_organ_unique['INFO'][group_mask] == 'negative').sum()

    return auroc, auroc_se, auprc, auprc_se, positives, negatives

# %%
track_results = pd.DataFrame()
for organ in ['All', 'Brain', 'Basal_ganglia', 'Cortex']:
    # if is directory
    if organ == 'All':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
    elif os.path.isdir(f'Data/source/eQTL/{organ}'):
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/{organ}/all.vcf', sep='\t')
        # get info
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/{organ}/info.csv')
    elif organ == 'Brain':
        # all brain eQTL
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(brain_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    elif organ == 'Basal_ganglia':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(basal_ganglia_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    elif organ == 'Cortex':
        eqtl_organ = pd.read_csv(f'Data/source/eQTL/all.vcf', sep='\t')
        eqtl_organ_info = pd.read_csv(f'Data/source/eQTL/info.csv')
        eqtl_organ_info = eqtl_organ_info[eqtl_organ_info['tissue'].isin(cortex_organs)]
        eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    else:
        continue
    # filter eqtl_organ and eqtl_organ_info by biotype_to_keep
    eqtl_organ_info = eqtl_organ_info.loc[(eqtl_organ_info['biotype'].isin(biotype_to_keep)) | (eqtl_organ['INFO'] == 'negative')]
    eqtl_organ = eqtl_organ.loc[eqtl_organ_info.index]
    eqtl_organ['group'] = eqtl_organ_info['group'].groupby(eqtl_organ_info['variant_id']).first().loc[eqtl_organ['ID']].values
    # drop duplicates
    eqtl_organ_unique = eqtl_organ.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT'])
    eqtl_scores = log_square[eqtl_info_df.index.get_indexer(eqtl_organ_unique['ID'])]
    # load track annotations
    track_anno = pd.read_csv('borzoi.published.targets.txt', index_col=0, sep='\t')
    track_anno['modality'] = track_anno['file'].str.split('/').str[6]
    # modality filter: keep CAGE, DNASE, RNA, and CHIP for H3K27ac/H3K27me3/H3K9me3 only
    chip_histone_pattern = r'H3K27ac|H3K27me3|H3K9me3'
    modality_mask = (
        track_anno['description'].str.contains(r'\bCAGE\b|\bDNASE\b|\bDNase\b|\bRNA\b', case=False, na=False) |
        track_anno['description'].str.contains(chip_histone_pattern, case=False, na=False)
    )
    modality_indices = track_anno.index[modality_mask].values
    track_anno = track_anno[modality_mask].reset_index(drop=True)
    eqtl_scores = eqtl_scores[:, modality_indices]
    if track_filter:
        filter_pattern_map = {
            'brain': brain_track_pattern,
            'basal_ganglia': basal_ganglia_track_pattern,
            'cortex': cortex_track_pattern,
            'gtex_brain': gtex_brain_track_pattern,
        }
        embryo_fetal_mask = track_anno['description'].str.contains(embryo_fetal_track_pattern, case=False, na=False)
        disease_mask = track_anno['description'].str.contains(disease_track_pattern, case=False, na=False)
        track_mask = track_anno['description'].str.contains(filter_pattern_map[track_filter], case=False, na=False) & ~embryo_fetal_mask & ~disease_mask
        track_indices = track_anno.index[track_mask].values
        track_anno = track_anno.loc[track_mask].reset_index(drop=True)
        # print number of tracks kept
        print(f'Organ: {organ}, Filter: {track_filter}, Tracks kept: {len(track_indices)}\n{track_anno["description"].values}')
        eqtl_scores = eqtl_scores[:, track_indices]
    track_anno['organ'] = organ
    # get overall precision recall
    for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
        track_anno['group'] = group
        label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
        for i, idx in enumerate(track_anno.index):
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, eqtl_scores[:, i], eqtl_organ_unique, group, n_bootstraps=None)
            track_anno.loc[idx, 'AUROC'] = auroc
            track_anno.loc[idx, 'AUROC_SE'] = auroc_se
            track_anno.loc[idx, 'AUPRC'] = auprc
            track_anno.loc[idx, 'AUPRC_SE'] = auprc_se
            track_anno.loc[idx, f'n_pos'] = positives
            track_anno.loc[idx, f'n_neg'] = negatives
        for i, modality in enumerate(track_anno['modality'].unique()):
            modality_idx = track_anno.index[track_anno['modality'] == modality].values
            # L2 norm across modality_idx
            modality_merged = np.linalg.norm(eqtl_scores[:, modality_idx], axis=1)
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
            track_results = pd.concat([track_results,
                                       pd.DataFrame({'identifier': f'borzoi_{modality}', 'modality': modality,
                                                     'organ': organ, 'group': group,
                                                     'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                     f'n_pos': positives, f'n_neg': negatives},
                                                    index=[0])], ignore_index=True)
        # get overall all modality
        modality_merged = np.linalg.norm(eqtl_scores, axis=1)
        auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
        track_results = pd.concat([track_results,
                                   pd.DataFrame({'identifier': f'borzoi_all', 'modality': 'ALL',
                                                 'organ': organ, 'group': group,
                                                 'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                 f'n_pos': positives, f'n_neg': negatives},
                                                index=[0])], ignore_index=True)
        track_results = pd.concat([track_results, track_anno])
# %% add annotation
track_results['mod'] = 'borzoi'
# %% save results
suffix = f'_{track_filter}' if track_filter else ''
track_results.to_csv(f'Data/source/eQTL/borzoi_track_results{suffix}.csv')
# %% plot
organs = ['All', 'Brain', 'Basal_ganglia']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = track_results[track_results['organ'] == organ].copy().dropna()
    sns.violinplot(data=to_plot, x='group', y='AUROC', ax=ax)
    ax.set_title(organ)
# %%
