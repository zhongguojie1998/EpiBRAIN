# %% import libs
import os
import sys

import h5py
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

PWD = f'{os.environ["workingHOME"]}/BICAN'
sys.path.append(f'{PWD}')
os.chdir(f'{PWD}')
# %% read in eQTL files
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')
# %% chrombpnet results
cell_types = []
for cell_type in os.listdir('Data/source/chrombpnet/BICAN/ATAC/'):
    if not os.path.isfile(f'Data/source/chrombpnet/BICAN/ATAC/{cell_type}/variant_scores.tsv'):
        continue
    cell_types.append(cell_type)
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
result_df = pd.DataFrame()
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
    # for each cell type get the chrombpnet scores
    for cell_type in cell_types + ['ATAC']:
        if cell_type == 'ATAC':
            # aggregate all cell types and get a L2 norm
            logfc_list = []
            variant_ids = None
            for ct in cell_types:
                ct_res = pd.read_csv(f'Data/source/chrombpnet/BICAN/ATAC/{ct}/variant_scores.tsv', sep='\t')
                # Get first logfc value for each variant
                ct_logfc = ct_res['logfc'].groupby(ct_res['variant_id']).first()
                if variant_ids is None:
                    variant_ids = ct_logfc.index
                logfc_list.append(ct_logfc.loc[variant_ids].values)

            # Stack logfc values and compute L2 norm across cell types
            logfc_array = np.stack(logfc_list, axis=1)  # shape: (n_variants, n_cell_types)
            l2_norm_scores = np.linalg.norm(logfc_array, axis=1)  # L2 norm across cell types

            # Create a DataFrame for compatibility with downstream code
            chrombpnet_res = pd.DataFrame({
                'variant_id': variant_ids,
                'logfc': l2_norm_scores
            })
        else:
            chrombpnet_res = pd.read_csv(f'Data/source/chrombpnet/BICAN/ATAC/{cell_type}/variant_scores.tsv', sep='\t')
        eqtl_scores = chrombpnet_res['logfc'].groupby(chrombpnet_res['variant_id']).first().loc[eqtl_organ_unique['ID']].values
        # get overall precision recall
        for group in ['all', '<3k', '3k-12k', '12k-35k', '>35k']:
            label = (eqtl_organ_unique['INFO'].values == 'positive').astype(int)
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, eqtl_scores, eqtl_organ_unique, group, n_bootstraps=50)
            result_df = pd.concat([result_df,
                                   pd.DataFrame({'exp': cell_type, 'modality': 'ATAC', 'organ': organ, 'group': group,
                                                 'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                 'positives': positives, 'negatives': negatives},
                                                index=[0])], ignore_index=True)
            
# %% save results
result_df.to_csv('Data/source/eQTL/chrombpnet_miniatlas_ATAC_results.csv')
# %% plot
organs = ['All', 'Brain', 'Basal_ganglia']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = result_df[result_df['organ'] == organ].copy().dropna()
    sns.violinplot(data=to_plot, x='group', y='AUROC', ax=ax)
    ax.set_title(organ)
# %%
