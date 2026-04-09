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
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='eQTL analysis')
parser.add_argument('--borzoi-filter', type=str, default=None,
                    choices=['brain', 'basal_ganglia', 'cortex', 'gtex_brain'],
                    help='Use filtered borzoi track results: brain, basal_ganglia, cortex, or gtex_brain')
parser.add_argument('--ag-filter', type=str, default=None,
                    choices=['brain', 'basal_ganglia', 'cortex', 'gtex_brain', 'gtex_basal_ganglia', 'gtex_cortex'],
                    help='Use filtered alphagenome track results')
parser.add_argument('--bican-filter', type=str, default=None,
                    choices=['basal_ganglia', 'cortex', 'basal_ganglia_rna', 'cortex_rna'],
                    help='Filter BICAN tracks: basal_ganglia (BasalGanglia), cortex (MiniAtlas), or _rna variants (RNA only)')
parser.add_argument('--organ', type=str, default='Basal_ganglia',
                    help='Organ to analyze (e.g. Basal_ganglia, Cortex, Brain)')
args = parser.parse_args()
borzoi_filter = args.borzoi_filter
ag_filter = args.ag_filter
bican_filter = args.bican_filter
organ = args.organ

# Make text editable in Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

PWD = f'{os.environ["workingHOME"]}/BICAN'
sys.path.append(f'{PWD}')
os.chdir(f'{PWD}')
bican_suffix = f'_{bican_filter}' if bican_filter else ''
track_results_path = f'Data/source/eQTL/full_finetune_original_loss_celltype_head_dim8_linear.track_results{bican_suffix}.csv'
if os.path.exists(track_results_path):
    track_results = pd.read_csv(track_results_path, index_col=0)
else:
    # %% read in eQTL files
    eqtl = pd.read_csv('Data/source/eQTL/all.v39.vcf', sep='\t')
    eqtl_info = pd.read_csv('Data/source/eQTL/info.v39.csv', sep=',')
    # %% read in results
    eqtl_h5 = h5py.File(f'Data/source/eQTL/full_finetune.dim8.chk20.h5', 'r')
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
        track_anno = pd.read_csv('./logs/full_finetune_original_loss_celltype_head_dim8_linear/regression_label_meta.csv')
        # Apply BICAN track filter
        if bican_filter:
            bican_filter_map = {
                'basal_ganglia': ('BasalGanglia', None),
                'cortex': ('MiniAtlas', None),
                'basal_ganglia_rna': ('BasalGanglia', r'RNA'),
                'cortex_rna': ('MiniAtlas', r'RNA'),
            }
            ct_prefix, mod_pattern = bican_filter_map[bican_filter]
            bican_mask = track_anno['cell_type'].str.startswith(ct_prefix, na=False)
            if mod_pattern:
                bican_mask = bican_mask & track_anno['modality'].str.contains(mod_pattern, case=False, na=False)
            bican_indices = track_anno.index[bican_mask].values
            track_anno = track_anno.loc[bican_mask].reset_index(drop=True)
            eqtl_scores = eqtl_scores[:, bican_indices]
            print(f'Organ: {organ}, BICAN filter: {bican_filter}, Tracks kept: {len(track_anno)}')
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
                                           pd.DataFrame({'exp': f'bican_{modality}', 'modality': modality, 'celltype': 'ALL',
                                                         'organ': organ, 'group': group,
                                                         'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                         f'n_pos': positives, f'n_neg': negatives},
                                                        index=[0])], ignore_index=True)
            for i, modality in enumerate(track_anno['cell_type'].unique()):
                modality_idx = track_anno.index[(track_anno['cell_type'] == modality) &
                                                (track_anno['modality'].str.contains('RNA') == False) &
                                                (track_anno['modality'].str.contains('K9Me3') == False)].values
                # L2 norm across modality_idx
                modality_merged = np.linalg.norm(eqtl_scores[:, modality_idx], axis=1)
                auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
                track_results = pd.concat([track_results,
                                           pd.DataFrame({'exp': f'bican_celltype_{modality}', 'modality': 'ALL', 'celltype': modality,
                                                         'organ': organ, 'group': group,
                                                         'AUROC': auroc, 'AUROC_SE': auroc_se, 'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                                                         f'n_pos': positives, f'n_neg': negatives},
                                                        index=[0])], ignore_index=True)
            # combine everything
            modality_merged = np.linalg.norm(eqtl_scores, axis=1)
            auroc, auroc_se, auprc, auprc_se, positives, negatives = compute_metrics(label, modality_merged, eqtl_organ_unique, group, n_bootstraps=50)
            track_results = pd.concat([track_results,
                                       pd.DataFrame({'exp': f'bican_all', 'modality': 'ALL', 'celltype': 'ALL',
                                                     'organ': organ, 'group': group, 'AUROC': auroc, 'AUROC_SE': auroc_se,
                                                     'AUPRC': auprc, 'AUPRC_SE': auprc_se, f'n_pos': positives, f'n_neg': negatives},
                                                    index=[0])], ignore_index=True)
            track_results = pd.concat([track_results, track_anno])
    # %% add annotation
    track_results.loc[track_results['exp'].isna(), 'exp'] = track_results['trial'][track_results['exp'].isna()].copy()
    # %% save results
    track_results.to_csv(track_results_path)
# %% Add borzoi results
borzoi_suffix = f'_{borzoi_filter}' if borzoi_filter else ''
borzoi = pd.read_csv(f'Data/source/eQTL/borzoi_track_results{borzoi_suffix}.csv', index_col=0)
borzoi['exp'] = borzoi['identifier'].copy()
borzoi.loc[borzoi['identifier'].str.startswith('borzoi'), 'celltype'] = 'ALL'
track_results = pd.concat([track_results, borzoi])
# %% Add alphagenome results
ag_suffix = f'_{ag_filter}' if ag_filter else ''
alphagenome = pd.read_csv(f'Data/source/eQTL/alphagenome_track_results{ag_suffix}.csv', index_col=0)
alphagenome['exp'] = alphagenome['identifier'].copy()
alphagenome.loc[alphagenome['identifier'].str.startswith('alphagenome', na=False), 'celltype'] = 'ALL'
track_results = pd.concat([track_results, alphagenome])
# %% Add chrombpnet results
# chrombpnet = pd.read_csv('Data/source/eQTL/chrombpnet_miniatlas_ATAC_results.csv', index_col=0)
# chrombpnet['mod'] = 'chrombpnet'
# chrombpnet.loc[chrombpnet['exp'].str.startswith('ATAC'), 'celltype'] = 'ALL'
# track_results = pd.concat([track_results, chrombpnet])
# %% split bican/borzoi/ATAC and track results
model_mask = track_results['exp'].str.startswith(('bican', 'borzoi', 'alphagenome', 'ATAC'), na=False)
track_results_model = track_results[model_mask].copy()
track_results_track = track_results[~model_mask].copy()
track_results_track.loc[track_results_track['mod'].isna(), 'mod'] = track_results_track['modality'][track_results_track['mod'].isna()].copy()
track_results_celltype = track_results_model[track_results_model['celltype'] != 'ALL'].copy()
track_results_overall = track_results_model[track_results_model['celltype'] == 'ALL'].copy()
# %% plot
organs = ['All', 'Brain', 'Basal_ganglia', 'Cortex']
for organ in organs:
    fig, ax = plt.subplots(figsize=(10, 6))
    # drop na values
    to_plot = track_results_track[track_results_track['organ'] == organ].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
    sns.violinplot(data=to_plot, x='group', y='AUROC', hue='mod', ax=ax)
    ax.set_title(organ)
# %% plot cell type performances
fig, ax = plt.subplots(figsize=(10, 6))
to_plot = track_results_celltype[(track_results_celltype['organ'] == organ) & 
                                 (track_results_celltype['group'] == 'all')].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
# rank by AUROC
to_plot = to_plot.sort_values(by='AUROC', ascending=False)
to_plot['type'] = 'borzoi'
to_plot.loc[to_plot['exp'].str.contains('bican_celltype_BasalGanglia'), 'type'] = 'bican:BasalGanglia'
to_plot.loc[to_plot['exp'].str.contains('bican_celltype_MiniAtlas'), 'type'] = 'bican:MiniAtlas'
to_plot.loc[to_plot['exp'] == 'ATAC', 'type'] = 'chrombpnet'
to_plot['exp'] = to_plot['exp'].str.replace('bican_celltype_', '')

# Create bar plot with error bars
x_pos = np.arange(len(to_plot))
colors = {'borzoi': '#1f77b4', 'bican:BasalGanglia': '#ff7f0e', 'bican:MiniAtlas': '#2ca02c', 'chrombpnet': '#d62728'}
bar_colors = [colors[t] for t in to_plot['type']]
bars = ax.bar(x_pos, to_plot['AUROC'], yerr=to_plot['AUROC_SE'], capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(to_plot['exp'], rotation=45, ha='right')
ax.set_ylabel('AUROC')
ax.set_title(organ)

# Add legend
legend_elements = [Patch(facecolor=colors[t], edgecolor='black', label=t) for t in to_plot['type'].unique()]
ax.legend(handles=legend_elements, title='Type')
# %% plot overall model performances
organ = args.organ
variant_groups = ['all', '<3k', '3k-12k', '12k-35k', '>35k']

# Prepare data for all groups
data_by_group = {}
for group in variant_groups:
    df = track_results_overall[(track_results_overall['organ'] == organ) &
                                (track_results_overall['group'] == group)].copy().dropna(axis=0, subset=['AUPRC', 'AUROC'])
    # Filter out RNAminus and RNAplus
    df = df[~df['exp'].str.contains('RNAminus|RNAplus', na=False)]

    df['type'] = 'borzoi'
    df.loc[df['exp'].str.contains('bican'), 'type'] = 'bican'
    df.loc[df['exp'].str.contains('alphagenome'), 'type'] = 'alphagenome'
    df.loc[df['exp'] == 'ATAC', 'type'] = 'chrombpnet'
    data_by_group[group] = df

# Get unique experiments (using 'all' group as reference) and their types
if 'all' in data_by_group and len(data_by_group['all']) > 0:
    df_all = data_by_group['all'].sort_values(by='AUROC', ascending=False)
    exp_order = df_all['exp'].tolist()
    exp_types = dict(zip(df_all['exp'], df_all['type']))
else:
    # Fallback: get all unique experiments across groups
    all_exps = set()
    exp_types = {}
    for df in data_by_group.values():
        all_exps.update(df['exp'].tolist())
        for _, row in df.iterrows():
            exp_types[row['exp']] = row['type']
    exp_order = sorted(list(all_exps))

# Create gradient colors for groups (from dark to light)
# All = darkest, >35k = lightest
borzoi_colors = plt.cm.Blues(np.linspace(0.9, 0.3, len(variant_groups)))  # Dark to light blue
bican_colors = plt.cm.Oranges(np.linspace(0.9, 0.3, len(variant_groups)))  # Dark to light orange
alphagenome_colors = plt.cm.Greens(np.linspace(0.9, 0.3, len(variant_groups)))  # Dark to light green
atac_colors = plt.cm.Reds(np.linspace(0.9, 0.3, len(variant_groups)))  # Dark to light red

# Create grouped bar plot
fig, ax = plt.subplots(figsize=(16, 6))
bar_width = 0.15
x_pos = np.arange(len(exp_order))

for idx, group in enumerate(variant_groups):
    if group not in data_by_group or len(data_by_group[group]) == 0:
        continue

    df = data_by_group[group]
    auroc_values = []
    auroc_se_values = []
    bar_colors = []

    for exp in exp_order:
        exp_data = df[df['exp'] == exp]
        if len(exp_data) > 0:
            auroc_values.append(exp_data['AUROC'].values[0])
            auroc_se_values.append(exp_data['AUROC_SE'].values[0])

            # Get color based on experiment type
            exp_type = exp_types.get(exp, 'borzoi')
            if exp_type == 'borzoi':
                bar_colors.append(borzoi_colors[idx])
            elif exp_type == 'bican':
                bar_colors.append(bican_colors[idx])
            elif exp_type == 'alphagenome':
                bar_colors.append(alphagenome_colors[idx])
            else:  # chrombpnet
                bar_colors.append(atac_colors[idx])
        else:
            auroc_values.append(0)
            auroc_se_values.append(0)
            bar_colors.append('gray')

    # Plot bars for this group
    offset = (idx - len(variant_groups)/2 + 0.5) * bar_width

    # Plot each bar individually to allow different colors
    for i, (val, se, color) in enumerate(zip(auroc_values, auroc_se_values, bar_colors)):
        if i == 0:  # Only add label once per group
            ax.bar(x_pos[i] + offset, val, bar_width,
                   yerr=se, capsize=3,
                   label=group, color=color,
                   alpha=0.8, edgecolor='black', linewidth=0.5)
        else:
            ax.bar(x_pos[i] + offset, val, bar_width,
                   yerr=se, capsize=3,
                   color=color,
                   alpha=0.8, edgecolor='black', linewidth=0.5)

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(exp_order, rotation=45, ha='right')
ax.set_ylabel('AUROC')
ax.set_ylim(ymin=0.4, ymax=1)
ax.set_title(f'{organ} - AUROC by Variant Distance Groups')
ax.legend(title='Distance Group', loc='upper right')
fig.tight_layout()
fig.savefig(f'figures/{organ}_overall_model_performance_by_variant_distance_groups_AUROC.pdf')
# %% plot comparison with borzoi in each group
exp_order = ['bican_all', 'borzoi_all', 'alphagenome_all']

fig, ax = plt.subplots(figsize=(10, 4))
bar_width = 0.25
x_pos = np.arange(len(variant_groups))

for exp_idx, exp in enumerate(exp_order):
    auroc_values = []
    auroc_se_values = []
    bar_colors = []

    # Get experiment type to determine color scheme
    exp_type = exp_types.get(exp, 'borzoi')

    for group_idx, group in enumerate(variant_groups):
        if group not in data_by_group or len(data_by_group[group]) == 0:
            auroc_values.append(0)
            auroc_se_values.append(0)
            bar_colors.append('gray')
            continue

        df = data_by_group[group]
        exp_data = df[df['exp'] == exp]

        if len(exp_data) > 0:
            auroc_values.append(exp_data['AUROC'].values[0])
            auroc_se_values.append(exp_data['AUROC_SE'].values[0])

            # Get gradient color based on experiment type and variant group
            if exp_type == 'borzoi':
                bar_colors.append(borzoi_colors[group_idx])
            elif exp_type == 'bican':
                bar_colors.append(bican_colors[group_idx])
            elif exp_type == 'alphagenome':
                bar_colors.append(alphagenome_colors[group_idx])
            else:  # chrombpnet
                bar_colors.append(atac_colors[group_idx])
        else:
            auroc_values.append(0)
            auroc_se_values.append(0)
            bar_colors.append('gray')

    # Plot bars for this experiment with gradient colors (no labels yet)
    offset = (exp_idx - len(exp_order)/2 + 0.5) * bar_width
    for i, (val, se, color) in enumerate(zip(auroc_values, auroc_se_values, bar_colors)):
        ax.bar(x_pos[i] + offset, val, bar_width,
               yerr=se, capsize=3,
               color=color,
               alpha=0.8, edgecolor='black', linewidth=0.5)

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(variant_groups, rotation=45, ha='right')
ax.set_ylabel('AUROC')
ax.set_ylim(ymin=0.3, ymax=1)
ax.set_title(f'{organ} - AUROC by Variant Distance Groups')

# Create custom legend with 3k-12k colors and clean labels
from matplotlib.patches import Patch
legend_color_idx = 2  # Index for '3k-12k' in variant_groups
legend_elements = []
for exp in exp_order:
    exp_type = exp_types.get(exp, 'borzoi')
    if exp_type == 'borzoi':
        legend_color = borzoi_colors[legend_color_idx]
    elif exp_type == 'bican':
        legend_color = bican_colors[legend_color_idx]
    elif exp_type == 'alphagenome':
        legend_color = alphagenome_colors[legend_color_idx]
    else:  # chrombpnet
        legend_color = atac_colors[legend_color_idx]
    # Remove '_all' suffix from label
    clean_label = exp.replace('_all', '')
    legend_elements.append(Patch(facecolor=legend_color, edgecolor='black', label=clean_label, alpha=0.8))

ax.legend(handles=legend_elements, title='Model', loc='upper right')
fig.tight_layout()
fig.savefig(f'figures/{organ}_comparison_by_variant_groups_AUROC.pdf')

# %%
