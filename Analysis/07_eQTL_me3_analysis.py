# %% import libs
"""
Explain why K27me3 and K9me3 tracks have higher AUROC scores in eQTL analysis.

This script directly fetches coverage from bigwig files at variant sites (±50bp)
and calculates AUROC scores to compare with model-based predictions.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# Make text editable in Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

PWD = f'/gpfs/commons/groups/ren_lab/guojiezhong/BICAN'
sys.path.append(f'{PWD}')
os.chdir(f'{PWD}')

# %% Define parameters
FLANK_SIZES = [50, 500, 5000, 50000]  # bp to extend on each side (100bp, 1kb, 10kb, 100kb windows)

# %% Read eQTL data
eqtl = pd.read_csv('Data/source/eQTL/all.vcf', sep='\t')
eqtl_info = pd.read_csv('Data/source/eQTL/info.csv', sep=',')

# Drop duplicates to get unique variants
eqtl_unique = eqtl.drop_duplicates(subset=['#CHROM', 'POS', 'REF', 'ALT']).copy()
print(f"Total unique variants: {len(eqtl_unique)}")
print(f"Positive: {(eqtl_unique['INFO'] == 'positive').sum()}")
print(f"Negative: {(eqtl_unique['INFO'] == 'negative').sum()}")

# %% Get bigwig file paths
k27ac_files = sorted(glob.glob('Data/source/bamCoverage_bw/*_merge_A.bw'))
k27me3_files = sorted(glob.glob('Data/source/bamCoverage_bw/*_merge_B.bw'))
k9me3_files = sorted(glob.glob('Data/source/bamCoverage_bw/*_merge_C.bw'))
atac_files = sorted(glob.glob('Data/source/ATAC_bw/*.bw'))

print(f"Found {len(k27ac_files)} K27ac tracks")
print(f"Found {len(k27me3_files)} K27me3 tracks")
print(f"Found {len(k9me3_files)} K9me3 tracks")
print(f"Found {len(atac_files)} ATAC tracks")

# %% Define helper functions
def fetch_coverage_at_variants(bw_file, variants_df, flank):
    """
    Fetch coverage from a bigwig file at variant positions ± flank.

    Parameters:
    -----------
    bw_file : str
        Path to bigwig file
    variants_df : pd.DataFrame
        DataFrame with '#CHROM' and 'POS' columns
    flank : int
        Base pairs to extend on each side

    Returns:
    --------
    np.ndarray
        Array of scores (mean coverage) for each variant
    """
    bw = pyBigWig.open(bw_file)
    scores = []

    for _, row in variants_df.iterrows():
        chrom = row['#CHROM']
        pos = row['POS']
        start = max(0, pos - flank)
        end = pos + flank

        try:
            # Fetch values in the region
            vals = bw.values(chrom, start, end)
            if vals is not None:
                # Use mean of non-NaN values as score
                vals = np.array(vals)
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    scores.append(np.mean(valid_vals))
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    bw.close()
    return np.array(scores)


def compute_auroc(labels, scores):
    """Compute AUROC with error handling."""
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return np.nan


def process_single_track(bw_file, variants_df, flank_size, labels, modality, suffix):
    """Process a single bigwig track and return results."""
    track_name = os.path.basename(bw_file).replace(suffix, '')
    scores = fetch_coverage_at_variants(bw_file, variants_df, flank_size)
    auroc = compute_auroc(labels, scores)
    result = {
        'track': track_name,
        'modality': modality,
        'flank_size': flank_size,
        'AUROC': auroc,
        'mean_score_pos': scores[labels == 1].mean(),
        'mean_score_neg': scores[labels == 0].mean()
    }
    return track_name, scores, result


# %% Fetch coverage and compute AUROC for each track and flank size
results = []
flank_results = []  # For tracking AUROC vs flank size
labels = (eqtl_unique['INFO'].values == 'positive').astype(int)

N_JOBS = 36  # Use all available cores

for flank_size in FLANK_SIZES:
    window_size = flank_size * 2
    print(f"\n{'='*60}")
    print(f"Processing with FLANK_SIZE = {flank_size} ({window_size}bp window)")
    print(f"{'='*60}")

    # Process K27ac tracks in parallel
    print("\nProcessing K27ac tracks...")
    k27ac_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_track)(bw_file, eqtl_unique, flank_size, labels, 'K27ac', '.bam_merge_A.bw')
        for bw_file in k27ac_files
    )
    k27ac_scores_all = [r[1] for r in k27ac_results]
    for track_name, scores, result in k27ac_results:
        results.append(result)
        print(f"  {track_name}: AUROC = {result['AUROC']:.4f}")

    # Process K27me3 tracks in parallel
    print("\nProcessing K27me3 tracks...")
    k27me3_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_track)(bw_file, eqtl_unique, flank_size, labels, 'K27me3', '.bam_merge_B.bw')
        for bw_file in k27me3_files
    )
    k27me3_scores_all = [r[1] for r in k27me3_results]
    for track_name, scores, result in k27me3_results:
        results.append(result)
        print(f"  {track_name}: AUROC = {result['AUROC']:.4f}")

    # Process K9me3 tracks in parallel
    print("\nProcessing K9me3 tracks...")
    k9me3_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_track)(bw_file, eqtl_unique, flank_size, labels, 'K9me3', '.bam_merge_C.bw')
        for bw_file in k9me3_files
    )
    k9me3_scores_all = [r[1] for r in k9me3_results]
    for track_name, scores, result in k9me3_results:
        results.append(result)
        print(f"  {track_name}: AUROC = {result['AUROC']:.4f}")

    # Process ATAC tracks in parallel
    print("\nProcessing ATAC tracks...")
    atac_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_track)(bw_file, eqtl_unique, flank_size, labels, 'ATAC', '.bw')
        for bw_file in atac_files
    )
    atac_scores_all = [r[1] for r in atac_results]
    for track_name, scores, result in atac_results:
        results.append(result)
        print(f"  {track_name}: AUROC = {result['AUROC']:.4f}")

    # Compute merged scores (L2 norm across tracks)
    print("\nComputing merged scores...")
    k27ac_merged = np.linalg.norm(np.array(k27ac_scores_all), axis=0)
    k27me3_merged = np.linalg.norm(np.array(k27me3_scores_all), axis=0)
    k9me3_merged = np.linalg.norm(np.array(k9me3_scores_all), axis=0)
    atac_merged = np.linalg.norm(np.array(atac_scores_all), axis=0)
    all_merged = np.linalg.norm(np.concatenate([k27ac_scores_all, k27me3_scores_all, k9me3_scores_all, atac_scores_all]), axis=0)

    # Store merged results
    merged_results = [
        ('ALL_K27ac', 'K27ac_merged', k27ac_merged),
        ('ALL_K27me3', 'K27me3_merged', k27me3_merged),
        ('ALL_K9me3', 'K9me3_merged', k9me3_merged),
        ('ALL_ATAC', 'ATAC_merged', atac_merged),
        ('ALL_marks', 'all_merged', all_merged)
    ]

    for track, modality, merged_scores in merged_results:
        auroc = compute_auroc(labels, merged_scores)
        results.append({
            'track': track,
            'modality': modality,
            'flank_size': flank_size,
            'AUROC': auroc,
            'mean_score_pos': merged_scores[labels == 1].mean(),
            'mean_score_neg': merged_scores[labels == 0].mean()
        })
        flank_results.append({
            'track': track,
            'flank_size': flank_size,
            'window_size': window_size,
            'AUROC': auroc
        })
        print(f"  {track}: AUROC = {auroc:.4f}")

# %% Create results DataFrame
results_df = pd.DataFrame(results)
flank_results_df = pd.DataFrame(flank_results)

print("\n" + "="*60)
print("Summary Results:")
print("="*60)
print(results_df.to_string())

# %% Save results
results_df.to_csv('Data/source/eQTL/me3_bigwig_auroc_results.csv', index=False)
flank_results_df.to_csv('Data/source/eQTL/me3_bigwig_auroc_by_flank.csv', index=False)
print(f"\nResults saved to Data/source/eQTL/me3_bigwig_auroc_results.csv")
print(f"Flank results saved to Data/source/eQTL/me3_bigwig_auroc_by_flank.csv")

# %% Plot AUROC vs Flank Size for ALL_* tracks
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors and markers for each track
track_styles = {
    'ALL_K27ac': {'color': '#e41a1c', 'marker': 'o'},
    'ALL_K27me3': {'color': '#1f77b4', 'marker': 's'},
    'ALL_K9me3': {'color': '#ff7f0e', 'marker': '^'},
    'ALL_ATAC': {'color': '#984ea3', 'marker': 'd'},
    'ALL_marks': {'color': '#2ca02c', 'marker': 'v'}
}

for track in ['ALL_K27ac', 'ALL_K27me3', 'ALL_K9me3', 'ALL_ATAC', 'ALL_marks']:
    track_data = flank_results_df[flank_results_df['track'] == track]
    ax.plot(track_data['window_size'], track_data['AUROC'],
            marker=track_styles[track]['marker'],
            color=track_styles[track]['color'],
            label=track, linewidth=2, markersize=8)

ax.set_xscale('log')
ax.set_xlabel('Window Size (bp)')
ax.set_ylabel('AUROC')
ax.set_title('AUROC vs Window Size for Merged Tracks')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.legend(loc='best')
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(['100bp', '1kb', '10kb', '100kb'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/me3_bigwig_auroc_vs_flank.pdf')
print("Plot saved to figures/me3_bigwig_auroc_vs_flank.pdf")

# %% Plot AUROC distribution by modality (for default flank size = 500)
default_flank = 500
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Filter for default flank size and individual tracks only (not merged)
results_default = results_df[results_df['flank_size'] == default_flank]
individual_tracks = results_default[~results_default['modality'].str.contains('merged')]

# Violin plot
ax = axes[0]
sns.violinplot(data=individual_tracks, x='modality', y='AUROC', ax=ax)
ax.set_title(f'AUROC Distribution by Modality (±{default_flank}bp)')
ax.set_ylabel('AUROC')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Bar plot of merged scores
ax = axes[1]
merged_tracks = results_default[results_default['modality'].str.contains('merged')]
colors = ['#e41a1c', '#1f77b4', '#ff7f0e', '#984ea3', '#2ca02c']  # K27ac, K27me3, K9me3, ATAC, all
bars = ax.bar(merged_tracks['track'], merged_tracks['AUROC'], color=colors)
ax.set_title(f'Merged AUROC Scores (±{default_flank}bp)')
ax.set_ylabel('AUROC')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xticklabels(merged_tracks['track'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/me3_bigwig_auroc_analysis.pdf')
print("Plot saved to figures/me3_bigwig_auroc_analysis.pdf")

# %% Compare positive vs negative mean coverage (for default flank size)
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(individual_tracks))
width = 0.35

ax.bar(x - width/2, individual_tracks['mean_score_pos'], width, label='Positive eQTLs')
ax.bar(x + width/2, individual_tracks['mean_score_neg'], width, label='Negative controls')

ax.set_ylabel('Mean Coverage')
ax.set_title(f'Mean Coverage at Variant Sites (±{default_flank}bp)')
ax.set_xticks(x)
ax.set_xticklabels(individual_tracks['track'], rotation=90, ha='center')
ax.legend()

plt.tight_layout()
plt.savefig('figures/me3_bigwig_coverage_comparison.pdf')
print("Plot saved to figures/me3_bigwig_coverage_comparison.pdf")

# %% Print interpretation
print("\n" + "="*60)
print("Interpretation:")
print("="*60)
print("""
The AUROC scores computed here represent how well the raw bigwig coverage
at variant sites (±50bp) can distinguish positive eQTLs from negative controls.

If these AUROC scores are similar to the model-based predictions from
07_eQTL_analysis.py, it suggests that the model is primarily learning to
recognize the baseline coverage patterns at K27me3/K9me3 regions, rather
than learning more complex regulatory features.

Higher coverage at certain genomic regions may simply correlate with:
1. Open chromatin regions (where eQTLs are more likely)
2. Active regulatory elements
3. Gene body regions

This baseline signal may explain why K27me3 and K9me3 tracks show higher
AUROC in the model predictions.
""")
# %%
