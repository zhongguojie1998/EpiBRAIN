#!/usr/bin/env python3
"""
Overlap EpiBRAIN-prioritized variants (top K27Ac quantile) with fine-mapped eQTL
variants via ATAC peaks, per cell type.

Usage:
    python 06_overlap_eQTL.py --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
                              --trait schizophrenia
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import pyranges as pr
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import importlib.util
from pathlib import Path

# Make text editable in Adobe Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

BASAL_GANGLIA_EQTL_TISSUES = [
    "Brain_Caudate_basal_ganglia",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
]


def load_variants_and_scores_from_h5(h5_file, trait_name):
    """Load hg38 variant positions and annotation matrix from HDF5 (no liftover)."""
    print(f"Loading variants and scores from {h5_file} for trait '{trait_name}'...")
    res = h5py.File(h5_file, mode='r')

    if f"experiments/{trait_name}" not in res:
        available = list(res['experiments'].keys()) if 'experiments' in res else []
        print(f"Error: Trait '{trait_name}' not found. Available: {available}")
        res.close()
        sys.exit(1)

    tracks = res["model_meta/trial_names"][:]
    track_names = [t.decode('utf-8') if isinstance(t, bytes) else t for t in tracks]

    exp_index = res[f"experiments/{trait_name}/index_key"][:]
    all_variant_index = res["variants/index_key"][:]

    all_variant_to_pos = {val: idx for idx, val in enumerate(all_variant_index)}
    positions = [all_variant_to_pos[val] for val in exp_index]
    sort_idx = np.argsort(positions)
    positions = np.array(positions)[sort_idx]

    # Annotation matrix (variants x tracks)
    exp_var_score = res["results/local_log_square"][positions, :]

    # Variant metadata in hg38
    variants = pd.DataFrame({
        'CHR': res['variants/chr'][positions],
        'BP': res['variants/pos'][positions],
        'SNP': res['variants/rsid'][positions],
    })
    res.close()

    variants['CHR'] = variants['CHR'].astype(str)
    variants['SNP'] = variants['SNP'].astype(str)
    variants['BP'] = variants['BP'].astype(int)

    print(f"  Loaded {len(variants):,} variants x {len(track_names)} tracks")
    return variants, exp_var_score, track_names


def load_atac_peaks(peaks_bed):
    """Load ATAC peaks BED file into PyRanges."""
    peaks = pd.read_csv(peaks_bed, sep='\t', header=None,
                        names=['Chromosome', 'Start', 'End', 'peak_id'])
    print(f"  Loaded {len(peaks):,} ATAC peaks from {peaks_bed}")
    return pr.PyRanges(peaks)


def load_subclass_peaks(peaks_tsv):
    """Load per-subclass MACS3 peaks TSV (has header) into PyRanges.

    Expected columns: chrom, start, end, name, score, strand, signal_value,
    p_value, q_value, peak.
    """
    peaks = pd.read_csv(peaks_tsv, sep='\t')
    peaks = peaks[['chrom', 'start', 'end']].copy()
    peaks.columns = ['Chromosome', 'Start', 'End']
    peaks['peak_id'] = (peaks['Chromosome'].astype(str) + ':' +
                        peaks['Start'].astype(str) + '-' +
                        peaks['End'].astype(str))
    return pr.PyRanges(peaks)


def load_eqtl_finemapped(eqtl_dir, tissues, pip_threshold):
    """Load and aggregate fine-mapped eQTL variants from multiple tissues."""
    all_variants = []
    for tissue in tissues:
        info_path = os.path.join(eqtl_dir, tissue, "info.csv")
        if not os.path.exists(info_path):
            print(f"  Warning: {info_path} not found, skipping")
            continue
        df = pd.read_csv(info_path)
        df_fm = df[df['pip'] > pip_threshold][['chr', 'pos', 'pip']].copy()
        print(f"  {tissue}: {len(df_fm):,} variants with PIP > {pip_threshold} (of {len(df):,} total)")
        all_variants.append(df_fm)

    if not all_variants:
        print("Error: No eQTL variants loaded")
        sys.exit(1)

    combined = pd.concat(all_variants, ignore_index=True)
    # Deduplicate by chr + pos, keep max PIP
    combined = combined.groupby(['chr', 'pos']).agg({'pip': 'max'}).reset_index()
    print(f"  Total unique fine-mapped eQTL variants: {len(combined):,}")
    return combined


def variants_to_pyranges(chr_col, pos_col):
    """Convert variant positions to single-bp PyRanges intervals."""
    df = pd.DataFrame({
        'Chromosome': chr_col.values,
        'Start': pos_col.values - 1,  # 0-based
        'End': pos_col.values,
    })
    return pr.PyRanges(df)


def overlap_peaks(variant_pr, peaks_pr):
    """Return set of peak_ids that overlap with any variant."""
    joined = peaks_pr.join(variant_pr)
    if len(joined) == 0:
        return set()
    return set(joined.df['peak_id'].values)


def main():
    parser = argparse.ArgumentParser(description='EpiBRAIN vs eQTL ATAC peak overlap')
    parser.add_argument('--h5-file', default='Data/source/GWAS/full_finetune.dim8.chk20.h5')
    parser.add_argument('--trait', default='Schizophrenia_fullinfo.sumstats')
    parser.add_argument('--atac-peaks', default='Data/source/ATAC_peak/Subclass.filtered.peaks.bed',
                        help='(Unused for per-cell-type overlap; kept for backward compat.)')
    parser.add_argument('--subclass-peaks-dir',
                        default='/gpfs/commons/groups/ren_lab/guojiezhong/Data/BICAN/ATAC-seq/Subclass.peaks',
                        help='Directory with per-subclass MACS3 peak TSVs: {Subclass}.macs3.peak.tsv')
    parser.add_argument('--eqtl-dir', default='Data/source/eQTL')
    parser.add_argument('--pip-threshold', type=float, default=0.5)
    parser.add_argument('--top-pct', type=float, default=5.0,
                        help='Top percentage of variants to use (default: 5%%)')
    parser.add_argument('--output-dir', default='Analysis/12_s_ldsc_continous/results_overlap')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Load data ---
    variants, scores, track_names = load_variants_and_scores_from_h5(args.h5_file, args.trait)

    print("\nLoading eQTL fine-mapped variants...")
    eqtl_df = load_eqtl_finemapped(args.eqtl_dir, BASAL_GANGLIA_EQTL_TISSUES, args.pip_threshold)
    eqtl_pr = variants_to_pyranges(eqtl_df['chr'], eqtl_df['pos'])

    # --- Step 2: Per-subclass K27Ac quantile → overlap (set_2 computed per cell type) ---
    k27ac_tracks = [(i, name) for i, name in enumerate(track_names)
                    if name.startswith('BasalGanglia-') and name.endswith('_K27Ac')]
    print(f"\nFound {len(k27ac_tracks)} BasalGanglia K27Ac tracks")

    results = []
    for track_idx, track_name in k27ac_tracks:
        # Extract subclass name: BasalGanglia-{Subclass}_K27Ac → Subclass
        subclass = track_name.replace('BasalGanglia-', '').replace('_K27Ac', '')

        # Load this subclass's ATAC peaks (cell-type-specific cCREs).
        # Track names use '-' as separator; peak filenames use '_'.
        subclass_fname = subclass.replace('-', '_')
        peaks_tsv = os.path.join(args.subclass_peaks_dir, f'{subclass_fname}.macs3.peak.tsv')
        if not os.path.exists(peaks_tsv):
            print(f"  {subclass}: peaks file {peaks_tsv} not found, skipping")
            continue
        peaks_pr = load_subclass_peaks(peaks_tsv)

        # set_2: eQTL cCREs restricted to this cell type's peaks
        set_2 = overlap_peaks(eqtl_pr, peaks_pr)

        k27ac_scores = scores[:, track_idx]

        # Select top N% of variants by K27Ac score
        threshold = np.percentile(k27ac_scores, 100 - args.top_pct)
        mask = k27ac_scores >= threshold

        top_variants = variants[mask]
        n_top = len(top_variants)

        if n_top == 0:
            print(f"  {subclass}: 0 variants in top {args.top_pct}%, skipping")
            continue

        # Overlap top variants with this cell type's ATAC peaks → set_1
        top_pr = variants_to_pyranges(top_variants['CHR'], top_variants['BP'])
        set_1 = overlap_peaks(top_pr, peaks_pr)

        # Compute overlap
        intersection = set_1 & set_2
        pct_set1_in_set2 = len(intersection) / len(set_1) * 100 if set_1 else 0.0
        pct_set2_in_set1 = len(intersection) / len(set_2) * 100 if set_2 else 0.0

        results.append({
            'subclass': subclass,
            'n_top_variants': n_top,
            'set_1_size': len(set_1),
            'set_2_size': len(set_2),
            'intersection_size': len(intersection),
            'pct_set1_in_set2': pct_set1_in_set2,
            'pct_set2_in_set1': pct_set2_in_set1,
        })
        print(f"  {subclass}: {n_top:,} top variants -> {len(set_1):,} peaks | "
              f"overlap={len(intersection):,} | "
              f"set1_in_set2={pct_set1_in_set2:.1f}% | set2_in_set1={pct_set2_in_set1:.1f}%")

    if not results:
        print("No results computed.")
        sys.exit(1)

    results_df = pd.DataFrame(results)

    # --- Step 4: Save CSV (include top_pct in filename) ---
    csv_path = os.path.join(args.output_dir, f'{args.trait}_eqtl_overlap_top{args.top_pct:.0f}pct.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # --- Step 5: Load subclass colors ---
    colors_file = os.path.join(os.path.dirname(__file__), '..', '00_basalganglia_subclass_colors.py')
    subclass_colors = {}
    if os.path.exists(colors_file):
        spec = importlib.util.spec_from_file_location("subclass_colors_module", colors_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Map: replace spaces with hyphens to match track naming
        subclass_colors = {k.replace(' ', '-'): v for k, v in mod.subclass_colors.items()}

    # --- Step 6: Plot ---
    results_df = results_df.sort_values('pct_set1_in_set2', ascending=False)

    x = np.arange(len(results_df))
    width = 0.35

    colors_set1 = [subclass_colors.get(s, '#808080') for s in results_df['subclass']]

    fig, ax = plt.subplots(figsize=(max(8, len(results_df) * 0.5), 5))
    bars1 = ax.bar(x - width / 2, results_df['pct_set1_in_set2'], width,
                   label='% EpiBRAIN peaks in eQTL peaks (set1 in set2)',
                   color=colors_set1, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, results_df['pct_set2_in_set1'], width,
                   label='% eQTL peaks in EpiBRAIN peaks (set2 in set1)',
                   color='#4a90d9', edgecolor='black', linewidth=0.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['subclass'], rotation=45, ha='right')
    ax.set_ylabel('Overlap Percentage (%)')
    ax.set_title(f'{args.trait} — EpiBRAIN (top {args.top_pct:.0f}% K27Ac) vs eQTL (PIP>{args.pip_threshold}) ATAC Peak Overlap')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    pdf_path = os.path.join(args.output_dir, f'{args.trait}_eqtl_overlap.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")
    plt.close(fig)


def plot_summary():
    """Plot summary across all thresholds. Run after 06_overlap_eQTL_by_quantile.sh."""
    parser = argparse.ArgumentParser(description='Plot summary of eQTL overlap across thresholds')
    parser.add_argument('--trait', default='Schizophrenia_fullinfo.sumstats')
    parser.add_argument('--output-dir', default='Analysis/12_s_ldsc_continous/results_overlap')
    args = parser.parse_args()

    # --- Load all threshold CSVs ---
    import glob
    csv_files = sorted(glob.glob(os.path.join(args.output_dir, f'{args.trait}_eqtl_overlap_top*pct.csv')))
    if not csv_files:
        print(f"No CSV files found in {args.output_dir} for trait {args.trait}")
        sys.exit(1)

    all_results = []
    for f in csv_files:
        basename = os.path.basename(f)
        # Extract top_pct from filename: {trait}_eqtl_overlap_top{N}pct.csv
        top_pct = float(basename.split('_top')[1].split('pct')[0])
        df = pd.read_csv(f)
        df['top_pct'] = top_pct
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    thresholds = sorted(combined['top_pct'].unique())
    subclasses = sorted(combined['subclass'].unique())
    print(f"Loaded {len(csv_files)} threshold files: {thresholds}")
    print(f"Cell types: {len(subclasses)}")

    # --- Load subclass colors ---
    colors_file = os.path.join(os.path.dirname(__file__), '..', '00_basalganglia_subclass_colors.py')
    subclass_colors = {}
    if os.path.exists(colors_file):
        spec = importlib.util.spec_from_file_location("subclass_colors_module", colors_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        subclass_colors = {k.replace(' ', '-'): v for k, v in mod.subclass_colors.items()}

    # --- Plot 1: Bar plot of set2_in_set1 per cell type (at top 5% threshold) ---
    df_5pct = combined[combined['top_pct'] == thresholds[0]].copy()
    df_5pct = df_5pct.sort_values('pct_set2_in_set1', ascending=False)
    colors_bar = [subclass_colors.get(s, '#808080') for s in df_5pct['subclass']]

    fig, ax = plt.subplots(figsize=(max(8, len(df_5pct) * 0.5), 5))
    x = np.arange(len(df_5pct))
    ax.bar(x, df_5pct['pct_set2_in_set1'], color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_5pct['subclass'], rotation=45, ha='right')
    ax.set_ylabel('% eQTL peaks in EpiBRAIN peaks')
    ax.set_title(f'{args.trait} — % eQTL peaks captured by EpiBRAIN top {thresholds[0]:.0f}% K27Ac')
    fig.tight_layout()
    pdf1 = os.path.join(args.output_dir, f'{args.trait}_eqtl_overlap_barplot.pdf')
    fig.savefig(pdf1, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot to {pdf1}")
    plt.close(fig)

    # --- Plot 2: Line plot per cell type across thresholds ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for subclass in subclasses:
        df_sub = combined[combined['subclass'] == subclass].sort_values('top_pct')
        color = subclass_colors.get(subclass, '#808080')
        axes[0].plot(df_sub['top_pct'], df_sub['pct_set1_in_set2'],
                     marker='o', markersize=3, color=color, label=subclass)
        axes[1].plot(df_sub['top_pct'], df_sub['pct_set2_in_set1'],
                     marker='o', markersize=3, color=color, label=subclass)

    axes[0].set_xlabel('Top % K27Ac variants')
    axes[0].set_ylabel('% EpiBRAIN peaks in eQTL peaks')
    axes[0].set_title('set1 in set2')
    axes[1].set_xlabel('Top % K27Ac variants')
    axes[1].set_ylabel('% eQTL peaks in EpiBRAIN peaks')
    axes[1].set_title('set2 in set1')

    # Single legend outside
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.suptitle(f'{args.trait} — EpiBRAIN vs eQTL overlap across thresholds', y=1.02)
    fig.tight_layout()
    pdf2 = os.path.join(args.output_dir, f'{args.trait}_eqtl_overlap_lineplot.pdf')
    fig.savefig(pdf2, dpi=300, bbox_inches='tight')
    print(f"Saved line plot to {pdf2}")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'plot-summary':
        sys.argv.pop(1)  # remove subcommand so argparse works
        plot_summary()
    else:
        main()
