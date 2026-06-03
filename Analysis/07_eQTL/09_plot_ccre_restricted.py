#!/usr/bin/env python
"""Side-by-side plot: AUROC (+AUPRC) vs distance bin, comparing 'all' vs
'ccre_only' subsets across EpiBRAIN / AlphaGenome / Borzoi.

Inputs: output.ag_like/{model}_{organ}_agg_ccre_restricted.csv (from 08_*).
Outputs: output.ag_like/figure/{organ}_ccre_restricted_{AUROC|AUPRC}.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

fm._load_fontmanager(try_read_cache=False)
_ARIAL_VARIANTS = ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf', 'ariblk.ttf']
_TTF_DIR = os.path.join(os.path.dirname(fm.__file__), 'mpl-data', 'fonts', 'ttf')
for _f in _ARIAL_VARIANTS:
    _p = os.path.join(_TTF_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

PWD = f'{os.environ["workingHOME"]}/BICAN'
OUT_DIR = os.path.join(PWD, 'Analysis/07_eQTL/output.ag_like')
FIG_DIR = os.path.join(OUT_DIR, 'figure')

ORGANS = ['Cortex', 'Basal_ganglia']
MODELS = ['bican', 'alphagenome_524k', 'borzoi']
DISPLAY = {'bican': 'EpiBRAIN', 'alphagenome_524k': 'AlphaGenome 524k', 'borzoi': 'Borzoi'}
COLORS = {'bican': '#f3701b', 'alphagenome_524k': '#4bb062', 'borzoi': '#4a98c9'}
BINS = ['<3k', '3k-12k', '12k-35k', '>35k']


_MODE_SUFFIX = {
    'pooled': 'agg_ccre_restricted',
    'per_tissue_avg': 'per_tissue_avg_ccre_restricted',
}
_MODE_TAG = {'pooled': 'pooled', 'per_tissue_avg': 'per_tissue'}


def _load(organ: str, mode: str) -> pd.DataFrame:
    suffix = _MODE_SUFFIX[mode]
    frames = []
    for m in MODELS:
        p = os.path.join(OUT_DIR, f'{m}_{organ}_{suffix}.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def _plot_one(organ: str, metric: str, mode: str) -> None:
    se_col = f'{metric}_SE'
    df = _load(organ, mode)
    df = df[df['group'].isin(BINS)].copy()

    subsets = ['all', 'ccre_only', 'pos_in_ccre', 'neg_in_ccre',
               'pos_notin_ccre', 'neg_notin_ccre']
    sub_df = df[df['subset'].isin(subsets)]
    vals = sub_df[metric].dropna().values
    errs = sub_df[se_col].fillna(0).values
    if len(vals) > 0:
        lo = max(0.0, float(np.nanmin(vals - errs)) - 0.02)
        hi = min(1.0, float(np.nanmax(vals + errs)) + 0.02)
    else:
        lo, hi = 0.0, 1.0
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharey=True)
    for ax, subset in zip(axes.flat, subsets):
        sub = df[df['subset'] == subset]
        x = np.arange(len(BINS))
        for m in MODELS:
            mdf = sub[sub['model'] == m].set_index('group').reindex(BINS)
            y = mdf[metric].values.astype(float)
            err = mdf[se_col].values.astype(float)
            err = np.where(np.isnan(err), 0.0, err)
            ax.plot(x, y, marker='o', color=COLORS[m], label=DISPLAY[m], linewidth=2)
            ax.fill_between(x, y - err, y + err, color=COLORS[m], alpha=0.2, linewidth=0)
        n_pos = sub[sub['model'] == MODELS[0]].set_index('group').reindex(BINS)['n_pos'].astype(int)
        n_neg = sub[sub['model'] == MODELS[0]].set_index('group').reindex(BINS)['n_neg'].astype(int)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b}\n({p}/{n})' for b, p, n in zip(BINS, n_pos, n_neg)])
        ax.set_xlabel('distance-to-TSS bin  (n_pos / n_neg)')
        ax.set_title(f'{organ} — subset: {subset}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(lo, hi)
        if metric == 'AUROC':
            ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
    for r in range(n_rows):
        axes[r, 0].set_ylabel(metric)
    axes[0, 0].legend(loc='lower left', frameon=False)
    fig.suptitle(f'{organ}  |  {metric} vs distance-to-TSS  |  {_MODE_TAG[mode]}')
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = _MODE_TAG[mode]
    out = os.path.join(FIG_DIR, f'{organ}_ccre_restricted_{tag}_{metric}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--organs', nargs='+', default=ORGANS)
    ap.add_argument('--metrics', nargs='+', default=['AUROC', 'AUPRC'])
    ap.add_argument('--modes', nargs='+', choices=['pooled', 'per_tissue_avg'],
                    default=['pooled', 'per_tissue_avg'])
    args = ap.parse_args()
    for organ in args.organs:
        for metric in args.metrics:
            for mode in args.modes:
                _plot_one(organ, metric, mode)


if __name__ == '__main__':
    main()
