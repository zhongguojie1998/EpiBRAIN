#!/usr/bin/env python
"""Evaluate and plot eQTL metrics restricted to test-set genome regions.

Filters variants to those whose positions fall within regions marked 'test'
in Data/basal_ganglia_miniatlas_drop_celltype_v1/sequences.bed.  For each
tissue, computes per-tissue AUROC/AUPRC on the test-set variants, then
averages across tissues (mean ± SE).  No cCRE partitioning.

Outputs:
  output.ag_like/{model}_{organ}_testset_per_tissue.csv
  output.ag_like/{model}_{organ}_testset_per_tissue_avg.csv
  output.ag_like/figure/{organ}_testset_{AUROC|AUPRC}.pdf
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

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

_CMAP_CYCLE = [
    plt.cm.Oranges, plt.cm.Reds, plt.cm.Greens,
    plt.cm.Purples, plt.cm.Blues, plt.cm.cool,
]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_data = importlib.import_module('Analysis.07_eQTL.data')
_cfg = importlib.import_module('Analysis.07_eQTL.config')
_metrics = importlib.import_module('Analysis.07_eQTL.metrics')

os.chdir(_cfg.PWD)

VARIANT_GROUPS = _cfg.VARIANT_GROUPS
ORGAN_TISSUE_MAP = _cfg.ORGAN_TISSUE_MAP
compute_metrics = _metrics.compute_metrics
load_tissue_vcf = _data.load_tissue_vcf
load_model = _data.load_model
set_paths_json = _data.set_paths_json

PWD = _cfg.PWD
OUT_DIR = os.path.join(PWD, 'Analysis/07_eQTL/output.ag_like')
FIG_DIR = os.path.join(OUT_DIR, 'figure')

SEQUENCES_BED = os.path.join(
    PWD, 'Data/basal_ganglia_miniatlas_drop_celltype_v1/sequences.bed'
)

ORGANS = ['Cortex', 'Basal_ganglia']
MODELS = ['bican', 'borzoi', 'alphagenome_524k']
DISPLAY = {'bican': 'EpiBRAIN', 'alphagenome_524k': 'AlphaGenome 524k', 'borzoi': 'Borzoi'}
BINS = ['<3k', '3k-12k', '12k-35k', '>35k']

MODEL_FILTER = {
    'Cortex': {'bican': 'cortex', 'borzoi': 'cortex', 'alphagenome_524k': 'cortex'},
    'Basal_ganglia': {'bican': 'basal_ganglia', 'borzoi': 'basal_ganglia',
                      'alphagenome_524k': 'basal_ganglia'},
}


# ---------------------------------------------------------------------------
# Test-region index
# ---------------------------------------------------------------------------

def _build_test_index(bed_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return chrom → (starts_sorted, ends_sorted) for all 'test' rows."""
    bed = pd.read_csv(bed_path, sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'split'])
    test = bed[bed['split'] == 'test'].copy()
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, grp in test.groupby('chrom'):
        starts = grp['start'].values
        order = np.argsort(starts)
        index[chrom] = (starts[order], grp['end'].values[order])
    return index


def _in_test_regions(variant_ids: pd.Series,
                     test_index: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return boolean mask: True if variant falls in a test region.

    Variant IDs are expected in the format chrN_POS_REF_ALT_b38.
    """
    mask = np.zeros(len(variant_ids), dtype=bool)
    for i, vid in enumerate(variant_ids):
        parts = vid.split('_')
        if len(parts) < 2:
            continue
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except ValueError:
            continue
        if chrom not in test_index:
            continue
        starts, ends = test_index[chrom]
        idx = np.searchsorted(starts, pos, side='right') - 1
        if idx >= 0 and pos < ends[idx]:
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _metric_rows(labels: np.ndarray, scores: np.ndarray,
                 vcf_df: pd.DataFrame, *, tag: dict) -> list[dict]:
    rows = []
    for group in VARIANT_GROUPS:
        auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
            labels, scores, vcf_df, group, n_bootstraps=100,
        )
        rows.append({**tag, 'group': group,
                     'AUROC': auroc, 'AUROC_SE': auroc_se,
                     'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                     'n_pos': n_pos, 'n_neg': n_neg})
    return rows


def eval_per_tissue_avg_testset(model_name: str, organ: str, json_path: str,
                                test_index: dict) -> pd.DataFrame:
    set_paths_json(json_path)
    track_filter = MODEL_FILTER[organ][model_name]
    print(f'\n=== TESTSET per-tissue | {model_name} | organ={organ} | '
          f'filter={track_filter} ===', flush=True)
    model_data = load_model(model_name, track_filter, add_addition=False)
    variant_index = model_data.variant_index
    log_square = model_data.log_square
    track_anno_base = model_data.track_anno
    tissue_filter_fn = model_data.tissue_filter_fn

    tissues = ORGAN_TISSUE_MAP[organ]
    per_tissue_rows: list[dict] = []

    for tissue in tissues:
        tissue_dir = os.path.join(PWD, 'Data/source/eQTL', tissue)
        if not os.path.isdir(tissue_dir):
            continue
        vcf_unique, labels = load_tissue_vcf(tissue_dir, add_addition=False)
        idx_in_h5 = variant_index.get_indexer(vcf_unique['ID'])
        keep = idx_in_h5 >= 0
        vcf_unique = vcf_unique.loc[keep].reset_index(drop=True)
        labels = labels[keep]
        h5_idx = idx_in_h5[keep]

        in_test = _in_test_regions(vcf_unique['ID'], test_index)
        vcf_unique = vcf_unique.loc[in_test].reset_index(drop=True)
        labels = labels[in_test]
        h5_idx = h5_idx[in_test]

        if len(labels) == 0:
            continue

        if tissue_filter_fn is not None:
            _, col_indices = tissue_filter_fn(tissue, track_anno_base)
            if len(col_indices) == 0:
                continue
            scores_mat = log_square[h5_idx][:, col_indices]
        else:
            scores_mat = log_square[h5_idx]
        l2 = np.linalg.norm(scores_mat, axis=1)

        print(f'  [{tissue}] n={len(labels)}, n_pos={int(labels.sum())}, '
              f'n_neg={int((labels == 0).sum())}', flush=True)

        tag = {'model': model_name, 'organ': organ, 'tissue': tissue,
               'track_filter': track_filter, 'n_tracks': scores_mat.shape[1]}
        per_tissue_rows.extend(_metric_rows(labels, l2, vcf_unique, tag=tag))

    per_tissue = pd.DataFrame(per_tissue_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    pt_out = os.path.join(OUT_DIR, f'{model_name}_{organ}_testset_per_tissue.csv')
    per_tissue.to_csv(pt_out, index=False)
    print(f'Saved: {pt_out}')

    # Average across tissues: mean; SE = (1/n)*sqrt(sum(SE_i^2)); sum n_pos/n_neg.
    key_cols = ['model', 'organ', 'group', 'track_filter']
    valid = per_tissue.dropna(subset=['AUROC'])
    agg_rows = []
    for keys, sub in valid.groupby(key_cols, dropna=False):
        row = dict(zip(key_cols, keys))
        row['n_tissues'] = len(sub)
        for m in ('AUROC', 'AUPRC'):
            vals = sub[m].dropna()
            se_vals = sub[f'{m}_SE'].dropna().values
            row[m] = float(vals.mean()) if len(vals) else np.nan
            row[f'{m}_SE'] = (float((1.0 / len(se_vals)) * np.sqrt(np.sum(se_vals ** 2)))
                              if len(se_vals) > 0 else np.nan)
        row['n_pos'] = int(sub['n_pos'].sum())
        row['n_neg'] = int(sub['n_neg'].sum())
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg_out = os.path.join(OUT_DIR, f'{model_name}_{organ}_testset_per_tissue_avg.csv')
    agg.to_csv(agg_out, index=False)
    print(f'Saved: {agg_out}')
    return agg


# ---------------------------------------------------------------------------
# Plot helpers (matching plot.py style)
# ---------------------------------------------------------------------------

def _annotate_bar(ax, x: float, bar_top: float, se: float, val: float,
                  fontsize: int = 6) -> None:
    if np.isnan(bar_top):
        return
    y = bar_top + (se if se and not np.isnan(se) else 0)
    ax.text(x, y, f'{val:.3f}', ha='center', va='bottom',
            fontsize=fontsize, rotation=0, clip_on=True)


def _annotate_group_counts(ax, x: float, n_pos, n_neg, fontsize: int = 6) -> None:
    try:
        label = f'+{int(n_pos)}\n-{int(n_neg)}'
    except (ValueError, TypeError):
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x, 0.01, label, ha='center', va='bottom',
            fontsize=fontsize, color='dimgray', transform=trans, clip_on=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _load_for_plot(organ: str) -> pd.DataFrame:
    frames = []
    for m in MODELS:
        p = os.path.join(OUT_DIR, f'{m}_{organ}_testset_per_tissue_avg.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_one(organ: str, metric: str,
              ymin: float | None = None, ymax: float | None = None) -> None:
    se_col = f'{metric}_SE'
    df = _load_for_plot(organ)
    if df.empty:
        print(f'No data for {organ}, skipping plot.')
        return
    groups = VARIANT_GROUPS
    df = df[df['group'].isin(groups)].copy()

    def _cmap_for(m: str):
        m_l = m.lower()
        if 'borzoi' in m_l:
            return plt.cm.Blues
        if 'alphagenome' in m_l:
            return plt.cm.Greens
        if 'bican' in m_l or 'epirain' in m_l:
            return plt.cm.Oranges
        return _CMAP_CYCLE[MODELS.index(m) % len(_CMAP_CYCLE)]
    color_maps = {m: _cmap_for(m) for m in MODELS}
    color_arrays = {m: color_maps[m](np.linspace(0.9, 0.3, len(groups))) for m in MODELS}

    bw = 0.15
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for mi, m in enumerate(MODELS):
        mdf = df[df['model'] == m].set_index('group').reindex(groups)
        vals = mdf[metric].values.astype(float)
        ses = mdf[se_col].values.astype(float)
        ses = np.where(np.isnan(ses), 0.0, ses)
        off = (mi - len(MODELS) / 2 + 0.5) * bw
        for gi, (v, s, c) in enumerate(zip(vals, ses, color_arrays[m])):
            kw = {'label': DISPLAY[m]} if gi == 0 else {}
            ax.bar(x[gi] + off, v, bw, yerr=s, capsize=3,
                   color=c, alpha=0.8, edgecolor='black', linewidth=0.5, **kw)
            _annotate_bar(ax, x[gi] + off, v, s, v)

    ref_model = MODELS[0]
    ref = df[df['model'] == ref_model].set_index('group').reindex(groups)
    for gi, g in enumerate(groups):
        row = ref.loc[g] if g in ref.index else None
        if row is not None and 'n_pos' in ref.columns:
            _annotate_group_counts(ax, x[gi], row['n_pos'], row['n_neg'])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_ylabel(f'{metric} (mean across tissues)')
    ax.set_ylim(0 if ymin is None else ymin, 1 if ymax is None else ymax)
    ax.set_title(f'{organ} - {metric} by Variant Distance Groups (test-set)')

    legend_elements = [
        Patch(facecolor=color_arrays[m][2], edgecolor='black',
              label=DISPLAY[m], alpha=0.8)
        for m in MODELS
    ]
    ax.legend(handles=legend_elements, title='Model', loc='upper right')

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'{organ}_testset_{metric}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(_SCRIPT_DIR, 'data_paths.ag_like.json'))
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--organs', nargs='+', default=ORGANS)
    ap.add_argument('--metrics', nargs='+', default=['AUROC', 'AUPRC'])
    ap.add_argument('--skip_eval', action='store_true',
                    help='Skip evaluation; plot from existing testset CSVs.')
    ap.add_argument('--ymin', type=float, default=None)
    ap.add_argument('--ymax', type=float, default=None)
    args = ap.parse_args()

    test_index = _build_test_index(SEQUENCES_BED)
    print(f'Loaded test regions: {sum(len(v[0]) for v in test_index.values())} intervals '
          f'across {len(test_index)} chromosomes', flush=True)

    if not args.skip_eval:
        for model in args.models:
            for organ in args.organs:
                eval_per_tissue_avg_testset(model, organ, args.json, test_index)

    for organ in args.organs:
        for metric in args.metrics:
            _plot_one(organ, metric, ymin=args.ymin, ymax=args.ymax)


if __name__ == '__main__':
    main()
