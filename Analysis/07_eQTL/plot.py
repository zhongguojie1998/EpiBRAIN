from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from .config import VARIANT_GROUPS, OUTPUT_DIR
from .data import _load_paths_cfg

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

_AGG_ORGANS = {'Brain_agg', 'Basal_ganglia_agg', 'Cortex_agg'}
_AGG_ORGAN_MAP = {
    'Basal_ganglia_agg': 'Basal_ganglia',
    'Cortex_agg': 'Cortex',
    'Brain_agg': 'Brain',
}

# Default colour cycle — one colormap per model, assigned in discovery order.
_CMAP_CYCLE = [
    plt.cm.Oranges, plt.cm.Reds, plt.cm.Greens,
    plt.cm.Purples, plt.cm.Blues, plt.cm.cool,
    plt.cm.autumn, plt.cm.winter,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name: str, output_dir: str | None = None):
    if output_dir is None:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure')
    else:
        d = os.path.join(output_dir, 'figure')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def _model_names() -> list[str]:
    """Return the model names defined in the active JSON config."""
    cfg = _load_paths_cfg()
    return list(cfg['models'].keys())


def _get_plot_names() -> dict[str, str]:
    """Return model_name → display_name mapping from optional JSON 'plot_name' field."""
    cfg = _load_paths_cfg()
    mapping = cfg.get('plot_name', {})
    # Fall back to model name itself for any model not in the mapping
    return {m: mapping.get(m, m) for m in cfg['models']}


def _annotate_bar(ax, x: float, bar_top: float, se: float, val: float,
                  fontsize: int = 6) -> None:
    """Write the numeric value horizontally just above the SE cap (or bar top)."""
    if np.isnan(bar_top):
        return
    y = bar_top + (se if se and not np.isnan(se) else 0)
    ax.text(x, y, f'{val:.3f}', ha='center', va='bottom',
            fontsize=fontsize, rotation=0, clip_on=True)


def _annotate_group_counts(ax, x: float, n_pos, n_neg,
                           fontsize: int = 6) -> None:
    """Write +n_pos / -n_neg centred at the bottom of the axes (axes-y coords)."""
    try:
        label = f'+{int(n_pos)}\n-{int(n_neg)}'
    except (ValueError, TypeError):
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(x, 0.01, label, ha='center', va='bottom',
            fontsize=fontsize, color='dimgray', transform=trans, clip_on=True)


def _display_exp(exp: str, plot_names: dict[str, str], models: list[str]) -> str:
    """Replace the model-name prefix in an exp string with its display name."""
    for m in sorted(models, key=len, reverse=True):
        if exp.startswith(m):
            suffix = exp[len(m):]
            return plot_names[m] + suffix
    return exp


def _build_color_maps(models: list[str]) -> dict[str, plt.cm.ScalarMappable]:
    return {m: _CMAP_CYCLE[i % len(_CMAP_CYCLE)] for i, m in enumerate(models)}


def _organ_key(organ: str) -> str:
    return _AGG_ORGAN_MAP.get(organ, organ)


def _get_organ_data(df: pd.DataFrame, organ: str, tissue: str | None = None) -> pd.DataFrame:
    key = _organ_key(organ)
    mask = df['organ'] == key
    if tissue is not None and 'tissue' in df.columns:
        mask = mask & (df['tissue'] == tissue)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Load and prepare combined results from CSVs in output_dir
# ---------------------------------------------------------------------------
def load_all_results(organ: str, output_dir: str | None = None,
                     bican_suffix: str = '', borzoi_suffix: str = '',
                     ag_suffix: str = '') -> pd.DataFrame:
    d = output_dir or OUTPUT_DIR
    is_agg = organ in _AGG_ORGANS
    models = _model_names()

    # Map model name → suffix
    suffix_map: dict[str, str] = {}
    for m in models:
        cfg = _load_paths_cfg()['models'][m]
        t = cfg.get('type', '')
        if t == 'bican':
            suffix_map[m] = bican_suffix
        elif t == 'borzoi':
            suffix_map[m] = borzoi_suffix
        else:
            suffix_map[m] = ag_suffix

    frames = []
    for name in models:
        suffix = suffix_map[name]
        if is_agg:
            path = os.path.join(d, f'{name}_{organ}{suffix}.csv')
        else:
            path = os.path.join(d, f'{name}_track_results{suffix}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if 'exp' not in df.columns and 'identifier' in df.columns:
                df['exp'] = df['identifier']
            frames.append(df)
        else:
            print(f'Not found, skipping: {path}')
    if not frames:
        raise FileNotFoundError(f'No result CSVs found in {d}')
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal: resolve metric column name and whether to show error bars
# ---------------------------------------------------------------------------
def _resolve_metric(metric: str) -> tuple[str, bool]:
    """Return (base_col, show_se). *_point variants suppress error bars."""
    if metric.endswith('_point'):
        return metric[:-len('_point')], False
    return metric, True


# ---------------------------------------------------------------------------
# 1. Overall model performance by variant distance groups
# ---------------------------------------------------------------------------
def plot_overall_comparison(combined: pd.DataFrame, organ: str,
                            output_dir: str | None = None,
                            metric: str = 'AUROC',
                            ymin: float | None = None,
                            ymax: float | None = None,
                            plot_num: bool = False):
    base_metric, show_se = _resolve_metric(metric)
    se_col = f'{base_metric}_SE'
    models = _model_names()
    plot_names = _get_plot_names()
    all_exps = [f'{m}_all' for m in models] + [f'{m}_' for m in models]

    model_mask = combined['exp'].str.startswith(tuple(f'{m}_' for m in models), na=False) \
                 & ~combined['exp'].str.contains('RNAminus|RNAplus', na=False)
    overall = combined[model_mask].copy()
    overall = overall[overall.get('celltype', 'ALL') == 'ALL']

    if organ in _AGG_ORGANS:
        agg = _get_organ_data(overall, organ)
    else:
        agg = _get_organ_data(overall, organ, tissue='all')

    def _type_from_exp(exp: str) -> str:
        for m in sorted(models, key=len, reverse=True):
            if m in exp:
                return m
        return models[0]

    agg['type'] = agg['exp'].apply(_type_from_exp)
    color_maps = _build_color_maps(models)

    data_by_group = {}
    for g in VARIANT_GROUPS:
        df = agg[agg['group'] == g].dropna(subset=[base_metric])
        if len(df):
            data_by_group[g] = df

    if not data_by_group:
        print(f'No data for organ={organ}')
        return

    if 'all' in data_by_group:
        exp_order = data_by_group['all'].sort_values(base_metric, ascending=False)['exp'].tolist()
    else:
        exp_order = sorted({e for d in data_by_group.values() for e in d['exp']})

    exp_types = {e: _type_from_exp(e) for e in exp_order}
    n_groups = len(VARIANT_GROUPS)
    color_arrays = {k: cm(np.linspace(0.9, 0.3, n_groups)) for k, cm in color_maps.items()}

    fig, ax = plt.subplots(figsize=(16, 6))
    bw = 0.15
    x = np.arange(len(exp_order))

    for gi, g in enumerate(VARIANT_GROUPS):
        if g not in data_by_group:
            continue
        df = data_by_group[g]
        vals, ses, colors = [], [], []
        for e in exp_order:
            row = df[df['exp'] == e]
            if len(row):
                vals.append(row[base_metric].values[0])
                ses.append(row[se_col].values[0] if show_se else 0)
                colors.append(color_arrays[exp_types[e]][gi])
            else:
                vals.append(0); ses.append(0); colors.append('gray')
        off = (gi - n_groups / 2 + 0.5) * bw
        for i, (v, s, c) in enumerate(zip(vals, ses, colors)):
            kw = dict(label=g) if i == 0 else {}
            ax.bar(x[i] + off, v, bw, yerr=(s if show_se else None), capsize=(3 if show_se else 0),
                   color=c, alpha=0.8, edgecolor='black', linewidth=0.5, **kw)
            if plot_num:
                _annotate_bar(ax, x[i] + off, v, s if show_se else 0, v)
                # n_pos/n_neg — one annotation per (exp×group) bar
                df_row = data_by_group[g][data_by_group[g]['exp'] == exp_order[i]]
                if len(df_row) and 'n_pos' in df_row.columns:
                    _annotate_group_counts(ax, x[i] + off,
                                           df_row['n_pos'].values[0],
                                           df_row['n_neg'].values[0])

    tick_labels = [_display_exp(e, plot_names, models) for e in exp_order]
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_ylabel(f'{base_metric} (mean across tissues)')
    ax.set_ylim(0 if ymin is None else ymin, 1 if ymax is None else ymax)
    ax.set_title(f'{organ} - {base_metric} by Variant Distance Groups')
    ax.legend(title='Distance Group', loc='upper right')
    _save(fig, f'{organ}_overall_model_performance_{metric}.pdf', output_dir)


# ---------------------------------------------------------------------------
# 2. Model comparison (L2 all-tracks score per model)
# ---------------------------------------------------------------------------
def plot_model_comparison(combined: pd.DataFrame, organ: str,
                          output_dir: str | None = None,
                          metric: str = 'AUROC',
                          ymin: float | None = None,
                          ymax: float | None = None,
                          plot_num: bool = False):
    base_metric, show_se = _resolve_metric(metric)
    se_col = f'{base_metric}_SE'
    models = _model_names()
    plot_names = _get_plot_names()
    model_exps = [f'{m}_all' for m in models]
    color_maps = _build_color_maps(models)

    model_mask = combined['exp'].isin(model_exps)
    overall = combined[model_mask].copy()

    if organ in _AGG_ORGANS:
        agg = _get_organ_data(overall, organ)
    else:
        agg = _get_organ_data(overall, organ, tissue='all')

    n_groups = len(VARIANT_GROUPS)
    color_arrays = {k: cm(np.linspace(0.9, 0.3, n_groups)) for k, cm in color_maps.items()}

    data_by_group = {}
    for g in VARIANT_GROUPS:
        df = agg[agg['group'] == g].dropna(subset=[base_metric])
        if len(df):
            data_by_group[g] = df

    fig, ax = plt.subplots(figsize=(10, 4))
    bw = 0.15
    x = np.arange(n_groups)

    for ei, exp in enumerate(model_exps):
        model_name = models[ei]
        vals, ses, colors = [], [], []
        for gi, g in enumerate(VARIANT_GROUPS):
            df = data_by_group.get(g, pd.DataFrame())
            row = df[df['exp'] == exp] if len(df) else pd.DataFrame()
            if len(row):
                vals.append(row[base_metric].values[0])
                ses.append(row[se_col].values[0] if show_se else 0)
                colors.append(color_arrays[model_name][gi])
            else:
                vals.append(0); ses.append(0); colors.append('gray')
        off = (ei - len(model_exps) / 2 + 0.5) * bw
        for i, (v, s, c) in enumerate(zip(vals, ses, colors)):
            ax.bar(x[i] + off, v, bw, yerr=(s if show_se else None), capsize=(3 if show_se else 0),
                   color=c, alpha=0.8, edgecolor='black', linewidth=0.5)
            if plot_num:
                _annotate_bar(ax, x[i] + off, v, s if show_se else 0, v)

    if plot_num:
        for gi, g in enumerate(VARIANT_GROUPS):
            df = data_by_group.get(g, pd.DataFrame())
            if len(df) and 'n_pos' in df.columns:
                _annotate_group_counts(ax, x[gi], df['n_pos'].values[0],
                                       df['n_neg'].values[0])

    ax.set_xticks(x); ax.set_xticklabels(VARIANT_GROUPS, rotation=45, ha='right')
    ax.set_ylabel(f'{base_metric} (mean across tissues)')
    ax.set_ylim(0 if ymin is None else ymin, 1 if ymax is None else ymax)
    ax.set_title(f'{organ} - {base_metric} by Variant Distance Groups')

    legend_elements = []
    for ei, exp in enumerate(model_exps):
        c = color_arrays[models[ei]][2]
        legend_elements.append(Patch(facecolor=c, edgecolor='black',
                                     label=plot_names[models[ei]], alpha=0.8))
    ax.legend(handles=legend_elements, title='Model', loc='upper right')
    _save(fig, f'{organ}_comparison_{metric}.pdf', output_dir)


# ---------------------------------------------------------------------------
# 3. Per-tissue bar chart (group='all', model comparison)
# ---------------------------------------------------------------------------
def plot_per_tissue_auroc(combined: pd.DataFrame, organ: str,
                          output_dir: str | None = None,
                          metric: str = 'AUROC',
                          ymin: float | None = None,
                          ymax: float | None = None,
                          plot_num: bool = False):
    base_metric, show_se = _resolve_metric(metric)
    se_col = f'{base_metric}_SE'
    models = _model_names()
    plot_names = _get_plot_names()
    model_exps = [f'{m}_all' for m in models]
    color_maps = _build_color_maps(models)
    model_colors = {f'{m}_all': color_maps[m](0.6) for m in models}

    key = _organ_key(organ)
    mask = (
        combined['exp'].isin(model_exps) &
        (combined['organ'] == key) &
        (combined['group'] == 'all') &
        combined['tissue'].notna() &
        (combined['tissue'] != 'all')
    )
    data = combined[mask].copy()

    valid_tissues = data.dropna(subset=[base_metric])['tissue'].unique()
    data = data[data['tissue'].isin(valid_tissues)]

    # Sort tissues by the first model's metric
    first_exp = model_exps[0]
    tissue_order = (
        data[data['exp'] == first_exp]
        .dropna(subset=[base_metric])
        .sort_values(base_metric, ascending=False)['tissue'].tolist()
    )
    if not tissue_order:
        tissue_order = data.dropna(subset=[base_metric])['tissue'].unique().tolist()

    labels = {f'{m}_all': plot_names[m] for m in models}
    n_t = len(tissue_order)
    bw = 0.15
    x = np.arange(n_t)

    fig, ax = plt.subplots(figsize=(max(6, n_t * 0.8 + 2), 5))
    for mi, exp in enumerate(model_exps):
        sub = data[data['exp'] == exp].set_index('tissue')
        vals = [sub.loc[t, base_metric] if t in sub.index else np.nan for t in tissue_order]
        ses = [sub.loc[t, se_col] if t in sub.index else np.nan for t in tissue_order] if show_se else None
        off = (mi - len(model_exps) / 2 + 0.5) * bw
        ax.bar(x + off, vals, bw, yerr=ses, capsize=(3 if show_se else 0),
               color=model_colors[exp], alpha=0.85, edgecolor='black',
               linewidth=0.5, label=labels[exp])
        if plot_num:
            se_vals = ses if (show_se and ses is not None) else [0] * len(tissue_order)
            for ti, (v, se_v) in enumerate(zip(vals, se_vals)):
                _annotate_bar(ax, x[ti] + off, v,
                               0 if np.isnan(se_v) else se_v, v)

    if plot_num:
        # n_pos/n_neg per tissue — same across models; use first model
        first_sub = data[data['exp'] == model_exps[0]].set_index('tissue')
        for ti, t in enumerate(tissue_order):
            if t in first_sub.index and 'n_pos' in first_sub.columns:
                _annotate_group_counts(ax, x[ti],
                                       first_sub.loc[t, 'n_pos'],
                                       first_sub.loc[t, 'n_neg'])

    ax.set_xticks(x); ax.set_xticklabels(tissue_order, rotation=45, ha='right')
    ax.set_ylabel(base_metric)
    ax.set_ylim(0 if ymin is None else ymin, 1 if ymax is None else ymax)
    ax.set_title(f'{organ} - Per-tissue {base_metric} (all variants)')
    ax.legend(loc='upper right', fontsize=8)
    _save(fig, f'{organ}_per_tissue_{metric}.pdf', output_dir)


# ---------------------------------------------------------------------------
# 4. Per-tissue comparison: one row per tissue, bars by distance group & model
# ---------------------------------------------------------------------------
def plot_per_tissue_comparison(combined: pd.DataFrame, organ: str,
                               output_dir: str | None = None,
                               metric: str = 'AUROC',
                               ymin: float | None = None,
                               ymax: float | None = None,
                               plot_num: bool = False):
    base_metric, show_se = _resolve_metric(metric)
    se_col = f'{base_metric}_SE'
    models = _model_names()
    plot_names = _get_plot_names()
    model_exps = [f'{m}_all' for m in models]
    color_maps = _build_color_maps(models)
    model_colors = {m: color_maps[m](0.6) for m in models}

    key = _organ_key(organ)
    mask = (
        combined['exp'].isin(model_exps) &
        (combined['organ'] == key) &
        combined['tissue'].notna() &
        (combined['tissue'] != 'all')
    )
    data = combined[mask].copy()

    tissues = sorted(data['tissue'].dropna().unique())
    if not tissues:
        print(f'No per-tissue data for organ={organ}')
        return

    n_tissues = len(tissues)
    n_models = len(model_exps)
    groups = VARIANT_GROUPS
    n_groups = len(groups)
    bw = 0.8 / n_models

    fig, axes = plt.subplots(n_tissues, 1, figsize=(max(6, n_groups * 1.5 + 2), 3.5 * n_tissues),
                             sharex=True, squeeze=False)

    for ti, tissue in enumerate(tissues):
        ax = axes[ti, 0]
        tissue_data = data[data['tissue'] == tissue]
        x = np.arange(n_groups)

        # Collect n_pos/n_neg per group from first model (same across models)
        first_sub = tissue_data[tissue_data['exp'] == model_exps[0]]
        group_counts: dict[str, tuple] = {}
        for g in groups:
            row = first_sub[first_sub['group'] == g]
            if len(row) and 'n_pos' in row.columns:
                group_counts[g] = (row['n_pos'].values[0], row['n_neg'].values[0])

        for mi, exp in enumerate(model_exps):
            model_name = models[mi]
            sub = tissue_data[tissue_data['exp'] == exp]
            vals, ses = [], []
            for g in groups:
                row = sub[sub['group'] == g]
                if len(row):
                    vals.append(row[base_metric].values[0])
                    if show_se:
                        ses.append(row[se_col].values[0] if pd.notna(row[se_col].values[0]) else 0)
                    else:
                        ses.append(0)
                else:
                    vals.append(np.nan)
                    ses.append(0)
            off = (mi - n_models / 2 + 0.5) * bw
            ax.bar(x + off, vals, bw, yerr=(ses if show_se else None), capsize=(3 if show_se else 0),
                   color=model_colors[model_name], alpha=0.85,
                   edgecolor='black', linewidth=0.5,
                   label=plot_names[model_name] if ti == 0 else None)
            if plot_num:
                for gi, (v, se_v) in enumerate(zip(vals, ses)):
                    _annotate_bar(ax, x[gi] + off, v,
                                   se_v if (show_se and not np.isnan(se_v)) else 0, v)

        if plot_num:
            for gi, g in enumerate(groups):
                if g in group_counts:
                    _annotate_group_counts(ax, x[gi], group_counts[g][0],
                                           group_counts[g][1])

        ax.set_ylabel(base_metric)
        ax.set_ylim(0 if ymin is None else ymin, 1 if ymax is None else ymax)
        ax.set_title(tissue.replace('Brain_', ''), fontsize=10)
        ax.set_xticks(x)
        if ti == n_tissues - 1:
            ax.set_xticklabels(groups, rotation=45, ha='right')

    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{organ} - Per-tissue {base_metric} by Distance Group', fontsize=12, y=1.0)
    _save(fig, f'{organ}_comparison_per_tissue_{metric}.pdf', output_dir)


# ---------------------------------------------------------------------------
# Run all plots
# ---------------------------------------------------------------------------
def plot_all(organ: str, output_dir: str | None = None,
             bican_suffix: str = '', borzoi_suffix: str = '',
             ag_suffix: str = '', metric: str = 'AUROC',
             ymin: float | None = None, ymax: float | None = None,
             plot_num: bool = False):
    combined = load_all_results(organ, output_dir, bican_suffix, borzoi_suffix, ag_suffix)
    plot_overall_comparison(combined, organ, output_dir, metric=metric, ymin=ymin, ymax=ymax, plot_num=plot_num)
    plot_model_comparison(combined, organ, output_dir, metric=metric, ymin=ymin, ymax=ymax, plot_num=plot_num)
    if organ not in _AGG_ORGANS:
        plot_per_tissue_auroc(combined, organ, output_dir, metric=metric, ymin=ymin, ymax=ymax, plot_num=plot_num)
        plot_per_tissue_comparison(combined, organ, output_dir, metric=metric, ymin=ymin, ymax=ymax, plot_num=plot_num)
