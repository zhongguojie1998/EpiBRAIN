#!/usr/bin/env python
"""cCRE-restricted AUROC/AUPRC evaluation.

For each (model, organ in {Cortex, Basal_ganglia}), pool variants across matched
GTEx tissues, compute L2-across-tracks score, and report metrics for every
combination of (distance-group, subset in {'all', 'ccre_only'}).

Writes: output.ag_like/{model}_{organ}_agg_ccre_restricted.csv
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

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

OUT_DIR = os.path.join(_cfg.PWD, 'Analysis/07_eQTL/output.ag_like')

MODEL_FILTER = {
    'Cortex': {'bican': 'cortex', 'borzoi': 'cortex', 'alphagenome_524k': 'cortex'},
    'Basal_ganglia': {'bican': 'basal_ganglia', 'borzoi': 'basal_ganglia',
                      'alphagenome_524k': 'basal_ganglia'},
}


def pool_variants(model_data, organ: str):
    """Pool variants across all matched GTEx tissues for this organ.

    Returns
    -------
    agg_vcf : DataFrame with columns ['ID', 'INFO', 'group']
    agg_labels : np.ndarray[int]
    scores_mat : (n_variants, n_tracks) score matrix
    track_anno : DataFrame of tracks (after tissue_filter_fn if applicable)
    """
    tissues = ORGAN_TISSUE_MAP[organ]
    log_square = model_data.log_square
    variant_index = model_data.variant_index
    track_anno_base = model_data.track_anno
    tissue_filter_fn = model_data.tissue_filter_fn

    if tissue_filter_fn is not None:
        agg_col_set: set[int] = set()
        for tissue in tissues:
            _, col_indices = tissue_filter_fn(tissue, track_anno_base)
            agg_col_set.update(col_indices.tolist())
        all_col_indices = np.array(sorted(agg_col_set))
        track_anno = track_anno_base.iloc[all_col_indices].reset_index(drop=True)
    else:
        track_anno = track_anno_base
        all_col_indices = None

    variant_map: dict = {}
    for tissue in tissues:
        tissue_dir = os.path.join(_cfg.PWD, 'Data/source/eQTL', tissue)
        if not os.path.isdir(tissue_dir):
            continue
        vcf_unique, labels = load_tissue_vcf(tissue_dir, add_addition=False)
        idx_in_h5 = variant_index.get_indexer(vcf_unique['ID'])
        if (idx_in_h5 < 0).all():
            continue
        for i, (vid, label, h5idx) in enumerate(zip(vcf_unique['ID'], labels, idx_in_h5)):
            if h5idx < 0:
                continue
            row = vcf_unique.iloc[i]
            if vid not in variant_map:
                variant_map[vid] = {
                    'h5_idx': h5idx, 'labels': [],
                    'info': row['INFO'], 'group': row.get('group', 'all'),
                }
            variant_map[vid]['labels'].append(int(label))
            if row['INFO'] == 'positive':
                variant_map[vid]['info'] = 'positive'

    vids = list(variant_map.keys())
    agg_h5_indices = np.array([variant_map[v]['h5_idx'] for v in vids])
    agg_labels = np.array([max(variant_map[v]['labels']) for v in vids])
    agg_vcf = pd.DataFrame({
        'ID': vids,
        'INFO': [variant_map[v]['info'] for v in vids],
        'group': [variant_map[v]['group'] for v in vids],
    })
    if all_col_indices is not None:
        scores_mat = log_square[agg_h5_indices][:, all_col_indices]
    else:
        scores_mat = log_square[agg_h5_indices]
    return agg_vcf, agg_labels, scores_mat, track_anno


def load_ccre_variants(organ: str) -> set[str]:
    path = os.path.join(OUT_DIR, f'variants_in_ccre_{organ}.tsv')
    return set(pd.read_csv(path)['variant_id'])


def _subset_masks(labels: np.ndarray, in_ccre: np.ndarray) -> dict[str, np.ndarray]:
    is_pos = labels.astype(bool)
    return {
        'all': np.ones(len(labels), dtype=bool),
        'ccre_only': in_ccre,
        'pos_in_ccre': (~is_pos) | (is_pos & in_ccre),
        'neg_in_ccre': is_pos | ((~is_pos) & in_ccre),
        'pos_notin_ccre': (~is_pos) | (is_pos & ~in_ccre),
        'neg_notin_ccre': is_pos | ((~is_pos) & ~in_ccre),
    }


def _metric_rows(labels, scores, vcf_df, in_ccre, *, tag: dict) -> list[dict]:
    rows = []
    for subset, mask in _subset_masks(labels, in_ccre).items():
        sub_vcf = vcf_df.loc[mask].reset_index(drop=True)
        sub_scores = scores[mask]
        sub_labels = labels[mask]
        for group in VARIANT_GROUPS:
            auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                sub_labels, sub_scores, sub_vcf, group, n_bootstraps=100,
            )
            rows.append({**tag, 'subset': subset, 'group': group,
                         'AUROC': auroc, 'AUROC_SE': auroc_se,
                         'AUPRC': auprc, 'AUPRC_SE': auprc_se,
                         'n_pos': n_pos, 'n_neg': n_neg})
    return rows


def eval_pooled(model_name: str, organ: str, json_path: str) -> pd.DataFrame:
    set_paths_json(json_path)
    track_filter = MODEL_FILTER[organ][model_name]
    print(f'\n=== POOLED | {model_name} | organ={organ} | filter={track_filter} ===', flush=True)
    model_data = load_model(model_name, track_filter, add_addition=False)
    agg_vcf, agg_labels, scores_mat, _ = pool_variants(model_data, organ)
    print(f'n_variants={len(agg_labels)}, n_pos={int(agg_labels.sum())}, '
          f'n_neg={int((agg_labels==0).sum())}, n_tracks={scores_mat.shape[1]}', flush=True)

    l2_scores = np.linalg.norm(scores_mat, axis=1)
    ccre_ids = load_ccre_variants(organ)
    in_ccre = agg_vcf['ID'].isin(ccre_ids).values

    tag = {'model': model_name, 'organ': organ,
           'track_filter': track_filter, 'n_tracks': scores_mat.shape[1]}
    df = pd.DataFrame(_metric_rows(agg_labels, l2_scores, agg_vcf, in_ccre, tag=tag))
    out_path = os.path.join(OUT_DIR, f'{model_name}_{organ}_agg_ccre_restricted.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    return df


def eval_per_tissue_avg(model_name: str, organ: str, json_path: str) -> pd.DataFrame:
    set_paths_json(json_path)
    track_filter = MODEL_FILTER[organ][model_name]
    print(f'\n=== PER-TISSUE | {model_name} | organ={organ} | filter={track_filter} ===', flush=True)
    model_data = load_model(model_name, track_filter, add_addition=False)
    variant_index = model_data.variant_index
    log_square = model_data.log_square
    track_anno_base = model_data.track_anno
    tissue_filter_fn = model_data.tissue_filter_fn

    ccre_ids = load_ccre_variants(organ)
    tissues = ORGAN_TISSUE_MAP[organ]

    per_tissue_rows: list[dict] = []
    for tissue in tissues:
        tissue_dir = os.path.join(_cfg.PWD, 'Data/source/eQTL', tissue)
        if not os.path.isdir(tissue_dir):
            continue
        vcf_unique, labels = load_tissue_vcf(tissue_dir, add_addition=False)
        idx_in_h5 = variant_index.get_indexer(vcf_unique['ID'])
        keep = idx_in_h5 >= 0
        vcf_unique = vcf_unique.loc[keep].reset_index(drop=True)
        labels = labels[keep]
        h5_idx = idx_in_h5[keep]

        if tissue_filter_fn is not None:
            _, col_indices = tissue_filter_fn(tissue, track_anno_base)
            if len(col_indices) == 0:
                continue
            scores_mat = log_square[h5_idx][:, col_indices]
        else:
            scores_mat = log_square[h5_idx]
        l2 = np.linalg.norm(scores_mat, axis=1)
        in_ccre = vcf_unique['ID'].isin(ccre_ids).values

        tag = {'model': model_name, 'organ': organ, 'tissue': tissue,
               'track_filter': track_filter, 'n_tracks': scores_mat.shape[1]}
        per_tissue_rows.extend(_metric_rows(labels, l2, vcf_unique, in_ccre, tag=tag))

    per_tissue = pd.DataFrame(per_tissue_rows)
    pt_out = os.path.join(OUT_DIR, f'{model_name}_{organ}_per_tissue_ccre_restricted.csv')
    per_tissue.to_csv(pt_out, index=False)
    print(f'Saved: {pt_out}')

    # Aggregate across tissues: mean for metric, SE = (1/n)*sqrt(sum(SE_i^2)); sum n_pos/n_neg.
    key_cols = ['model', 'organ', 'subset', 'group', 'track_filter']
    valid = per_tissue.dropna(subset=['AUROC'])
    agg_rows = []
    for keys, sub in valid.groupby(key_cols, dropna=False):
        row = dict(zip(key_cols, keys))
        n_t = len(sub)
        row['n_tissues'] = n_t
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
    agg_out = os.path.join(OUT_DIR, f'{model_name}_{organ}_per_tissue_avg_ccre_restricted.csv')
    agg.to_csv(agg_out, index=False)
    print(f'Saved: {agg_out}')
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(_SCRIPT_DIR, 'data_paths.ag_like.json'))
    ap.add_argument('--models', nargs='+',
                    default=['bican', 'alphagenome_524k', 'borzoi'])
    ap.add_argument('--organs', nargs='+',
                    default=['Cortex', 'Basal_ganglia'])
    ap.add_argument('--mode', nargs='+', choices=['pooled', 'per_tissue_avg'],
                    default=['pooled', 'per_tissue_avg'])
    args = ap.parse_args()

    for model in args.models:
        for organ in args.organs:
            if 'pooled' in args.mode:
                eval_pooled(model, organ, args.json)
            if 'per_tissue_avg' in args.mode:
                eval_per_tissue_avg(model, organ, args.json)


if __name__ == '__main__':
    main()
