from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import PWD, ORGAN_TISSUE_MAP, VARIANT_GROUPS
from .metrics import compute_metrics
from .data import ModelData, load_tissue_vcf

_AGG_ORGAN_MAP = {
    'Basal_ganglia_agg': 'Basal_ganglia',
    'Cortex_agg': 'Cortex',
    'Brain_agg': 'Brain',
}


def evaluate_model(model: ModelData, organ: str | None = None, suffix: str = '', output_dir: str | None = None, add_addition: bool = False) -> pd.DataFrame:
    if output_dir is None:
        from .config import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    name = model.name
    log_square = model.log_square
    variant_index = model.variant_index
    track_anno_base = model.track_anno

    tissue_filter_fn = model.tissue_filter_fn

    organ_map = {organ: ORGAN_TISSUE_MAP[organ]} if organ else ORGAN_TISSUE_MAP
    all_rows: list[pd.DataFrame] = []

    for organ_name, tissues in organ_map.items():
        for tissue in tissues:
            tissue_dir = os.path.join(PWD, 'Data/source/eQTL', tissue)
            if not os.path.isdir(tissue_dir):
                print(f'Missing tissue dir: {tissue_dir}, skipping')
                continue

            vcf_unique, labels = load_tissue_vcf(tissue_dir, add_addition=add_addition)

            idx_in_h5 = variant_index.get_indexer(vcf_unique['ID'])
            if (idx_in_h5 < 0).all():
                print(f'No variants found for tissue {tissue}, skipping')
                continue

            # Resolve per-tissue track filtering (e.g. alphagenome GTEx)
            if tissue_filter_fn is not None:
                track_anno, col_indices = tissue_filter_fn(tissue, track_anno_base)
                if len(col_indices) == 0:
                    print(f'No matching tracks for tissue {tissue}, skipping')
                    continue
                tissue_scores = log_square[idx_in_h5][:, col_indices]
            else:
                track_anno = track_anno_base
                tissue_scores = log_square[idx_in_h5]

            print(f'Organ: {organ_name}, Tissue: {tissue}, '
                  f'n_variants: {len(vcf_unique)}, n_pos: {labels.sum()}, '
                  f'n_neg: {(labels == 0).sum()}, n_tracks: {tissue_scores.shape[1]}')

            for group in VARIANT_GROUPS:
                # --- Per-track AUROC ---
                track_group = track_anno.copy()
                track_group['organ'] = organ_name
                track_group['tissue'] = tissue
                track_group['group'] = group
                for i, idx in enumerate(track_group.index):
                    auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                        labels, tissue_scores[:, i], vcf_unique, group, n_bootstraps=None,
                    )
                    track_group.loc[idx, 'AUROC'] = auroc
                    track_group.loc[idx, 'AUROC_SE'] = auroc_se
                    track_group.loc[idx, 'AUPRC'] = auprc
                    track_group.loc[idx, 'AUPRC_SE'] = auprc_se
                    track_group.loc[idx, 'n_pos'] = n_pos
                    track_group.loc[idx, 'n_neg'] = n_neg
                all_rows.append(track_group)

                # --- L2 norm across all tracks ---
                l2_scores = np.linalg.norm(tissue_scores, axis=1)
                auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                    labels, l2_scores, vcf_unique, group, n_bootstraps=50,
                )
                all_rows.append(_metric_row(
                    exp=f'{name}_all', modality='ALL', celltype='ALL',
                    organ=organ_name, tissue=tissue, group=group,
                    auroc=auroc, auroc_se=auroc_se, auprc=auprc, auprc_se=auprc_se,
                    n_pos=n_pos, n_neg=n_neg,
                ))

                # --- Per-modality L2 ---
                for modality in track_anno['modality'].unique():
                    mod_idx = track_anno.index[track_anno['modality'] == modality].values
                    mod_scores = np.linalg.norm(tissue_scores[:, mod_idx], axis=1)
                    auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                        labels, mod_scores, vcf_unique, group, n_bootstraps=50,
                    )
                    all_rows.append(_metric_row(
                        exp=f'{name}_{modality}', modality=modality, celltype='ALL',
                        organ=organ_name, tissue=tissue, group=group,
                        auroc=auroc, auroc_se=auroc_se, auprc=auprc, auprc_se=auprc_se,
                        n_pos=n_pos, n_neg=n_neg,
                    ))

    track_results = pd.concat(all_rows, ignore_index=True)
    track_results['mod'] = name

    # Fill exp for per-track rows that don't have it
    if 'exp' in track_results.columns and 'trial' in track_results.columns:
        track_results.loc[track_results['exp'].isna(), 'exp'] = \
            track_results['trial'][track_results['exp'].isna()].copy()
    if 'identifier' in track_results.columns:
        track_results.loc[track_results['exp'].isna(), 'exp'] = \
            track_results['identifier'][track_results['exp'].isna()].copy()

    # --- Cross-tissue aggregation: mean across tissues, SE = (1/n)*sqrt(sum(SE_i^2)) ---
    def _combined_se(se_vals: pd.Series) -> float:
        vals = se_vals.dropna().values
        n = len(vals)
        return (1.0 / n) * np.sqrt(np.sum(vals ** 2)) if n > 0 else np.nan

    grp_cols = ['exp', 'modality', 'celltype', 'organ', 'group']
    valid = track_results.dropna(subset=['AUROC'])
    agg_auroc = valid.groupby(grp_cols, dropna=False).agg(
        AUROC=('AUROC', 'mean'),
        AUROC_SE=('AUROC_SE', _combined_se),
    ).reset_index()
    valid_prc = track_results.dropna(subset=['AUPRC'])
    agg_auprc = valid_prc.groupby(grp_cols, dropna=False).agg(
        AUPRC=('AUPRC', 'mean'),
        AUPRC_SE=('AUPRC_SE', _combined_se),
    ).reset_index()
    agg = agg_auroc.merge(agg_auprc, on=grp_cols, how='outer')
    agg['tissue'] = 'all'
    agg['mod'] = name
    track_results = pd.concat([track_results, agg], ignore_index=True)

    out_path = os.path.join(output_dir, f'{name}_track_results{suffix}.csv')
    track_results.to_csv(out_path)
    print(f'Saved: {out_path}')
    return track_results


def evaluate_organ_agg(model: ModelData, organ: str, suffix: str = '', output_dir: str | None = None, add_addition: bool = False) -> pd.DataFrame:
    """Pool all tissue variants for an organ, then run full evaluation pipeline on pooled data."""
    if output_dir is None:
        from .config import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    organ_key = _AGG_ORGAN_MAP[organ]
    tissues = ORGAN_TISSUE_MAP[organ_key]

    name = model.name
    log_square = model.log_square
    variant_index = model.variant_index
    track_anno_base = model.track_anno

    tissue_filter_fn = model.tissue_filter_fn

    # For gtex organ_agg (alphagenome only): collect union of col_indices across all tissues.
    if tissue_filter_fn is not None:
        agg_col_set: set[int] = set()
        for tissue in tissues:
            _, col_indices = tissue_filter_fn(tissue, track_anno_base)
            agg_col_set.update(col_indices.tolist())
        all_col_indices = np.array(sorted(agg_col_set))
        track_anno = track_anno_base.iloc[all_col_indices].reset_index(drop=True)
        print(f'GTEx organ_agg [{organ}]: total tracks across tissues = {len(all_col_indices)}')
    else:
        track_anno = track_anno_base

    # ---- Pool tissue variants ------------------------------------------------
    variant_map: dict[str, dict] = {}
    for tissue in tissues:
        tissue_dir = os.path.join(PWD, 'Data/source/eQTL', tissue)
        if not os.path.isdir(tissue_dir):
            print(f'Missing tissue dir: {tissue_dir}, skipping')
            continue
        vcf_unique, labels = load_tissue_vcf(tissue_dir, add_addition=add_addition)
        idx_in_h5 = variant_index.get_indexer(vcf_unique['ID'])
        if (idx_in_h5 < 0).all():
            print(f'No variants found for tissue {tissue}, skipping')
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

    # Sanity check: conflicting labels across tissues
    conflict_count = sum(
        1 for v in variant_map.values() if 1 in v['labels'] and 0 in v['labels']
    )
    print(f'Sanity check [{organ}]: variants positive in some tissues but negative in others: {conflict_count}')

    # Build aggregated arrays
    vids = list(variant_map.keys())
    agg_h5_indices = np.array([variant_map[v]['h5_idx'] for v in vids])
    agg_labels = np.array([max(variant_map[v]['labels']) for v in vids])
    agg_vcf = pd.DataFrame({
        'ID': vids,
        'INFO': [variant_map[v]['info'] for v in vids],
        'group': [variant_map[v]['group'] for v in vids],
    })

    # Score matrix
    if tissue_filter_fn is not None:
        scores_mat = log_square[agg_h5_indices][:, all_col_indices]
    else:
        scores_mat = log_square[agg_h5_indices]

    print(f'Organ agg [{organ}]: n_variants={len(agg_labels)}, '
          f'n_pos={int(agg_labels.sum())}, n_neg={int((agg_labels == 0).sum())}, '
          f'n_tracks={scores_mat.shape[1]}')

    # ---- Full evaluation pipeline (same as evaluate_model inner loop) --------
    all_rows: list[pd.DataFrame] = []

    for group in VARIANT_GROUPS:
        # --- Per-track AUROC ---
        track_group = track_anno.copy()
        track_group['organ'] = organ_key
        track_group['group'] = group
        for i, idx in enumerate(track_group.index):
            auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                agg_labels, scores_mat[:, i], agg_vcf, group, n_bootstraps=None,
            )
            track_group.loc[idx, 'AUROC'] = auroc
            track_group.loc[idx, 'AUROC_SE'] = auroc_se
            track_group.loc[idx, 'AUPRC'] = auprc
            track_group.loc[idx, 'AUPRC_SE'] = auprc_se
            track_group.loc[idx, 'n_pos'] = n_pos
            track_group.loc[idx, 'n_neg'] = n_neg
        all_rows.append(track_group)

        # --- L2 norm across all tracks ---
        l2_scores = np.linalg.norm(scores_mat, axis=1)
        auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
            agg_labels, l2_scores, agg_vcf, group, n_bootstraps=50,
        )
        all_rows.append(_metric_row(
            exp=f'{name}_all', modality='ALL', celltype='ALL',
            organ=organ_key, group=group,
            auroc=auroc, auroc_se=auroc_se, auprc=auprc, auprc_se=auprc_se,
            n_pos=n_pos, n_neg=n_neg,
        ))

        # --- Per-modality L2 ---
        for modality in track_anno['modality'].unique():
            mod_idx = track_anno.index[track_anno['modality'] == modality].values
            mod_scores = np.linalg.norm(scores_mat[:, mod_idx], axis=1)
            auroc, auroc_se, auprc, auprc_se, n_pos, n_neg = compute_metrics(
                agg_labels, mod_scores, agg_vcf, group, n_bootstraps=50,
            )
            all_rows.append(_metric_row(
                exp=f'{name}_{modality}', modality=modality, celltype='ALL',
                organ=organ_key, group=group,
                auroc=auroc, auroc_se=auroc_se, auprc=auprc, auprc_se=auprc_se,
                n_pos=n_pos, n_neg=n_neg,
            ))

    results = pd.concat(all_rows, ignore_index=True)
    results['mod'] = name

    # Fill exp for per-track rows
    if 'exp' in results.columns and 'trial' in results.columns:
        results.loc[results['exp'].isna(), 'exp'] = \
            results['trial'][results['exp'].isna()].copy()
    if 'identifier' in results.columns:
        results.loc[results['exp'].isna(), 'exp'] = \
            results['identifier'][results['exp'].isna()].copy()

    out_path = os.path.join(output_dir, f'{name}_{organ}{suffix}.csv')
    results.to_csv(out_path)
    print(f'Saved: {out_path}')
    return results


def _metric_row(*, exp, modality, celltype, organ, group,
                auroc, auroc_se, auprc, auprc_se, n_pos, n_neg,
                tissue=None) -> pd.DataFrame:
    d = {
        'exp': exp, 'modality': modality, 'celltype': celltype,
        'organ': organ, 'group': group,
        'AUROC': auroc, 'AUROC_SE': auroc_se,
        'AUPRC': auprc, 'AUPRC_SE': auprc_se,
        'n_pos': n_pos, 'n_neg': n_neg,
    }
    if tissue is not None:
        d['tissue'] = tissue
    return pd.DataFrame(d, index=[0])
