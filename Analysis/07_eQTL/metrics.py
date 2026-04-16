from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def _stratified_resample(V: pl.DataFrame, seed: int) -> pl.DataFrame:
    V_pos = V.filter(pl.col("label")).sample(fraction=1.0, with_replacement=True, seed=seed)
    V_neg = V.filter(~pl.col("label")).sample(fraction=1.0, with_replacement=True, seed=seed)
    return pl.concat([V_pos, V_neg])


def compute_auroc_with_se(
    labels: np.ndarray, scores: np.ndarray, n_bootstraps: int | None = 100,
) -> tuple[float, float]:
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        return np.nan, np.nan
    if n_bootstraps is None:
        return auroc, np.nan

    V = pl.DataFrame({"label": np.asarray(labels).astype(bool), "score": scores})
    boot = []
    for i in range(n_bootstraps):
        Vb = _stratified_resample(V, i)
        try:
            boot.append(roc_auc_score(Vb["label"].cast(pl.Int64), Vb["score"]))
        except ValueError:
            continue
    se = pl.Series(boot).std() if boot else np.nan
    return auroc, se


def compute_auprc_with_se(
    labels: np.ndarray, scores: np.ndarray, n_bootstraps: int | None = 100,
) -> tuple[float, float]:
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        return np.nan, np.nan
    if n_bootstraps is None:
        return auprc, np.nan

    V = pl.DataFrame({"label": np.asarray(labels).astype(bool), "score": scores})
    boot = []
    for i in range(n_bootstraps):
        Vb = _stratified_resample(V, i)
        try:
            boot.append(average_precision_score(Vb["label"].cast(pl.Int64), Vb["score"]))
        except ValueError:
            continue
    se = pl.Series(boot).std() if boot else np.nan
    return auprc, se


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    vcf_unique: pd.DataFrame,
    group: str,
    n_bootstraps: int | None = 100,
) -> tuple[float, float, float, float, int, int]:
    if group == 'all':
        mask = np.ones(len(labels), dtype=bool)
    else:
        mask = (vcf_unique['group'] == group).values
        if mask.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan, 0, 0

    # Drop variants with NaN scores (e.g. gene_lfc undefined for some variants).
    valid = ~np.isnan(scores[mask])
    masked_labels = labels[mask][valid]
    masked_scores = scores[mask][valid]
    masked_info = vcf_unique['INFO'].values[mask][valid]

    n_pos = int((masked_info == 'positive').sum())
    n_neg = int((masked_info == 'negative').sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan, n_pos, n_neg
    auroc, auroc_se = compute_auroc_with_se(masked_labels, masked_scores, n_bootstraps)
    auprc, auprc_se = compute_auprc_with_se(masked_labels, masked_scores, n_bootstraps)
    return auroc, auroc_se, auprc, auprc_se, n_pos, n_neg


def aggregate_tissue_aurocs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        tissue_aurocs = sub['AUROC'].dropna()
        if len(tissue_aurocs) == 0:
            continue
        mean_auroc = tissue_aurocs.mean()
        se_auroc = (tissue_aurocs.std() / np.sqrt(len(tissue_aurocs))
                    if len(tissue_aurocs) > 1 else np.nan)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row['AUROC'] = mean_auroc
        row['AUROC_SE'] = se_auroc
        row['n_tissues'] = len(tissue_aurocs)
        rows.append(row)
    return pd.DataFrame(rows)
