from __future__ import annotations

import json
import os
from typing import Any, Callable, NamedTuple

import h5py
import numpy as np
import pandas as pd

from .config import (
    PWD, BIOTYPE_TO_KEEP,
    EMBRYO_FETAL_TRACK_PATTERN, DISEASE_TRACK_PATTERN, REGION_FILTER_MAP,
)


class ModelData(NamedTuple):
    name: str                           # e.g. "bican" | "borzoi" | "alphagenome131k"
    log_square: np.ndarray              # (n_variants, n_tracks)
    variant_index: pd.Index             # variant IDs aligned to rows
    track_anno: pd.DataFrame            # must have 'modality' column
    tissue_filter_fn: Callable | None = None  # fn(tissue, track_anno) -> (filtered_anno, col_indices)


# ---------------------------------------------------------------------------
# Path config
# ---------------------------------------------------------------------------
_DEFAULT_PATHS_JSON = os.path.join(os.path.dirname(__file__), 'data_paths.json')
_PATHS_CFG: dict[str, Any] | None = None


def _load_paths_cfg(paths_json: str | None = None) -> dict[str, Any]:
    global _PATHS_CFG, _NEG_ADD_VCF, _EXCLUDE_SET
    if _PATHS_CFG is None or paths_json is not None:
        with open(paths_json or _DEFAULT_PATHS_JSON) as f:
            _PATHS_CFG = json.load(f)
        _NEG_ADD_VCF = None  # invalidate cached VCF — may point elsewhere now
        _EXCLUDE_SET = None  # invalidate cached exclude set
    return _PATHS_CFG


def set_paths_json(path: str) -> None:
    """Override the default data_paths.json. Call once at startup."""
    _load_paths_cfg(os.path.abspath(path))


def get_output_dir(add_addition: bool = False) -> str:
    """Return the configured output directory (absolute), respecting add_addition."""
    cfg = _load_paths_cfg()
    key = 'output_dir_addition' if add_addition else 'output_dir'
    if key not in cfg:
        # Fall back to base output_dir if the *_addition variant wasn't defined.
        if add_addition and 'output_dir' in cfg:
            key = 'output_dir'
        else:
            raise KeyError(f"JSON config is missing '{key}'")
    return _abs(cfg[key])


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PWD, path)


def _model_cfg(name: str) -> dict[str, Any]:
    cfg = _load_paths_cfg()
    if name not in cfg['models']:
        raise KeyError(f"Unknown model '{name}'. Known: {list(cfg['models'])}")
    return cfg['models'][name]


# ---------------------------------------------------------------------------
# Shared VCF loader
# ---------------------------------------------------------------------------
_NEG_ADD_VCF: pd.DataFrame | None = None


def _load_neg_addition_vcf() -> pd.DataFrame:
    global _NEG_ADD_VCF
    if _NEG_ADD_VCF is None:
        cfg = _load_paths_cfg()
        _NEG_ADD_VCF = pd.read_csv(_abs(cfg['neg_addition_vcf']), sep='\t')
    return _NEG_ADD_VCF


_EXCLUDE_SET: set[str] | None = None


def _load_exclude_set() -> set[str]:
    global _EXCLUDE_SET
    if _EXCLUDE_SET is None:
        cfg = _load_paths_cfg()
        path = cfg.get('exclude')
        if path:
            with open(_abs(path)) as f:
                _EXCLUDE_SET = {line.strip() for line in f if line.strip()}
        else:
            _EXCLUDE_SET = set()
    return _EXCLUDE_SET


def _distance_to_group(distance: np.ndarray) -> np.ndarray:
    return np.where(distance < 3000, '<3k',
           np.where(distance < 12000, '3k-12k',
           np.where(distance < 35000, '12k-35k', '>35k')))


def load_tissue_vcf(tissue_dir: str, add_addition: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = _load_paths_cfg()
    tissue_name = os.path.basename(tissue_dir)

    if 'tissue_vcf_files' not in cfg:
        # Fallback: single global info CSV covering all tissues (e.g. AG dataset).
        info = pd.read_csv(_abs(cfg['info']))
        info_t = info[info['tissue'] == tissue_name]
        if len(info_t) == 0:
            return pd.DataFrame(columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'INFO', 'group']), np.array([])
        # Collapse multi-gene rows: one row per variant, keep min distance.
        agg = (info_t.groupby('variant_id', sort=False)
               .agg(label=('label', 'first'),
                    distance=('distance', 'min'),
                    chr=('chr', 'first'),
                    pos=('pos', 'first'),
                    ref=('ref', 'first'),
                    alt=('alt', 'first'))
               .reset_index())
        vcf = pd.DataFrame({
            '#CHROM': agg['chr'].values,
            'POS': agg['pos'].astype(int).values,
            'REF': agg['ref'].values,
            'ALT': agg['alt'].values,
            'ID': [v if v.endswith('_b38') else v + '_b38' for v in agg['variant_id'].values],
            'INFO': agg['label'].values,
            'group': _distance_to_group(agg['distance'].values),
        })
        conflict_ids = set(vcf.loc[vcf['INFO']=='positive','ID']) & set(vcf.loc[vcf['INFO']=='negative','ID'])
        vcf = vcf.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT']).reset_index(drop=True)
        print(f'[{tissue_name}] {len(vcf)} variants ({(vcf["INFO"]=="positive").sum()} pos, {(vcf["INFO"]=="negative").sum()} neg), {len(conflict_ids)} conflicts')
    else:
        # Standard path: per-tissue VCF + info files.
        vcf_fname = cfg['tissue_vcf_files']['vcf']
        info_fname = cfg['tissue_vcf_files']['info']

        vcf = pd.read_csv(os.path.join(tissue_dir, vcf_fname), sep='\t')
        info = pd.read_csv(os.path.join(tissue_dir, info_fname))

        variant_to_group = info.set_index('variant_id')['group']
        vcf['group'] = variant_to_group.reindex(vcf['ID']).values

        # Append additional negatives for this tissue.
        if add_addition:
            neg_add = _load_neg_addition_vcf()
            neg_tissue = neg_add[neg_add['tissue'] == tissue_name].copy()
            if len(neg_tissue) > 0:
                pos_ids = set(vcf.loc[vcf['INFO'] == 'positive', 'ID'])
                conflict_ids = pos_ids & set(neg_tissue['ID'])
                if conflict_ids:
                    vcf = vcf[~vcf['ID'].isin(conflict_ids)]
                    neg_tissue = neg_tissue[~neg_tissue['ID'].isin(conflict_ids)]
                shared_cols = [c for c in vcf.columns if c in neg_tissue.columns]
                vcf = pd.concat([vcf, neg_tissue[shared_cols]], ignore_index=True)

        keep = vcf['biotype'].isin(BIOTYPE_TO_KEEP) | (vcf['INFO'] == 'negative')
        vcf = vcf[keep].reset_index(drop=True)
        vcf = vcf.drop_duplicates(subset=['#CHROM', 'REF', 'POS', 'ALT']).reset_index(drop=True)

    # Exclude variants listed in the config (e.g. NaN gene_lfc variants).
    exclude = _load_exclude_set()
    if exclude:
        vcf = vcf[~vcf['ID'].isin(exclude)].reset_index(drop=True)

    labels = (vcf['INFO'].values == 'positive').astype(int)
    return vcf, labels


# ---------------------------------------------------------------------------
# Helper: build variant index from h5 arrays
# ---------------------------------------------------------------------------
def _build_variant_index(chr_arr, pos_arr, ref_arr, alt_arr) -> pd.Index:
    def _decode(arr):
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    chrs = _decode(chr_arr)
    refs = _decode(ref_arr)
    alts = _decode(alt_arr)
    ids = [f'{c}_{p}_{r}_{a}_b38' for c, p, r, a in zip(chrs, pos_arr, refs, alts)]
    return pd.Index(ids)


def _dedup_variants(log_square: np.ndarray, variant_index: pd.Index) -> tuple[np.ndarray, pd.Index]:
    seen = set()
    keep_rows = []
    for i, vid in enumerate(variant_index):
        if vid not in seen:
            seen.add(vid)
            keep_rows.append(i)
    keep_rows = np.array(keep_rows)
    return log_square[keep_rows], variant_index[keep_rows]


# ---------------------------------------------------------------------------
# Low-level h5 readers
# ---------------------------------------------------------------------------
def _load_log_square_h5(h5_path: str, score_name: str = 'results/log_square') -> tuple[np.ndarray, pd.Index]:
    """Read <score_name> + variants/{chr,pos,ref,alt} (BICAN/Borzoi format)."""
    with h5py.File(h5_path, 'r') as f:
        log_square = f[score_name][:]
        variant_index = _build_variant_index(
            f['variants/chr'][:], f['variants/pos'][:],
            f['variants/ref'][:], f['variants/alt'][:],
        )
    return log_square, variant_index


def _load_alphagenome_h5(
    h5_path: str,
    ag_heads: list[str],
    score_name: str = 'scores/{head}/l2_sum',
) -> tuple[np.ndarray, pd.Index]:
    """Read <score_name> per head (the template must contain {head}) + variants/{chrom,pos,ref,alt}."""
    with h5py.File(h5_path, 'r') as f:
        score_blocks = [f[score_name.format(head=head)][:] for head in ag_heads]
        log_square = np.concatenate(score_blocks, axis=1)
        variant_index = _build_variant_index(
            f['variants/chrom'][:], f['variants/pos'][:],
            f['variants/ref'][:], f['variants/alt'][:],
        )
    return log_square, variant_index


_SCORE_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'abs': np.abs,
}


def _apply_score_transform(scores: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    t = cfg.get('score_transform')
    if t is None:
        return scores
    if t not in _SCORE_TRANSFORMS:
        raise ValueError(f"Unknown score_transform '{t}'. Known: {list(_SCORE_TRANSFORMS)}")
    return _SCORE_TRANSFORMS[t](scores)


def _load_with_addition(
    reader: Callable[[str], tuple[np.ndarray, pd.Index]],
    h5_path: str,
    h5_add_path: str | None,
    add_addition: bool,
) -> tuple[np.ndarray, pd.Index]:
    log_square, variant_index = reader(h5_path)
    if add_addition and h5_add_path and os.path.exists(h5_add_path):
        log_square_add, variant_index_add = reader(h5_add_path)
        log_square = np.concatenate([log_square, log_square_add], axis=0)
        variant_index = variant_index.append(variant_index_add)
    return _dedup_variants(log_square, variant_index)


# ---------------------------------------------------------------------------
# BICAN loader
# ---------------------------------------------------------------------------
def load_bican(name: str = 'bican', bican_filter: str | None = None, add_addition: bool = True) -> ModelData:
    cfg = _model_cfg(name)
    score_name = cfg.get('score_name', 'results/log_square')
    h5_add = _abs(cfg['h5_addition']) if cfg.get('h5_addition') else None
    log_square, variant_index = _load_with_addition(
        lambda p: _load_log_square_h5(p, score_name),
        _abs(cfg['h5']), h5_add, add_addition,
    )
    log_square = _apply_score_transform(log_square, cfg)

    track_anno = pd.read_csv(_abs(cfg['track_anno']))

    if bican_filter:
        bican_filter_map = {
            'basal_ganglia': ('BasalGanglia', None),
            'cortex': ('MiniAtlas', None),
            'basal_ganglia_rna': ('BasalGanglia', r'RNA'),
            'cortex_rna': ('MiniAtlas', r'RNA'),
            'rna': ('', r'RNA'),
        }
        ct_prefix, mod_pattern = bican_filter_map[bican_filter]
        mask = track_anno['cell_type'].str.startswith(ct_prefix, na=False)
        if mod_pattern:
            mask = mask & track_anno['modality'].str.contains(mod_pattern, case=False, na=False)
        col_indices = track_anno.index[mask].values
        log_square = log_square[:, col_indices]
        track_anno = track_anno.loc[mask].reset_index(drop=True)
        print(f'BICAN filter: {bican_filter}, Tracks kept: {len(track_anno)}')

    return ModelData(
        name=name,
        log_square=log_square,
        variant_index=variant_index,
        track_anno=track_anno,
    )


# ---------------------------------------------------------------------------
# Borzoi loader
# ---------------------------------------------------------------------------
def load_borzoi(name: str = 'borzoi', track_filter: str | None = None, add_addition: bool = True) -> ModelData:
    cfg = _model_cfg(name)
    score_name = cfg.get('score_name', 'results/log_square')
    h5_add = _abs(cfg['h5_addition']) if cfg.get('h5_addition') else None
    log_square, variant_index = _load_with_addition(
        lambda p: _load_log_square_h5(p, score_name),
        _abs(cfg['h5']), h5_add, add_addition,
    )
    log_square = _apply_score_transform(log_square, cfg)

    track_anno = pd.read_csv(_abs(cfg['track_anno']), index_col=0, sep='\t')
    track_anno['modality'] = track_anno['file'].str.split('/').str[6]

    # Keep CAGE, DNASE, RNA, CHIP H3K27ac/H3K27me3/H3K9me3
    chip_pattern = r'H3K27ac|H3K27me3|H3K9me3'
    mod_mask = (
        track_anno['description'].str.contains(r'\bCAGE\b|\bDNASE\b|\bDNase\b|\bRNA\b', case=False, na=False) |
        track_anno['description'].str.contains(chip_pattern, case=False, na=False)
    )
    mod_indices = track_anno.index[mod_mask].values
    track_anno = track_anno[mod_mask].reset_index(drop=True)

    if track_filter:
        pattern = REGION_FILTER_MAP[track_filter]
        embryo_mask = track_anno['description'].str.contains(EMBRYO_FETAL_TRACK_PATTERN, case=False, na=False)
        disease_mask = track_anno['description'].str.contains(DISEASE_TRACK_PATTERN, case=False, na=False)
        filt_mask = (
            track_anno['description'].str.contains(pattern, case=False, na=False) &
            ~embryo_mask & ~disease_mask
        )
        filt_indices = track_anno.index[filt_mask].values
        track_anno = track_anno.loc[filt_mask].reset_index(drop=True)
        final_indices = mod_indices[filt_indices]
        print(f'Borzoi filter: {track_filter}, Tracks kept: {len(track_anno)}')
    else:
        final_indices = mod_indices

    log_square = log_square[:, final_indices]

    return ModelData(
        name=name,
        log_square=log_square,
        variant_index=variant_index,
        track_anno=track_anno,
    )


# ---------------------------------------------------------------------------
# AlphaGenome loader (single entry point — name selects the JSON-defined variant)
# ---------------------------------------------------------------------------
def load_alphagenome(
    name: str,
    track_filter: str | None = None,
    add_addition: bool = True,
) -> ModelData:
    cfg = _model_cfg(name)
    if cfg.get('type') != 'alphagenome':
        raise ValueError(f"Model '{name}' is not an alphagenome variant (type={cfg.get('type')})")

    ag_heads: list[str] = cfg['heads']
    score_name = cfg.get('score_name', 'scores/{head}/l2_sum')
    h5_add = _abs(cfg['h5_addition']) if cfg.get('h5_addition') else None

    log_square, variant_index = _load_with_addition(
        lambda p: _load_alphagenome_h5(p, ag_heads, score_name),
        _abs(cfg['h5']), h5_add, add_addition,
    )
    log_square = _apply_score_transform(log_square, cfg)

    catalog = pd.read_csv(_abs(cfg['track_anno']))
    track_anno = catalog[catalog['output_type'].isin(ag_heads)].reset_index(drop=True)
    track_anno = track_anno.rename(columns={'output_type': 'modality'})

    # Keep CAGE, DNASE, RNA, ATAC, CHIP H3K27ac/H3K27me3/H3K9me3
    chip_pattern = r'H3K27ac|H3K27me3|H3K9me3'
    mod_mask = (
        track_anno['modality'].isin(['cage', 'dnase', 'rna_seq', 'atac']) |
        track_anno['histone_mark'].str.contains(chip_pattern, case=False, na=False)
    )
    mod_indices = track_anno.index[mod_mask].values
    track_anno_mod = track_anno.loc[mod_mask].reset_index(drop=True)
    track_desc = (track_anno_mod['biosample_name'].fillna('') + ' ' +
                  track_anno_mod['track_name'].fillna(''))

    tissue_filter_fn = None
    if track_filter == 'gtex':
        def _gtex_filter(tissue, ta):
            tissue_pattern = f'gtex {tissue}'
            mask = track_desc.str.contains(tissue_pattern, case=False, na=False)
            indices = ta.index[mask].values
            return ta.loc[mask].reset_index(drop=True), indices
        tissue_filter_fn = _gtex_filter
        track_anno_final = track_anno_mod
        final_indices = mod_indices
    elif track_filter and track_filter in REGION_FILTER_MAP:
        pattern = REGION_FILTER_MAP[track_filter]
        embryo_mask = track_desc.str.contains(EMBRYO_FETAL_TRACK_PATTERN, case=False, na=False)
        disease_mask = track_desc.str.contains(DISEASE_TRACK_PATTERN, case=False, na=False)
        filt_mask = (
            track_desc.str.contains(pattern, case=False, na=False) &
            ~embryo_mask & ~disease_mask
        )
        filt_indices = track_anno_mod.index[filt_mask].values
        track_anno_final = track_anno_mod.loc[filt_mask].reset_index(drop=True)
        final_indices = mod_indices[filt_indices]
        print(f'{name} filter: {track_filter}, Tracks kept: {len(track_anno_final)}')
    else:
        track_anno_final = track_anno_mod
        final_indices = mod_indices

    log_square = log_square[:, final_indices]
    track_anno_final['identifier'] = track_anno_final['track_name']

    return ModelData(
        name=name,
        log_square=log_square,
        variant_index=variant_index,
        track_anno=track_anno_final,
        tissue_filter_fn=tissue_filter_fn,
    )


# ---------------------------------------------------------------------------
# AlphaGenome paper loader
#
# Input is a long-format CSV of per-(variant, gene, tissue) predictions
# published with the AlphaGenome paper, not an h5 of predicted tracks.
# Columns of interest: chr, pos, ref, alt, tissue, gene_id, prediction.
# We pivot to (n_variants, n_tissues) so it fits the ModelData interface;
# each "track" is one GTEx tissue (modality=rna_seq).
# ---------------------------------------------------------------------------
def load_alphagenome_paper(
    name: str,
    track_filter: str | None = None,
    add_addition: bool = True,  # unused: paper CSV already contains all variants
) -> ModelData:
    del add_addition  # accepted for API parity with other loaders
    cfg = _model_cfg(name)
    if cfg.get('type') != 'alphagenome_paper':
        raise ValueError(f"Model '{name}' is not alphagenome_paper (type={cfg.get('type')})")

    df = pd.read_csv(_abs(cfg['h5']))

    required = {'chr', 'pos', 'ref', 'alt', 'tissue', 'prediction'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'alphagenome_paper CSV missing columns: {sorted(missing)}')

    df = df.dropna(subset=['chr', 'pos', 'ref', 'alt', 'tissue', 'prediction'])
    df['vid'] = (df['chr'].astype(str) + '_' +
                 df['pos'].astype('int64').astype(str) + '_' +
                 df['ref'].astype(str) + '_' +
                 df['alt'].astype(str) + '_b38')

    # Collapse multi-gene / multi-metric rows to one score per (variant, tissue).
    # Take max across genes so the strongest effect for a variant is retained.
    dedup_keys = ['vid', 'tissue'] + (['gene_id'] if 'gene_id' in df.columns else [])
    df_dedup = df.drop_duplicates(subset=dedup_keys)
    pivot = (df_dedup.groupby(['vid', 'tissue'])['prediction']
             .max()
             .unstack(fill_value=0.0))

    variant_index = pd.Index(pivot.index.tolist())
    log_square = pivot.values.astype(np.float32)
    tissues = list(pivot.columns)

    track_anno = pd.DataFrame({
        'track_name': tissues,
        'biosample_name': tissues,
        'modality': 'rna_seq',
        'identifier': tissues,
    })

    # Per-tissue filter: pick the column whose identifier exactly matches the
    # GTEx tissue being evaluated. Mirrors the gtex branch of load_alphagenome.
    def _tissue_filter(tissue, ta):
        mask = ta['identifier'] == tissue
        return ta.loc[mask].reset_index(drop=True), ta.index[mask].values

    if track_filter is not None and track_filter != 'gtex':
        print(f"alphagenome_paper: ignoring track_filter='{track_filter}' "
              f"(only per-tissue filtering is supported)")

    return ModelData(
        name=name,
        log_square=log_square,
        variant_index=variant_index,
        track_anno=track_anno,
        tissue_filter_fn=_tissue_filter,
    )


# ---------------------------------------------------------------------------
# Type-dispatching entry point
# ---------------------------------------------------------------------------
def load_model(name: str, track_filter: str | None = None, add_addition: bool = True) -> ModelData:
    """Dispatch to the right loader based on the model's 'type' field in JSON."""
    cfg = _model_cfg(name)
    t = cfg.get('type')
    if t == 'bican':
        return load_bican(name, track_filter, add_addition)
    if t == 'borzoi':
        return load_borzoi(name, track_filter, add_addition)
    if t == 'alphagenome':
        return load_alphagenome(name, track_filter, add_addition)
    if t == 'alphagenome_paper':
        return load_alphagenome_paper(name, track_filter, add_addition)
    raise ValueError(f"Unknown model type '{t}' for model '{name}'")
