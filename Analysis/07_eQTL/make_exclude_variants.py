"""Generate exclude-variant list from one or more H5 files defined in a data_paths JSON.

For each model entry in the JSON:
  - Read the dataset(s) pointed to by `score_name` (supports {head} template for
    alphagenome type where scores are split per head).
  - A variant is excluded from this model if ALL tracks are NaN for that variant.
  - Variant IDs are normalised to chr_pos_ref_alt_b38 (BICAN/Borzoi use `chr`;
    alphagenome uses `chrom` — both are handled).

The final exclude list is the union across all models and is written to the path
specified by `"exclude"` in the JSON, or to --output if given.

Usage:
    python make_exclude_variants.py data_paths.ag_like.json
    python make_exclude_variants.py data_paths.ag_like.json --output path/to/out.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PWD, path)


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _build_variant_ids(f: h5py.File) -> list[str]:
    """Return chr_pos_ref_alt_b38 IDs from a BICAN/Borzoi or alphagenome H5."""
    vg = f['variants']
    # alphagenome uses 'chrom'; BICAN/Borzoi use 'chr'
    chr_key = 'chrom' if 'chrom' in vg else 'chr'
    chrs = _decode(vg[chr_key][:])
    pos  = vg['pos'][:]
    refs = _decode(vg['ref'][:])
    alts = _decode(vg['alt'][:])
    return [f'{c}_{p}_{r}_{a}_b38' for c, p, r, a in zip(chrs, pos, refs, alts)]


def _all_nan_mask(f: h5py.File, score_name: str, heads: list[str] | None) -> np.ndarray:
    """Return boolean array (n_variants,): True where ALL tracks are NaN.

    For alphagenome (heads is not None) scores are concatenated across heads
    before the row-wise all-NaN check.
    """
    if heads:
        # Template e.g. "scores/{head}/gene_abs_lfc"
        blocks = []
        for head in heads:
            path = score_name.format(head=head)
            if path not in f:
                print(f'  [warn] path not found: {path} — skipping head {head}', file=sys.stderr)
                continue
            blocks.append(f[path][:])
        if not blocks:
            return np.zeros(len(f['variants/pos'][:]), dtype=bool)
        scores = np.concatenate(blocks, axis=1)
    else:
        if score_name not in f:
            print(f'  [warn] score_name not found: {score_name} — skipping', file=sys.stderr)
            return np.zeros(len(f['variants/pos'][:]), dtype=bool)
        scores = f[score_name][:]

    return np.all(np.isnan(scores), axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('json', help='Path to data_paths JSON (e.g. data_paths.ag_like.json)')
    parser.add_argument('--output', default=None,
                        help='Override output path (default: use "exclude" key in JSON)')
    args = parser.parse_args()

    with open(args.json) as fh:
        cfg = json.load(fh)

    out_path = args.output or cfg.get('exclude')
    if not out_path:
        sys.exit('No output path: set "exclude" in the JSON or pass --output.')
    out_path = _abs(out_path)

    exclude: set[str] = set()

    for model_name, model_cfg in cfg.get('models', {}).items():
        h5_path = _abs(model_cfg['h5'])
        score_name = model_cfg.get('score_name', '')
        model_type = model_cfg.get('type', '')
        heads = model_cfg.get('heads') if model_type == 'alphagenome' else None

        print(f'[{model_name}] {os.path.basename(h5_path)}  score={score_name}')

        if not os.path.exists(h5_path):
            print(f'  [warn] H5 not found: {h5_path} — skipping', file=sys.stderr)
            continue

        with h5py.File(h5_path, 'r') as f:
            all_nan = _all_nan_mask(f, score_name, heads)
            variant_ids = _build_variant_ids(f)

        n_excl = int(all_nan.sum())
        print(f'  all-NaN variants: {n_excl} / {len(variant_ids)}')

        excl_ids = {vid for vid, flag in zip(variant_ids, all_nan) if flag}
        exclude |= excl_ids

    print(f'\nTotal exclude variants (union): {len(exclude)}')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        for vid in sorted(exclude):
            fh.write(vid + '\n')

    print(f'Written → {out_path}')


if __name__ == '__main__':
    main()
