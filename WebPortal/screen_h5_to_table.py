#!/usr/bin/env python
"""Extract a variant-by-track fold-change table from the variant-effect-screen
HDF5 produced by ``Analysis/03_variant_effect_screen`` (script.sh + merge).

HDF5 layout (see Analysis/03_variant_effect_screen/README.md and init_tasks.py):
  model_meta/trial_names         track names      [n_tracks]   (column headers)
  variants/{index_key,rsid,chr,pos,ref,alt}       [n_variants] (full variant table)
  results/{score_name}           score matrix     [n_variants x n_tracks]
  experiments/{exp}/index_key    variants of this run                (row subset)
  experiments/{exp}/reverse_map  allele-orientation flag (optional, GWAS only)

Deliverable: a CSV with one row per variant of the requested experiment and one
column per track, holding the chosen fold-change score. Leading columns carry
variant identifiers. ``--score auto`` prefers ``gene_lfc`` when it exists and is
not entirely NaN, else falls back to ``local_raw_log_diff``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ID_COLS = ("index_key", "rsid", "chr", "pos", "ref", "alt")
_AUTO_PREFERENCE = ("gene_lfc", "local_raw_log_diff")


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode an HDF5 string dataset to a numpy array of python str."""
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def _select_score(results_grp: h5py.Group, requested: str) -> str:
    available = list(results_grp.keys())
    if requested != "auto":
        if requested not in available:
            raise KeyError(f"score {requested!r} not in HDF5 (available: {available})")
        return requested
    for cand in _AUTO_PREFERENCE:
        if cand in available:
            data = results_grp[cand]
            # gene_lfc rows are NaN where no exon overlap; skip if entirely NaN.
            if not np.all(np.isnan(data[: min(len(data), 1000)])):
                return cand
    raise KeyError(
        f"no usable score among {_AUTO_PREFERENCE}; available: {available}"
    )


def extract_table(h5_path: Path, experiment: str, score: str, output: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        if "model_meta/trial_names" not in f:
            raise KeyError("model_meta/trial_names missing; not a screen HDF5")
        tracks = _decode(f["model_meta/trial_names"][:])

        exp_grp_path = f"experiments/{experiment}"
        if exp_grp_path not in f:
            avail = list(f["experiments"].keys()) if "experiments" in f else []
            raise KeyError(f"experiment {experiment!r} not found (have: {avail})")

        score_name = _select_score(f["results"], score)
        scores = f[f"results/{score_name}"][:].astype(np.float64)  # [n_var, n_track]

        all_index = _decode(f["variants/index_key"][:])
        variants = {c: _decode(f[f"variants/{c}"][:]) for c in ID_COLS}

        exp_index = _decode(f[f"{exp_grp_path}/index_key"][:])
        reverse_map = (
            f[f"{exp_grp_path}/reverse_map"][:].astype(bool)
            if "reverse_map" in f[exp_grp_path]
            else None
        )

    if scores.shape[1] != len(tracks):
        raise ValueError(
            f"score matrix has {scores.shape[1]} columns but {len(tracks)} track names"
        )

    pos_of = {key: i for i, key in enumerate(all_index)}
    missing = [k for k in exp_index if k not in pos_of]
    if missing:
        print(
            f"WARNING: {len(missing)} experiment variants absent from variants table "
            f"(first: {missing[:3]})",
            file=sys.stderr,
        )
    keep = [k for k in exp_index if k in pos_of]
    rows = np.array([pos_of[k] for k in keep], dtype=np.int64)

    sub = scores[rows, :]
    if reverse_map is not None:
        order = {k: i for i, k in enumerate(exp_index)}
        flip = np.array([reverse_map[order[k]] for k in keep], dtype=bool)
        # Allele swap (A1/A2 reversed) flips the sign of the ALT-vs-REF fold change.
        sub = sub * (1.0 - 2.0 * flip).reshape(-1, 1)

    id_frame = pd.DataFrame({c: variants[c][rows] for c in ID_COLS})
    score_frame = pd.DataFrame(sub, columns=list(tracks))
    table = pd.concat([id_frame.reset_index(drop=True), score_frame], axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(
        f"Wrote {table.shape[0]} variants x {len(tracks)} tracks "
        f"(score={score_name}) -> {output}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, type=Path, help="Merged screen HDF5 file")
    p.add_argument("--experiment", required=True, help="Experiment name used in the screen")
    p.add_argument("--score", default="auto",
                   help="Score dataset name, or 'auto' (gene_lfc -> local_raw_log_diff)")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path")
    args = p.parse_args()

    try:
        extract_table(args.h5, args.experiment, args.score, args.output)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
