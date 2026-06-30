#!/usr/bin/env python
"""Convert per-base attribution (importance) scores produced by
``Analysis/02_motif_gene_diff_interpretation.py`` (Step 5 of the variant
analysis pipeline) into a bigWig track for the web portal.

Step 5 writes, per interpreted region:
  - ``{data_dir}/{name_base}_metadata.npy``                  (dict, allow_pickle)
  - ``{data_dir}/{name_base}_{baseline}_importance.npy``     (float32 [L])

``importance`` is the bp-resolution signal ``(attribution * one_hot).sum(-1)``,
already trimmed to the prediction window. Its genomic span is reconstructed from
the metadata exactly as ``02_motif_interpretation_plot.py`` does:

    trim            = (context_length // window_size - n_window) // 2
    total_bp        = n_window * window_size
    region_start_bp = real_start + trim * window_size          # 0-based
    region_end_bp   = region_start_bp + total_bp

The stored signal is at bp resolution (length ``total_bp``, one value per base).
For a compact bigWig the track is aggregated to ``window_size`` (e.g. 32 bp)
bins by averaging each window, giving ``n_window`` entries with
``span = step = window_size``. Already-binned inputs (length ``n_window``) are
written directly at the same bin resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyBigWig


def _load_chrom_sizes(fai_path: Path) -> dict[str, int]:
    """Read chromosome sizes from a ``.fai`` index (cols: name, length, ...)."""
    if not fai_path.is_file():
        raise FileNotFoundError(f"FASTA index not found: {fai_path}")
    sizes: dict[str, int] = {}
    with fai_path.open() as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise ValueError(f"No chromosome sizes parsed from {fai_path}")
    return sizes


def _reconstruct_span(metadata: dict) -> tuple[str, int, int, int]:
    """Return (chrom, region_start_bp, total_bp, window_size) from Step 5 metadata."""
    required = ("chr_name", "real_start", "window_size", "n_window", "context_length")
    missing = [k for k in required if k not in metadata]
    if missing:
        raise KeyError(f"metadata is missing required keys: {missing}")

    window_size = int(metadata["window_size"])
    n_window = int(metadata["n_window"])
    context_length = int(metadata["context_length"])
    real_start = int(metadata["real_start"])

    trim = (context_length // window_size - n_window) // 2
    total_bp = n_window * window_size
    region_start_bp = real_start + trim * window_size
    return str(metadata["chr_name"]), region_start_bp, total_bp, window_size


def _bin_signal(values: np.ndarray, total_bp: int, window_size: int) -> tuple[np.ndarray, int]:
    """Aggregate the stored signal to ``window_size`` bins for the bigWig.

    Returns (binned_values, span) where span is the bp covered per value.
    - bp resolution (len == total_bp): average each window -> n_window values, span=window_size
    - already binned (len == total_bp/window_size): used as-is, span=window_size
    - other length: written at bp resolution (span=1) as a safe fallback
    """
    n_window = total_bp // window_size
    if values.shape[0] == total_bp:
        return values.reshape(n_window, window_size).mean(axis=1), window_size
    if values.shape[0] == n_window:
        return values, window_size
    print(
        f"WARNING: importance length ({values.shape[0]}) matches neither bp "
        f"({total_bp}) nor bin ({n_window}) resolution; writing at bp resolution.",
        file=sys.stderr,
    )
    return values, 1


def attribution_to_bigwig(
    data_dir: Path,
    name_base: str,
    baseline: str,
    output: Path,
    fai_path: Path,
) -> None:
    metadata_file = data_dir / f"{name_base}_metadata.npy"
    importance_file = data_dir / f"{name_base}_{baseline}_importance.npy"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_file}")
    if not importance_file.is_file():
        raise FileNotFoundError(f"importance scores not found: {importance_file}")

    metadata = np.load(metadata_file, allow_pickle=True).item()
    values = np.load(importance_file).astype(np.float64).ravel()

    chrom, region_start_bp, total_bp, window_size = _reconstruct_span(metadata)
    values, span = _bin_signal(values, total_bp, window_size)

    chrom_sizes = _load_chrom_sizes(fai_path)
    if chrom not in chrom_sizes:
        raise KeyError(f"chromosome {chrom!r} absent from {fai_path}")
    chrom_size = chrom_sizes[chrom]

    region_end_bp = region_start_bp + values.shape[0] * span
    if region_start_bp < 0 or region_end_bp > chrom_size:
        raise ValueError(
            f"reconstructed span {chrom}:{region_start_bp}-{region_end_bp} "
            f"is outside chromosome bounds (0-{chrom_size})"
        )

    # bigWig cannot store NaN/Inf; clamp them to 0 so the track stays loadable.
    n_bad = int((~np.isfinite(values)).sum())
    if n_bad:
        print(f"WARNING: zeroing {n_bad} non-finite attribution values.", file=sys.stderr)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(output), "w")
    try:
        bw.addHeader([(chrom, chrom_size)])
        bw.addEntries(
            chrom,
            int(region_start_bp),
            values=values,
            span=span,
            step=span,
        )
    finally:
        bw.close()

    res = "bp" if span == 1 else f"{span}bp-bin"
    print(
        f"Wrote attribution bigWig: {output} "
        f"({chrom}:{region_start_bp}-{region_end_bp}, {values.shape[0]} values @ {res})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing Step 5 *_metadata.npy / *_importance.npy")
    p.add_argument("--name-base", required=True,
                   help="Name base printed by Step 5 (e.g. chr11_..._STR-D1-MSN)")
    p.add_argument("--baseline", default="random",
                   help="Baseline type used in Step 5 (default: random)")
    p.add_argument("--output", required=True, type=Path, help="Output .bw path")
    p.add_argument("--fai", type=Path,
                   default=Path("Data/source/hg38/hg38.fa.fai"),
                   help="FASTA .fai index for chromosome sizes")
    args = p.parse_args()

    try:
        attribution_to_bigwig(
            data_dir=args.data_dir,
            name_base=args.name_base,
            baseline=args.baseline,
            output=args.output,
            fai_path=args.fai,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
