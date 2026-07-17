"""Shared helpers for cCRE-rank screening (gradient×input and in-silico CRISPRi).

Both screens operate on the ranked enhancer table
(``enhancer_candidates.ranked.tsv``) and centre a ``context_length`` window on
the midpoint between the cCRE centre and the target gene body midpoint, so that
both the cCRE and the gene fall inside a single model input.

Task expansion (per the study design):
  * gradient×input : (cCRE) × (each abc_target_gene) × (each supporting_celltype)
                     → one attribution score per triple.
  * CRISPRi        : (cCRE) × (each abc_target_gene)
                     → one silencing run scored against ALL model cell types.

Coordinate conventions:
  * ranked.tsv ``start``/``end``  : 0-based half-open (BED-style), as in
    11_TFMotif/08_abc_cCRE_attribution.py.
  * GTF genes/exons              : parsed to 0-based half-open.
"""
import gzip
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Standard chromosomes (matches Model/data/data_utils.STD_CHR) — defined locally
# so the CPU-only `build` step avoids importing torch/pysam.
STD_CHR = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# ranked.tsv cell types (L6IT-1, L23IT, ...) are the MiniAtlas cortical set.
MINIATLAS_PREFIX = "MiniAtlas-"

RANKED_DEFAULT = "Analysis/18_cCRE_rank/enhancer_candidates.ranked.tsv"
GTF_DEFAULT = "Data/source/gencode.v48.annotation.gtf.gz"
OUT_ROOT_DEFAULT = "Analysis/18_cCRE_rank/output"


# --------------------------------------------------------------------------- #
# GTF parsing (matches 11_TFMotif/00_interpret_gene_RNA.py)
# --------------------------------------------------------------------------- #
def parse_gtf(gtf_path: str, gene_set: set) -> dict:
    """Parse GTF for protein-coding genes in ``gene_set``.

    Returns {gene_name: {'chr','start','end','strand','exons':[(s,e),...]}} with
    0-based half-open coordinates. Only genes on standard chromosomes are kept.
    """
    genes: dict = {}
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            feat = cols[2]
            if feat not in ("gene", "exon"):
                continue
            attr = cols[8]
            if "protein_coding" not in attr:
                continue
            m = re.search(r'gene_name "([^"]+)"', attr)
            if not m:
                continue
            gname = m.group(1)
            if gname not in gene_set:
                continue
            chrom = cols[0]
            start = int(cols[3]) - 1  # GTF 1-based → 0-based
            end = int(cols[4])        # GTF end inclusive → 0-based half-open
            strand = cols[6]
            if feat == "gene":
                if chrom in STD_CHR:
                    genes[gname] = {
                        "chr": chrom, "start": start, "end": end,
                        "strand": strand, "exons": [],
                    }
            elif feat == "exon" and gname in genes:
                genes[gname]["exons"].append((start, end))
    return genes


def exon_bin_range(exons, real_start: int, window_size: int, n_output_bins: int):
    """Sorted array of output-bin indices covered by any exon; None if empty.

    Bins are expressed in the un-cropped model output frame (crop disabled),
    relative to ``real_start`` (the 0-based genomic start of the input window).
    """
    bins: set = set()
    for exon_start, exon_end in exons:
        rel_s = max(0, exon_start - real_start)
        rel_e = min(n_output_bins * window_size, exon_end - real_start)
        if rel_s >= rel_e:
            continue
        b_s = rel_s // window_size
        b_e = (rel_e - 1) // window_size
        for b in range(b_s, min(b_e + 1, n_output_bins)):
            bins.add(b)
    return np.array(sorted(bins)) if bins else None


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def gene_midpoint(ginfo: dict) -> int:
    return (int(ginfo["start"]) + int(ginfo["end"])) // 2


def window_center(ccre_start: int, ccre_end: int, gene_mid: int) -> int:
    """Midpoint between the cCRE centre and the gene body midpoint."""
    ccre_center = (int(ccre_start) + int(ccre_end)) // 2
    return (ccre_center + gene_mid) // 2


def fits_window(ccre_start: int, ccre_end: int, ginfo: dict, context_length: int) -> bool:
    """True iff both the cCRE and the whole gene body fit in a window centred on
    ``window_center`` with width ``context_length``."""
    gene_mid = gene_midpoint(ginfo)
    center = window_center(ccre_start, ccre_end, gene_mid)
    half = context_length // 2
    lo, hi = center - half, center + half
    return (int(ccre_start) >= lo and int(ccre_end) <= hi
            and int(ginfo["start"]) >= lo and int(ginfo["end"]) <= hi)


def rna_trial(cell_type: str, strand: str) -> str:
    """+ strand → RNAminus track; - strand → RNAplus track (matches 00_interpret_gene_RNA)."""
    modality = "RNAminus" if strand == "+" else "RNAplus"
    return f"{cell_type}_{modality}"


# --------------------------------------------------------------------------- #
# Task-list construction
# --------------------------------------------------------------------------- #
def load_ranked(ranked_path: str) -> pd.DataFrame:
    df = pd.read_csv(ranked_path, sep="\t")
    need = {"chr", "start", "end", "cCRE_id", "abc_target_genes"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"ranked table missing columns: {sorted(missing)}")
    return df


def _split_field(val) -> list:
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]


def build_gene_set(ranked: pd.DataFrame) -> set:
    genes: set = set()
    for v in ranked["abc_target_genes"]:
        genes.update(_split_field(v))
    return genes


def write_chunks(rows: list, columns: list, out_tasks_dir: str, chunk_size: int) -> int:
    """Write ``rows`` to numbered chunk TSVs (chunk_0001.tsv, ...). Returns n_chunks."""
    os.makedirs(out_tasks_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    n = len(df)
    n_chunks = (n + chunk_size - 1) // chunk_size if n else 0
    for i in range(n_chunks):
        sub = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        sub.to_csv(os.path.join(out_tasks_dir, f"chunk_{i + 1:04d}.tsv"),
                   sep="\t", index=False)
    with open(os.path.join(out_tasks_dir, "n_chunks.txt"), "w") as fh:
        fh.write(str(n_chunks) + "\n")
    return n_chunks
