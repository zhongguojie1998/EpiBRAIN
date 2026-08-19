"""Element-centred in-silico CRISPRi screen: every gene in the locus × every cell type.

One task = one 500 bp element. The element is placed at the **centre** of a
``context_length`` (524,288 bp) window, its bases are replaced with pad tokens
(0.25), and reference vs. silenced RNA predictions are compared over the exon
bins of **every protein-coding gene whose exons fall in that window**, for all 70
model cell types (strand-aware RNA track). Four forward passes per element
(ref/CRISPRi × forward/reverse-complement) therefore serve the whole locus.

Unlike ``02_crispri_cCRE_screen.py`` this screen is hypothesis-free: it needs no
ABC link list, and the window geometry depends only on the element.

Results are HDF5, one **tidy table per element** — one row per (gene, cell_type),
gene-major, with the per-gene metadata repeated down each cell-type block:

    /cell_types                [n_ct] utf-8
    /elements/<enh_id>/table   compound [n_genes * n_ct], LONG_DTYPE:
        gene S32 | cell_type S40 | strand S1 | tss i8 | dist_to_element i8
        | n_exon_bins i4 | exon_bin_frac f4 | fully_inside bool
        | pred_ref f4 | pred_crispri f4 | log2fc f4
    /elements/<enh_id>.attrs   chr, start, end, class, win_start, win_end
    /.attrs                    layout="long", context_length, window_size, exp_name, chk

  dist_to_element : signed, tss - element centre
  n_exon_bins     : exon bins of that gene inside the window
  exon_bin_frac   : in-window / total exon bins (< 1 ⇒ transcript truncated by the
                    window edge, so pred_ref covers only part of the gene)
  fully_inside    : whole gene body inside the window

``read_element(f, enh_id)`` returns that table as a DataFrame with the element's
``enh_id/chr/start/end/class`` prepended, and also reads the older wide
``cell_type × gene`` layout (rebuilding the long form on the fly). Use ``convert``
to migrate wide chunk files in place of re-running the screen.

Note on track indexing: ``regression_label_meta.csv`` carries two indices — ``dim``
is the model **prediction** dim (the head emits every modality for every cell type,
420 tracks) and ``label_dim`` indexes the label matrix. Predictions are sliced with
``dim``, as in ``02_crispri_cCRE_screen.py``.

Workflow:
  python Analysis/18_cCRE_rank/04_crispri_element_screen.py build \\
      --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \\
      --chk 17 --chunk_size 2000
  python Analysis/18_cCRE_rank/04_crispri_element_screen.py run \\
      --exp_name ... --chk 17 --chunk_id 1 --device cuda:0
  python Analysis/18_cCRE_rank/04_crispri_element_screen.py merge --exp_name ... --chk 17
  # migrate pre-existing wide results, then merge from the converted dir:
  python Analysis/18_cCRE_rank/04_crispri_element_screen.py convert --exp_name ... --chk 17
  python Analysis/18_cCRE_rank/04_crispri_element_screen.py merge --exp_name ... --chk 17 \
      --results_subdir results_long --force_restart
"""
import glob
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis" / "18_cCRE_rank"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import _cCRE_rank_common as C  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import BaseLogger  # noqa: E402

BED_DEFAULT = "Analysis/18_cCRE_rank/epibrain_elements.500bp.bed"
BED_COLUMNS = ["chr", "start", "end", "enh_id", "class", "width"]
TASK_COLUMNS = ["enh_id", "chr", "start", "end", "class", "genes"]
EPS = 1e-6
STR_DT = h5py.string_dtype(encoding="utf-8")

# One row per (gene, cell_type). Fixed-width bytes for the string fields so the
# table is a plain packed record array (no vlen heap); the heavy duplication of
# the gene metadata across cell types compresses away with gzip-1.
LONG_DTYPE = np.dtype([
    ("gene", "S32"), ("cell_type", "S40"), ("strand", "S1"),
    ("tss", "<i8"), ("dist_to_element", "<i8"),
    ("n_exon_bins", "<i4"), ("exon_bin_frac", "<f4"), ("fully_inside", "?"),
    ("pred_ref", "<f4"), ("pred_crispri", "<f4"), ("log2fc", "<f4"),
])
LONG_STR_COLS = ("gene", "cell_type", "strand")


def _out_dirs(out_root, exp_name, chk):
    base = os.path.join(out_root, f"{exp_name}_{chk}", "crispri_element")
    return base, os.path.join(base, "tasks"), os.path.join(base, "results")


def _gene_db_path(base):
    return os.path.join(base, "gene_db.pkl")


def window_bounds(el_start: int, el_end: int, context_length: int):
    """Window a centred ``context_length`` input spans, matching FastaInterval's
    symmetric extension of a short interval (extra//2 to the left)."""
    extra = context_length - (int(el_end) - int(el_start))
    win_start = int(el_start) - extra // 2
    return win_start, win_start + context_length


def build_long_table(meta: list, cell_types_b: np.ndarray,
                     ref_mat: np.ndarray, cri_mat: np.ndarray, log2fc_mat: np.ndarray):
    """Assemble one element's (gene × cell_type) rows as a ``LONG_DTYPE`` record array.

    ``meta`` is the per-gene metadata list (column order); the three matrices are
    ``[n_cell_types, n_genes]``. Rows come out gene-major — all cell types of gene 0,
    then gene 1, ... — which also keeps the duplicated string columns run-length
    friendly for the compressor.
    """
    n_ct = len(cell_types_b)
    tab = np.empty(len(meta) * n_ct, dtype=LONG_DTYPE)
    tab["gene"] = np.repeat(np.array([m["gene"].encode() for m in meta], dtype="S32"), n_ct)
    tab["cell_type"] = np.tile(cell_types_b, len(meta))
    tab["strand"] = np.repeat(np.array([m["strand"].encode() for m in meta], dtype="S1"), n_ct)
    tab["tss"] = np.repeat(np.array([m["tss"] for m in meta], dtype=np.int64), n_ct)
    tab["dist_to_element"] = np.repeat(np.array([m["dist"] for m in meta], dtype=np.int64), n_ct)
    tab["n_exon_bins"] = np.repeat(np.array([m["n_exon_bins"] for m in meta], dtype=np.int32), n_ct)
    tab["exon_bin_frac"] = np.repeat(np.array([m["exon_bin_frac"] for m in meta], dtype=np.float32), n_ct)
    tab["fully_inside"] = np.repeat(np.array([m["fully_inside"] for m in meta], dtype=bool), n_ct)
    # .T.ravel() → gene-major, matching the repeat/tile order above
    tab["pred_ref"] = ref_mat.T.ravel()
    tab["pred_crispri"] = cri_mat.T.ravel()
    tab["log2fc"] = log2fc_mat.T.ravel()
    return tab


def read_element(f: h5py.File, enh_id: str) -> pd.DataFrame:
    """One element's screen result as a tidy DataFrame: one row per (gene, cell_type).

    Works on both layouts — the long ``table`` written by `run`/`convert`, and the
    older wide ``cell_type × gene`` matrices (rebuilt on the fly). The element's own
    coordinates are injected as leading columns so the frame stands alone when
    concatenated across elements.
    """
    g = f[f"elements/{enh_id}"]
    if "table" in g:
        df = pd.DataFrame(g["table"][:])
        for c in LONG_STR_COLS:
            df[c] = df[c].str.decode("utf-8")
    else:  # legacy wide layout
        meta = [{"gene": gene, "strand": st, "tss": t, "dist": d,
                 "n_exon_bins": nb, "exon_bin_frac": fr, "fully_inside": fi}
                for gene, st, t, d, nb, fr, fi in zip(
                    g["genes"].asstr()[:], g["gene_strand"].asstr()[:], g["gene_tss"][:],
                    g["dist_to_element"][:], g["n_exon_bins"][:], g["exon_bin_frac"][:],
                    g["fully_inside"][:])]
        cts_b = np.array([c.encode() for c in f["cell_types"].asstr()[:]], dtype="S40")
        df = pd.DataFrame(build_long_table(meta, cts_b, g["pred_ref"][:],
                                           g["pred_crispri"][:], g["log2fc"][:]))
        for c in LONG_STR_COLS:
            df[c] = df[c].str.decode("utf-8")
    df.insert(0, "enh_id", enh_id)
    for i, k in enumerate(("chr", "start", "end", "class")):
        v = g.attrs[k]
        df.insert(1 + i, k, v.decode() if isinstance(v, bytes) else v)
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.option("--bed", default=BED_DEFAULT)
@click.option("--gtf", default=C.GTF_DEFAULT)
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--log_base", default="./logs")
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--chunk_size", type=int, default=2000)
def build(bed, gtf, exp_name, chk, log_base, out_root, chunk_size):
    """Assign genes to each element's window and split the elements into chunks."""
    import pyranges as pr

    logger = BaseLogger(name="CRISPRi-elem-build", level=logging.INFO)
    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    context_length = int(load_config(config_name=config_path, skip_validation=True).data.context_length)

    elems = pd.read_csv(bed, sep="\t", header=None, names=BED_COLUMNS,
                        dtype={"chr": str, "enh_id": str, "class": str})
    n_all = len(elems)
    elems = elems[elems["chr"].isin(C.STD_CHR)].reset_index(drop=True)
    logger.info(f"{len(elems)} elements on standard chromosomes (of {n_all})")

    # Full protein-coding gene set; cached so GPU tasks never re-parse the GTF.
    base, tasks_dir, _ = _out_dirs(out_root, exp_name, chk)
    os.makedirs(base, exist_ok=True)
    gene_db = C.parse_gtf(gtf, None)
    gene_db = {g: v for g, v in gene_db.items() if v["exons"]}
    with open(_gene_db_path(base), "wb") as fh:
        pickle.dump(gene_db, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"GTF: {len(gene_db)} protein-coding genes with exons → {_gene_db_path(base)}")

    win = np.array([window_bounds(s, e, context_length)
                    for s, e in zip(elems["start"], elems["end"])], dtype=np.int64)
    # Clip the query to >= 0: negative window coordinates are pad, never sequence.
    elems_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": elems["chr"], "Start": np.maximum(win[:, 0], 0), "End": win[:, 1],
        "enh_id": elems["enh_id"],
    }))
    genes_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": [v["chr"] for v in gene_db.values()],
        "Start": [v["start"] for v in gene_db.values()],
        "End": [v["end"] for v in gene_db.values()],
        "gene": list(gene_db.keys()),
    }))
    hits = elems_pr.join(genes_pr).df
    logger.info(f"{len(hits)} (element, gene) window overlaps")

    # Column order = gene start, so each element's matrix reads left→right along the locus.
    hits = hits.sort_values(["enh_id", "Start_b", "gene"])
    genes_by_elem = hits.groupby("enh_id", sort=False)["gene"].apply(
        lambda s: ",".join(dict.fromkeys(s)))

    elems["genes"] = elems["enh_id"].map(genes_by_elem)
    n_no_gene = int(elems["genes"].isna().sum())
    kept = elems.dropna(subset=["genes"])

    n_chunks = C.write_chunks(kept[TASK_COLUMNS].values.tolist(), TASK_COLUMNS, tasks_dir, chunk_size)
    logger.info(f"elements={len(kept)}  skipped: no_gene={n_no_gene}")
    logger.info(f"genes per element: mean={kept['genes'].str.count(',').add(1).mean():.2f} "
                f"max={kept['genes'].str.count(',').add(1).max()}")
    logger.info(f"wrote {n_chunks} chunks (size {chunk_size}) → {tasks_dir}")
    logger.info(f"submit: sbatch --array=1-{n_chunks}%40 "
                f"Analysis/18_cCRE_rank/slurm_03_crispri_additional.sh "
                f"--exp_name {exp_name} --chk {chk}")


@cli.command()
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--log_base", default="./logs")
@click.option("--chk_base", default="./Chk")
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--chunk_id", type=int, required=True)
@click.option("--device", default="cuda:0")
@click.option("--use_head", default="regression")
@click.option("--force_restart", is_flag=True)
def run(exp_name, chk, log_base, chk_base, out_root, chunk_id, device, use_head, force_restart):
    """Process one chunk of elements → per-element cell_type × gene matrices."""
    import torch
    from data.tokenizer import FastaInterval
    from model.model_building_block import TargetLengthCrop
    from model.model_utils import setup_model

    logger = BaseLogger(name=f"CRISPRi-elem-run[{chunk_id}]", level=logging.INFO)

    base, tasks_dir, results_dir = _out_dirs(out_root, exp_name, chk)
    os.makedirs(results_dir, exist_ok=True)
    chunk_path = os.path.join(tasks_dir, f"chunk_{chunk_id:04d}.tsv")
    out_path = os.path.join(results_dir, f"chunk_{chunk_id:04d}.h5")
    if os.path.exists(out_path) and not force_restart:
        logger.info(f"exists, skipping: {out_path}")
        return
    # "class" is a keyword → rename so itertuples exposes a usable attribute.
    tasks = pd.read_csv(chunk_path, sep="\t", dtype={"chr": str, "enh_id": str, "class": str}
                        ).rename(columns={"class": "cls"})
    logger.info(f"{len(tasks)} elements from {chunk_path}")

    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    myconfig = load_config(config_name=config_path, skip_validation=True)
    context_length = int(myconfig.data.context_length)
    window_size = int(myconfig.data.preprocess.window_size)
    n_output_bins = context_length // window_size

    label_meta = pd.read_csv(os.path.join(log_base, exp_name, "regression_label_meta.csv"))
    rc_org, rc_swap = C.build_rc_swap_index(label_meta)
    # Canonical cell-type order (from RNAplus dims) shared by both strand readouts,
    # so every element's matrix rows carry the same meaning.
    plus = label_meta[label_meta["modality"] == "RNAplus"].sort_values("dim")
    minus = label_meta[label_meta["modality"] == "RNAminus"].set_index("cell_type")["dim"]
    cell_types = plus["cell_type"].tolist()
    if sorted(cell_types) != sorted(minus.index.tolist()):
        raise ValueError("RNAplus/RNAminus cell types differ in regression_label_meta.csv")
    dims_plus = plus["dim"].astype(int).to_numpy()
    dims_minus = minus.loc[cell_types].astype(int).to_numpy()
    logger.info(f"{len(cell_types)} cell types; {n_output_bins} output bins of {window_size} bp")

    with open(_gene_db_path(base), "rb") as fh:
        gene_db = pickle.load(fh)

    checkpoint = torch.load(os.path.join(chk_base, exp_name, f"chk_epoch_{chk}.pt"), map_location="cpu")
    myconfig.model.use_compile = False
    model = setup_model(myconfig, logger)
    load_target = model._orig_mod if hasattr(model, "_orig_mod") else model
    load_target.load_state_dict(checkpoint["model_state_dict"])
    # Disable crop so exon bins index the full n_output_bins frame directly.
    if hasattr(load_target, "crop"):
        load_target.crop = TargetLengthCrop(-1)
    else:
        model.crop = TargetLengthCrop(-1)
    model.eval().to(device)

    tokenizer = FastaInterval(fasta_file=os.path.abspath(myconfig.data.refer_genom),
                              context_length=context_length)

    buffered, n_skip_elem, n_skip_gene = [], 0, 0
    t0 = time.time()
    for i, t in enumerate(tasks.itertuples(index=False)):
        el_start, el_end = int(t.start), int(t.end)
        tok = tokenizer(chr_name=t.chr, start=el_start, end=el_end,
                        return_augs=False, return_rela_idx=True)
        ref_onehot = tok["one_hot"]
        s_idx, e_idx = tok["rela_idx"]
        # Genomic coordinate of one-hot index 0 — the output-bin frame origin.
        # Derived from rela_idx (not real_region) so left padding is accounted for.
        origin = el_start - int(s_idx)
        win_start, win_end = origin, origin + context_length

        crispri_onehot = ref_onehot.clone()
        crispri_onehot[int(s_idx):int(e_idx)] = 0.25

        try:
            pred_ref = C.predict_fwd_rc(model, ref_onehot, use_head, device, rc_org, rc_swap)
            pred_cri = C.predict_fwd_rc(model, crispri_onehot, use_head, device, rc_org, rc_swap)
        except Exception as e:
            logger.warning(f"forward failed {t.enh_id}: {type(e).__name__}: {e}")
            continue
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        el_center = (el_start + el_end) // 2
        cols, ref_cols, cri_cols = [], [], []
        for gene in str(t.genes).split(","):
            ginfo = gene_db.get(gene)
            if ginfo is None:
                n_skip_gene += 1
                continue
            all_bins = C.exon_bins(ginfo["exons"], origin, window_size)
            in_bins = np.array(sorted(b for b in all_bins if 0 <= b < n_output_bins))
            if in_bins.size == 0:
                n_skip_gene += 1
                continue
            dims = dims_minus if ginfo["strand"] == "+" else dims_plus
            ref_cols.append(pred_ref[np.ix_(in_bins, dims)].mean(axis=0))
            cri_cols.append(pred_cri[np.ix_(in_bins, dims)].mean(axis=0))
            cols.append({
                "gene": gene, "strand": ginfo["strand"],
                "tss": C.gene_tss(ginfo),
                "dist": C.gene_tss(ginfo) - el_center,
                "n_exon_bins": int(in_bins.size),
                "exon_bin_frac": float(in_bins.size) / float(len(all_bins)),
                "fully_inside": bool(ginfo["start"] >= win_start and ginfo["end"] <= win_end),
            })

        if not cols:
            n_skip_elem += 1
            continue
        ref_mat = np.stack(ref_cols, axis=1).astype(np.float32)
        cri_mat = np.stack(cri_cols, axis=1).astype(np.float32)
        buffered.append({
            "enh_id": t.enh_id, "chr": t.chr, "start": el_start, "end": el_end,
            "cls": t.cls,
            "win_start": win_start, "win_end": win_end,
            "pred_ref": ref_mat, "pred_crispri": cri_mat,
            "log2fc": np.log2((cri_mat + EPS) / (ref_mat + EPS)).astype(np.float32),
            "meta": cols,
        })
        if (i + 1) % 100 == 0:
            logger.info(f"{i + 1}/{len(tasks)} elements  ({(time.time() - t0) / (i + 1):.2f} s/element)")

    # Single write at the end → a killed job never leaves a partial result file.
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("cell_types", data=np.array(cell_types, dtype=object), dtype=STR_DT)
        f.attrs["context_length"] = context_length
        f.attrs["window_size"] = window_size
        f.attrs["exp_name"] = exp_name
        f.attrs["chk"] = str(chk)
        f.attrs["layout"] = "long"
        cts_b = np.array([c.encode() for c in cell_types], dtype="S40")
        eg = f.create_group("elements")
        for rec in buffered:
            g = eg.create_group(rec["enh_id"])
            tab = build_long_table(rec["meta"], cts_b,
                                   rec["pred_ref"], rec["pred_crispri"], rec["log2fc"])
            # gzip-1: the per-gene columns repeat once per cell type, so the table
            # compresses ~4x; level 1 gets essentially all of it at the lowest cost.
            g.create_dataset("table", data=tab, compression="gzip", compression_opts=1)
            g.attrs["chr"] = rec["chr"]
            g.attrs["start"] = rec["start"]
            g.attrs["end"] = rec["end"]
            g.attrs["class"] = rec["cls"]
            g.attrs["win_start"] = rec["win_start"]
            g.attrs["win_end"] = rec["win_end"]
    os.replace(tmp_path, out_path)

    n_cols = sum(r["log2fc"].shape[1] for r in buffered)
    logger.info(f"wrote {len(buffered)} elements ({n_cols} element-gene columns) → {out_path}  "
                f"[{(time.time() - t0) / max(1, len(tasks)):.2f} s/element; "
                f"skipped: elements={n_skip_elem} genes={n_skip_gene}]")


@cli.command()
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--out_name", default="elements.h5")
@click.option("--results_subdir", default="results",
              help="source dir under crispri_element/ (e.g. results_long)")
@click.option("--force_restart", is_flag=True)
def merge(exp_name, chk, out_root, out_name, results_subdir, force_restart):
    """Concatenate per-chunk HDF5 results into one file (optional)."""
    logger = BaseLogger(name="CRISPRi-elem-merge", level=logging.INFO)
    base, _, results_dir = _out_dirs(out_root, exp_name, chk)
    results_dir = os.path.join(base, results_subdir)
    out_path = os.path.join(base, out_name)
    if os.path.exists(out_path) and not force_restart:
        logger.info(f"exists, skipping: {out_path}")
        return

    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.h5")))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.h5 under {results_dir}")

    tmp_path = out_path + ".tmp"
    n_elem = 0
    with h5py.File(tmp_path, "w") as out:
        eg = out.create_group("elements")
        for ci, path in enumerate(chunks):
            with h5py.File(path, "r") as src:
                if ci == 0:
                    src.copy("cell_types", out)
                    for k, v in src.attrs.items():
                        out.attrs[k] = v
                for enh_id in src["elements"]:
                    src.copy(f"elements/{enh_id}", eg, name=enh_id)
                    n_elem += 1
            if (ci + 1) % 20 == 0:
                logger.info(f"{ci + 1}/{len(chunks)} chunks  ({n_elem} elements)")
    os.replace(tmp_path, out_path)
    logger.info(f"merged {len(chunks)} chunks, {n_elem} elements → {out_path}")


@cli.command()
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--src_subdir", default="results")
@click.option("--dst_subdir", default="results_long")
@click.option("--force_restart", is_flag=True)
def convert(exp_name, chk, out_root, src_subdir, dst_subdir, force_restart):
    """Rewrite wide ``cell_type × gene`` chunk results into the long table layout.

    CPU-only — no model, no GPU. Use it to migrate results produced before the
    layout change; `run` now writes the long table directly.
    """
    logger = BaseLogger(name="CRISPRi-elem-convert", level=logging.INFO)
    base, _, _ = _out_dirs(out_root, exp_name, chk)
    src_dir, dst_dir = os.path.join(base, src_subdir), os.path.join(base, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    chunks = sorted(glob.glob(os.path.join(src_dir, "chunk_*.h5")))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.h5 under {src_dir}")
    logger.info(f"{len(chunks)} chunks: {src_dir} → {dst_dir}")

    t0, n_elem, n_rows = time.time(), 0, 0
    for ci, src_path in enumerate(chunks):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        if os.path.exists(dst_path) and not force_restart:
            continue
        tmp_path = dst_path + ".tmp"
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            src.copy("cell_types", dst)
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["layout"] = "long"
            cts_b = np.array([c.encode() for c in src["cell_types"].asstr()[:]], dtype="S40")
            eg = dst.create_group("elements")
            for enh_id in src["elements"]:
                g = src["elements"][enh_id]
                meta = [{"gene": gene, "strand": st, "tss": t, "dist": d,
                         "n_exon_bins": nb, "exon_bin_frac": fr, "fully_inside": fi}
                        for gene, st, t, d, nb, fr, fi in zip(
                            g["genes"].asstr()[:], g["gene_strand"].asstr()[:], g["gene_tss"][:],
                            g["dist_to_element"][:], g["n_exon_bins"][:], g["exon_bin_frac"][:],
                            g["fully_inside"][:])]
                tab = build_long_table(meta, cts_b, g["pred_ref"][:],
                                       g["pred_crispri"][:], g["log2fc"][:])
                og = eg.create_group(enh_id)
                og.create_dataset("table", data=tab, compression="gzip", compression_opts=1)
                for k, v in g.attrs.items():
                    og.attrs[k] = v
                n_elem += 1
                n_rows += len(tab)
        os.replace(tmp_path, dst_path)
        if (ci + 1) % 10 == 0:
            logger.info(f"{ci + 1}/{len(chunks)} chunks  ({n_elem} elements, {n_rows:,} rows, "
                        f"{time.time() - t0:.0f}s)")
    logger.info(f"converted {n_elem} elements / {n_rows:,} rows → {dst_dir}  "
                f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    cli()
