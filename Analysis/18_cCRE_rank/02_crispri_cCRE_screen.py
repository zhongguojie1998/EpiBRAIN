"""In-silico CRISPRi screen of cCRE → target-gene effect across all cell types.

One task = one (cCRE, target_gene) pair. A ``context_length`` window is centred
on the midpoint between the cCRE centre and the target gene body midpoint; the
cCRE bases are replaced with pad tokens (0.25) and reference vs. silenced RNA
predictions are compared over the gene's exon bins. A single silencing run is
scored against ALL model cell types (strand-aware RNA track), so one run yields
one (cCRE, gene, cell_type) row per cell type.

Predictions use forward + reverse-complement averaging (RNAplus/RNAminus tracks
swapped on the reverse strand), matching Analysis/03_0_crispri.py.

Results are written per chunk and NOT merged into the ranked table.

Workflow:
  python Analysis/18_cCRE_rank/02_crispri_cCRE_screen.py build \\
      --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \\
      --chk 17 --chunk_size 25
  python Analysis/18_cCRE_rank/02_crispri_cCRE_screen.py run \\
      --exp_name ... --chk 17 --chunk_id 1 --device cuda:0
"""
import logging
import os
import sys
import warnings
from pathlib import Path

import click
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

TASK_COLUMNS = ["cCRE_id", "chr", "start", "end", "gene"]
EPS = 1e-6


def _out_dirs(out_root, exp_name, chk):
    base = os.path.join(out_root, f"{exp_name}_{chk}", "crispri")
    return os.path.join(base, "tasks"), os.path.join(base, "results")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--ranked", default=C.RANKED_DEFAULT)
@click.option("--gtf", default=C.GTF_DEFAULT)
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--log_base", default="./logs")
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--chunk_size", type=int, default=25)
def build(ranked, gtf, exp_name, chk, log_base, out_root, chunk_size):
    """Expand ranked table → (cCRE, gene) pairs and split into chunks."""
    logger = BaseLogger(name="CRISPRi-build", level=logging.INFO)
    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    context_length = int(load_config(config_name=config_path, skip_validation=True).data.context_length)

    df = C.load_ranked(ranked)
    gene_db = C.parse_gtf(gtf, C.build_gene_set(df))
    logger.info(f"GTF: {len(gene_db)} target genes resolved on standard chromosomes")

    rows, n_no_gene, n_far = [], 0, 0
    for _, r in df.iterrows():
        for gene in C._split_field(r["abc_target_genes"]):
            ginfo = gene_db.get(gene)
            if ginfo is None:
                n_no_gene += 1
                continue
            if r["chr"] != ginfo["chr"]:
                continue
            if not C.fits_window(r["start"], r["end"], ginfo, context_length):
                n_far += 1
                continue
            rows.append([r["cCRE_id"], r["chr"], int(r["start"]), int(r["end"]), gene])

    tasks_dir, _ = _out_dirs(out_root, exp_name, chk)
    n_chunks = C.write_chunks(rows, TASK_COLUMNS, tasks_dir, chunk_size)
    logger.info(f"pairs={len(rows)}  skipped: no_gene={n_no_gene} far={n_far}")
    logger.info(f"wrote {n_chunks} chunks (size {chunk_size}) → {tasks_dir}")
    logger.info(f"submit: sbatch --array=1-{n_chunks} Analysis/18_cCRE_rank/slurm_02_crispri.sh "
                f"--exp_name {exp_name} --chk {chk}")


@cli.command()
@click.option("--gtf", default=C.GTF_DEFAULT)
@click.option("--exp_name", required=True)
@click.option("--chk", required=True)
@click.option("--log_base", default="./logs")
@click.option("--chk_base", default="./Chk")
@click.option("--out_root", default=C.OUT_ROOT_DEFAULT)
@click.option("--chunk_id", type=int, required=True)
@click.option("--device", default="cuda:0")
@click.option("--use_head", default="regression")
@click.option("--force_restart", is_flag=True)
def run(gtf, exp_name, chk, log_base, chk_base, out_root, chunk_id, device, use_head, force_restart):
    """Process one chunk of (cCRE, gene) pairs → effect scores across all cell types."""
    import torch
    from data.tokenizer import FastaInterval
    from model.model_building_block import TargetLengthCrop
    from model.model_utils import setup_model

    logger = BaseLogger(name=f"CRISPRi-run[{chunk_id}]", level=logging.INFO)

    tasks_dir, results_dir = _out_dirs(out_root, exp_name, chk)
    os.makedirs(results_dir, exist_ok=True)
    chunk_path = os.path.join(tasks_dir, f"chunk_{chunk_id:04d}.tsv")
    out_path = os.path.join(results_dir, f"chunk_{chunk_id:04d}.csv")
    if os.path.exists(out_path) and not force_restart:
        logger.info(f"exists, skipping: {out_path}")
        return
    tasks = pd.read_csv(chunk_path, sep="\t")
    logger.info(f"{len(tasks)} (cCRE,gene) pairs from {chunk_path}")

    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    myconfig = load_config(config_name=config_path, skip_validation=True)
    context_length = int(myconfig.data.context_length)
    window_size = int(myconfig.data.preprocess.window_size)
    n_output_bins = context_length // window_size

    label_meta = pd.read_csv(os.path.join(log_base, exp_name, "regression_label_meta.csv"))
    rc_org, rc_swap = C.build_rc_swap_index(label_meta)
    # Pre-group RNA readout tracks by modality: modality → [(cell_type, dim), ...]
    rna_tracks = {
        mod: [(row["cell_type"], int(row["dim"]))
              for _, row in label_meta[label_meta["modality"] == mod].iterrows()]
        for mod in ("RNAplus", "RNAminus")
    }

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
    gene_db = C.parse_gtf(gtf, set(tasks["gene"].unique()))

    out_rows = []
    for _, t in tasks.iterrows():
        if t["chr"] not in C.STD_CHR:
            continue
        gene = t["gene"]
        ginfo = gene_db.get(gene)
        if ginfo is None:
            logger.warning(f"gene {gene} not in GTF; skip")
            continue
        strand = ginfo["strand"]
        modality = "RNAminus" if strand == "+" else "RNAplus"

        center = C.window_center(t["start"], t["end"], C.gene_midpoint(ginfo))
        tok = tokenizer(chr_name=t["chr"], start=center, end=center,
                        return_augs=False, return_rela_idx=True)
        ref_onehot = tok["one_hot"]
        real_start, real_end = tok["real_region"]

        bin_range = C.exon_bin_range(ginfo["exons"], int(real_start), window_size, n_output_bins)
        if bin_range is None or bin_range.min() < 0 or bin_range.max() >= n_output_bins:
            logger.warning(f"no valid exon bins for {gene}; skip")
            continue

        c_s = max(0, int(t["start"]) - int(real_start))
        c_e = min(int(t["end"]) - int(real_start), len(ref_onehot))
        if c_e <= c_s:
            logger.warning(f"cCRE out of window for {t['cCRE_id']}/{gene}; skip")
            continue

        crispri_onehot = ref_onehot.clone()
        crispri_onehot[c_s:c_e] = 0.25

        try:
            pred_ref = C.predict_fwd_rc(model, ref_onehot, use_head, device, rc_org, rc_swap)
            pred_crispri = C.predict_fwd_rc(model, crispri_onehot, use_head, device, rc_org, rc_swap)
        except Exception as e:
            logger.warning(f"forward failed {t['cCRE_id']}/{gene}: {type(e).__name__}: {e}")
            continue
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # readout: mean predicted RNA over gene exon bins, per cell type (this strand)
        ref_bins = pred_ref[bin_range, :]
        cri_bins = pred_crispri[bin_range, :]
        for cell_type, dim in rna_tracks[modality]:
            a_ref = float(ref_bins[:, dim].mean())
            a_cri = float(cri_bins[:, dim].mean())
            out_rows.append({
                "cCRE_id": t["cCRE_id"], "chr": t["chr"], "start": int(t["start"]), "end": int(t["end"]),
                "gene": gene, "cell_type": cell_type, "trial": f"{cell_type}_{modality}", "strand": strand,
                "win_start": int(real_start), "win_end": int(real_end),
                "pred_ref": a_ref, "pred_crispri": a_cri,
                "delta": a_cri - a_ref,
                "log2fc": float(np.log2((a_cri + EPS) / (a_ref + EPS))),
            })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    logger.info(f"scored {len(out_rows)} rows from {len(tasks)} pairs → {out_path}")


if __name__ == "__main__":
    cli()
