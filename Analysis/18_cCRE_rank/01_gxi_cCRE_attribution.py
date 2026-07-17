"""Gradient×input screen of cCRE → target-gene attribution.

One task = one (cCRE, target_gene, cell_type) triple → one attribution score.
For each triple a ``context_length`` window is centred on the midpoint between
the cCRE centre and the target gene body midpoint. Gradient×input attribution is
computed for the gene's predicted RNA expression (aggregated over its exon bins,
strand-aware track) and the per-position genome attribution is sliced over the
cCRE window; its mean/max/min/sum are recorded as the score.

Cell types come from the ranked table's ``supporting_celltypes`` (mapped to the
MiniAtlas track prefix). Results are written per chunk and NOT merged into the
ranked table.

Workflow:
  # 1. expand + split into chunks (prints n_chunks for the SLURM array)
  python Analysis/18_cCRE_rank/01_gxi_cCRE_attribution.py build \\
      --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \\
      --chk 17 --chunk_size 50

  # 2. run one chunk (invoked by slurm_01_gxi.sh array tasks)
  python Analysis/18_cCRE_rank/01_gxi_cCRE_attribution.py run \\
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

TASK_COLUMNS = ["cCRE_id", "chr", "start", "end", "gene", "cell_type"]


def _out_dirs(out_root, exp_name, chk):
    base = os.path.join(out_root, f"{exp_name}_{chk}", "gxi")
    return os.path.join(base, "tasks"), os.path.join(base, "results")


# --------------------------------------------------------------------------- #
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
@click.option("--chunk_size", type=int, default=50)
def build(ranked, gtf, exp_name, chk, log_base, out_root, chunk_size):
    """Expand ranked table → (cCRE, gene, celltype) triples and split into chunks."""
    logger = BaseLogger(name="GxI-build", level=logging.INFO)
    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    context_length = int(load_config(config_name=config_path, skip_validation=True).data.context_length)

    label_meta = pd.read_csv(os.path.join(log_base, exp_name, "regression_label_meta.csv"))
    valid_ct = set(label_meta["cell_type"].unique())

    df = C.load_ranked(ranked)
    gene_db = C.parse_gtf(gtf, C.build_gene_set(df))
    logger.info(f"GTF: {len(gene_db)} target genes resolved on standard chromosomes")

    rows, n_no_gene, n_no_ct, n_far = [], 0, 0, 0
    for _, r in df.iterrows():
        genes = C._split_field(r["abc_target_genes"])
        cts = C._split_field(r.get("supporting_celltypes"))
        for gene in genes:
            ginfo = gene_db.get(gene)
            if ginfo is None:
                n_no_gene += 1
                continue
            if r["chr"] != ginfo["chr"]:
                continue
            if not C.fits_window(r["start"], r["end"], ginfo, context_length):
                n_far += 1
                continue
            for ct in cts:
                mapped = f"{C.MINIATLAS_PREFIX}{ct}"
                if mapped not in valid_ct:
                    n_no_ct += 1
                    continue
                rows.append([r["cCRE_id"], r["chr"], int(r["start"]), int(r["end"]), gene, mapped])

    tasks_dir, _ = _out_dirs(out_root, exp_name, chk)
    n_chunks = C.write_chunks(rows, TASK_COLUMNS, tasks_dir, chunk_size)
    logger.info(f"triples={len(rows)}  skipped: no_gene={n_no_gene} far={n_far} no_celltype={n_no_ct}")
    logger.info(f"wrote {n_chunks} chunks (size {chunk_size}) → {tasks_dir}")
    logger.info(f"submit: sbatch --array=1-{n_chunks} Analysis/18_cCRE_rank/slurm_01_gxi.sh "
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
@click.option("--force_restart", is_flag=True)
def run(gtf, exp_name, chk, log_base, chk_base, out_root, chunk_id, device, force_restart):
    """Process one chunk of triples → one attribution score each."""
    import importlib.util as ilu

    import torch
    from data.tokenizer import FastaInterval
    from model.model_building_block import TargetLengthCrop
    from model.model_utils import setup_model

    spec = ilu.spec_from_file_location(
        "gi_interp", str(ROOT / "Analysis" / "02_motif_interpretation_gradient_input.py"))
    _gi_mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(_gi_mod)
    gradients_input_attribution = _gi_mod.gradients_input_attribution

    logger = BaseLogger(name=f"GxI-run[{chunk_id}]", level=logging.INFO)

    tasks_dir, results_dir = _out_dirs(out_root, exp_name, chk)
    os.makedirs(results_dir, exist_ok=True)
    chunk_path = os.path.join(tasks_dir, f"chunk_{chunk_id:04d}.tsv")
    out_path = os.path.join(results_dir, f"chunk_{chunk_id:04d}.csv")
    if os.path.exists(out_path) and not force_restart:
        logger.info(f"exists, skipping: {out_path}")
        return
    tasks = pd.read_csv(chunk_path, sep="\t")
    logger.info(f"{len(tasks)} triples from {chunk_path}")

    config_path = os.path.join(log_base, exp_name, "overall_setting.yaml")
    myconfig = load_config(config_name=config_path, skip_validation=True)
    context_length = int(myconfig.data.context_length)
    window_size = int(myconfig.data.preprocess.window_size)
    n_output_bins = context_length // window_size

    label_meta = pd.read_csv(os.path.join(log_base, exp_name, "regression_label_meta.csv"))

    checkpoint = torch.load(os.path.join(chk_base, exp_name, f"chk_epoch_{chk}.pt"), map_location="cpu")
    model = setup_model(myconfig, logger)
    load_target = model._orig_mod if hasattr(model, "_orig_mod") else model
    load_target.load_state_dict(checkpoint["model_state_dict"])
    # Disable crop so outputs span the full n_output_bins frame (as in 00_interpret_gene_RNA).
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
        gene = t["gene"]
        ginfo = gene_db.get(gene)
        if ginfo is None:
            logger.warning(f"gene {gene} not in GTF; skip")
            continue
        strand = ginfo["strand"]
        trial = C.rna_trial(t["cell_type"], strand)
        match = label_meta[label_meta["trial"] == trial]
        if match.empty:
            logger.warning(f"trial {trial} not in label_meta; skip")
            continue
        trial_dim = int(match["dim"].values[0])
        label_meta_row = match.iloc[0]

        center = C.window_center(t["start"], t["end"], C.gene_midpoint(ginfo))
        tok = tokenizer(chr_name=t["chr"], start=center, end=center,
                        return_augs=False, return_rela_idx=True)
        seq_onehot = tok["one_hot"]
        real_start, real_end = tok["real_region"]

        bin_range = C.exon_bin_range(ginfo["exons"], int(real_start), window_size, n_output_bins)
        if bin_range is None or bin_range.min() < 0 or bin_range.max() >= n_output_bins:
            logger.warning(f"no valid exon bins for {gene}; skip")
            continue

        inp = seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device).requires_grad_(True)
        try:
            gi = gradients_input_attribution(
                model=model, seq_input=inp, output_key="regression",
                target_dim=trial_dim, bin_range=bin_range, label_meta_row=label_meta_row,
                pseudo_count=0.0, no_untransform=False, use_mean=True,
                subtract_avg=True, input_gate=True,
            )
            gi_1d = gi.detach().cpu().squeeze(0).sum(dim=-1).numpy().astype(np.float32)  # [L]
        except Exception as e:
            logger.warning(f"grad×input failed {gene}@{trial}: {type(e).__name__}: {e}")
            continue
        finally:
            del inp
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if not np.isfinite(gi_1d).all():
            logger.warning(f"NaN attribution {gene}@{trial}; skip")
            continue

        arr_s = max(0, int(t["start"]) - int(real_start))
        arr_e = min(int(t["end"]) - int(real_start), len(gi_1d))
        if arr_e <= arr_s:
            logger.warning(f"cCRE out of window for {t['cCRE_id']}/{gene}; skip")
            continue
        w = gi_1d[arr_s:arr_e]
        out_rows.append({
            "cCRE_id": t["cCRE_id"], "chr": t["chr"], "start": int(t["start"]), "end": int(t["end"]),
            "gene": gene, "cell_type": t["cell_type"], "trial": trial, "strand": strand,
            "win_start": int(real_start), "win_end": int(real_end),
            "attr_mean": float(w.mean()), "attr_max": float(w.max()),
            "attr_min": float(w.min()), "attr_sum": float(w.sum()),
        })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    logger.info(f"scored {len(out_rows)}/{len(tasks)} → {out_path}")


if __name__ == "__main__":
    cli()
