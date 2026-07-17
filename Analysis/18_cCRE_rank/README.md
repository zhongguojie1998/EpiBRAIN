# 18_cCRE_rank — cCRE → target-gene screening

Two independent screens that score the cCRE→gene links in
`enhancer_candidates.ranked.tsv` with the trained model. Both centre a
`context_length` (524,288 bp) window on the **midpoint between the cCRE centre
and the target gene body midpoint**, so the cCRE and the gene co-occur in a
single input. Target genes are expanded from `abc_target_genes`. Results are
written per chunk under `output/<exp>_<chk>/` and are **not** merged into the
ranked table.

Pairs/triples whose cCRE and gene cannot both fit in one window are dropped at
`build` time (reported as `far=`).

## 1. Gradient×input attribution (`01_gxi_cCRE_attribution.py`)

- Unit of work: one **(cCRE, target_gene, cell_type)** triple → one score.
- Cell types: expanded from the ranked table's `supporting_celltypes`
  (mapped to `MiniAtlas-*` tracks).
- Score: gradient×input attribution of the gene's predicted RNA (aggregated over
  its exon bins, strand-aware track), summed over nucleotides and sliced over the
  cCRE window → `attr_mean/max/min/sum`.

```bash
PY=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python3
# build (prints n_chunks + the sbatch line)
$PY Analysis/18_cCRE_rank/01_gxi_cCRE_attribution.py build \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \
    --chk 17 --chunk_size 50
# submit array (N from build)
sbatch --array=1-N Analysis/18_cCRE_rank/slurm_01_gxi.sh \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas --chk 17
```

## 2. In-silico CRISPRi (`02_crispri_cCRE_screen.py`)

- Unit of work: one **(cCRE, target_gene)** pair → one silencing run.
- The cCRE bases are replaced with pad tokens (0.25); reference vs. silenced RNA
  predictions (forward + reverse-complement averaged) are compared over the
  gene's exon bins for **all 70 cell types** (strand-aware track) → one
  `(cCRE, gene, cell_type)` row each with `pred_ref`, `pred_crispri`, `delta`,
  `log2fc`.

```bash
$PY Analysis/18_cCRE_rank/02_crispri_cCRE_screen.py build \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \
    --chk 17 --chunk_size 25
sbatch --array=1-N Analysis/18_cCRE_rank/slurm_02_crispri.sh \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas --chk 17
```

## Layout

```
output/<exp>_<chk>/gxi/tasks/chunk_NNNN.tsv        # (cCRE,gene,celltype) triples
output/<exp>_<chk>/gxi/results/chunk_NNNN.csv      # attr_mean/max/min/sum
output/<exp>_<chk>/crispri/tasks/chunk_NNNN.tsv    # (cCRE,gene) pairs
output/<exp>_<chk>/crispri/results/chunk_NNNN.csv  # pred_ref/pred_crispri/delta/log2fc
```

Concatenate `results/chunk_*.csv` for the final screen table. Both `run` steps
skip chunks whose result CSV already exists (use `--force_restart` to redo).
```
