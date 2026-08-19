# 18_cCRE_rank — cCRE → target-gene screening

Screens that score cCRE→gene effects with the trained model. Results are written
per chunk under `output/<exp>_<chk>/` and are **not** merged into the ranked table.

Screens 1–2 are **link-driven**: they score the cCRE→gene links in
`enhancer_candidates.ranked.tsv`, centring a `context_length` (524,288 bp) window
on the **midpoint between the cCRE centre and the target gene body midpoint** so
the cCRE and the gene co-occur in a single input. Target genes are expanded from
`abc_target_genes`, and pairs/triples whose cCRE and gene cannot both fit in one
window are dropped at `build` time (reported as `far=`).

Screen 3 is **hypothesis-free**: it centres the window on the element itself and
reads out every gene in the resulting locus.

Shared helpers live in `_cCRE_rank_common.py` (GTF parsing, exon→bin mapping,
window geometry, RC-averaged prediction, chunk writing).

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

## 3. Element-centred in-silico CRISPRi (`04_crispri_element_screen.py`)

- Input: `epibrain_elements.500bp.bed` (581,959 × 500 bp elements; col5 =
  `both`/`adult_only`/`iN_only`). No ABC links involved.
- Unit of work: one **element** → one silencing run. The element sits at the exact
  centre of the 524 kb window, and the ref/CRISPRi prediction pair is read out for
  **every protein-coding gene with ≥1 exon bin in the window** × all 70 cell types.
  Four forward passes therefore cover the whole locus (5.84 genes on average).
- Genes at the window edge are kept, not dropped; `exon_bin_frac`, `fully_inside`
  and `dist_to_element` are stored so they can be filtered downstream.
- `build` caches the full protein-coding gene set to `gene_db.pkl` so the array
  tasks never re-parse the GTF, and drops elements with no gene in the window.

```bash
$PY Analysis/18_cCRE_rank/04_crispri_element_screen.py build \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas \
    --chk 17 --chunk_size 2000
# any GPU in the gpu partition (l40s or b6k); %40 throttles concurrency
sbatch --array=1-N%40 Analysis/18_cCRE_rank/slurm_03_crispri_additional.sh \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas --chk 17
# optional: fold the per-chunk HDF5 files into one elements.h5
$PY Analysis/18_cCRE_rank/04_crispri_element_screen.py merge \
    --exp_name full_finetune_original_loss_celltype_head_dim8_linear_full_atlas --chk 17
```

Measured on the full BED (job 20871462, 135 × 4000-element tasks, all COMPLETED):
538,173 of 581,959 elements screened (43,404 dropped as gene deserts at `build`, 382
more had no exon bin in window), 3,143,616 element-gene pairs ⇒ 220 M table rows.
0.89 s/element ⇒ ~136 GPU-hours; 38 min – 1 h 53 min per task; ~6 GB of HDF5.

HDF5 layout — one **tidy table per element**, one row per (gene, cell_type):

```
/cell_types                [70] utf-8
/elements/<enh_id>/table   compound [n_genes * 70]
/elements/<enh_id>.attrs   chr, start, end, class, win_start, win_end
/.attrs                    layout="long", context_length, window_size, exp_name, chk
```

`table` columns (`LONG_DTYPE`, 110 bytes/row, gzip-1):

| column | dtype | meaning |
|---|---|---|
| `gene`, `cell_type`, `strand` | S32 / S40 / S1 | row identity |
| `tss` | i8 | 0-based TSS |
| `dist_to_element` | i8 | signed, `tss - element centre` |
| `n_exon_bins` | i4 | exon bins of the gene inside the window |
| `exon_bin_frac` | f4 | in-window / total exon bins; `< 1` ⇒ transcript truncated by the window edge |
| `fully_inside` | bool | whole gene body inside the window |
| `pred_ref`, `pred_crispri` | f4 | mean predicted RNA over the in-window exon bins |
| `log2fc` | f4 | `log2((pred_crispri + 1e-6) / (pred_ref + 1e-6))` |

Rows are gene-major — all 70 cell types of gene 1, then gene 2, … — so the per-gene
metadata repeats down each 70-row block. That duplication is what makes the table
directly viewable; gzip-1 compresses it back out (~4x).

```python
# read one element back as a tidy DataFrame
import h5py, importlib.util as ilu
spec = ilu.spec_from_file_location("es", "Analysis/18_cCRE_rank/04_crispri_element_screen.py")
es = ilu.module_from_spec(spec); spec.loader.exec_module(es)
with h5py.File(".../crispri_element/elements.h5") as f:
    df = es.read_element(f, "enh_1248")   # (n_genes*70) x 16, enh_id/chr/start/end/class prepended
    clean = df[df.exon_bin_frac == 1.0]   # drop window-truncated transcripts
```

`read_element` also reads the older wide `cell_type × gene` layout, rebuilding the
long form on the fly, so pre-conversion files stay usable.

## Layout

```
output/<exp>_<chk>/gxi/tasks/chunk_NNNN.tsv               # (cCRE,gene,celltype) triples
output/<exp>_<chk>/gxi/results/chunk_NNNN.csv             # attr_mean/max/min/sum
output/<exp>_<chk>/crispri/tasks/chunk_NNNN.tsv           # (cCRE,gene) pairs
output/<exp>_<chk>/crispri/results/chunk_NNNN.csv         # pred_ref/pred_crispri/delta/log2fc
output/<exp>_<chk>/crispri_element/gene_db.pkl            # cached protein-coding gene/exon db
output/<exp>_<chk>/crispri_element/tasks/chunk_NNNN.tsv   # elements + their in-window genes
output/<exp>_<chk>/crispri_element/results/chunk_NNNN.h5   # per-chunk results
output/<exp>_<chk>/crispri_element/elements.h5             # optional merged single file
```

`convert` rewrites wide `cell_type × gene` chunk results into the long table layout
(CPU only, no GPU) — used to migrate the completed screen after the layout change:

```bash
$PY Analysis/18_cCRE_rank/04_crispri_element_screen.py convert --exp_name <EXP> --chk 17
$PY Analysis/18_cCRE_rank/04_crispri_element_screen.py merge --exp_name <EXP> --chk 17 \
    --results_subdir results_long --force_restart
```

Concatenate `results/chunk_*.csv` (screens 1–2) for the final screen table. All
`run` steps skip chunks whose result file already exists (use `--force_restart` to
redo); the element screen writes its HDF5 atomically via a `.tmp` rename, so a
killed task never leaves a partial file behind.

## Track indexing gotcha

`regression_label_meta.csv` has two index columns: **`dim` is the prediction-head
dim** (the head emits all 6 modalities × 70 cell types = 420 tracks) and
`label_dim` indexes the 386-column label matrix. Slice model predictions with
`dim`.
