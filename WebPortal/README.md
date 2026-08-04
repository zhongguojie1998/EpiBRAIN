# WebPortal pipelines

SLURM-ready wrappers that turn the analysis pipelines under `Analysis/` into
self-contained jobs producing web-portal deliverables. Submit from the project
root (so `SLURM_SUBMIT_DIR` resolves correctly).

| Script | Wraps | Input | Deliverables |
|---|---|---|---|
| `run_variant_analysis.slurm` | `Analysis/variant_analysis_pipeline.sh` | one variant (+ optional track / gene / region) | predicted, diff, and (optional) attribution bigWig tracks |
| `run_variant_effect_screen.slurm` | `Analysis/03_variant_effect_screen` | a VCF | variant-by-track fold-change table (CSV) |

Helper scripts (called by the wrappers, not submitted directly):
`attribution_to_bigwig.py`, `screen_h5_to_table.py`.

---

## 1. `run_variant_analysis.slurm` — single-variant tracks

Runs inference for one variant and, when a track is given, its attribution.
Resources: 1 GPU / 64 GB / 24 h (`--partition=gpu`).

### Input

| Argument | Required | Meaning |
|---|---|---|
| `--variant chr:pos:ref:alt` | yes | the variant (e.g. `chr12:40208963:C:T`) |
| `--track NAME` | no | attribution track (a `trial` in the model's `label_meta`); **no attribution if omitted** |
| `--gene SYMBOL` | no | aggregate attribution over this gene; **requires `--track`**; overrides `--region` |
| `--region chr:start-end` | no | aggregate attribution over this region; **requires `--track`** |
| `--sat` | no | also run saturation mutagenesis (Step 2) |
| `--no-labels` | no | skip `inference/label/` (observed coverage). These are identical for every variant in a region — pass this and serve them statically to save ~13s per run |
| `--exp-name`, `--checkpoint` | no | model (default: `full_finetune_original_loss_celltype_head_dim8_linear_full_atlas`, ckpt 17) |
| `--output-base`, `--num-gpus`, `--disease`, `--baseline`, `--fai` | no | passthrough / overrides |

`--gene` and `--region` only influence attribution scope, so they require a
`--track`. If both are given, `--gene` wins.

### Deliverables

Written to `<output-base>/<variant>_<gene|region>/` (default `output-base` is
`Analysis/figures`):

- `inference/pred/<track>.bw` — reference (predicted) track
- `inference/alt/<track>.bw` — alternative-allele track
- `inference/diff/<track>.bw` — alt − ref difference track
- `inference/label/<track>.bw` — observed coverage; variant-independent, suppressed by `--no-labels`
- `attribution_score.bw` — *(with `--track`)* per-base attribution aggregated to `window_size` (32 bp) bins
- `motif_interpretation_zoom.pdf` — *(with `--track`)* motif zoom plot, variant ± 50 bp
- `sat_mutagenesis/` — *(with `--sat`)*

Visualization PDF (Step 3), mutagenesis-viz PDF (Step 4), full motif plot
(Step 6 full) and TOMTOM (Step 7) are always skipped — deliverables are tracks.

### Example

```bash
# predicted + diff tracks only
sbatch WebPortal/run_variant_analysis.slurm --variant chr12:40208963:C:T

# + attribution bigWig + zoom plot, aggregated over LRRK2
sbatch WebPortal/run_variant_analysis.slurm \
    --variant chr12:40208963:C:T --track BasalGanglia-Microglia_RNAminus --gene LRRK2 \
    --exp-name full_finetune_original_loss_celltype_head_dim8_linear --checkpoint 20
```

---

## 2. `run_variant_effect_screen.slurm` — VCF fold-change table

Orchestrates the distributed variant-effect screen on a VCF, then extracts a
fold-change table. This is a **driver** job: it submits one GPU job per chunk,
polls until they finish, merges to HDF5, and extracts the table — so the driver
itself needs no GPU (4 CPU / 32 GB / 24 h). No bigWig is produced.

### Input

| Argument | Required | Meaning |
|---|---|---|
| `--vcf FILE[.gz]` | yes | input VCF (standard 8-column; `INFO` may carry `gene_ID=`/`gene_name=`) |
| `--model-preset full_atlas\|dim8_ckpt20` | no | model + matching `label_meta` (default `full_atlas`) |
| `--model`, `--config`, `--label_meta` | no | explicit overrides (a `.pt` model needs `--config`) |
| `--gene-lfc auto\|on\|off` | no | fold-change choice (default `auto`, detected from the VCF) |
| `--gtf GTF` | no | exon annotation for `gene_lfc` (default `Data/source/gencode.v48.annotation.gtf.gz`) |
| `--output H5`, `--table CSV`, `--experiment`, `--chunks`, `--force` | no | paths / scale / passthrough |

**Fold-change selection** (the deliverable values):

- VCF `INFO` carries `gene_ID`/`gene_name` → `gene_lfc` — `log(mean ALT exon bins) − log(mean REF exon bins)`, aggregated over that gene's exons (needs GTF).
- plain VCF → `local_raw_log_diff` — `log2(1+ALT) − log2(1+REF)` summed over ±15 bins (~992 bp) around the variant.

The screen has no arbitrary custom-region score; `--gene-lfc off` forces the
local LFC.

### Deliverables

Written to `<output dir>/` (default `Analysis/figures/variant_effect_screen/<vcf-stem>/`):

- `<vcf-stem>.h5` — merged HDF5 (all score matrices, `model_meta/trial_names`, `experiments/<exp>`)
- `<vcf-stem>_<score>_table.csv` — **the table**: one row per variant
  (`index_key, rsid, chr, pos, ref, alt` + one column per track), values = chosen fold change

### Example

```bash
# plain VCF -> local LFC table
sbatch WebPortal/run_variant_effect_screen.slurm --vcf variants.vcf.gz

# gene-annotated VCF -> gene_lfc table (auto-detected)
sbatch WebPortal/run_variant_effect_screen.slurm --vcf annotated.vcf.gz --model-preset full_atlas
```

---

## Helper scripts

- **`attribution_to_bigwig.py`** — converts Step-5 attribution
  (`{name_base}_metadata.npy` + `{name_base}_{baseline}_importance.npy`) to a
  bigWig. Reconstructs genomic coordinates from metadata
  (`region_start_bp = real_start + trim·window_size`) and aggregates bp-level
  scores to `window_size` bins. Chromosome sizes from a FASTA `.fai`.

- **`screen_h5_to_table.py`** — reads the merged screen HDF5 and writes the
  variant-by-track CSV for one experiment. `--score auto` prefers `gene_lfc`,
  falling back to `local_raw_log_diff`. Applies the `reverse_map` sign flip when
  present (GWAS inputs).

## Tests

`test_run_variant_analysis.sh` and `test_run_variant_effect_screen.sh`
syntax-check the wrappers and exercise the new helper code against real
reference data (LRRK2 attribution npy; an existing eQTL screen HDF5). The
second test also builds a small gene-annotated VCF fixture under `test_data/`
(git-ignored, regenerated on each run). The full GPU pipelines are not run by
the tests; each prints the `sbatch` command to launch them.

```bash
bash WebPortal/test_run_variant_analysis.sh
bash WebPortal/test_run_variant_effect_screen.sh
```
