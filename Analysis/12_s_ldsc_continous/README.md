# Analysis 12: S-LDSC continuous & variant-overlap pipelines

Two independent pipelines organized into subdirectories. All scripts expect the
working directory to be the **project root** (`/gpfs/commons/groups/ren_lab/guojiezhong/BICAN`)
unless noted.

```
heritability/      # Pipeline A — partitioned heritability (S-LDSC)
overlap/           # Pipeline B — variant overlap counts (no heritability)
annotations_by_trait/ results_by_trait/ quantile_results_by_trait/
ccre_ldsc/  results_overlap/  logs/  track_lists/
listHM3.txt  w_hm3.snplist  hapmap3_snps/
archive/           # Legacy scripts (run_pipeline.sh, 05_plot_quantile_enrichment.R)
```

---

## Pipeline A — Heritability

Entry point: `heritability/run_heritability_pipeline.sh` (was `run_pipeline_ultra_parallel.sh`).
It orchestrates the four annotation variants below by driving the `continuous/` subfolder.

### A1. Continuous annotation from model `.h5` predictions (default, 20-quantile)
`heritability/continuous/`
- `00_extract_trait_from_h5.py` — extract trait predictions + sumstats from `.h5`; liftover hg38→hg19.
- `01_create_annotation_files.py` — per-chrom per-track `.annot.gz` files.
- `02_compute_ld_scores.sh` (+ `02a_slurm_compute_ld_per_chrom.sh`) — LD scores per track×chrom.
- `03_run_sldsc_regression.sh` (+ `03a_slurm_regression_per_track.sh`) — S-LDSC regression.
- `04_quantile_enrichment.sh` — quantile-stratified h² (default 20 quantiles).
- `05_plot_quantile_enrichment.py` — enrichment plots. *Run from `Analysis/12_s_ldsc_continous/`.*

Outputs: `annotations_by_trait/annotations_<TRAIT>/`, `results_by_trait/results_<TRAIT>/`, `quantile_results_by_trait/quantile_results_<TRAIT>/`.

### A2. Binary cCRE annotations
`heritability/binary/`
- `00_create_binary_annot.py` — 0/1 annotation from ATAC peak TSVs (optionally eQTL-restricted).
- `01_ccre_full_ldsc.sh` — S-LDSC on full (unfiltered) cCRE annotations, per cell type (36-subclass SLURM array).
- `02_ccre_eqtl_ldsc.sh` — S-LDSC on eQTL-restricted cCRE annotations.

Outputs: `ccre_ldsc/{full,eqtl}/`, `ccre_ldsc/results_{full,eqtl}/`.

### A3. Borzoi brain L2-sum annotation
`heritability/borzoi_brain/`
- `00_borzoi_brain_annot.py` — sum L2 of brain-related Borzoi tracks → continuous annotation.
- `01_borzoi_brain_ldsc.sh` — S-LDSC + quantile enrichment on that annotation. Supports `--eqtl-matched` (reads `ccre_ldsc/results_eqtl/`).

Outputs: `ccre_ldsc/borzoi_brain/`, `ccre_ldsc/results_borzoi_brain/`, `quantile_results_by_trait/quantile_results_<TRAIT>/borzoi_brain_*`.

### A4. Variable-quantile top-bin (eQTL-matched proportion)
`heritability/variable_quantile/`
- `00_compute_topbin_quantile_M.py` — 2-bin quantile_M in single pass.
- `01_quantile_enrichment_based_on_eqtl.sh` — top-bin enrichment using proportion from `ccre_ldsc/results_eqtl/`.

Outputs: `quantile_results_by_trait/quantile_results_<TRAIT>/` (2-bin).

---

## Pipeline B — Variant overlap counts (no heritability)

`overlap/`
- `00_overlap_eQTL.py` — count top-K% prioritized variants overlapping eQTL cCREs per cell type; two sources (EpiBRAIN K27Ac, Borzoi L2-sum).
- `01_overlap_eQTL_by_quantile.sh` — SLURM array over thresholds {0.5, 1, 2, 3, 4, 5, 10}%.
- `02_overlap_eQTL_by_quantile_plot.py` — bar plots comparing EpiBRAIN vs Borzoi.
- `03_number_of_variants_overlap_eQTL_cCRE.py` — summary stats per subclass.
- `04_number_of_variants_overlap_eQTL_cCRE.sh` — SLURM wrapper.

Outputs: `results_overlap/`.

---

## Reference data (top level, unchanged)
- `listHM3.txt`, `w_hm3.snplist`, `hapmap3_snps/` — HapMap3 SNP references.
- `track_lists/` — per-trait track and track×chrom lists.
- `example_visualizations.sh` — pyGenomeTracks example calls.

## Archive
`archive/` holds superseded scripts (`run_pipeline.sh`, `05_plot_quantile_enrichment.R`).
