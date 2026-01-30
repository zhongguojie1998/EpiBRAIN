# S-LDSC Pipeline for Continuous Annotations

This directory contains a complete pipeline for stratified LD score regression (S-LDSC) analysis using continuous annotations.

## Overview

The pipeline follows the procedure described in Gazal et al. (2017) and the LDSC wiki for analyzing partitioned heritability using continuous annotations.

## Pipeline Steps

### Step 1: Create Annotation Files
**Script**: `01_create_annotation_files.py`

Creates annotation files from continuous values.

**Output**: `annotations/*/` - Annotation files for each track

### Step 2: Compute LD Scores
**Script**: `slurm_02_compute_ld_per_chrom.sh`

Computes LD scores for each track-chromosome combination.

**Output**: `annotations/*/*.l2.ldscore.gz`

### Step 3: Run S-LDSC Regression
**Script**: `slurm_03_regression_per_track.sh`

Runs partitioned heritability analysis.

**Output**: `results/*.results`, `results/*.part_delete`

### Step 4: Quantile Enrichment Analysis
**Script**: `04_quantile_enrichment.sh`

Partitions continuous annotations into quantiles and computes enrichment.

**Output**: `quantile_results/*.quantile_h2g.txt`, `quantile_results/summary_enrichment.txt`

### Step 5: Visualization
**Script**: `05_plot_quantile_enrichment.R`

Creates plots of quantile enrichment results.

**Output**: `quantile_results/enrichment_plot_*.pdf`

## Running the Pipeline

### Full Pipeline
```bash
./run_pipeline_ultra_parallel.sh
```

### Only "all" Track
```bash
./run_pipeline_ultra_parallel.sh --only-all
```

### Individual Steps
```bash
# Step 4: Quantile enrichment
./04_quantile_enrichment.sh

# Step 5: Visualization
Rscript 05_plot_quantile_enrichment.R
```

## Interpreting Results

### Quantile Enrichment

- **enr > 1**: Over-representation of heritability
- **enr < 1**: Under-representation of heritability  
- **enr_pval < 0.05**: Statistically significant enrichment

## Important Notes

1. **Continuous annotations**: Use quantile-based enrichment, not standard enrichment from .results files
2. **Multiple testing**: Consider Bonferroni correction (p < 0.05/N_tracks)
3. **Frequency files**: Generated automatically if missing


## Requirements

- **Conda environment**: `ldsc` (provides plink, perl, R, and LDSC tools)
  ```bash
  conda activate ldsc
  ```

- **Python packages**: numpy, pandas (for Step 1)
- **R packages**: ggplot2, dplyr, tidyr (for Step 5 visualization)

