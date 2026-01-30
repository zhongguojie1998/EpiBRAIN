# S-LDSC Pipeline with HDF5 Trait Extraction

This pipeline now supports extracting GWAS trait data from HDF5 files and running S-LDSC analysis on any trait, not just Schizophrenia.

## Overview

The pipeline has been enhanced with three new capabilities:

1. **Extract trait data from HDF5 files** - New script `00_extract_trait_from_h5.py`
2. **Configurable data sources** - Modified `01_create_annotation_files.py` to accept any trait data
3. **Integrated pipeline** - Updated `run_pipeline_ultra_parallel.sh` to handle everything end-to-end

## Quick Start

### List Available Traits in HDF5 File

```bash
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --list-traits
```

### Run S-LDSC on a Trait from HDF5 File

```bash
# Extract and process a specific trait (e.g., "my_trait")
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait my_trait
```

### Run Only the 'all' Track (L2 norm)

```bash
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait my_trait \
    --only-all
```

### Use Default Schizophrenia Data (Backward Compatible)

```bash
# This still works exactly as before
bash run_pipeline_ultra_parallel.sh
bash run_pipeline_ultra_parallel.sh --only-all
```

## Detailed Usage

### Step 0: Extract Trait Data (Optional - Standalone)

If you want to extract trait data separately before running the pipeline:

```bash
# Extract a specific trait
python3 00_extract_trait_from_h5.py \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait schizophrenia \
    --output-dir Data/source/schizophrenia_extracted

# List available traits
python3 00_extract_trait_from_h5.py \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --list-traits
```

**Output files:**
- `{trait_name}.npy` - Annotation matrix (variants × tracks)
- `{trait_name}.tracks.csv` - Track names/metadata
- `{trait_name}.variants.csv` - Variant information (CHR, BP, SNP, A1, A2)
- `{trait_name}.sumstats.tsv.gz` - Summary statistics
- `{trait_name}_combined.tsv.gz` - Combined data for reference

### Step 1: Create Annotations (Optional - Standalone)

If you want to create annotation files separately:

```bash
# Using extracted trait data
python3 01_create_annotation_files.py \
    --data-dir Data/source/my_trait_extracted \
    --trait-name my_trait \
    --n-jobs 36

# Using existing data directory
python3 01_create_annotation_files.py \
    --data-dir Data/source/Schizophrenia \
    --trait-name Schizophrenia \
    --n-jobs 8
```

### Full Pipeline

The main pipeline script now handles everything automatically:

```bash
# Complete workflow from HDF5 to S-LDSC results
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait alzheimers
```

**What happens:**
1. Checks if trait data is already extracted
2. If not, extracts it from HDF5 file
3. Creates annotation files for all tracks
4. Computes LD scores for each track × chromosome
5. Runs S-LDSC regression for each track
6. Computes quantile-based enrichment
7. Generates summary results

## Command-Line Options

### Main Pipeline (`run_pipeline_ultra_parallel.sh`)

| Option | Description |
|--------|-------------|
| `--h5-file FILE` | Path to HDF5 file containing GWAS traits |
| `--trait NAME` | Name of trait to extract from HDF5 file |
| `--list-traits` | List available traits in HDF5 file and exit |
| `--only-all` | Run only the 'all' track (L2 norm) |

### Extraction Script (`00_extract_trait_from_h5.py`)

| Option | Description |
|--------|-------------|
| `--h5-file FILE` | Path to input HDF5 file (required) |
| `--trait NAME` | Name of trait to extract (required unless --list-traits) |
| `--output-dir DIR` | Output directory for extracted data (required) |
| `--chain-file FILE` | Path to hg38ToHg19 liftover chain file (optional) |
| `--list-traits` | List available traits and exit |

### Annotation Creation Script (`01_create_annotation_files.py`)

| Option | Description |
|--------|-------------|
| `--data-dir DIR` | Path to directory containing trait data files (required) |
| `--trait-name NAME` | Name of the trait (required) |
| `--ref-dir DIR` | Path to reference directory (optional) |
| `--output-dir DIR` | Path to output directory (optional) |
| `--n-jobs N` | Number of parallel jobs (default: -1 = all cores) |

## Examples

### Example 1: Quick Analysis on New Trait

```bash
# List what's available
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --list-traits

# Run analysis on specific trait
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait parkinsons \
    --only-all
```

### Example 2: Multi-Trait Analysis

```bash
# Process multiple traits sequentially
for trait in schizophrenia alzheimers parkinsons; do
    echo "Processing ${trait}..."
    bash run_pipeline_ultra_parallel.sh \
        --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
        --trait ${trait} \
        --only-all
done
```

### Example 3: Custom Workflow

```bash
# 1. Extract trait data first
python3 00_extract_trait_from_h5.py \
    --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 \
    --trait depression \
    --output-dir Data/source/depression_custom

# 2. Create annotations with custom settings
python3 01_create_annotation_files.py \
    --data-dir Data/source/depression_custom \
    --trait-name depression \
    --n-jobs 16

# 3. Run the rest of the pipeline manually
# (Steps 2-4 from run_pipeline_ultra_parallel.sh)
```

## Output Structure

When using `--trait my_trait`, the pipeline creates:

```
Data/source/my_trait_extracted/          # Extracted trait data
├── my_trait.npy                          # Annotation matrix
├── my_trait.tracks.csv                   # Track metadata
├── my_trait.variants.csv                 # Variant information
├── my_trait.sumstats.tsv.gz              # Summary statistics
└── my_trait_combined.tsv.gz              # Combined data

Analysis/12_s_ldsc_continous/
├── annotations/                           # Annotation files by track
│   ├── all/                              # L2 norm track
│   ├── track1/
│   └── track2/
├── results/                              # S-LDSC results
│   ├── all.results
│   ├── all.log
│   ├── all.part_delete                   # Jackknife values
│   └── summary_results.txt               # Summary table
├── quantile_results/                     # Quantile enrichment
│   ├── all.quantile_M.txt
│   ├── all.quantile_h2g.txt
│   └── summary_enrichment.txt
└── logs/                                 # SLURM job logs
```

## Notes

- **Coordinate Conversion**: The extraction script automatically converts coordinates from hg38 to hg19 using the liftover chain file
- **Caching**: Extracted data is cached and reused if you run the pipeline multiple times with the same trait
- **Backward Compatibility**: Running the pipeline without `--h5-file` still uses the default Schizophrenia data
- **Memory Requirements**: Large HDF5 files may require substantial memory for extraction

## Troubleshooting

### Missing Chain File Error

If you see an error about missing chain file:

```bash
# Ensure the chain file exists at the default location
ls ../Data/Ref/hg38ToHg19.over.chain.gz

# Or specify a custom location
python3 00_extract_trait_from_h5.py \
    --chain-file /path/to/hg38ToHg19.over.chain.gz \
    ...
```

### Trait Not Found

```bash
# List available traits to see the exact names
bash run_pipeline_ultra_parallel.sh \
    --h5-file Data/source/GWAS/file.h5 \
    --list-traits
```

### Missing .part_delete File

If you see warnings about missing `.part_delete` files, the regression step was skipped. Delete the `.results` file and rerun:

```bash
rm Analysis/12_s_ldsc_continous/results/{trait}.results
bash run_pipeline_ultra_parallel.sh --h5-file ... --trait {trait}
```

## References

- Original split script: `Analysis/03_3_split_GWAS_by_experiment.py`
- S-LDSC documentation: [github.com/bulik/ldsc](https://github.com/bulik/ldsc)
