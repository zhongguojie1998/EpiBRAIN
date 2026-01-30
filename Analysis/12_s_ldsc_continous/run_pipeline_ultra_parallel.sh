#!/bin/bash
#
# Ultra-parallel S-LDSC pipeline using SLURM job arrays
#
# This pipeline supports two modes:
# 1. Use existing trait data (default Schizophrenia)
# 2. Extract trait from HDF5 file and process
#
# This is the FASTEST execution mode (~2-3 hours total)
#
# Usage:
#   # Use existing Schizophrenia data (backward compatible)
#   ./run_pipeline_ultra_parallel.sh
#   ./run_pipeline_ultra_parallel.sh --only-all
#
#   # Extract and process a trait from HDF5 file
#   ./run_pipeline_ultra_parallel.sh --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 --trait schizophrenia
#   ./run_pipeline_ultra_parallel.sh --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 --trait my_trait --only-all
#
#   # Use custom number of quantiles (default is 10)
#   ./run_pipeline_ultra_parallel.sh --nb-quantile 20
#   ./run_pipeline_ultra_parallel.sh --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 --trait my_trait --nb-quantile 5
#
#   # Exclude zeros from quantile computation (recommended for sparse annotations)
#   ./run_pipeline_ultra_parallel.sh --nb-quantile 20 --exclude-zero
#
#   # List available traits in HDF5 file
#   ./run_pipeline_ultra_parallel.sh --h5-file Data/source/GWAS/full_finetune.dim8.chk20.h5 --list-traits
#

set -e

# Parse command-line arguments
ONLY_ALL=0
H5_FILE=""
TRAIT_NAME=""
LIST_TRAITS=0
DATA_DIR=""
NB_QUANTILE=10  # Default number of quantiles (deciles)
EXCLUDE_ZERO=0  # Default: include zeros in quantile computation

while [[ $# -gt 0 ]]; do
    case $1 in
        --only-all)
            ONLY_ALL=1
            shift
            ;;
        --h5-file)
            H5_FILE="$2"
            shift 2
            ;;
        --trait)
            TRAIT_NAME="$2"
            shift 2
            ;;
        --list-traits)
            LIST_TRAITS=1
            shift
            ;;
        --nb-quantile)
            NB_QUANTILE="$2"
            shift 2
            ;;
        --exclude-zero)
            EXCLUDE_ZERO=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --h5-file FILE       Path to HDF5 file containing GWAS traits"
            echo "  --trait NAME         Name of trait to extract from HDF5 file"
            echo "  --list-traits        List available traits in HDF5 file and exit"
            echo "  --only-all           Run only the 'all' track (L2 norm)"
            echo "  --nb-quantile N      Number of quantiles for enrichment analysis (default: 10)"
            echo "  --exclude-zero       Exclude SNPs with annotation value=0 from quantile computation"
            echo ""
            echo "Examples:"
            echo "  # Use default Schizophrenia data"
            echo "  $0"
            echo ""
            echo "  # Extract and process a trait from HDF5"
            echo "  $0 --h5-file Data/source/GWAS/file.h5 --trait my_trait"
            echo ""
            echo "  # Use 20 quantiles instead of default 10"
            echo "  $0 --nb-quantile 20"
            echo ""
            echo "  # Exclude zeros from quantile computation"
            echo "  $0 --nb-quantile 20 --exclude-zero"
            echo ""
            echo "  # List available traits"
            echo "  $0 --h5-file Data/source/GWAS/file.h5 --list-traits"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Handle --list-traits mode
if [ ${LIST_TRAITS} -eq 1 ]; then
    if [ -z "${H5_FILE}" ]; then
        echo "Error: --list-traits requires --h5-file"
        exit 1
    fi
    echo "Listing available traits in ${H5_FILE}..."
    python3 ${SCRIPT_DIR}/00_extract_trait_from_h5.py --h5-file "${H5_FILE}" --list-traits
    exit 0
fi

# Validate arguments
if [ -n "${H5_FILE}" ] && [ -z "${TRAIT_NAME}" ]; then
    echo "Error: --h5-file requires --trait"
    exit 1
fi

if [ -z "${H5_FILE}" ] && [ -n "${TRAIT_NAME}" ]; then
    echo "Error: --trait requires --h5-file"
    exit 1
fi

# Determine data directory and trait name
if [ -n "${H5_FILE}" ]; then
    # Extract trait from HDF5 file
    DATA_DIR="${BASE_DIR}/Data/source/s_ldsc/${TRAIT_NAME}"
    echo "========================================================================"
    echo "Extracting trait data from HDF5 file"
    echo "========================================================================"
    echo "HDF5 file: ${H5_FILE}"
    echo "Trait: ${TRAIT_NAME}"
    echo "Output directory: ${DATA_DIR}"
    echo ""

    # Check if data already exists
    if [ -f "${DATA_DIR}/${TRAIT_NAME}.npy" ] && \
       [ -f "${DATA_DIR}/${TRAIT_NAME}.tracks.csv" ] && \
       [ -f "${DATA_DIR}/${TRAIT_NAME}.variants.csv" ]; then
        echo "Extracted data already exists, skipping extraction..."
    else
        echo "Running extraction script..."
        python3 ${SCRIPT_DIR}/00_extract_trait_from_h5.py \
            --h5-file "${H5_FILE}" \
            --trait "${TRAIT_NAME}" \
            --output-dir "${DATA_DIR}"

        if [ $? -ne 0 ]; then
            echo "Error: Failed to extract trait data from HDF5 file"
            exit 1
        fi
    fi
    echo ""
    echo "Trait extraction complete!"
    echo ""
else
    # Use default Schizophrenia data (backward compatible)
    DATA_DIR="${BASE_DIR}/Data/source/Schizophrenia"
    TRAIT_NAME="Schizophrenia"
    echo "Using default Schizophrenia data"
fi

# Track start time
START_TIME=$(date +%s)

# Determine output prefix based on H5 file
OUTPUT_PREFIX=""
if [ -n "${H5_FILE}" ]; then
    # Extract the basename of the H5 file (without path and extension)
    H5_BASENAME=$(basename "${H5_FILE}" .h5)
    if [ "${H5_BASENAME}" = "borzoi" ]; then
        OUTPUT_PREFIX="borzoi_"
    fi
fi

# Create trait-specific output directories
mkdir -p "${SCRIPT_DIR}/annotations_by_trait"
mkdir -p "${SCRIPT_DIR}/results_by_trait"
mkdir -p "${SCRIPT_DIR}/quantile_results_by_trait"

ANNOT_DIR="${SCRIPT_DIR}/annotations_by_trait/annotations_${OUTPUT_PREFIX}${TRAIT_NAME}"
RESULTS_DIR="${SCRIPT_DIR}/results_by_trait/results_${OUTPUT_PREFIX}${TRAIT_NAME}"
QUANTILE_DIR="${SCRIPT_DIR}/quantile_results_by_trait/quantile_results_${OUTPUT_PREFIX}${TRAIT_NAME}"

# Create logs directory and trait-specific directories
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${ANNOT_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${QUANTILE_DIR}

echo "========================================================================"
echo "S-LDSC Pipeline for Continuous Annotations (ULTRA-PARALLEL MODE)"
echo "========================================================================"
echo ""
echo "Working directory: ${SCRIPT_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Trait: ${TRAIT_NAME}"
echo "Date: $(date)"
echo "Number of quantiles: ${NB_QUANTILE}"
if [ ${EXCLUDE_ZERO} -eq 1 ]; then
    echo "Exclude zeros: yes (only non-zero annotation values)"
else
    echo "Exclude zeros: no (include all values)"
fi
if [ ${ONLY_ALL} -eq 1 ]; then
    echo "Mode: ONLY 'all' TRACK (L2 norm across all tracks)"
else
    echo "Mode: ALL TRACKS"
fi
echo ""
echo "Output directories (trait-specific):"
echo "  - Annotations: ${ANNOT_DIR}"
echo "  - Results: ${RESULTS_DIR}"
echo "  - Quantile results: ${QUANTILE_DIR}"
echo ""

# Step 1: Create annotation files
echo "========================================================================"
echo "Step 1: Creating annotation files from numpy matrix"
echo "========================================================================"

# Check if annotations directory exists and has all required files
ANNOT_EXISTS=0
if [ ${ONLY_ALL} -eq 1 ]; then
    # When --only-all is specified, only check for 'all' track
    if [ -d "${ANNOT_DIR}/all" ]; then
        # Check if all 22 chromosome annotation files exist for 'all' track
        MISSING_ANNOT=0
        for chr in {1..22}; do
            if [ ! -f "${ANNOT_DIR}/all/all.${chr}.annot.gz" ]; then
                MISSING_ANNOT=$((MISSING_ANNOT + 1))
            fi
        done

        if [ ${MISSING_ANNOT} -eq 0 ]; then
            ANNOT_EXISTS=1
            echo "Found complete annotation files for 'all' track (22 chromosomes)"
        else
            echo "Found 'all' track but missing ${MISSING_ANNOT} chromosome annotation files"
        fi
    fi
else
    # When processing all tracks, check if annotation directory is populated
    if [ -d "${ANNOT_DIR}" ] && [ -n "$(ls -A ${ANNOT_DIR} 2>/dev/null)" ]; then
        # Check if 'all' track exists as a basic sanity check
        if [ -d "${ANNOT_DIR}/all" ]; then
            ANNOT_EXISTS=1
            echo "Found existing annotation directory with multiple tracks"
        fi
    fi
fi

if [ ${ANNOT_EXISTS} -eq 0 ]; then
    if [ ${ONLY_ALL} -eq 1 ]; then
        echo "Creating annotation files for 'all' track only..."
        echo "WARNING: --only-all flag specified, but annotation script will create all tracks."
        echo "         This is expected behavior. Subsequent steps will only process 'all' track."
    else
        echo "Creating annotation files for all tracks..."
    fi

    python3 ${SCRIPT_DIR}/01_create_annotation_files.py \
        --data-dir "${DATA_DIR}" \
        --trait-name "${TRAIT_NAME}" \
        --output-dir "${ANNOT_DIR}" \
        --n-jobs 36

    if [ $? -ne 0 ]; then
        echo "Error in Step 1: Annotation file creation failed"
        exit 1
    fi
else
    if [ ${ONLY_ALL} -eq 1 ]; then
        echo "Annotation file for 'all' track already exists, skipping creation..."
    else
        echo "Annotation files already exist, skipping creation..."
    fi
fi

echo ""
echo "Step 1 complete!"
echo ""

# Create track list for job arrays
mkdir -p "${SCRIPT_DIR}/track_lists"
TRACK_LIST="${SCRIPT_DIR}/track_lists/track_list_${TRAIT_NAME}.txt"
echo "Creating track list for SLURM job arrays..."
if [ ${ONLY_ALL} -eq 1 ]; then
    # Only include the "all" track
    if [ ! -d "${ANNOT_DIR}/all" ]; then
        echo "Error: 'all' track directory not found at ${ANNOT_DIR}/all"
        echo "Please ensure Step 1 has created the 'all' track (L2 norm)."
        exit 1
    fi
    echo "all" > ${TRACK_LIST}
    echo "Filtering for 'all' track only (L2 norm)"
else
    # Include all tracks
    ls -d ${ANNOT_DIR}/*/ | xargs -n 1 basename > ${TRACK_LIST}
fi
NUM_TRACKS=$(wc -l < ${TRACK_LIST})
echo "Found ${NUM_TRACKS} tracks"

# Create track-chromosome mapping for ultra-parallel LD score computation
TRACK_CHR_LIST="${SCRIPT_DIR}/track_lists/track_chr_list_${TRAIT_NAME}.txt"
echo "Creating track-chromosome mapping..."
> ${TRACK_CHR_LIST}  # Clear file

for track in $(cat ${TRACK_LIST}); do
    for chr in {1..22}; do
        echo "${track} ${chr}" >> ${TRACK_CHR_LIST}
    done
done

NUM_JOBS=$(wc -l < ${TRACK_CHR_LIST})
if [ ${ONLY_ALL} -eq 1 ]; then
    echo "Created ${NUM_JOBS} track-chromosome combinations (1 track × 22 chromosomes)"
else
    echo "Created ${NUM_JOBS} track-chromosome combinations (${NUM_TRACKS} tracks × 22 chromosomes)"
fi

# Create HapMap3 SNP list if needed
HAPMAP3_LIST="${SCRIPT_DIR}/listHM3.txt"
if [ ! -f "${HAPMAP3_LIST}" ]; then
    echo "Creating HapMap3 SNP list..."
    WEIGHTS_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/12_ldsc/reference/weights_hm3_no_hla"
    for chr in {1..22}; do
        zcat "${WEIGHTS_DIR}/weights.hm3_noMHC.${chr}.l2.ldscore.gz" | tail -n +2 | cut -f2
    done > "${HAPMAP3_LIST}"
    echo "Created ${HAPMAP3_LIST}"
fi

# Step 2: Compute LD scores in ultra-parallel mode
echo ""
echo "========================================================================"
echo "Step 2: Computing LD scores (ULTRA-PARALLEL MODE)"
echo "========================================================================"

# Check which LD score files are missing and create filtered list
MISSING_TRACK_CHR_LIST="${SCRIPT_DIR}/track_lists/track_chr_list_${TRAIT_NAME}_missing.txt"
> ${MISSING_TRACK_CHR_LIST}  # Clear file

MISSING_LDSCORES=0
while read line; do
    track=$(echo ${line} | cut -d' ' -f1)
    chr=$(echo ${line} | cut -d' ' -f2)
    ldscore_file="${ANNOT_DIR}/${track}/${track}.${chr}.l2.ldscore.gz"
    if [ ! -f "${ldscore_file}" ]; then
        echo "${track} ${chr}" >> ${MISSING_TRACK_CHR_LIST}
        MISSING_LDSCORES=$((MISSING_LDSCORES + 1))
    fi
done < ${TRACK_CHR_LIST}

if [ ${MISSING_LDSCORES} -eq 0 ]; then
    echo "All LD score files already exist (${NUM_JOBS} files), skipping..."
    echo ""
    echo "Step 2 complete!"
    echo ""
    FAILED_LD=0
else
    echo "Found ${MISSING_LDSCORES}/${NUM_JOBS} LD score files missing. Running computation for missing files only..."
    echo ""

echo "Submitting SLURM job array for LD score computation..."
echo "  Array size: 1-${MISSING_LDSCORES} (one job per missing track-chromosome)"
echo "  Time per job: 12 hours"
echo "  Memory per job: 4GB"
echo "  Total jobs to run: ${MISSING_LDSCORES} (skipping ${NUM_JOBS} - ${MISSING_LDSCORES} = $((NUM_JOBS - MISSING_LDSCORES)) existing files)"

LD_JOB_ID=$(sbatch --parsable --array=1-${MISSING_LDSCORES}%500 \
    --export=ANNOT_DIR="${ANNOT_DIR}",TRACK_CHR_LIST="${MISSING_TRACK_CHR_LIST}",TRAIT_NAME="${TRAIT_NAME}" \
    ${SCRIPT_DIR}/slurm_02_compute_ld_per_chrom.sh)

if [ -z "${LD_JOB_ID}" ]; then
    echo "Error: Failed to submit LD score computation jobs"
    exit 1
fi

echo "  Job ID: ${LD_JOB_ID}"
echo "  Note: Using %500 throttle to limit concurrent jobs"
echo "  Waiting for LD score computation to complete..."
echo ""
echo "  Monitor progress with:"
echo "    squeue -j ${LD_JOB_ID}"
echo "    squeue -j ${LD_JOB_ID} | wc -l  # Count remaining jobs"
echo "  Check logs in: ${SCRIPT_DIR}/logs/ld_scores_${LD_JOB_ID}_*.{out,err}"

# Wait for LD score jobs to complete
LAST_COUNT=-1
while squeue -j ${LD_JOB_ID} 2>/dev/null | grep -q ${LD_JOB_ID}; do
    CURRENT_COUNT=$(squeue -j ${LD_JOB_ID} 2>/dev/null | grep -c ${LD_JOB_ID} || echo "0")
    if [ ${CURRENT_COUNT} -ne ${LAST_COUNT} ]; then
        COMPLETED=$((MISSING_LDSCORES - CURRENT_COUNT))
        echo "  Progress: ${COMPLETED}/${MISSING_LDSCORES} completed ($(echo "scale=1; ${COMPLETED}*100/${MISSING_LDSCORES}" | bc)%)"
        LAST_COUNT=${CURRENT_COUNT}
    fi
    sleep 30
done

echo ""
echo "LD score computation jobs completed!"

# Check for failures
FAILED_LD=$(find ${SCRIPT_DIR}/logs/ -name "ld_scores_${LD_JOB_ID}_*.err" -exec grep -l "Failed\|Error" {} \; 2>/dev/null | wc -l)
if [ ${FAILED_LD} -gt 0 ]; then
    echo "Warning: ${FAILED_LD} LD score jobs failed. Check logs for details."
    echo "Failed job logs:"
    find ${SCRIPT_DIR}/logs/ -name "ld_scores_${LD_JOB_ID}_*.err" -exec grep -l "Failed\|Error" {} \;
fi

echo ""
echo "Step 2 complete!"
echo ""
fi  # End of LD score check

# Step 3: Run S-LDSC regression in parallel
echo "========================================================================"
if [ ${ONLY_ALL} -eq 1 ]; then
    echo "Step 3: Running S-LDSC regression for 'all' track only (L2 norm)"
else
    echo "Step 3: Running S-LDSC regression for ${NUM_TRACKS} tracks (PARALLEL)"
fi
echo "========================================================================"

# Check if all result files already exist

ALL_RESULTS_EXIST=1
MISSING_RESULTS=0
while read track; do
    # Check for both .results and .part_delete files (latter needed for quantile enrichment)
    if [ ! -f "${RESULTS_DIR}/${track}.results" ] || [ ! -f "${RESULTS_DIR}/${track}.part_delete" ]; then
        ALL_RESULTS_EXIST=0
        MISSING_RESULTS=$((MISSING_RESULTS + 1))
    fi
done < ${TRACK_LIST}

if [ ${ALL_RESULTS_EXIST} -eq 1 ]; then
    echo "All regression results already exist (${NUM_TRACKS} tracks with .results and .part_delete files), skipping..."
    echo ""
    echo "Step 3 complete!"
    echo ""
    FAILED_REG=0
else
    echo "Found ${MISSING_RESULTS}/${NUM_TRACKS} tracks missing results. Running regression..."
    echo ""

echo "Submitting SLURM job array for S-LDSC regression..."
echo "  Array size: 1-${NUM_TRACKS}"
echo "  Time per job: 12 hours"
echo "  Memory per job: 8GB"

REG_JOB_ID=$(sbatch --parsable --array=1-${NUM_TRACKS} \
    --export=ANNOT_DIR="${ANNOT_DIR}",RESULTS_DIR="${RESULTS_DIR}",TRACK_LIST="${TRACK_LIST}",TRAIT_NAME="${TRAIT_NAME}" \
    ${SCRIPT_DIR}/slurm_03_regression_per_track.sh)

if [ -z "${REG_JOB_ID}" ]; then
    echo "Error: Failed to submit regression jobs"
    exit 1
fi

echo "  Job ID: ${REG_JOB_ID}"
echo "  Waiting for S-LDSC regression to complete..."
echo ""
echo "  Monitor progress with:"
echo "    squeue -j ${REG_JOB_ID}"
echo "    squeue -j ${REG_JOB_ID} | wc -l  # Count remaining jobs"
echo "  Check logs in: ${SCRIPT_DIR}/logs/regression_${REG_JOB_ID}_*.{out,err}"

# Wait for regression jobs to complete with progress
LAST_COUNT=-1
while squeue -j ${REG_JOB_ID} 2>/dev/null | grep -q ${REG_JOB_ID}; do
    CURRENT_COUNT=$(squeue -j ${REG_JOB_ID} 2>/dev/null | grep -c ${REG_JOB_ID} || echo "0")
    if [ ${CURRENT_COUNT} -ne ${LAST_COUNT} ]; then
        COMPLETED=$((NUM_TRACKS - CURRENT_COUNT))
        echo "  Progress: ${COMPLETED}/${NUM_TRACKS} completed ($(echo "scale=1; ${COMPLETED}*100/${NUM_TRACKS}" | bc)%)"
        LAST_COUNT=${CURRENT_COUNT}
    fi
    sleep 30
done

echo ""
echo "S-LDSC regression jobs completed!"

# Check for failures
FAILED_REG=$(find ${SCRIPT_DIR}/logs/ -name "regression_${REG_JOB_ID}_*.err" -exec grep -l "Failed\|Error" {} \; 2>/dev/null | wc -l)
if [ ${FAILED_REG} -gt 0 ]; then
    echo "Warning: ${FAILED_REG} regression jobs failed. Check logs for details."
    echo "Failed job logs:"
    find ${SCRIPT_DIR}/logs/ -name "regression_${REG_JOB_ID}_*.err" -exec grep -l "Failed\|Error" {} \;
fi

echo ""
echo "Step 3 complete!"
echo ""
fi  # End of regression check

# Step 4: Compute quantile enrichment
echo "========================================================================"
echo "Step 4: Computing quantile-based enrichment"
echo "========================================================================"

if [ ${EXCLUDE_ZERO} -eq 1 ]; then
    echo "Running quantile enrichment analysis with ${NB_QUANTILE} quantiles (excluding zeros)..."
else
    echo "Running quantile enrichment analysis with ${NB_QUANTILE} quantiles..."
fi
ANNOT_DIR="${ANNOT_DIR}" RESULTS_DIR="${RESULTS_DIR}" QUANTILE_DIR="${QUANTILE_DIR}" \
    TRACK_LIST="${TRACK_LIST}" TRAIT_NAME="${TRAIT_NAME}" NB_QUANTILE="${NB_QUANTILE}" \
    EXCLUDE_ZERO="${EXCLUDE_ZERO}" \
    ${SCRIPT_DIR}/04_quantile_enrichment.sh

if [ $? -eq 0 ]; then
    echo "Quantile enrichment analysis complete!"
else
    echo "Warning: Quantile enrichment analysis encountered errors"
fi

echo ""
echo "Step 4 complete!"
echo ""

# Create summary table
echo "========================================================================"
echo "Creating summary results table"
echo "========================================================================"

SUMMARY_FILE="${RESULTS_DIR}/summary_results.txt"

echo -e "Track\tCoefficient\tCoefficient_SE\tCoefficient_Z" > ${SUMMARY_FILE}

for result_file in ${RESULTS_DIR}/*.results; do
    if [ -f "${result_file}" ]; then
        track_name=$(basename "${result_file}" .results)
        # Extract the last line (custom annotation) from results
        # Columns: 1=Category, 8=Coefficient, 9=Coefficient_std_error, 10=Coefficient_z-score
        tail -n 1 "${result_file}" | \
            awk -v track="${track_name}" '{printf "%s\t%s\t%s\t%s\n", track, $8, $9, $10}' \
            >> ${SUMMARY_FILE}
    fi
done

# Final summary
echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  Total tracks: ${NUM_TRACKS}"
echo "  Total LD score jobs: ${NUM_JOBS}"
echo "  Failed LD score jobs: ${FAILED_LD}"
echo "  Failed regression jobs: ${FAILED_REG}"
echo "  Successful results: $(ls ${RESULTS_DIR}/*.results 2>/dev/null | wc -l)"
echo ""
echo "Output directories (trait: ${TRAIT_NAME}):"
echo "  - Annotations: ${ANNOT_DIR}/"
echo "  - Results: ${RESULTS_DIR}/"
echo "  - Quantile enrichment: ${QUANTILE_DIR}/"
echo "  - Logs: ${SCRIPT_DIR}/logs/"
echo ""
echo "Key files:"
echo "  - Summary table: ${SUMMARY_FILE}"
echo "  - Individual results: ${RESULTS_DIR}/*.results"
echo "  - Quantile enrichment: ${QUANTILE_DIR}/*.quantile_h2g.txt"
echo "  - Enrichment summary: ${QUANTILE_DIR}/summary_enrichment.txt"
echo ""

if [ -f "${SUMMARY_FILE}" ]; then
    echo "Top 10 results by absolute Z-score:"
    head -1 ${SUMMARY_FILE}
    tail -n +2 ${SUMMARY_FILE} | awk '{print $0"\t"sqrt($4*$4)}' | sort -t$'\t' -k6 -rn | head -10 | cut -f1-5
fi

echo ""
echo "========================================================================"
echo "Total runtime: $(( $(date +%s) - START_TIME )) seconds"
echo "========================================================================"
