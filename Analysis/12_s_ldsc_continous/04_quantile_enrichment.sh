#!/bin/bash
#
# Step 4: Compute quantile-based enrichment for continuous annotations
#
# Requirements: ldsc conda environment (provides plink, perl, and R)
#
# This script:
# 1. Generates frequency files (if needed) using plink
# 2. Computes annotation sums stratified by quantiles using quantile_M.pl
# 3. Computes quantile-specific heritability using quantile_h2g.r
#
# File Organization:
# - annotations_by_trait/  : Trait-specific annotation directories (115 traits)
# - results_by_trait/      : Trait-specific results directories (115 traits)
# - quantile_results_by_trait/ : Trait-specific quantile results (114 traits)
# - track_lists/          : Track list and track-chr list files (229 files total)
# - archive/              : Old working directories (annotations/, results/, quantile_results/)
#
# Usage:
#   bash 04_quantile_enrichment.sh -t <trait_name>
#   or
#   bash 04_quantile_enrichment.sh -a <annot_dir> -r <results_dir> -q <quantile_dir>
#
# Arguments:
#   -t, --trait TRAIT_NAME           Trait name (constructs paths automatically)
#   -a, --annot-dir ANNOT_DIR        Custom annotation directory path
#   -r, --results-dir RESULTS_DIR    Custom results directory path
#   -q, --quantile-dir QUANTILE_DIR  Custom quantile results directory path
#   -l, --track-list TRACK_LIST      Custom track list file path
#   -n, --num-quantiles N            Number of quantiles (default: 10)
#   -e, --exclude-zero               Exclude zero annotation values
#   -h, --help                       Show this help message
#
# Example:
#   bash 04_quantile_enrichment.sh -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
TRAIT_NAME=""
ANNOT_DIR=""
RESULTS_DIR=""
QUANTILE_DIR=""
TRACK_LIST=""
NB_QUANTILE=10
EXCLUDE_ZERO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--trait)
            TRAIT_NAME="$2"
            shift 2
            ;;
        -a|--annot-dir)
            ANNOT_DIR="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -q|--quantile-dir)
            QUANTILE_DIR="$2"
            shift 2
            ;;
        -l|--track-list)
            TRACK_LIST="$2"
            shift 2
            ;;
        -n|--num-quantiles)
            NB_QUANTILE="$2"
            shift 2
            ;;
        -e|--exclude-zero)
            EXCLUDE_ZERO=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --trait TRAIT_NAME           Trait name (constructs paths automatically)"
            echo "  -a, --annot-dir ANNOT_DIR        Custom annotation directory path"
            echo "  -r, --results-dir RESULTS_DIR    Custom results directory path"
            echo "  -q, --quantile-dir QUANTILE_DIR  Custom quantile results directory path"
            echo "  -l, --track-list TRACK_LIST      Custom track list file path"
            echo "  -n, --num-quantiles N            Number of quantiles (default: 10)"
            echo "  -e, --exclude-zero               Exclude zero annotation values"
            echo "  -h, --help                       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats"
            echo "  $0 -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats -n 20 -e"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Activate conda environment
source /gpfs/commons/home/guojiezhong/miniconda3/etc/profile.d/conda.sh
conda activate ldsc

# Configuration
BASE_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN"

# Construct paths from TRAIT_NAME if not provided directly
if [ -z "${ANNOT_DIR}" ] || [ -z "${RESULTS_DIR}" ] || [ -z "${QUANTILE_DIR}" ]; then
    if [ -z "${TRAIT_NAME}" ]; then
        echo "ERROR: Either --trait or (--annot-dir, --results-dir, and --quantile-dir) must be specified"
        echo "Run '$0 --help' for usage information"
        exit 1
    fi
    ANNOT_DIR="${SCRIPT_DIR}/annotations_by_trait/annotations_${TRAIT_NAME}"
    RESULTS_DIR="${SCRIPT_DIR}/results_by_trait/results_${TRAIT_NAME}"
    QUANTILE_DIR="${SCRIPT_DIR}/quantile_results_by_trait/quantile_results_${TRAIT_NAME}"
fi

# Set TRACK_LIST if not already set
if [ -z "${TRACK_LIST}" ]; then
    if [ ! -z "${TRAIT_NAME}" ]; then
        TRACK_LIST="${SCRIPT_DIR}/track_lists/track_list_${TRAIT_NAME}.txt"
    else
        TRACK_LIST="${SCRIPT_DIR}/track_lists/track_list.txt"
    fi
fi

# Set TRAIT_NAME if not set (for display purposes)
if [ -z "${TRAIT_NAME}" ]; then
    TRAIT_NAME="unknown"
fi

# Check if directories exist
if [ ! -d "${ANNOT_DIR}" ]; then
    echo "ERROR: Annotation directory not found: ${ANNOT_DIR}"
    exit 1
fi
if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

# Create quantile results directory if it doesn't exist
mkdir -p "${QUANTILE_DIR}"

REF_DIR="${BASE_DIR}/Analysis/12_ldsc/reference"
LDSC_DIR="${REF_DIR}/ldsc"
PLINK_DIR="${REF_DIR}/1000G_EUR_Phase3_plink"
BASELINE_DIR="${REF_DIR}/1000G_EUR_Phase3_baseline"

# Scripts
QUANTILE_M="${LDSC_DIR}/ContinuousAnnotations/quantile_M.pl"
QUANTILE_H2G="${LDSC_DIR}/ContinuousAnnotations/quantile_h2g.r"

# Parameters (already set from command line arguments)
MAF_THRESHOLD=0.05

# Create output directory
mkdir -p "${QUANTILE_DIR}"

echo "========================================================================"
echo "Quantile-Based Enrichment Analysis for Continuous Annotations"
echo "========================================================================"
echo ""
echo "Working directory: ${SCRIPT_DIR}"
echo "Trait: ${TRAIT_NAME}"
echo "Date: $(date)"
echo "Number of quantiles: ${NB_QUANTILE}"
echo "MAF threshold: ${MAF_THRESHOLD}"
if [ ${EXCLUDE_ZERO} -eq 1 ]; then
    echo "Exclude zeros: yes (only non-zero annotation values)"
else
    echo "Exclude zeros: no (include all values)"
fi
echo ""
echo "Output directories:"
echo "  - Annotations: ${ANNOT_DIR}"
echo "  - Results: ${RESULTS_DIR}"
echo "  - Quantile results: ${QUANTILE_DIR}"
echo ""

# Step 1: Generate frequency files if they don't exist
echo "========================================================================"
echo "Step 1: Checking/generating frequency files"
echo "========================================================================"

FRQ_DIR="${PLINK_DIR}"
FRQ_PREFIX="${FRQ_DIR}/1000G.EUR.QC."

# Check if frequency files exist
NEED_FRQ=0
for chr in {1..22}; do
    if [ ! -f "${FRQ_PREFIX}${chr}.frq" ]; then
        NEED_FRQ=1
        break
    fi
done

if [ ${NEED_FRQ} -eq 1 ]; then
    echo "Frequency files not found. Generating using plink..."

    # Generate frequency files for each chromosome
    for chr in {1..22}; do
        if [ ! -f "${FRQ_PREFIX}${chr}.frq" ]; then
            echo "  Generating frequency file for chromosome ${chr}..."
            plink --bfile "${PLINK_DIR}/1000G.EUR.QC.${chr}" \
                  --freq \
                  --out "${PLINK_DIR}/1000G.EUR.QC.${chr}" \
                  --silent
        fi
    done

    echo "Frequency files generated successfully!"
else
    echo "Frequency files already exist. Skipping generation."
fi

echo ""

# Step 2: Get list of tracks
# If TRACK_LIST is not already set (from environment), generate it
if [ ! -f "${TRACK_LIST}" ]; then
    echo "Track list not found at ${TRACK_LIST}. Generating from annotation directory..."
    mkdir -p "${SCRIPT_DIR}/track_lists"
    TRACK_LIST="${SCRIPT_DIR}/track_lists/track_list_${TRAIT_NAME}.txt"
    ls -d ${ANNOT_DIR}/*/ 2>/dev/null | xargs -n 1 basename > ${TRACK_LIST}
fi

if [ ! -f "${TRACK_LIST}" ] || [ ! -s "${TRACK_LIST}" ]; then
    echo "ERROR: Track list is empty or not found: ${TRACK_LIST}"
    echo "Please ensure annotations have been created first."
    exit 1
fi

NUM_TRACKS=$(wc -l < ${TRACK_LIST})
echo "Found ${NUM_TRACKS} tracks to analyze"
echo ""

# Step 3: Run quantile analysis for each track
echo "========================================================================"
echo "Step 2-3: Computing quantile enrichment for each track"
echo "========================================================================"

TRACK_NUM=0
while read TRACK_NAME; do
    TRACK_NUM=$((TRACK_NUM + 1))

    echo ""
    echo "Processing track ${TRACK_NUM}/${NUM_TRACKS}: ${TRACK_NAME}"
    echo "--------------------------------------------------------------------"

    # Check if results exist
    RESULT_FILE="${RESULTS_DIR}/${TRACK_NAME}.results"
    if [ ! -f "${RESULT_FILE}" ]; then
        echo "  WARNING: Results file not found: ${RESULT_FILE}"
        echo "  Skipping ${TRACK_NAME}"
        continue
    fi

    # Check if part_delete file exists
    DELETE_FILE="${RESULTS_DIR}/${TRACK_NAME}.part_delete"
    if [ ! -f "${DELETE_FILE}" ]; then
        echo "  WARNING: Jackknife delete file not found: ${DELETE_FILE}"
        echo "  Skipping ${TRACK_NAME}"
        continue
    fi

    # Output files
    QUANTILE_M_OUT="${QUANTILE_DIR}/${TRACK_NAME}.quantile_M.txt"
    QUANTILE_H2G_OUT="${QUANTILE_DIR}/${TRACK_NAME}.quantile_h2g.txt"

    # Skip if already computed with the same number of quantiles
    SKIP_COMPUTATION=0
    if [ -f "${QUANTILE_H2G_OUT}" ]; then
        # Count existing quantiles (subtract 1 for header line)
        EXISTING_QUANTILES=$(($(wc -l < "${QUANTILE_H2G_OUT}") - 1))
        if [ ${EXISTING_QUANTILES} -eq ${NB_QUANTILE} ]; then
            echo "  Quantile enrichment already computed with ${NB_QUANTILE} quantiles. Skipping..."
            SKIP_COMPUTATION=1
        else
            echo "  Existing results have ${EXISTING_QUANTILES} quantiles, but ${NB_QUANTILE} requested. Recomputing..."
        fi
    fi

    if [ ${SKIP_COMPUTATION} -eq 1 ]; then
        continue
    fi

    echo "  Step 2a: Computing annotation sums by quantile..."

    # Build quantile_M.pl command
    QUANTILE_M_CMD="perl ${QUANTILE_M} \
        --frqfile-chr ${FRQ_PREFIX} \
        --ref-annot-chr ${BASELINE_DIR}/baseline.,${ANNOT_DIR}/${TRACK_NAME}/${TRACK_NAME}. \
        --annot-header ${TRACK_NAME} \
        --nb-quantile ${NB_QUANTILE} \
        --maf ${MAF_THRESHOLD}"

    # Add --exclude0 flag if requested
    if [ ${EXCLUDE_ZERO} -eq 1 ]; then
        QUANTILE_M_CMD="${QUANTILE_M_CMD} --exclude0"
    fi

    QUANTILE_M_CMD="${QUANTILE_M_CMD} --out ${QUANTILE_M_OUT}"

    # Run quantile_M.pl
    eval ${QUANTILE_M_CMD}

    if [ $? -eq 0 ]; then
        echo "  ✓ Quantile M computation complete"
    else
        echo "  ✗ Quantile M computation failed"
        continue
    fi

    echo "  Step 2b: Computing quantile-specific heritability..."

    # Run quantile_h2g.r
    Rscript ${QUANTILE_H2G} \
        ${QUANTILE_M_OUT} \
        ${RESULTS_DIR}/${TRACK_NAME} \
        ${QUANTILE_H2G_OUT}

    if [ $? -eq 0 ]; then
        echo "  ✓ Quantile h2g computation complete"
        echo "  Results saved to: ${QUANTILE_H2G_OUT}"
    else
        echo "  ✗ Quantile h2g computation failed"
        continue
    fi

done < ${TRACK_LIST}

echo ""
echo "========================================================================"
echo "Quantile enrichment analysis complete!"
echo "========================================================================"
echo ""
echo "Results directory: ${QUANTILE_DIR}"
echo ""
echo "Output files:"
echo "  - Quantile M files: ${QUANTILE_DIR}/*.quantile_M.txt"
echo "  - Quantile h2g files: ${QUANTILE_DIR}/*.quantile_h2g.txt"
echo ""

# Generate summary
echo "Generating summary plot data..."
SUMMARY_FILE="${QUANTILE_DIR}/summary_enrichment.txt"

echo -e "Track\tQuantile\th2g\th2g_se\tprop_h2g\tprop_h2g_se\tenr\tenr_se\tenr_pval" > ${SUMMARY_FILE}

for h2g_file in ${QUANTILE_DIR}/*.quantile_h2g.txt; do
    if [ -f "${h2g_file}" ]; then
        track_name=$(basename "${h2g_file}" .quantile_h2g.txt)
        tail -n +2 "${h2g_file}" | awk -v track="${track_name}" 'BEGIN{q=1}{print track"\t"q"\t"$0; q++}'
    fi
done >> ${SUMMARY_FILE}

echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "You can visualize the enrichment with:"
echo "  - Plot prop_h2g vs quantile to see heritability distribution"
echo "  - Plot enr vs quantile to see enrichment across quantiles"
echo ""
