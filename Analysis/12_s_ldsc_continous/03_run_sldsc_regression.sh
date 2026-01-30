#!/bin/bash
#
# Run S-LDSC regression for each continuous annotation
#
# This script:
# 1. Loops through all annotation tracks
# 2. Runs LDSC heritability analysis with baseline + continuous annotation
# 3. Generates results files with regression coefficients
#
# File Organization:
# - annotations_by_trait/  : Trait-specific annotation directories (115 traits)
# - results_by_trait/      : Trait-specific results directories (115 traits)
# - quantile_results_by_trait/ : Trait-specific quantile results (114 traits)
# - track_lists/          : Track list and track-chr list files (229 files total)
# - archive/              : Old working directories (annotations/, results/, quantile_results/)
#
# Usage:
#   bash 03_run_sldsc_regression.sh -t <trait_name>
#   or
#   bash 03_run_sldsc_regression.sh -a <annot_dir> -r <results_dir>
#
# Arguments:
#   -t, --trait TRAIT_NAME         Trait name (constructs paths automatically)
#   -a, --annot-dir ANNOT_DIR      Custom annotation directory path
#   -r, --results-dir RESULTS_DIR  Custom results directory path
#   -h, --help                     Show this help message
#
# Example:
#   bash 03_run_sldsc_regression.sh -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
TRAIT_NAME=""
ANNOT_DIR=""
RESULTS_DIR=""

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --trait TRAIT_NAME         Trait name (constructs paths automatically)"
            echo "  -a, --annot-dir ANNOT_DIR      Custom annotation directory path"
            echo "  -r, --results-dir RESULTS_DIR  Custom results directory path"
            echo "  -h, --help                     Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats"
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
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ldsc

# Configuration
BASE_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN"

# Construct paths from TRAIT_NAME if not provided directly
if [ -z "${ANNOT_DIR}" ] || [ -z "${RESULTS_DIR}" ]; then
    if [ -z "${TRAIT_NAME}" ]; then
        echo "ERROR: Either --trait or (--annot-dir and --results-dir) must be specified"
        echo "Run '$0 --help' for usage information"
        exit 1
    fi
    ANNOT_DIR="${SCRIPT_DIR}/annotations_by_trait/annotations_${TRAIT_NAME}"
    RESULTS_DIR="${SCRIPT_DIR}/results_by_trait/results_${TRAIT_NAME}"
fi

# Check if annotation directory exists
if [ ! -d "${ANNOT_DIR}" ]; then
    echo "ERROR: Annotation directory not found: ${ANNOT_DIR}"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"
REF_DIR="${BASE_DIR}/Analysis/12_ldsc/reference"
LDSC="${REF_DIR}/ldsc/ldsc.py"
WEIGHTS_DIR="${REF_DIR}/weights_hm3_no_hla"
BASELINE_DIR="${REF_DIR}/1000G_EUR_Phase3_baseline"
FRQFILE_PREFIX="${REF_DIR}/1000G_EUR_Phase3_plink/1000G.EUR.QC"
SUMSTATS="${BASE_DIR}/Data/source/GWAS/processed/PGC_Nature_2014_Schizophrenia_fullinfo.sumstats.gz"

# Check required files/directories
if [ ! -d "${ANNOT_DIR}" ]; then
    echo "Error: Annotation directory not found: ${ANNOT_DIR}"
    echo "Please run 01_create_annotation_files.py first"
    exit 1
fi

if [ ! -f "${SUMSTATS}" ]; then
    echo "Error: GWAS summary statistics not found: ${SUMSTATS}"
    exit 1
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "========================================================================"
echo "Running S-LDSC regression for continuous annotations"
echo "GWAS: Schizophrenia (PGC Nature 2014)"
echo "========================================================================"

# Count total number of tracks
total_tracks=$(ls -d ${ANNOT_DIR}/*/ | wc -l)
track_num=0

# Loop through each annotation track directory
for track_dir in ${ANNOT_DIR}/*/; do
    track_num=$((track_num + 1))
    track_name=$(basename "${track_dir}")

    echo ""
    echo "Processing track ${track_num}/${total_tracks}: ${track_name}"
    echo "------------------------------------------------------------------------"

    # Check if LD scores exist for all chromosomes
    missing_ldscore=0
    for chr in {1..22}; do
        ldscore_file="${track_dir}${track_name}.${chr}.l2.ldscore.gz"
        if [ ! -f "${ldscore_file}" ]; then
            echo "  Warning: LD score file missing for chr${chr}"
            missing_ldscore=1
        fi
    done

    if [ ${missing_ldscore} -eq 1 ]; then
        echo "  Skipping track ${track_name}: Missing LD score files"
        echo "  Please run 02_compute_ld_scores.sh first"
        continue
    fi

    # Output file
    output_file="${RESULTS_DIR}/${track_name}"

    # Skip if already computed
    if [ -f "${output_file}.results" ]; then
        echo "  Already computed, skipping"
        continue
    fi

    echo "  Running S-LDSC regression..."

    # Run LDSC
    # Use baseline annotations + continuous annotation
    # --ref-ld-chr specifies baseline and custom annotation (comma-separated prefixes)
    # Note: Need to specify the prefix for each chromosome (LDSC will append .CHR.l2.ldscore.gz)
    python ${LDSC} \
        --h2 ${SUMSTATS} \
        --ref-ld-chr ${BASELINE_DIR}/baseline.,${track_dir}${track_name}. \
        --w-ld-chr ${WEIGHTS_DIR}/weights.hm3_noMHC. \
        --overlap-annot \
        --frqfile-chr ${FRQFILE_PREFIX}. \
        --out ${output_file} \
        --print-coefficients \
        2>&1 | tail -20

    # Check if output was created
    if [ -f "${output_file}.results" ]; then
        echo "  ✓ Complete - Results saved to ${output_file}.results"

        # Print key results
        echo ""
        echo "  Key results:"
        tail -n +2 "${output_file}.results" | grep "${track_name}" | \
            awk '{printf "    Coefficient: %.6e, SE: %.6e, Z-score: %.3f, P-value: %.3e\n", $2, $3, $4, $5}' || \
            echo "    (See ${output_file}.results for details)"
    else
        echo "  ✗ Failed"
    fi

    echo ""
done

echo ""
echo "========================================================================"
echo "S-LDSC regression complete!"
echo "Results directory: ${RESULTS_DIR}"
echo "========================================================================"

# Create summary table
echo ""
echo "Creating summary table..."
SUMMARY_FILE="${RESULTS_DIR}/summary_results.txt"

echo -e "Track\tCoefficient\tStdError\tZ_score\tP_value" > ${SUMMARY_FILE}

for result_file in ${RESULTS_DIR}/*.results; do
    if [ -f "${result_file}" ]; then
        track_name=$(basename "${result_file}" .results)
        # Extract the last line (custom annotation) from results
        tail -n 1 "${result_file}" | \
            awk -v track="${track_name}" '{printf "%s\t%s\t%s\t%s\t%s\n", track, $2, $3, $4, $5}' \
            >> ${SUMMARY_FILE}
    fi
done

if [ -f "${SUMMARY_FILE}" ]; then
    echo "Summary table created: ${SUMMARY_FILE}"
    echo ""
    echo "Top 10 results by Z-score:"
    head -1 ${SUMMARY_FILE}
    tail -n +2 ${SUMMARY_FILE} | sort -t$'\t' -k4 -rn | head -10
fi
