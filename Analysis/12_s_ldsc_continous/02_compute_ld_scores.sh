#!/bin/bash
#
# Compute LD scores for each continuous annotation
#
# This script:
# 1. Loops through all annotation directories
# 2. For each annotation and chromosome, computes LD scores using LDSC
# 3. Uses HapMap3 SNPs and 1cM window
#
# File Organization:
# - annotations_by_trait/  : Trait-specific annotation directories (115 traits)
# - results_by_trait/      : Trait-specific results directories (115 traits)
# - quantile_results_by_trait/ : Trait-specific quantile results (114 traits)
# - track_lists/          : Track list and track-chr list files (229 files total)
# - archive/              : Old working directories (annotations/, results/, quantile_results/)
#
# Usage:
#   bash 02_compute_ld_scores.sh -t <trait_name>
#   or
#   bash 02_compute_ld_scores.sh -a <annot_dir>
#
# Arguments:
#   -t, --trait TRAIT_NAME       Trait name (constructs path automatically)
#   -a, --annot-dir ANNOT_DIR    Custom annotation directory path
#   -h, --help                   Show this help message
#
# Example:
#   bash 02_compute_ld_scores.sh -t PGC_Nature_2014_Schizophrenia_fullinfo.sumstats
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments
TRAIT_NAME=""
ANNOT_DIR=""

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --trait TRAIT_NAME       Trait name (constructs path automatically)"
            echo "  -a, --annot-dir ANNOT_DIR    Custom annotation directory path"
            echo "  -h, --help                   Show this help message"
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

# Construct ANNOT_DIR from TRAIT_NAME if not provided directly
if [ -z "${ANNOT_DIR}" ]; then
    if [ -z "${TRAIT_NAME}" ]; then
        echo "ERROR: Either --trait or --annot-dir must be specified"
        echo "Run '$0 --help' for usage information"
        exit 1
    fi
    ANNOT_DIR="${SCRIPT_DIR}/annotations_by_trait/annotations_${TRAIT_NAME}"
fi

# Check if annotation directory exists
if [ ! -d "${ANNOT_DIR}" ]; then
    echo "ERROR: Annotation directory not found: ${ANNOT_DIR}"
    echo ""
    if [ ! -z "${TRAIT_NAME}" ]; then
        echo "Available traits in annotations_by_trait/:"
        ls "${SCRIPT_DIR}/annotations_by_trait/" | sed 's/annotations_/  /' | head -10
        echo "  ..."
    fi
    exit 1
fi
REF_DIR="${BASE_DIR}/Analysis/12_ldsc/reference"
LDSC="${REF_DIR}/ldsc/ldsc.py"
PLINK_DIR="${REF_DIR}/1000G_EUR_Phase3_plink"
WEIGHTS_DIR="${REF_DIR}/weights_hm3_no_hla"

# Check if annotation directory exists
if [ ! -d "${ANNOT_DIR}" ]; then
    echo "Error: Annotation directory not found: ${ANNOT_DIR}"
    echo "Please run 01_create_annotation_files.py first"
    exit 1
fi

# Get list of HapMap3 SNPs
HAPMAP3_SNPS="${WEIGHTS_DIR}/weights.hm3_noMHC.1.l2.ldscore.gz"
if [ ! -f "${HAPMAP3_SNPS}" ]; then
    echo "Error: HapMap3 weights file not found"
    exit 1
fi

# Create a list of HapMap3 SNPs (needed for --print-snps)
HAPMAP3_LIST="${BASE_DIR}/Analysis/12_s_ldsc_continous/listHM3.txt"
if [ ! -f "${HAPMAP3_LIST}" ]; then
    echo "Creating HapMap3 SNP list..."
    for chr in {1..22}; do
        zcat "${WEIGHTS_DIR}/weights.hm3_noMHC.${chr}.l2.ldscore.gz" | tail -n +2 | cut -f2
    done > "${HAPMAP3_LIST}"
    echo "Created ${HAPMAP3_LIST}"
fi

echo "========================================================================"
echo "Computing LD scores for continuous annotations"
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

    # Loop through chromosomes
    for chr in {1..22}; do
        annot_file="${track_dir}${track_name}.${chr}.annot.gz"

        # Check if annotation file exists
        if [ ! -f "${annot_file}" ]; then
            echo "  Warning: Annotation file not found for chr${chr}, skipping"
            continue
        fi

        # Output files will be created with this prefix
        output_prefix="${track_dir}${track_name}.${chr}"

        # Skip if already computed
        if [ -f "${output_prefix}.l2.ldscore.gz" ]; then
            echo "  chr${chr}: Already computed, skipping"
            continue
        fi

        echo "  chr${chr}: Computing LD scores..."

        # Run LDSC to compute LD scores
        python ${LDSC} \
            --l2 \
            --bfile ${PLINK_DIR}/1000G.EUR.QC.${chr} \
            --ld-wind-cm 1 \
            --annot ${annot_file} \
            --out ${output_prefix} \
            --print-snps ${HAPMAP3_LIST} \
            2>&1 | grep -E "(SNPs|annotations|Reading|After)" || true

        # Check if output was created
        if [ -f "${output_prefix}.l2.ldscore.gz" ]; then
            echo "  chr${chr}: ✓ Complete"
        else
            echo "  chr${chr}: ✗ Failed"
        fi
    done

    echo "  Track ${track_name} complete"
done

echo ""
echo "========================================================================"
echo "LD score computation complete!"
echo "========================================================================"
