#!/bin/bash
#SBATCH --job-name=ldsc_reg
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/12_s_ldsc_continous/logs/regression_%A_%a.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/12_s_ldsc_continous/logs/regression_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

#
# SLURM job script to run S-LDSC regression for a single track
# This is designed to be run as an array job (1 job per track)
#

set -e

# Activate conda environment
source /gpfs/commons/home/guojiezhong/miniconda3/etc/profile.d/conda.sh
conda activate ldsc

# Configuration
BASE_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN"
SCRIPT_DIR="${BASE_DIR}/Analysis/12_s_ldsc_continous"

# Use environment variables if set (passed from main script), otherwise require them
# These must be provided by the calling script (run_pipeline_ultra_parallel.sh)
if [ -z "${ANNOT_DIR}" ]; then
    echo "ERROR: ANNOT_DIR must be set (usually by run_pipeline_ultra_parallel.sh)"
    exit 1
fi
if [ -z "${RESULTS_DIR}" ]; then
    echo "ERROR: RESULTS_DIR must be set (usually by run_pipeline_ultra_parallel.sh)"
    exit 1
fi
if [ -z "${TRACK_LIST}" ]; then
    echo "ERROR: TRACK_LIST must be set (usually by run_pipeline_ultra_parallel.sh)"
    exit 1
fi
if [ -z "${TRAIT_NAME}" ]; then
    TRAIT_NAME="unknown"
fi

REF_DIR="${BASE_DIR}/Analysis/12_ldsc/reference"
LDSC="${REF_DIR}/ldsc/ldsc.py"
WEIGHTS_DIR="${REF_DIR}/weights_hm3_no_hla"
BASELINE_DIR="${REF_DIR}/1000G_EUR_Phase3_baseline"
FRQFILE_PREFIX="${REF_DIR}/1000G_EUR_Phase3_plink/1000G.EUR.QC"
SUMSTATS="${BASE_DIR}/Analysis/12_ldsc/reference/GWAStraits/PGC.Nature.2014.Schizophrenia.sumstats.gz"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Get track name from array index
TRACK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${TRACK_LIST})

if [ -z "${TRACK}" ]; then
    echo "Error: Could not find track name for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "========================================================================"
echo "Running S-LDSC regression"
echo "Trait: ${TRAIT_NAME}"
echo "Track: ${TRACK}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"

TRACK_DIR="${ANNOT_DIR}/${TRACK}"

# Check if LD scores exist for all chromosomes
missing_ldscore=0
for chr in {1..22}; do
    ldscore_file="${TRACK_DIR}/${TRACK}.${chr}.l2.ldscore.gz"
    if [ ! -f "${ldscore_file}" ]; then
        echo "  Warning: LD score file missing for chr${chr}"
        missing_ldscore=1
    fi
done

if [ ${missing_ldscore} -eq 1 ]; then
    echo "  Error: Missing LD score files for track ${TRACK}"
    echo "  Please run LD score computation first"
    exit 1
fi

# Output file
output_file="${RESULTS_DIR}/${TRACK}"

# Skip if already computed (need both .results and .part_delete files)
if [ -f "${output_file}.results" ] && [ -f "${output_file}.part_delete" ]; then
    echo "  Already computed, skipping"
    exit 0
fi

echo "  Running S-LDSC regression..."

# Run LDSC
python ${LDSC} \
    --h2 ${SUMSTATS} \
    --ref-ld-chr ${BASELINE_DIR}/baseline.,${TRACK_DIR}/${TRACK}. \
    --w-ld-chr ${WEIGHTS_DIR}/weights. \
    --overlap-annot \
    --not-M-5-50 \
    --out ${output_file} \
    --print-coefficients \
    --print-delete-vals

# Check if output was created
if [ -f "${output_file}.results" ]; then
    echo "  ✓ Complete - Results saved to ${output_file}.results"

    # Print key results
    echo ""
    echo "  Key results:"
    tail -n +2 "${output_file}.results" | grep "${TRACK}" | \
        awk '{printf "    Coefficient: %.6e, SE: %.6e, Z-score: %.3f, P-value: %.3e\n", $2, $3, $4, $5}' || \
        echo "    (See ${output_file}.results for details)"
else
    echo "  ✗ Failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "S-LDSC regression complete for trait ${TRAIT_NAME}, track: ${TRACK}"
echo "========================================================================"
