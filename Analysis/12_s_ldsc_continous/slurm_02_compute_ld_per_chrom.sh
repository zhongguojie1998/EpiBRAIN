#!/bin/bash
#SBATCH --job-name=ldsc_ld
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/12_s_ldsc_continous/logs/ld_scores_%A_%a.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/12_s_ldsc_continous/logs/ld_scores_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

#
# SLURM job script to compute LD scores for a single track-chromosome combination
# This is designed to be run as an array job (1 job per track per chromosome)
# Array size: 1 to (NUM_TRACKS * 22)
#

set -e

# Activate conda environment
source /gpfs/commons/home/guojiezhong/miniconda3/etc/profile.d/conda.sh
conda activate ldsc

# Configuration
BASE_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN"
SCRIPT_DIR="${BASE_DIR}/Analysis/12_s_ldsc_continous"

# Use environment variables if set (passed from main script), otherwise require them
# ANNOT_DIR and TRACK_CHR_LIST must be provided by the calling script (run_pipeline_ultra_parallel.sh)
if [ -z "${ANNOT_DIR}" ]; then
    echo "ERROR: ANNOT_DIR must be set (usually by run_pipeline_ultra_parallel.sh)"
    exit 1
fi
if [ -z "${TRACK_CHR_LIST}" ]; then
    echo "ERROR: TRACK_CHR_LIST must be set (usually by run_pipeline_ultra_parallel.sh)"
    exit 1
fi
if [ -z "${TRAIT_NAME}" ]; then
    TRAIT_NAME="unknown"
fi

REF_DIR="${BASE_DIR}/Analysis/12_ldsc/reference"
LDSC="${REF_DIR}/ldsc/ldsc.py"
PLINK_DIR="${REF_DIR}/1000G_EUR_Phase3_plink"
WEIGHTS_DIR="${REF_DIR}/weights_hm3_no_hla"
HAPMAP3_LIST="${SCRIPT_DIR}/listHM3.txt"

# Get track name and chromosome from array index
# The track-chromosome mapping file should be created before submitting the array job

if [ ! -f "${TRACK_CHR_LIST}" ]; then
    echo "Error: Track-chromosome list not found: ${TRACK_CHR_LIST}"
    exit 1
fi

# Read the track name and chromosome for this array task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${TRACK_CHR_LIST})
TRACK_NAME=$(echo ${LINE} | cut -d' ' -f1)
CHR=$(echo ${LINE} | cut -d' ' -f2)

if [ -z "${TRACK_NAME}" ] || [ -z "${CHR}" ]; then
    echo "Error: Could not parse track name and chromosome for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "========================================================================"
echo "Computing LD scores"
echo "Trait: ${TRAIT_NAME}"
echo "Track: ${TRACK_NAME}"
echo "Chromosome: ${CHR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"

TRACK_DIR="${ANNOT_DIR}/${TRACK_NAME}"
annot_file="${TRACK_DIR}/${TRACK_NAME}.${CHR}.annot.gz"

# Check if annotation file exists
if [ ! -f "${annot_file}" ]; then
    echo "Error: Annotation file not found: ${annot_file}"
    exit 1
fi

# Output files will be created with this prefix
output_prefix="${TRACK_DIR}/${TRACK_NAME}.${CHR}"

# Skip if already computed
if [ -f "${output_prefix}.l2.ldscore.gz" ]; then
    echo "Already computed, skipping"
    exit 0
fi

echo "Computing LD scores..."

# Run LDSC to compute LD scores
python ${LDSC} \
    --l2 \
    --bfile ${PLINK_DIR}/1000G.EUR.QC.${CHR} \
    --ld-wind-cm 1 \
    --annot ${annot_file} \
    --out ${output_prefix} \
    --print-snps ${HAPMAP3_LIST}

# Check if output was created
if [ -f "${output_prefix}.l2.ldscore.gz" ]; then
    echo "✓ Complete"
else
    echo "✗ Failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "LD score computation complete for ${TRACK_NAME} chr${CHR}"
echo "========================================================================"
