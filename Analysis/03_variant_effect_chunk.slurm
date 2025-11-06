#!/bin/bash
#SBATCH --job-name=var_eff_chunk
#SBATCH --output=logs/var_eff_chunk_%a.out
#SBATCH --error=logs/var_eff_chunk_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-6

# This script processes VCF chunks in parallel using SLURM array jobs
# Usage: sbatch Analysis/03_variant_effect_chunk.slurm

# Configuration
EXP_NAME="full_finetune_original_loss_celltype_head_dim8_linear"
CHK="20"
VCF_BASE="Data/source/Jang2025_SingleBrain/finemapped_variants"
LOG_BASE="./logs"
CHK_BASE="./Chk"
RES_BASE="./Res"

# Get chunk number from SLURM array task ID
CHUNK_NUM=${SLURM_ARRAY_TASK_ID}
VCF_FILE="${VCF_BASE}_chunk${CHUNK_NUM}.vcf"

echo "=========================================="
echo "Processing chunk ${CHUNK_NUM}"
echo "VCF file: ${VCF_FILE}"
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint: ${CHK}"
echo "=========================================="

# Run variant effect analysis
# Note: GPU mode automatically uses cuda:0 with num_processes=1
python3 Analysis/03_variant_effect.py \
  --vcf ${VCF_FILE} \
  --exp_name ${EXP_NAME} \
  --chk ${CHK} \
  --log_base ${LOG_BASE} \
  --chk_base ${CHK_BASE} \
  --res_base ${RES_BASE} \
  --processor gpu \
  --num_processes 1 \
  --use_head regression

echo "Chunk ${CHUNK_NUM} completed!"
