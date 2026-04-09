#!/bin/bash
#SBATCH --job-name=eqtl_overlap
#SBATCH --output=logs/eqtl_overlap_%A_%a.out
#SBATCH --error=logs/eqtl_overlap_%A_%a.err
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Thresholds: 5, 10, 15, ..., 100
THRESHOLDS=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
TOP_PCT=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

TRAIT="${1:-Schizophrenia_fullinfo.sumstats}"
PROJECT_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN"
cd "$PROJECT_DIR"

echo "Running top_pct=${TOP_PCT}% for trait=${TRAIT}"

python Analysis/12_s_ldsc_continous/06_overlap_eQTL.py \
    --trait "$TRAIT" \
    --top-pct "$TOP_PCT"

echo "Done top_pct=${TOP_PCT}%"
