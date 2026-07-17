#!/bin/bash
# In-silico CRISPRi cCRE→gene screen (all cell types) — one SLURM array task per chunk.
#
# 1. build chunks:
#      PY=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python3
#      $PY Analysis/18_cCRE_rank/02_crispri_cCRE_screen.py build \
#          --exp_name <EXP> --chk <CHK> --chunk_size 25
# 2. submit (N = n_chunks printed by build):
#      sbatch --array=1-N Analysis/18_cCRE_rank/slurm_02_crispri.sh --exp_name <EXP> --chk <CHK>
#SBATCH --job-name=crispri_cCRE
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=Analysis/18_cCRE_rank/logs/02_crispri/%A_%a.out
#SBATCH --error=Analysis/18_cCRE_rank/logs/02_crispri/%A_%a.err

set -euo pipefail
mkdir -p Analysis/18_cCRE_rank/logs/02_crispri

EXP_NAME=""; CHK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --chk)      CHK="$2";      shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$EXP_NAME" && -n "$CHK" ]] || { echo "need --exp_name and --chk" >&2; exit 2; }

PY=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python3

echo "[$(date)] crispri chunk ${SLURM_ARRAY_TASK_ID} exp=${EXP_NAME} chk=${CHK}"
$PY Analysis/18_cCRE_rank/02_crispri_cCRE_screen.py run \
    --exp_name "$EXP_NAME" --chk "$CHK" \
    --chunk_id "${SLURM_ARRAY_TASK_ID}" --device cuda:0
echo "[$(date)] done chunk ${SLURM_ARRAY_TASK_ID}"
