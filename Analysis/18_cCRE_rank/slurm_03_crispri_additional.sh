#!/bin/bash
# Element-centred in-silico CRISPRi screen (all genes in the 524kb locus × all cell
# types) — one SLURM array task per chunk of elements. Any GPU in the `gpu`
# partition is fine (l40s or b6k), so no --gres type constraint is set.
#
# 1. build chunks (CPU; also caches gene_db.pkl):
#      PY=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python3
#      $PY Analysis/18_cCRE_rank/04_crispri_element_screen.py build \
#          --exp_name <EXP> --chk <CHK> --chunk_size 2000
# 2. submit (N = n_chunks printed by build; %40 throttles concurrent tasks):
#      sbatch --array=1-N%40 Analysis/18_cCRE_rank/slurm_03_crispri_additional.sh \
#          --exp_name <EXP> --chk <CHK>
# 3. optional single-file merge:
#      $PY Analysis/18_cCRE_rank/04_crispri_element_screen.py merge --exp_name <EXP> --chk <CHK>
#SBATCH --job-name=crispri_elem
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=Analysis/18_cCRE_rank/logs/03_crispri_element/%A_%a.out
#SBATCH --error=Analysis/18_cCRE_rank/logs/03_crispri_element/%A_%a.err

set -euo pipefail
mkdir -p Analysis/18_cCRE_rank/logs/03_crispri_element

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

echo "[$(date)] crispri-element chunk ${SLURM_ARRAY_TASK_ID} exp=${EXP_NAME} chk=${CHK} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$PY Analysis/18_cCRE_rank/04_crispri_element_screen.py run \
    --exp_name "$EXP_NAME" --chk "$CHK" \
    --chunk_id "${SLURM_ARRAY_TASK_ID}" --device cuda:0
echo "[$(date)] done chunk ${SLURM_ARRAY_TASK_ID}"
