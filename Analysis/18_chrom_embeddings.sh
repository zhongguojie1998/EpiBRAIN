#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MACHINES=(shannon euler neumann turing)

WORKDIR=/share/vault/Users/gz2294/BICAN
PYTHON=/share/vault/Users/gz2294/miniconda3/envs/BICAN/bin/python
BED=Data/source/MiniAtlas_ATAC_peak/merged_all_peaks.bed
OUTPUT=Analysis/chrom_emb/merged_peaks.h5
CHECKPOINT=Chk/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/chk_epoch_17.pt
CONFIG=logs/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/overall_setting.yaml
LOG_DIR=Analysis/chrom_emb/logs

mkdir -p "$WORKDIR/$LOG_DIR"

# ---------------------------------------------------------------------------
# Step 1: Probe GPU counts on each machine
# ---------------------------------------------------------------------------
declare -A MACHINE_N_GPUS
ACTIVE_MACHINES=()
TOTAL_GPUS=0

echo "=== Probing machines ==="
for m in "${MACHINES[@]}"; do
    count=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$m" \
        "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l" \
        2>/dev/null || echo 0)
    count=$(echo "$count" | tr -d '[:space:]')
    if [ "${count:-0}" -gt 0 ]; then
        MACHINE_N_GPUS[$m]=$count
        ACTIVE_MACHINES+=("$m")
        echo "  $m: ${count} GPU(s)"
        TOTAL_GPUS=$((TOTAL_GPUS + count))
    else
        echo "  $m: unreachable or no GPUs — skipping"
    fi
done

if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "No GPUs found on any machine." >&2
    exit 1
fi

echo "Total world_size: $TOTAL_GPUS"

# ---------------------------------------------------------------------------
# Step 2: Launch workers on each machine via SSH
#
# Each SSH session receives its parameters as positional arguments ($1..$9)
# so that the heredoc can be quoted (no local variable expansion leaks).
# ---------------------------------------------------------------------------
launch_on_machine() {
    # Args: machine start_rank n_gpus world_size workdir bed output ckpt cfg logdir
    ssh "$1" "bash -s" -- "$@" <<'REMOTE'
        MACHINE=$1
        START_RANK=$2
        N_GPUS=$3
        WORLD_SIZE=$4
        WORKDIR=$5
        BED=$6
        OUTPUT=$7
        CHECKPOINT=$8
        CONFIG=$9
        LOG_DIR=${10}
        PYTHON=${11}

        cd "$WORKDIR"
        PIDS=()
        for ((g=0; g<N_GPUS; g++)); do
            RANK=$((START_RANK + g))
            LOG="${LOG_DIR}/rank${RANK}.log"
            echo "[$(hostname)] Rank ${RANK} -> cuda:${g}  (log: ${LOG})"
            "$PYTHON" Analysis/18_chrom_embeddings.py \
                --bed        "$BED"        \
                --output     "$OUTPUT"     \
                --checkpoint "$CHECKPOINT" \
                --config     "$CONFIG"     \
                --device     "cuda:${g}"   \
                --rank       "$RANK"       \
                --world_size "$WORLD_SIZE" \
                > "$LOG" 2>&1 &
            PIDS+=($!)
        done

        FAILED=0
        for i in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$i]}"; then
                echo "[$(hostname)] ERROR: rank $((START_RANK + i)) failed — see ${LOG_DIR}/rank$((START_RANK + i)).log"
                FAILED=1
            fi
        done
        exit $FAILED
REMOTE
}

echo ""
echo "=== Launching workers ==="
RANK_OFFSET=0
SSH_PIDS=()
SSH_MACHINES=()

for m in "${ACTIVE_MACHINES[@]}"; do
    N=${MACHINE_N_GPUS[$m]}
    echo "  $m: ranks ${RANK_OFFSET}..$((RANK_OFFSET + N - 1))"
    launch_on_machine \
        "$m" "$RANK_OFFSET" "$N" "$TOTAL_GPUS" \
        "$WORKDIR" "$BED" "$OUTPUT" "$CHECKPOINT" "$CONFIG" "$LOG_DIR" "$PYTHON" \
        &
    SSH_PIDS+=($!)
    SSH_MACHINES+=("$m")
    RANK_OFFSET=$((RANK_OFFSET + N))
done

# ---------------------------------------------------------------------------
# Step 3: Wait and report
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for all workers ==="
FAILED=0
for i in "${!SSH_PIDS[@]}"; do
    m=${SSH_MACHINES[$i]}
    if ! wait "${SSH_PIDS[$i]}"; then
        echo "ERROR: workers on $m failed — check ${LOG_DIR}/rank*.log"
        FAILED=1
    else
        echo "  $m: done"
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "All workers finished successfully."
    echo "Output shards: ${WORKDIR}/${OUTPUT%.h5}_rank*.h5  (per cell type)"

    # ---------------------------------------------------------------------------
    # Step 4: Merge rank shards into per-cell-type files
    # ---------------------------------------------------------------------------
    echo ""
    echo "=== Merging rank shards ==="
    OUTPUT_DIR=$(dirname "${WORKDIR}/${OUTPUT}")
    "$PYTHON" "${WORKDIR}/Analysis/18_chrom_embeddings_merge.py" \
        --input_dir  "Analysis/chrom_emb" \
        --output_dir "Analysis/chrom_emb" \
        --world_size "$TOTAL_GPUS" \
        --n_workers  8
    echo "Merge complete."
else
    echo "" >&2
    echo "One or more workers failed. Check logs in ${WORKDIR}/${LOG_DIR}/" >&2
    exit 1
fi
