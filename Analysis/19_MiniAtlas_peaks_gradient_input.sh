#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MACHINES=(shannon euler neumann turing)

WORKDIR=/share/vault/Users/gz2294/BICAN
PYTHON=/share/vault/Users/gz2294/miniconda3/envs/BICAN/bin/python
EXP_NAME=full_finetune_original_loss_celltype_head_dim8_linear_full_atlas
CHK=17
LRTEST_DIR=Data/source/MiniAtlas_LRtest
BED_DIR=Analysis/gradient_input_miniatlas
BED_OUT=${BED_DIR}/regions.bed
LOG_DIR=${BED_DIR}/logs

mkdir -p "$WORKDIR/$BED_DIR"
mkdir -p "$WORKDIR/$LOG_DIR"

# ---------------------------------------------------------------------------
# Step 1: Generate cross-product BED file from LRtest CSVs
# ---------------------------------------------------------------------------
echo "=== Generating BED file from LRtest CSVs ==="
> "$WORKDIR/$BED_OUT"

for f in "$WORKDIR/$LRTEST_DIR"/K27ac_hba_ccre_LR_*.csv; do
    # Extract cell type from filename: K27ac_hba_ccre_LR_{CT}.csv
    CT=$(basename "$f" .csv | sed 's/^K27ac_hba_ccre_LR_//')
    TRIAL="MiniAtlas-${CT}_K27Ac"
    # Parse CSV: skip header, extract "feature name" (col 1) as chr:start-end
    awk -F',' -v trial="$TRIAL" 'NR > 1 {
        split($1, a, ":");
        chr = a[1];
        split(a[2], b, "-");
        start = b[1];
        end = b[2];
        name = chr ":" start "-" end;
        print chr "\t" start "\t" end "\t" name "\t" trial;
    }' "$f"
done > "$WORKDIR/$BED_OUT"

N_REGIONS=$(wc -l < "$WORKDIR/$BED_OUT")
echo "  Generated ${N_REGIONS} region-trial pairs in ${BED_OUT}"

# ---------------------------------------------------------------------------
# Step 2: Probe GPU counts on each machine
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

echo "Total GPUs: $TOTAL_GPUS"

# ---------------------------------------------------------------------------
# Step 3: Split BED file into per-machine chunks (proportional to GPU count)
# ---------------------------------------------------------------------------
echo "=== Splitting BED into per-machine chunks ==="
OFFSET=0
declare -A MACHINE_BED

for m in "${ACTIVE_MACHINES[@]}"; do
    N_GPUS=${MACHINE_N_GPUS[$m]}
    CHUNK_SIZE=$(( (N_REGIONS * N_GPUS + TOTAL_GPUS - 1) / TOTAL_GPUS ))
    # Ensure we don't exceed remaining lines
    REMAINING=$((N_REGIONS - OFFSET))
    if [ "$CHUNK_SIZE" -gt "$REMAINING" ]; then
        CHUNK_SIZE=$REMAINING
    fi
    CHUNK_FILE="${BED_DIR}/regions_${m}.bed"
    MACHINE_BED[$m]=$CHUNK_FILE
    sed -n "$((OFFSET + 1)),$((OFFSET + CHUNK_SIZE))p" "$WORKDIR/$BED_OUT" > "$WORKDIR/$CHUNK_FILE"
    CHUNK_ACTUAL=$(wc -l < "$WORKDIR/$CHUNK_FILE")
    echo "  $m: ${CHUNK_ACTUAL} regions (${N_GPUS} GPUs) -> ${CHUNK_FILE}"
    OFFSET=$((OFFSET + CHUNK_SIZE))
done

# ---------------------------------------------------------------------------
# Step 4: Launch workers on each machine via SSH
# ---------------------------------------------------------------------------
launch_on_machine() {
    # Args: machine workdir python exp_name chk bed_file n_gpus logdir
    ssh "$1" "bash -s" -- "$@" <<'REMOTE'
        MACHINE=$1
        WORKDIR=$2
        PYTHON=$3
        EXP_NAME=$4
        CHK=$5
        BED_FILE=$6
        N_GPUS=$7
        LOG_DIR=$8

        cd "$WORKDIR"
        LOG="${LOG_DIR}/machine_${MACHINE}.log"
        echo "[$(hostname)] Starting ${N_GPUS} GPU(s), bed=${BED_FILE}  (log: ${LOG})"

        "$PYTHON" Analysis/02_motif_interpretation_gradient_input.py \
            --region_bed "$BED_FILE" \
            --exp_name   "$EXP_NAME" \
            --chk        "$CHK" \
            --processor  gpu \
            --num_processes "$N_GPUS" \
            --save_raw \
            --use_head   regression \
            > "$LOG" 2>&1
REMOTE
}

echo ""
echo "=== Launching workers ==="
SSH_PIDS=()
SSH_MACHINES=()

for m in "${ACTIVE_MACHINES[@]}"; do
    N=${MACHINE_N_GPUS[$m]}
    BED_FILE=${MACHINE_BED[$m]}
    echo "  $m: ${N} GPU(s), bed=${BED_FILE}"
    launch_on_machine \
        "$m" "$WORKDIR" "$PYTHON" "$EXP_NAME" "$CHK" "$BED_FILE" "$N" "$LOG_DIR" \
        &
    SSH_PIDS+=($!)
    SSH_MACHINES+=("$m")
done

# ---------------------------------------------------------------------------
# Step 5: Wait and report
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for all workers ==="
FAILED=0
for i in "${!SSH_PIDS[@]}"; do
    m=${SSH_MACHINES[$i]}
    if ! wait "${SSH_PIDS[$i]}"; then
        echo "ERROR: workers on $m failed — check ${LOG_DIR}/machine_${m}.log"
        FAILED=1
    else
        echo "  $m: done"
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "All workers finished successfully."
    echo "Results in: Res/${EXP_NAME}/analysis_${CHK}/"
else
    echo "" >&2
    echo "One or more workers failed. Check logs in ${WORKDIR}/${LOG_DIR}/" >&2
    exit 1
fi
