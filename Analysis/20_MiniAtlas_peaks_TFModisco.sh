#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# TF-MoDISco pipeline for MiniAtlas peaks gradient x input
#
# Phase 1: prepare  — Build NPZ + BED per cell type (SLURM on nova)
# Phase 2: modisco  — Run modisco motifs per cell type (SLURM on nova)
# Phase 3: postprocess — Extract seqlets, CWMs, TOMTOM (SLURM on nova)
# Phase 4: aggregate  — Cluster CWMs across all cell types, visualize
# ---------------------------------------------------------------------------

# Config
WORKDIR=/share/vault/Users/gz2294/BICAN
PYTHON=/share/vault/Users/gz2294/miniconda3/envs/BICAN/bin/python
MODISCO=/share/vault/Users/gz2294/miniconda3/envs/BICAN/bin/modisco
EXP_NAME=full_finetune_original_loss_celltype_head_dim8_linear_full_atlas
CHK=17
CENTER_BP=500
NUM_SEQLETS=20000
MEME_DB=Data/source/meme-5.4.1/motif_databases/HOCOMOCO/H12CORE_meme_format.meme

SCRIPT=Analysis/20_MiniAtlas_peaks_TFModisco.py
BASE_DIR=Res/${EXP_NAME}/analysis_${CHK}/modisco_miniatlas
LOG_DIR=${BASE_DIR}/logs

# Parse phase argument
PHASE=${1:-all}
echo "=== TF-MoDISco Pipeline — Phase: ${PHASE} ==="

mkdir -p "$WORKDIR/$LOG_DIR"

# ---------------------------------------------------------------------------
# Phase 1: Prepare NPZ files via SLURM
# ---------------------------------------------------------------------------
run_prepare() {
    echo ""
    echo "=== Phase 1: Prepare NPZ files via SLURM ==="

    # Discover all cell types from .pt filenames
    PT_DIR="Res/${EXP_NAME}/analysis_${CHK}/raw_data/interp_gradient_input"
    ALL_CTS=$(ls "$WORKDIR/$PT_DIR" | grep '_grad_input\.pt$' \
        | sed 's/_grad_input\.pt$//' \
        | grep -oP 'MiniAtlas-\K[^_]+(?=_K27Ac$)' \
        | sort -u \
        | paste -sd ',')
    echo "Cell types: $ALL_CTS"
    IFS=',' read -ra CT_ARRAY <<< "$ALL_CTS"
    N_CTS=${#CT_ARRAY[@]}
    echo "Total cell types: $N_CTS"

    # Submit one SLURM job per cell type
    echo "--- Submitting prepare jobs to nova ---"
    JOB_IDS=()

    for CT in "${CT_ARRAY[@]}"; do
        JOB_NAME="prepare_${CT}"
        LOG_FILE="${WORKDIR}/${LOG_DIR}/prepare_${CT}.log"

        # Count peaks for this cell type to size memory
        N_PEAKS=$(ls "$WORKDIR/$PT_DIR" | grep -c "_MiniAtlas-${CT}_K27Ac_grad_input\\.pt$" || true)

        # Memory: output arrays ~ n_peaks * 500 * 4 * 4bytes * 2 (attrib+onehot)
        # Plus overhead. Each .pt loaded one at a time (~8MB transient).
        if [ "$N_PEAKS" -gt 20000 ]; then
            MEM="16G"
        elif [ "$N_PEAKS" -gt 5000 ]; then
            MEM="8G"
        else
            MEM="4G"
        fi

        echo "  $CT ($N_PEAKS peaks, ${MEM} mem)"
        JOB_ID=$(ssh nova "cd $WORKDIR && sbatch --parsable \
            --job-name=$JOB_NAME \
            --cpus-per-task=1 \
            --mem=$MEM \
            --time=4:00:00 \
            --output=$LOG_FILE \
            --error=$LOG_FILE \
            --wrap='$PYTHON $SCRIPT prepare \
                --exp_name $EXP_NAME --chk $CHK \
                --celltype $CT --center_bp $CENTER_BP --n_jobs 1'")
        JOB_IDS+=("$JOB_ID")
        echo "    -> job $JOB_ID"
    done

    echo ""
    echo "Phase 1: ${#JOB_IDS[@]} prepare jobs submitted to nova."
    echo "Monitor with: ssh nova squeue -u \$USER"
    echo "When all jobs complete, run: bash $0 modisco"
}

# ---------------------------------------------------------------------------
# Phase 2: Run modisco via SLURM on nova
# ---------------------------------------------------------------------------
run_modisco() {
    echo ""
    echo "=== Phase 2: Run modisco via SLURM ==="

    # Find cell types with prepared NPZ files
    CELLTYPES=()
    for d in "$WORKDIR/$BASE_DIR"/*/; do
        CT=$(basename "$d")
        if [ -f "$d/attributions.npz" ] && [ ! -f "$d/modisco_results.h5" ]; then
            CELLTYPES+=("$CT")
        fi
    done

    if [ ${#CELLTYPES[@]} -eq 0 ]; then
        echo "No cell types need modisco (all done or none prepared)."
        return
    fi
    echo "Submitting ${#CELLTYPES[@]} modisco jobs to nova"

    # Count peaks per cell type for memory allocation
    for CT in "${CELLTYPES[@]}"; do
        CT_DIR="$WORKDIR/$BASE_DIR/$CT"
        N_PEAKS=$(wc -l < "$CT_DIR/peaks.bed")

        # Memory: 200G for >20K peaks, 100G for <5K, 150G otherwise
        if [ "$N_PEAKS" -gt 20000 ]; then
            MEM="200G"
        elif [ "$N_PEAKS" -lt 5000 ]; then
            MEM="100G"
        else
            MEM="150G"
        fi

        JOB_NAME="modisco_${CT}"
        LOG_FILE="${WORKDIR}/${LOG_DIR}/modisco_${CT}.log"

        echo "  $CT ($N_PEAKS peaks, ${MEM} mem)"
        ssh nova "cd $WORKDIR && sbatch \
            --job-name=$JOB_NAME \
            --cpus-per-task=8 \
            --mem=$MEM \
            --time=24:00:00 \
            --output=$LOG_FILE \
            --error=$LOG_FILE \
            --wrap='cd $CT_DIR && $MODISCO motifs \
                -s sequences.npz -a attributions.npz \
                -n $NUM_SEQLETS -w $CENTER_BP \
                -o modisco_results.h5 -v'"
    done

    echo ""
    echo "Phase 2: ${#CELLTYPES[@]} jobs submitted to nova."
    echo "Monitor with: ssh nova squeue -u \$USER"
    echo "When all jobs complete, run: bash $0 postprocess"
}

# ---------------------------------------------------------------------------
# Phase 3: Post-process
# ---------------------------------------------------------------------------
run_postprocess() {
    echo ""
    echo "=== Phase 3: Post-process ==="

    "$PYTHON" "$SCRIPT" postprocess \
        --exp_name "$EXP_NAME" --chk "$CHK" \
        --meme_db "$MEME_DB" --center_bp "$CENTER_BP" --n_jobs 8

    echo "Phase 3 complete."
}

# ---------------------------------------------------------------------------
# Phase 4: Aggregate CWMs across cell types
# ---------------------------------------------------------------------------
run_aggregate() {
    echo ""
    echo "=== Phase 4: Aggregate and cluster CWMs ==="

    shift || true  # consume 'aggregate' from $@
    "$PYTHON" "$SCRIPT" aggregate \
        --exp_name "$EXP_NAME" --chk "$CHK" "$@"

    echo "Phase 4 complete."
    echo "Outputs in: ${BASE_DIR}/aggregate/"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "$PHASE" in
    prepare)
        run_prepare
        ;;
    modisco)
        run_modisco
        ;;
    postprocess)
        run_postprocess
        ;;
    aggregate)
        run_aggregate "$@"
        ;;
    all)
        run_prepare
        run_modisco
        echo ""
        echo "Waiting for SLURM jobs to complete before postprocessing."
        echo "Run 'bash $0 postprocess' when all modisco jobs finish."
        echo "Then run: bash $0 aggregate"
        ;;
    *)
        echo "Usage: $0 {prepare|modisco|postprocess|aggregate|all}" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
