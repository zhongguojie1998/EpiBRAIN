#!/bin/bash
# Submit ABC analysis jobs for all 6 K27Ac splits

EXP_NAME="basal_ganglia_miniatlas_drop_celltype_v1"
CHK_NUM="best_valid_loss"

for i in {0..5}; do
    BED_FILE="Data/source/ABC/abc_k27ac_split_${i}.bed"

    echo "Submitting job for split ${i}: ${BED_FILE}"
    sbatch Analysis/09_ABC_run.slurm "$BED_FILE" "$EXP_NAME" "$CHK_NUM"

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All 6 jobs submitted!"
