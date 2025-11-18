#!/bin/bash
#SBATCH --job-name=smooth_grad_diff
#SBATCH --output=smooth_grad_diff_%j.out
#SBATCH --error=smooth_grad_diff_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Usage: bash run_smooth_gradient_diff.sh <bed_file> [exp_name] [checkpoint]
# Example: bash run_smooth_gradient_diff.sh regions.bed my_experiment 100

# Check if BED file is provided
if [ -z "$1" ]; then
    echo "Error: BED file not provided"
    echo "Usage: bash run_smooth_gradient_diff.sh <bed_file> [exp_name] [checkpoint]"
    exit 1
fi

BED_FILE=$1
EXP_NAME=${2:-"your_experiment_name"}  # Default experiment name
CHK=${3:-"best"}  # Default checkpoint

# Parameters matching borzoi_satg_gene_smooth.py example
N_SAMPLES=16
SAMPLE_PROB_VALUES=(0.95 0.98)  # Array of sample_prob values to test
PSEUDO_COUNT=20.0
SAMPLE_VALUE=1.0  # 1.0 for uniform distribution
SAMPLE_SEED=42

# Directory structure
LOG_BASE="./logs"
CHK_BASE="./Chk"
RES_BASE="./Res"

# Processing parameters
PROCESSOR="gpu"
NUM_PROCESSES=1  # For GPU, typically use 1 process per GPU
NUM_THREADS=8
USE_HEAD="regression"

# Script location
PYTHON_SCRIPT="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/02_motif_bed_diff_interpretation_gradient_input_smooth.py"

# Check if BED file exists
if [ ! -f "$BED_FILE" ]; then
    echo "Error: BED file '$BED_FILE' not found"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found"
    exit 1
fi

# Create necessary directories
mkdir -p "${LOG_BASE}"
mkdir -p "${RES_BASE}/${EXP_NAME}/analysis_${CHK}/plot/interp_diff_smooth_gradient_input"
mkdir -p "${RES_BASE}/${EXP_NAME}/analysis_${CHK}/raw_data/interp_diff_smooth_gradient_input"
mkdir -p "${RES_BASE}/${EXP_NAME}/analysis_${CHK}/raw_data/label"

echo "=========================================="
echo "Running Smooth Gradient Differential Analysis"
echo "=========================================="
echo "BED file: $BED_FILE"
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CHK"
echo "N samples: $N_SAMPLES"
echo "Sample prob values: ${SAMPLE_PROB_VALUES[@]}"
echo "Pseudo count: $PSEUDO_COUNT"
echo "Sample value: $SAMPLE_VALUE"
echo "Processor: $PROCESSOR"
echo "=========================================="

# Loop over different sample_prob values
for SAMPLE_PROB in "${SAMPLE_PROB_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing with sample_prob = $SAMPLE_PROB"
    echo "=========================================="

    # Run the Python script
    python $PYTHON_SCRIPT \
        --region_bed "$BED_FILE" \
        --exp_name "$EXP_NAME" \
        --chk "$CHK" \
        --log_base "$LOG_BASE" \
        --chk_base "$CHK_BASE" \
        --res_base "$RES_BASE" \
        --processor "$PROCESSOR" \
        --num_processes "$NUM_PROCESSES" \
        --num_threads "$NUM_THREADS" \
        --use_head "$USE_HEAD" \
        --pseudo_count "$PSEUDO_COUNT" \
        --use_mean \
        --input_gate \
        --rc \
        --save_raw \
        --n_samples "$N_SAMPLES" \
        --sample_prob "$SAMPLE_PROB" \
        --sample_value "$SAMPLE_VALUE" \
        --sample_seed "$SAMPLE_SEED"

    if [ $? -eq 0 ]; then
        echo "Completed sample_prob = $SAMPLE_PROB successfully"
    else
        echo "ERROR: Failed for sample_prob = $SAMPLE_PROB"
    fi
done

echo ""
echo "=========================================="
echo "All Smooth Gradient Analyses Complete!"
echo "Results saved to: ${RES_BASE}/${EXP_NAME}/analysis_${CHK}/"
echo "Processed sample_prob values: ${SAMPLE_PROB_VALUES[@]}"
echo "=========================================="
