#!/bin/bash
#
#SBATCH --job-name=infer_over_time
#SBATCH --mail-user=gz2294@cumc.columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240gb
#SBATCH --time 24:00:00
#SBATCH --output=%x-%j.out

date;hostname;pwd

PYTHON=$(which python)

EXP_NAME=${1:?"Usage: $0 <exp_name> [epoch_start] [epoch_end] [world_size]"}
EPOCH_START=${2:-1}
EPOCH_END=${3:-20}
WORLD_SIZE=${4:-4}

CONFIG_PATH="logs/${EXP_NAME}/overall_setting.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found at $CONFIG_PATH"
    exit 1
fi

echo "Experiment: $EXP_NAME"
echo "Epochs: $EPOCH_START to $EPOCH_END"
echo "World size: $WORLD_SIZE"

# Step 1: Run inference for each epoch
for EPOCH in $(seq $EPOCH_START $EPOCH_END); do
    PRED_FILE="Res/${EXP_NAME}/Test_preds_epoch_${EPOCH}.pt"
    if [ -f "$PRED_FILE" ]; then
        echo "Epoch $EPOCH: predictions already exist, skipping inference"
        continue
    fi

    CHK_FILE="Chk/${EXP_NAME}/chk_epoch_${EPOCH}.pt"
    if [ ! -f "$CHK_FILE" ]; then
        echo "Epoch $EPOCH: checkpoint not found at $CHK_FILE, skipping"
        continue
    fi

    echo "=== Running inference for epoch $EPOCH ==="
    $PYTHON Model/train.py \
        -c "$CONFIG_PATH" \
        -x "training.test_only=True" \
        -x "data.used_dataset=[test]" \
        -x "training.load_checkpoint=$EPOCH" \
        -x "training.world_size=$WORLD_SIZE"
done

# Step 2: Aggregate correlations over time
echo "=== Aggregating correlations over epochs ==="
$PYTHON -m Analysis.01_1_test_correlation_over_time \
    -e "$EXP_NAME" \
    --epoch_start "$EPOCH_START" \
    --epoch_end "$EPOCH_END" \
    -s Test \
    --res_base ./Res \
    --log_base ./logs \
    --data_base ./Data \
    -t none -t log -t log_quantile -t log_quantile_substract_mean

date
