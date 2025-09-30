# !/bin/bash
# --vcf is input vcf
# --output is output h5
# --model is the model pickle file, it it is .pt, we will call prebuild_model.py to build a packaged model
# --config is the config file for .pt models (required if model is .pt)
# --label_meta is model label_meta file
# --experiment is experiment name, optional, default "variant_effect_screen"
# --chunks is number of chunks to split the vcf into, optional, default 1
# --job_script_path is path for job scripts, optional, default "$(dirname H5_FILE)/job_script"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vcf)
            VCF_FILE="$2"
            shift 2
            ;;
        --output)
            H5_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --label_meta)
            LABEL_META="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --chunks)
            CHUNKS="$2"
            shift 2
            ;;
        --job_script_path)
            JOB_SCRIPT_PATH="$2"
            shift 2
            ;;
        --machine)
            MACHINES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Set defaults
EXPERIMENT=${EXPERIMENT:-"variant_effect_screen"}
CHUNKS=${CHUNKS:-1}
JOB_SCRIPT_PATH=${JOB_SCRIPT_PATH:-"$(dirname "$H5_FILE")/job_script"}
MACHINES=${MACHINES:-"turing,neumann,euler"}

# Handle .pt model files - convert to packaged model
if [[ "$MODEL_FILE" == *.pt ]]; then
    echo "Detected .pt model file. Converting to packaged model..."

    # Check if config file is provided
    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: --config parameter is required when using .pt model files"
        exit 1
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file $CONFIG_FILE not found"
        exit 1
    fi

    PACKAGED_MODEL="${MODEL_FILE%.pt}_packaged.pkl"
    /gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/03_variant_effect_screen/prebuild_model.py --config "$CONFIG_FILE" --checkpoint "$MODEL_FILE" --output "$PACKAGED_MODEL"

    # Use the packaged model for the rest of the script
    MODEL_FILE="$PACKAGED_MODEL"
fi

/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/03_variant_effect_screen/init_tasks.py -f "$VCF_FILE" \
    -h5 "$H5_FILE" \
    -l "$LABEL_META" \
    -e "$EXPERIMENT" -s raw_diff -s l1_sum -s l2_sum -s log_square -s local_raw_diff -s local_l1_sum -s local_l2_sum -s local_log_square --force

# Build -g arguments for multiple chunks
G_ARGS=""
IFS=',' read -ra MACHINE_ARRAY <<< "$MACHINES"
for machine in "${MACHINE_ARRAY[@]}"; do
    G_ARGS="$G_ARGS -g ${machine}:4:1:gpu"
done

/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/03_variant_effect_screen/assign_tasks.py -h5 "$H5_FILE" \
    -o "$JOB_SCRIPT_PATH" -m "$MODEL_FILE" -c Analysis/03_variant_effect_screen/compute.py $G_ARGS --abs_path --use_head human

# do slurm submission
for ((i=0; i<$CHUNKS; i++)); do
    # Check if chunk results already exist
    files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
    if [ -e "${files[0]}" ]; then
        echo "Chunk $i results already exist, skipping job submission"
    else
        echo "Submitting job for chunk $i"
        sbatch --job-name="variant_chunk_$i" --partition="gpu" --mem="24G" --cpus-per-task="8" --time="12:00:00" --gres="gpu:1" "$JOB_SCRIPT_PATH/run_chunk_$i.sh"
    fi
done

# wait for all chunks to complete
RESULTS_DIR="$(realpath "$(dirname "$H5_FILE")")/chunk_results"
echo "Waiting for all chunks to complete. Checking results in: $RESULTS_DIR"

# Timeout after 24 hours (86400 seconds)
TIMEOUT_SECONDS=86400
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT_SECONDS ]; then
        echo "Error: Timeout after 24:00:00. Not all chunks completed within time limit."
        exit 1
    fi

    completed=0
    for ((i=0; i<$CHUNKS; i++)); do
        files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
        if [ -e "${files[0]}" ]; then
            completed=$((completed + 1))
        fi
    done

    echo "Completed chunks: $completed/$CHUNKS (Elapsed: $((ELAPSED/3600))h $((ELAPSED%3600/60))m)"

    if [ $completed -eq $CHUNKS ]; then
        echo "All chunks completed!"
        break
    fi

    sleep 300  # Wait 5 minutes before checking again
done

# wait for jobs to finish, then collate results
/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/03_variant_effect_screen/merge_results.py -h5 "$H5_FILE"

# cleanup: delete the results directory and job scripts after merging
echo "Cleaning up intermediate results directory: $RESULTS_DIR"
rm -r "$RESULTS_DIR"
echo "Cleaning up job scripts directory: $JOB_SCRIPT_PATH"
rm -r "$JOB_SCRIPT_PATH"
