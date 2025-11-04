#!/bin/bash
# Unified script for variant effect screen
# Supports both VCF files (--vcf) and file lists (--filelist)
#
# --vcf is input vcf (for VCF mode)
# --filelist is input filelist (for filelist mode)
# --output is output h5
# --model is the model pickle file, it it is .pt, we will call prebuild_model.py to build a packaged model
# --config is the config file for .pt models (required if model is .pt)
# --label_meta is model label_meta file
# --experiment is experiment name, optional, default "variant_effect_screen"
# --chunks is number of chunks to split the vcf into, optional, default 1
# --job_script_path is path for job scripts, optional, default "$(dirname H5_FILE)/job_script"
# --machine is comma-separated list of machines, optional, default "turing,neumann,euler"
#           Machines with "-vpn" in name are treated as remote (files synced via rsync)
#           Other machines are treated as local network machines (shared filesystem)
# --mode is execution mode: "ssh", "slurm", or "ssh+slurm", optional, default "slurm"
#        - "ssh": submit all chunks to SSH machines (each machine gets 4 GPUs)
#        - "slurm": submit all chunks to SLURM
#        - "ssh+slurm": submit first N chunks to SSH machines, remaining chunks to SLURM
# --merge if set, only run the merge step (skip all processing steps)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vcf)
            VCF_FILE="$2"
            shift 2
            ;;
        --filelist)
            FILE_LIST="$2"
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
        --mode)
            MODE="$2"
            shift 2
            ;;
        --force)
            FORCE="true"
            shift 1
            ;;
        --merge)
            MERGE_ONLY="true"
            shift 1
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
H5_BASENAME=$(basename "$H5_FILE" .h5)

# Convert paths to absolute paths if they're relative
if [[ "$H5_FILE" != /* ]]; then
    H5_FILE="$PWD/$H5_FILE"
fi
if [[ "$MODEL_FILE" != /* ]]; then
    MODEL_FILE="$PWD/$MODEL_FILE"
fi

JOB_SCRIPT_PATH=${JOB_SCRIPT_PATH:-"$(dirname "$H5_FILE")/${H5_BASENAME}_job_script"}
# Convert JOB_SCRIPT_PATH to absolute path if it's relative
if [[ "$JOB_SCRIPT_PATH" != /* ]]; then
    JOB_SCRIPT_PATH="$PWD/$JOB_SCRIPT_PATH"
fi
MACHINES=${MACHINES:-"turing,neumann,euler"}
MODE=${MODE:-"slurm"}

# Get workingHOME from environment (set in ~/.bashrc)
WORKING_HOME=${workingHOME:?Error: workingHOME environment variable not set}

# Remote workingHOME (same for all machines)
REMOTE_WORKING_HOME="/share/vault/Users/gz2294"

# Function to translate local path to remote path
translate_path() {
    local path="$1"
    local remote_home="$2"
    echo "$path" | sed "s|^${WORKING_HOME}|${remote_home}|"
}

# Function to sync files to remote VPN machine
sync_to_remote() {
    local machine="$1"
    local remote_home="$2"

    echo "Syncing files to remote machine $machine"

    # Translate paths to remote paths
    local remote_job_script_path=$(translate_path "$JOB_SCRIPT_PATH" "$remote_home")
    local remote_h5_file=$(translate_path "$H5_FILE" "$remote_home")
    local remote_model_file=$(translate_path "$MODEL_FILE" "$remote_home")

    # Create remote directories if needed
    ssh "$machine" "mkdir -p $(dirname "$remote_job_script_path")" || { echo "Failed to create remote job script directory"; exit 1; }
    ssh "$machine" "mkdir -p $(dirname "$remote_h5_file")" || { echo "Failed to create remote h5 directory"; exit 1; }
    ssh "$machine" "mkdir -p $(dirname "$remote_model_file")" || { echo "Failed to create remote model directory"; exit 1; }

    # Sync job scripts
    echo "  - Syncing job scripts: $JOB_SCRIPT_PATH -> $machine:$remote_job_script_path"
    rsync -avz --progress "$JOB_SCRIPT_PATH/" "$machine:$remote_job_script_path/" || { echo "Failed to sync job scripts"; exit 1; }

    # Update paths in job scripts on remote machine
    echo "  - Updating paths in job scripts (replacing local paths with remote paths)"
    ssh "$machine" "find $remote_job_script_path -type f -name '*.sh' -exec sed -i -e 's|$WORKING_HOME|$remote_home|g' -e 's|/gpfs/commons/home/guojiezhong|$remote_home|g' {} +" || { echo "Failed to update paths in job scripts"; exit 1; }

    # Sync h5 file
    echo "  - Syncing h5 file: $H5_FILE -> $machine:$remote_h5_file"
    rsync -avz --progress "$H5_FILE" "$machine:$remote_h5_file" || { echo "Failed to sync h5 file"; exit 1; }

    # Sync model file
    echo "  - Syncing model file: $MODEL_FILE -> $machine:$remote_model_file"
    rsync -avz --progress "$MODEL_FILE" "$machine:$remote_model_file" || { echo "Failed to sync model file"; exit 1; }

    echo "Sync to $machine completed"
}

# Function to sync chunk results back from remote VPN machine
sync_results_from_remote() {
    local machine="$1"
    local remote_home="$2"

    # Translate local results dir to remote path
    local remote_results_dir=$(translate_path "$RESULTS_DIR" "$remote_home")

    echo "  - Syncing results from $machine:$remote_results_dir -> $RESULTS_DIR"
    # Use rsync with -a to preserve timestamps, -z for compression, --ignore-missing-args to not fail if dir doesn't exist yet
    rsync -avz --progress --ignore-missing-args "$machine:$remote_results_dir/" "$RESULTS_DIR/" 2>/dev/null || true
}

# Skip validation if only merging
if [ "$MERGE_ONLY" != "true" ]; then
    # Validate input: either --vcf or --filelist must be provided
    if [ -z "$VCF_FILE" ] && [ -z "$FILE_LIST" ]; then
        echo "Error: Either --vcf or --filelist must be provided"
        exit 1
    fi

    if [ -n "$VCF_FILE" ] && [ -n "$FILE_LIST" ]; then
        echo "Error: Cannot use both --vcf and --filelist at the same time"
        exit 1
    fi
fi

# Skip all processing steps if only merging
if [ "$MERGE_ONLY" != "true" ]; then
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
        python Analysis/03_variant_effect_screen/prebuild_model.py --config "$CONFIG_FILE" --checkpoint "$MODEL_FILE" --output "$PACKAGED_MODEL"

        # Use the packaged model for the rest of the script
        MODEL_FILE="$PACKAGED_MODEL"
    fi

    # Initialize tasks based on input type
    FORCE_FLAG=""
    if [ "$FORCE" = "true" ]; then
        FORCE_FLAG="--force"
    fi

    # Only run init_tasks if H5 file doesn't exist or force flag is on
    if [ ! -f "$H5_FILE" ] || [ "$FORCE" = "true" ]; then
        if [ -n "$VCF_FILE" ]; then
            # VCF mode
            python Analysis/03_variant_effect_screen/init_tasks.py -f "$VCF_FILE" \
                -h5 "$H5_FILE" \
                -l "$LABEL_META" \
                -e "$EXPERIMENT" -s raw_diff -s l1_sum -s l2_sum -s log_square -s local_raw_diff -s local_l1_sum -s local_l2_sum -s local_log_square $FORCE_FLAG
        else
            # Filelist mode
            python Analysis/03_variant_effect_screen/init_tasks.py -fl "$FILE_LIST" \
                -h5 "$H5_FILE" \
                -l "$LABEL_META" \
                -e "$EXPERIMENT" -s raw_diff -s l1_sum -s l2_sum -s log_square -s local_raw_diff -s local_l1_sum -s local_l2_sum -s local_log_square $FORCE_FLAG
        fi
    else
        echo "H5 file already exists and --force not specified. Skipping init_tasks."
    fi

    # Build -g arguments based on mode
    G_ARGS=""
    if [ "$MODE" = "ssh" ]; then
        # SSH mode: use machine names
        IFS=',' read -ra MACHINE_ARRAY <<< "$MACHINES"
        for machine in "${MACHINE_ARRAY[@]}"; do
            G_ARGS="$G_ARGS -g ${machine}:4:1:gpu"
        done
    elif [ "$MODE" = "ssh+slurm" ]; then
        # Hybrid mode: first use SSH machines, then SLURM for remaining chunks
        IFS=',' read -ra MACHINE_ARRAY <<< "$MACHINES"
        SSH_CHUNKS=0
        for machine in "${MACHINE_ARRAY[@]}"; do
            G_ARGS="$G_ARGS -g ${machine}:4:1:gpu"
            SSH_CHUNKS=$((SSH_CHUNKS + 4))
        done
        # Remaining chunks go to SLURM
        SLURM_CHUNKS=$((CHUNKS - SSH_CHUNKS))
        for ((i=0; i<$SLURM_CHUNKS; i++)); do
            G_ARGS="$G_ARGS -g nygc$i:1:1:gpu"
        done
    else
        # SLURM mode: use nygc naming
        for ((i=0; i<$CHUNKS; i++)); do
            G_ARGS="$G_ARGS -g nygc$i:1:1:gpu"
        done
    fi

    python Analysis/03_variant_effect_screen/assign_tasks.py -h5 "$H5_FILE" \
        -o "$JOB_SCRIPT_PATH" -m "$MODEL_FILE" -c Analysis/03_variant_effect_screen/compute.py $G_ARGS --abs_path

    # Job submission based on mode
    RESULTS_DIR="$(realpath "$(dirname "$H5_FILE")")/${H5_BASENAME}_chunk_results"

    # Track VPN machines and their remote homes for syncing results back
    declare -A VPN_MACHINES_MAP

    if [ "$MODE" = "ssh" ]; then
        # SSH to each machine and run jobs
        CURRENT_MACHINE=$(hostname)
        for machine in "${MACHINE_ARRAY[@]}"; do
            echo "Submitting jobs to $machine"
            if [ "$machine" = "$CURRENT_MACHINE" ]; then
                # Run locally if machine is current machine
                echo "Running locally on $machine (current machine)"
                bash "$JOB_SCRIPT_PATH/run_all_${machine}.sh" &
            else
                # Check if this is a VPN machine (remote)
                if [[ "$machine" == *"-vpn"* ]]; then
                    echo "Detected VPN machine: $machine"
                    echo "Remote workingHOME: $REMOTE_WORKING_HOME"

                    # Track this VPN machine for result syncing
                    VPN_MACHINES_MAP["$machine"]="$REMOTE_WORKING_HOME"

                    # Sync files to remote machine
                    sync_to_remote "$machine" "$REMOTE_WORKING_HOME"

                    # Translate paths for remote execution
                    REMOTE_PWD=$(translate_path "$PWD" "$REMOTE_WORKING_HOME")
                    REMOTE_JOB_SCRIPT=$(translate_path "$JOB_SCRIPT_PATH" "$REMOTE_WORKING_HOME")

                    # SSH to remote machine with translated paths
                    echo "Running on remote machine $machine"
                    ssh "$machine" "cd $REMOTE_PWD; bash $REMOTE_JOB_SCRIPT/run_all_${machine}.sh" &
                else
                    # Local network machine - direct SSH
                    echo "Running on local network machine $machine"
                    ssh "$machine" "cd $PWD; bash $JOB_SCRIPT_PATH/run_all_${machine}.sh" &
                fi
            fi
        done
    elif [ "$MODE" = "ssh+slurm" ]; then
        # Hybrid mode: submit SSH jobs first, then SLURM jobs
        echo "Running hybrid ssh+slurm mode"
        echo "SSH machines: ${MACHINES}"
        echo "SSH chunks: $SSH_CHUNKS"
        echo "SLURM chunks: $SLURM_CHUNKS"

        # Submit SSH jobs
        CURRENT_MACHINE=$(hostname)
        for machine in "${MACHINE_ARRAY[@]}"; do
            echo "Submitting jobs to $machine via SSH"
            if [ "$machine" = "$CURRENT_MACHINE" ]; then
                # Run locally if machine is current machine
                echo "Running locally on $machine (current machine)"
                bash "$JOB_SCRIPT_PATH/run_all_${machine}.sh" &
            else
                # Check if this is a VPN machine (remote)
                if [[ "$machine" == *"-vpn"* ]]; then
                    echo "Detected VPN machine: $machine"
                    echo "Remote workingHOME: $REMOTE_WORKING_HOME"

                    # Track this VPN machine for result syncing
                    VPN_MACHINES_MAP["$machine"]="$REMOTE_WORKING_HOME"

                    # Sync files to remote machine
                    sync_to_remote "$machine" "$REMOTE_WORKING_HOME"

                    # Translate paths for remote execution
                    REMOTE_PWD=$(translate_path "$PWD" "$REMOTE_WORKING_HOME")
                    REMOTE_JOB_SCRIPT=$(translate_path "$JOB_SCRIPT_PATH" "$REMOTE_WORKING_HOME")

                    # SSH to remote machine with translated paths
                    echo "Running on remote machine $machine"
                    ssh "$machine" "cd $REMOTE_PWD; bash $REMOTE_JOB_SCRIPT/run_all_${machine}.sh" &
                else
                    # Local network machine - direct SSH
                    echo "Running on local network machine $machine"
                    ssh "$machine" "cd $PWD; bash $JOB_SCRIPT_PATH/run_all_${machine}.sh" &
                fi
            fi
        done

        # Submit SLURM jobs for remaining chunks (starting from SSH_CHUNKS)
        for ((i=$SSH_CHUNKS; i<$CHUNKS; i++)); do
            # Check if chunk results already exist
            files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
            if [ -e "${files[0]}" ]; then
                echo "SLURM chunk $i results already exist, skipping job submission"
            else
                echo "Submitting SLURM job for chunk $i"
                # Use different memory based on input type
                if [ -n "$VCF_FILE" ]; then
                    MEM="24G"
                    TIME="12:00:00"
                else
                    MEM="64G"
                    TIME="24:00:00"
                fi
                sbatch --job-name="variant_chunk_$i" --partition="gpu" --mem="$MEM" --cpus-per-task="8" --time="$TIME" --gres="gpu:1" "$JOB_SCRIPT_PATH/run_chunk_$i.sh"
            fi
        done
    else
        # SLURM submission
        for ((i=0; i<$CHUNKS; i++)); do
            # Check if chunk results already exist
            files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
            if [ -e "${files[0]}" ]; then
                echo "Chunk $i results already exist, skipping job submission"
            else
                echo "Submitting job for chunk $i"
                # Use different memory based on input type
                if [ -n "$VCF_FILE" ]; then
                    MEM="24G"
                    TIME="12:00:00"
                else
                    MEM="64G"
                    TIME="24:00:00"
                fi
                sbatch --job-name="variant_chunk_$i" --partition="gpu" --mem="$MEM" --cpus-per-task="8" --time="$TIME" --gres="gpu:1" "$JOB_SCRIPT_PATH/run_chunk_$i.sh"
            fi
        done
    fi

    # wait for all chunks to complete
    echo "Waiting for all chunks to complete. Checking results in: $RESULTS_DIR"

    # Timeout based on input type
    if [ -n "$VCF_FILE" ]; then
        TIMEOUT_SECONDS=86400  # 24 hours
    else
        TIMEOUT_SECONDS=172800  # 48 hours
    fi
    START_TIME=$(date +%s)

    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ $ELAPSED -gt $TIMEOUT_SECONDS ]; then
            echo "Error: Timeout after $((TIMEOUT_SECONDS/3600)):00:00. Not all chunks completed within time limit."
            exit 1
        fi

        # Sync results from VPN machines if any
        if [ ${#VPN_MACHINES_MAP[@]} -gt 0 ]; then
            echo "Syncing results from remote VPN machines..."
            for machine in "${!VPN_MACHINES_MAP[@]}"; do
                remote_home="${VPN_MACHINES_MAP[$machine]}"
                sync_results_from_remote "$machine" "$remote_home"
            done
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

    # Verify all chunks are actually completed before proceeding to merge
    echo "Verifying all chunks are complete..."
    final_completed=0
    for ((i=0; i<$CHUNKS; i++)); do
        files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
        if [ -e "${files[0]}" ]; then
            final_completed=$((final_completed + 1))
        fi
    done

    if [ $final_completed -ne $CHUNKS ]; then
        echo "Error: Not all chunks completed. Only $final_completed/$CHUNKS chunks are ready."
        echo "Cannot proceed to merge. Exiting."
        exit 1
    fi

    echo "Verification passed: All $CHUNKS chunks are ready for merging."
else
    echo "Running in merge-only mode. Skipping all processing steps."
    # Still need to set RESULTS_DIR for the merge step
    RESULTS_DIR="$(realpath "$(dirname "$H5_FILE")")/${H5_BASENAME}_chunk_results"

    # In merge-only mode, also verify all chunks exist
    echo "Verifying all chunks exist before merging..."
    final_completed=0
    for ((i=0; i<$CHUNKS; i++)); do
        files=(${RESULTS_DIR}/chunk_${i}_part_*.h5)
        if [ -e "${files[0]}" ]; then
            final_completed=$((final_completed + 1))
        fi
    done

    if [ $final_completed -ne $CHUNKS ]; then
        echo "Error: Not all chunks found. Only $final_completed/$CHUNKS chunks exist."
        echo "Cannot proceed to merge. Exiting."
        exit 1
    fi

    echo "Verification passed: All $CHUNKS chunks are ready for merging."
fi

# wait for jobs to finish, then collate results
if python Analysis/03_variant_effect_screen/merge_results.py -h5 "$H5_FILE" --chunk_dir "$RESULTS_DIR"; then
    # cleanup: delete the results directory and job scripts after successful merging
    echo "Cleaning up intermediate results directory: $RESULTS_DIR"
    rm -r "$RESULTS_DIR"
    echo "Cleaning up job scripts directory: $JOB_SCRIPT_PATH"
    rm -r "$JOB_SCRIPT_PATH"
else
    echo "Error: merge_results.py failed. Keeping intermediate files for debugging."
    echo "Results directory: $RESULTS_DIR"
    echo "Job scripts directory: $JOB_SCRIPT_PATH"
    exit 1
fi