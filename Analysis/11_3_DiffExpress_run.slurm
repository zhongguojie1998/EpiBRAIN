#!/bin/bash
#
#SBATCH --job-name=diffexpress              # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                            # nodes
#SBATCH --ntasks-per-node=1                  # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                    # number of cores per tasks
#SBATCH --gres=gpu:1                         # number of gpus
#SBATCH --mem=64gb
#SBATCH --time 24:00:00                      # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                   # output file name

# Parse command line arguments (passed after script name in sbatch)
PART_NUM=${1}
EXP_NAME=${2}
CHK_NUM=${3}

# Validate required parameters
if [ -z "$PART_NUM" ] || [ -z "$EXP_NAME" ] || [ -z "$CHK_NUM" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: sbatch 11_3_DiffExpress_run.slurm <part_num> <exp_name> <checkpoint_num>"
    echo "Example: sbatch 11_3_DiffExpress_run.slurm 01 baskerville_atac_only 50"
    exit 1
fi

date;hostname;pwd

export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_BUILD_VERBOSE=1
export NCCL_DEBUG=INFO
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include

# Set BED file path based on part number
BED_FILE="Data/source/DiffExpress/split_files/part_${PART_NUM}.bed"

echo "Processing part ${PART_NUM}"
echo "BED file: ${BED_FILE}"
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint: ${CHK_NUM}"

# Run the motif bed interpretation script
/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/02_motif_bed_diff_interpretation.py \
    --region_bed "$BED_FILE" \
    --exp_name "$EXP_NAME" \
    --chk "$CHK_NUM" \
    --baseline random \
    --log_base ./logs \
    --chk_base ./Chk \
    --res_base ./Res \
    --processor gpu \
    --save_raw \
    --num_processes 1

date
