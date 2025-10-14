#!/bin/bash
#
#SBATCH --job-name=abc_analysis              # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                            # nodes
#SBATCH --ntasks-per-node=1                  # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                    # number of cores per tasks
##SBATCH --gres=gpu:1                         # number of gpus
#SBATCH --mem=64gb
#SBATCH --time 24:00:00                      # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                   # output file name

# Parse command line arguments (passed after script name in sbatch)
BED_FILE=${1}
INPUT_NAME=${2}
CHK_NUM=${3}

# Validate required parameters
if [ -z "$BED_FILE" ] || [ -z "$INPUT_NAME" ] || [ -z "$CHK_NUM" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: sbatch 09_ABC_run.sh <bed_file> <input_name> <checkpoint_file>"
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

# Run the motif interpretation script
/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Analysis/02_motif_diff_interpretation.py \
    --region_bed "$BED_FILE" \
    --exp_name "$INPUT_NAME" \
    --chk "$CHK_NUM" \
    --save_raw \
    --num_processes 1

date
