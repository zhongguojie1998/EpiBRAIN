#!/bin/bash
#
#SBATCH --job-name=bican                    # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu    # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=2                               # number of nodes
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24                       # number of cores per tasks
#SBATCH --gres=gpu:3                            # number of gpus per node
#SBATCH --mem=300gb
#SBATCH --time 144:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                      # output file name

# Parse experiment name argument
EXP_NAME=${1:-"basal_ganglia_miniatlas_drop_celltype_v1"}

date;hostname;pwd

export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_BUILD_VERBOSE=1
export NCCL_DEBUG=INFO
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include

# Get node list and setup multinode variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "SLURM_PROCID: $SLURM_PROCID"

srun --cpu-bind=none --jobid $SLURM_JOBID bash -c "torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=3 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=\$MASTER_ADDR \
    --master_port=\$MASTER_PORT \
    Model/train.py -c default -x \"logging.exp_name=$EXP_NAME\" \
        -x \"training=full_finetune_nygc\" --torchrun \
        -x \"data.storage_path=./Data/$EXP_NAME\" \
        -x \"model=full_finetune\" \
        -x \"model.use_compile=True\" \
        -x \"model.output_heads.use_cell_encoder=True\" \
        -x \"data.preprocess.trial_summary_path=./Data/data_config/$EXP_NAME.csv\""

date
