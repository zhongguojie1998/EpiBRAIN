#!/bin/bash
#
#SBATCH --job-name=deepspeed                    # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu    # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=2                               # nodes
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                       # number of cores per tasks
#SBATCH --gres=gpu:1                            # number of gpus
#SBATCH --mem=36gb
#SBATCH --time 20:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                      # output file name

date;hostname;pwd

export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/deepspeed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_BUILD_VERBOSE=1
export NCCL_DEBUG=INFO
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/deepspeed
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include

export GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --cpu-bind=none --jobid $SLURM_JOBID bash -c '/gpfs/commons/home/guojiezhong/miniconda3/envs/deepspeed/bin/python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
Model/train.py --config_dir=./Config -d -x logging=debug -x logging.exp_name=debug'

date
