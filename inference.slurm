#!/bin/bash
#
#SBATCH --job-name=bican                        # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu    # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                               # nodes
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32                       # number of cores per tasks
##SBATCH --gres=gpu:1                            # number of gpus
#SBATCH --mem=240gb
#SBATCH --time 24:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                      # output file name

date;hostname;pwd

export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_BUILD_VERBOSE=1
export NCCL_DEBUG=INFO
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include

/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python Model/train.py -c default \
    -x "logging.exp_name=basal_ganglia_miniatlas_drop_celltype_v1_no_celltype_head" \
    -x "training=default_nygc" \
    -x "training.world_size=1" -x "training.gpu_id=0" \
    -x "model=no_celltype_heads" \
    -x "training.test_only=True" -x "data=test" \
    -x "training.load_checkpoint=135"

date
