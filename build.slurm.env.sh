mamba create -n deepspeed nvidia/label/cuda-12.6.0::cuda-nvdisasm nvidia/label/cuda-12.6.0::cuda-nvcc nvidia/label/cuda-12.6.0::cuda-toolkit conda-forge::hydra-core bioconda::pybigwig bioconda::pysam conda-forge::tensorboard conda-forge::pyyaml conda-forge::pytorch=2.7.1=cuda126_mkl_py39_hd241233_300 conda-forge::pandas conda-forge::intervaltree conda-forge::peft conda-forge::flash-attn bioconda::pyfaidx conda-forge::click conda-forge::h5py conda-forge::torchmetrics

conda activate deepspeed

pip install deepspeed
