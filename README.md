# Data Pipeline

1. Org data format: bw (UCSC bigwig file). Get the data summary stats (serve as **label**) with [`pyBigWig`](https://github.com/deeptools/pyBigWig)
2. preprocess the data and get the train/valid/test in the .bed file (essentially a tsv file, contain the chr, start, end info)
3. DNA Tokenizer, Dataset (/share/vault/Users/gz2294/GenoSCOPE/data/Data.py, `MYEnformerTokenizer`, `gnomAD`), get the tokenized DNA (one hot, L * 4) and the label of the correponding region (cell type * modality * central bin num) (data["sequence"], data["target"])

Final data point: a 196,608 length DNA window, further truncated into 128 (bin width) * 896 (bin num), for each bin, calculate the label (for each cell type, each modality, we have a label value). When training, we only calculate the loss for the central bins, not for the marginal bins (marginal ones don't have enough information)

Presample all the data points (across all the genomes) and Precompute all the labels in the preprocessing stage, reference the sample script (https://github.com/calico/basenji/blob/master/tutorials/preprocess.ipynb), save in the torch pt files

For now no need to have 6 training files as done in the reference preprocessing file. For now no need to include the chrX and chrY (leave this feature)

# Model Setting

1. Model backbone: [borzoi-pytorch](https://github.com/johahi/borzoi-pytorch) with flash atten
   - remove human/mouse head, only keep one head (human), but need to change the output channel number to fit our data (cell type * modality * central bin num)
   - check the model crop behavior
2. Training code (reference: https://github.com/boxiangliu/enformer-pytorch/blob/main/bin/train.py, Premode trainer)
   - config file
   - trainer function
   - logs
   - checkpoint autosave
   - continue training
   - DDP (if use DDP, set the model forward, `data_parallel_training = True`)

# Q & A

1. model, set_track_subset? Annotated version?
   - used to add prompt to the enformer, leave it now

# possible issue

1. flash atten
2. cuda memory

# Installation

```bash
conda create -n bican python=3.12 ipykernel
conda activate bican

# data processing
conda install pybigwig -c conda-forge -c bioconda  # get label from bw files
pip install pyfaidx  # reading genome data

# model
pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cu122 # need to first install torch to avoid automatic cpu version installation
pip install "einops >= 0.5" "transformers >= 4.34.1" numpy pandas 

## install flash attention
pip install ninja # for fast compile
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # for cuda toolkit, need nvcc
pip install flash-attn --no-build-isolation  # flash attention

# util packages
pip install click # command line tool
pip install hydra-core  # better config
```