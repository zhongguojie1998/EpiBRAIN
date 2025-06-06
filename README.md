# Installation

```bash
conda create -n bican python=3.12 ipykernel
conda activate bican

# data processing
conda install -c conda-forge -c bioconda pybigwig  # get label from bw files
conda install -c conda-forge -c bioconda pysam  # reading genome data
conda install -c bioconda bedtools  # process bed files
pip install pyfaidx  # reading genome data

# model
pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cu122 # need to first install torch to avoid automatic cpu version installation
pip install "einops >= 0.5" "transformers >= 4.34.1" "intervaltree~=3.1.0" numpy pandas h5py torchmetrics peft

## install flash attention
pip install ninja # for fast compile
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # for cuda toolkit, need nvcc
pip install flash-attn --no-build-isolation  # flash attention

# util packages
pip install click # command line tool
pip install hydra-core  # better config
pip install tensorboard  # better logging
```

# Usage

## Quick start

- For training from scratch

Config used: data (default), model (default), training (default), logging (default)

```bash
python Model/train.py -c default -x "logging.exp_name=your_exp_name"
```

- For fine tuning from Borzoi

Config used: data (default), model (finetune, extra configs for building finetune model), training (finetune), logging (default)

```bash
python Model/train.py -c finetune -x "logging.exp_name=your_exp_name"
```

## Multi-GPU training

- Single machine

```bash
python Model/train.py -c default -x "training.world_size=your_gpu_num"
```

- Multiple machine

```bash
python launchjob.py -e your_exp_name -m your_machine_1 -m your_machine_2
```

## For developer

Always use logging=debug for more information about the training

```bash
python Model/train.py -c finetune -x "logging=debug" -x "logging.exp_name=250605_finetune"
```

# Developer notes

## Data Preprocessing

1. Just in case, we may need to remap the raw reads to the genome to get better signal calls (follow [Basenji](https://pmc.ncbi.nlm.nih.gov/articles/PMC5932613/) pipeline)
2. Exlude region: By avoiding assembly gaps and unmappable regions >1 kb, we extracted (217=) 131-kb nonoverlapping sequences across the chromosomes. We discarded sequences with >35% unmappable sequence, leaving 14,533 sequences. (Basenji)
3. Correct read value ([Basenji2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050#sec017)): 
   - For ENCODE blacklist, RepeatMasker satellite repeats, unmappable regions of >32 bp where 24-mers align to >10 genomic sites using Umap mappability tracks (all are false positive regions), set signal values overlapping these regions 25th percentile value of each dataset (background value)
   - soft clipped high values with the function f(x) = min(x, tc + sqrt(max(0, x − tc))), to reduce the contribution of rare very large values that one would not expect to generalize to other genomic locations. Via this procedure, we decided to clip all CAGE data with tc = 384, ENCODE with tc = 32, and GEO with tc = 64.


## Data Pipeline

1. Org data format: bw (UCSC bigwig file). Get the data summary stats (serve as **label**) with [`pyBigWig`](https://github.com/deeptools/pyBigWig)
2. preprocess the data and get the train/valid/test in the .bed file (essentially a tsv file, contain the chr, start, end info). Presample all the data points (across all the genomes) and Precompute all the labels in the preprocessing stage, reference the [script](https://github.com/calico/basenji/blob/master/tutorials/preprocess.ipynb), save in the torch pt files
3. [DNA Tokenizer](https://github.com/lucidrains/enformer-pytorch/blob/5a5974d2821c728f93294731c50b55f1f55fd86d/enformer_pytorch/data.py), Dataset, get the tokenized DNA (one hot, L * 4) and the label of the correponding region (cell type * modality * central bin num)

Final data point: a 196,608 length DNA window, further truncated into 128 (bin width) * 896 (bin num), for each bin, calculate the label (for each cell type, each modality, we have a label value). When training, we only calculate the loss for the central bins, not for the marginal bins (marginal ones don't have enough information)


## Model Setting

1. Model backbone: [borzoi-pytorch](https://github.com/johahi/borzoi-pytorch) with flash atten
   - remove human/mouse head, only keep one head (human), but need to change the output channel number to fit our data (cell type * modality * central bin num)
   - check the model crop behavior
2. Training code (Premode trainer)
   - config file
   - trainer function
   - logs (tensorboard)
   - checkpoint autosave
   - continue training
   - DDP (if use DDP, set the model forward, `data_parallel_training = True`)
   - multi machine training

## Q & A

1. model, set_track_subset? Annotated version?
   - used to add prompt to the enformer, leave it now
