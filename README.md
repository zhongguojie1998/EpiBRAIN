# Data Preprocessing

1. Just in case, we may need to remap the raw reads to the genome to get better signal calls (follow [Basenji](https://pmc.ncbi.nlm.nih.gov/articles/PMC5932613/) pipeline)
2. Exlude region: By avoiding assembly gaps and unmappable regions >1 kb, we extracted (217=) 131-kb nonoverlapping sequences across the chromosomes. We discarded sequences with >35% unmappable sequence, leaving 14,533 sequences. (Basenji)
3. Correct read value ([Basenji2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050#sec017)): 
   - For ENCODE blacklist, RepeatMasker satellite repeats, unmappable regions of >32 bp where 24-mers align to >10 genomic sites using Umap mappability tracks (all are false positive regions), set signal values overlapping these regions 25th percentile value of each dataset (background value)
   - soft clipped high values with the function f(x) = min(x, tc + sqrt(max(0, x − tc))), to reduce the contribution of rare very large values that one would not expect to generalize to other genomic locations. Via this procedure, we decided to clip all CAGE data with tc = 384, ENCODE with tc = 32, and GEO with tc = 64.


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
   - logs (tensorboard)
   - checkpoint autosave
   - continue training
   - DDP (if use DDP, set the model forward, `data_parallel_training = True`)


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
pip install "einops >= 0.5" "transformers >= 4.34.1" "intervaltree~=3.1.0" numpy pandas h5py torchmetrics

## install flash attention
pip install ninja # for fast compile
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # for cuda toolkit, need nvcc
pip install flash-attn --no-build-isolation  # flash attention

# util packages
pip install click # command line tool
pip install hydra-core  # better config
pip install tensorboard  # better logging
```

# Q & A

1. model, set_track_subset? Annotated version?
   - used to add prompt to the enformer, leave it now

# possible issue

1. flash atten
2. cuda memory
