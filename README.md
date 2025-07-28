- [Installation](#installation)
  - [Optional packages](#optional-packages)
- [Data Pipeline](#data-pipeline)
- [Usage](#usage)
  - [Only to generate labels](#only-to-generate-labels)
  - [Only testing](#only-testing)
  - [Quick start](#quick-start)
  - [Multi-GPU training](#multi-gpu-training)
  - [For developer](#for-developer)
- [Analysis](#analysis)
- [Developer notes](#developer-notes)
  - [Exp Notes](#exp-notes)
    - [Data](#data)
    - [Training](#training)
      - [Current Hyperparameter Setting](#current-hyperparameter-setting)
      - [Milestones](#milestones)
  - [Data Preprocessing](#data-preprocessing)
  - [Data Pipeline](#data-pipeline-1)
  - [Model Setting](#model-setting)
  - [Q \& A](#q--a)


# Installation

```bash
conda create -n bican python=3.12
conda activate bican

# data processing
conda install -c conda-forge -c bioconda pybigwig  # get label from bw files
conda install -c conda-forge -c bioconda pysam  # reading genome data
conda install -c bioconda bedtools  # process bed files
pip install pyfaidx==0.8.1.4  # reading genome data

# model
pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cu122 # need to first install torch to avoid automatic cpu version installation
pip install "einops >= 0.5" "transformers >= 4.34.1" "intervaltree~=3.1.0" numpy pandas h5py torchmetrics peft==0.15.2

## install flash attention
pip install ninja # for fast compile
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # for cuda toolkit, need nvcc
pip install flash-attn --no-build-isolation  # flash attention

## install deepspeed, make sure you've already installed cuda-toolkit in the former step
pip install deepspeed
pip install cupy-cuda12x # if want to use onebitadam

# util packages
pip install click # command line tool
pip install hydra-core  # better config
pip install tensorboard  # better logging
```

## Optional packages

```bash
pip install matplotlib seaborn scikit-learn ipykernel 
pip install cyvcf2   # for reading vcf files
pip install captum   # for model interpretation
pip install modisco  # for visualization
```

# Data Pipeline

Write all the transformation configuration into one csv file and update the csv file in `data.preprocess.trial_summary_path` field

The configuration can have the following fields:

- `exp` (required): the name of the trial
- `file` (required): the path to the raw bigwig file
- `sum_stat` (optional, default: sum): in each bin, how to aggregate the raw reads into a summary
- `baseline_pct` (optional, default: 0.5): set the nan/blacklist region value to this quantile of all values
- `umap_pct` (optional, default: 0.5): set the umap region value to this quantile of all values
- `scale` (optional, default: 1): scale the raw reads
- `extreme_clip_pct`(optional, default: None): final hard clip all values above this quantile to the corresponding value. If not provided, skip.
- `offset` (optional, default: None): shift the value by a given value (org_read - offset), which can be used to reduce some noise. If not provided, skip.
- `anchor_target` (optional, default: None): after aggregating the data, anchor the given quantile of the value to this target. If not provided, skip.
- `anchor_pct` (optional, default: 0.999): the given quantile for anchoring
- `clip_soft` (optional, default: None): soft clip the aggregated value ($t_c - 1 + \sqrt{x - t_c + 1}$ for all $x > t_c$) to the given threshold. If not provided, skip.
- `clip` (optional, default: None): hard clip the aggregated value to the given threshold. If not provided, skip.

The data preprocess pipeline would be: 

1. impute nan / reset value in the blacklist region to the `baseline_pct` of the whole context length (eg. 196608)
2. scale the data based on `scale`
3. aggregate the data for given pool width into bins with the given `sum_stat`
4. clip the umap region to the `umap_pct` of the whole context length (if umap bedfile is provided)
5. hard clip extreme value above `extreme_clip_pct` to the corresponding value (if applicable)
6. Subtract the original value by `offset` (if applicable)
7. anchor the value at the `anchor_pct` to `anchor_target` (if applicable)
8. soft clip based on threshold `clip_soft` (if applicable)
9. hard clip based on threshold `clip` (if applicable)

# Usage

## Only to generate labels

```bash
python Model/train.py -x "logging=debug" -x "logging.exp_name=data_generation" --only_data
```

The data generation will not re-process if the data have already been generated. Force start with

```bash
python Model/train.py -x "logging=debug" -x "logging.exp_name=data_generation" -x "data.preprocess.force_restart=True" --only_data
```

## Only testing

```bash
python Model/train.py -c default -x "logging=debug" -x "logging.exp_name=test" -x "training.test_only=True" -x "training.load_checkpoint=path_to_your_chk"
```

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

# Analysis

- [Plot data distribution](./Analysis/00_visualize_data.py)
- [Analyze model performance based on pearson correlation](./Analysis/01_test_correlation.py)
- [Important motif identification](./Analysis/02_motif_interpretation.py)


# Developer notes

## Exp Notes

### Data

1. Data v1: Default pipeline, sum_stat mean, baseline_pct 0.25, scale 1, extreme_clip_pct 0.9999999, anchor_target 100, anchor_pct 0.999, clip_threshold None, softclip_threshold 300
2. Data v2: Default pipeline, sum_stat mean, baseline_pct 0.25, scale 1, extreme_clip_pct 0.9999999, offset `out_peak_non_zero_median`, anchor_target 100, anchor_pct 0.999, clip_threshold None, softclip_threshold 300
3. Data v3: Default pipeline, sum_stat mean, baseline_pct 0.25, scale 1, extreme_clip_pct 0.9999999, offset `out_peak_mean`, anchor_target 100, anchor_pct 0.999, clip_threshold None, softclip_threshold 300
4. Data v4: Default pipeline, sum_stat mean, baseline_pct 0.25, scale 1, extreme_clip_pct None, offset None, anchor_target 100, anchor_pct 0.999, clip_threshold None, softclip_threshold 300 (New source: MACS2_bw, discard GP-GABA-Glut trials)
5. Data v5: Default pipeline, sum_stat sum, baseline_pct 0.5, umap_pct 0.5, scale 1, extreme_clip_pct None, offset None, anchor_target None, anchor_pct 0.999, clip_threshold None, softclip_threshold None (New source: bamCoverage_bw, discard GP-GABA-Glut trials, change blacklist (blacklist.bed), add umap (umap_k36_t10_l32.bed))
6. Data v6: Default pipeline 2.0 (put scale in the very end, after clip), sum_stat mean, baseline_pct 0.5, umap_pct 0.5, scale 2, extreme_clip_pct None, offset None, anchor_target None, anchor_pct 0.999, clip_threshold 128, softclip_threshold 32 (New source: bamCoverage_bw, discard GP-GABA-Glut, SMC trials, change blacklist (blacklist.bed), add umap (umap_k36_t10_l32.bed))

### Training

Details see the config generated in the corresponding folder

#### Current Hyperparameter Setting

- batch size: 196 (12 (btz) * 8 (gpu) * 2 (accum step))
  - per epoch, 202 step (btz 96), 101 step (btz 196)
- lr: 1e-4 (both for scratch / finetune)

#### Milestones

1. `250606_finetune_new_data`
   - Data: v1
   - Model: Finetune (Lora)
   - Training (no trick)
2. `250614_scratch_gc`
   - Data: v1
   - Model: Full
   - Training (with gradient compression)


## Data Preprocessing

1. Just in case, we may need to remap the raw reads to the genome to get better signal calls (follow [Basenji](https://pmc.ncbi.nlm.nih.gov/articles/PMC5932613/) pipeline)
2. Exlude region: By avoiding assembly gaps and unmappable regions >1 kb, we extracted (217=) 131-kb nonoverlapping sequences across the chromosomes. We discarded sequences with >35% unmappable sequence, leaving 14,533 sequences. (Basenji)
3. Correct read value ([Basenji2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050#sec017)): 
   - For ENCODE blacklist, RepeatMasker satellite repeats, unmappable regions of >32 bp where 24-mers align to >10 genomic sites using Umap mappability tracks (all are false positive regions), set signal values overlapping these regions 25th percentile value of each dataset (background value)
   - soft clipped high values with the function f(x) = min(x, tc + sqrt(max(0, x − tc))), to reduce the contribution of rare very large values that one would not expect to generalize to other genomic locations. Via this procedure, we decided to clip all CAGE data with tc = 384, ENCODE with tc = 32, and GEO with tc = 64.


## Data Pipeline

1. Org data format: bw (UCSC bigwig file). Get the data summary stats (serve as **label**) with [`pyBigWig`](https://github.com/deeptools/pyBigWig)
2. preprocess the data and get the train/valid/test in the .bed file (essentially a tsv file, contain the chr, start, end info). Presample all the data points (across all the genomes) and Precompute all the labels in the preprocessing stage, reference the [preprocess script](https://github.com/calico/basenji/blob/master/tutorials/preprocess.ipynb), save in the torch pt files
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