# EpiBRAIN
This is the repository for EpiBRAIN (Epigenomics-based Brain Regulation Attention Inference Network), for our manuscript "Single-cell Analysis of Chromatin State and Transcriptome in Human Basal Ganglia" (in submission)

# EpiBRAIN
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
conda create -f EpiBRAIN.yml
conda activate EpiBRAIN
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

- `exp` (required): the name of the trial, **please name it in `{celltype}_{modality}` style**
- `file` (required): the path to the raw bigwig file
- `task` (required): the final target is regression or classification task
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

## Only do variant effect predictions

1. Download pretrained model weights, see `Chk/full_finetune_original_loss_celltype_head_dim8_linear/README.md` and `Chk/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/README.md` for details.
2. Download gencode v48 from GENCODE website: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz, and put it under `Data/source/`
3. Run the `Analysis/variant_analysis_pipeline.sh`, see the script for details of documentation and configuration.
```bash
bash Analysis/variant_analysis_pipeline.sh --variant chrXX:POS:REF:ALT --track CellTypeName --variant-name rsXXX(anything) --gene GeneName --disease DiseaseName(Optional,anything) --method gradient_input --tomtom-db /path/to/meme
```

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

- **00 - Data Visualization & Conversion**: [Plot data distribution](./Analysis/00_visualize_data.py), [IGV visualization](./Analysis/00_igv_visualization.py), [pyGenomeTracks visualization](./Analysis/00_visualize_data_pygenometrack.py), [TF to PyTorch conversion](./Analysis/00_tf_to_torch/)
- **01 - Model Inference & Performance**: [Quick inference to BigWig](./Analysis/01_0_quick_inference_bigwig.py), [Bin-level correlation](./Analysis/01_1_test_correlation.py), [Cross-celltype correlation](./Analysis/01_2_test_correlation_across_celltypes.py), [Gene-level correlation](./Analysis/01_5_test_correlation_by_gene.py), [Performance plots](./Analysis/01_6_plots.py)
- **02 - Motif Interpretation**: [DeepLift motif interpretation](./Analysis/02_motif_interpretation_DeepLift.py), [Gradient-input motif interpretation](./Analysis/02_motif_interpretation_gradient_input.py), [Differential motif interpretation](./Analysis/02_motif_diff_interpretation_DeepLift.py), [TOMTOM motif validation](./Analysis/02_motif_region_tomtom.py)
- **03 - Variant Effect Prediction**: [Variant effect prediction](./Analysis/03_0_variant_effect.py), [CRISPRi validation](./Analysis/03_0_crispri.py), [Variant effect visualization](./Analysis/03_1_variant_effect_viz.ipynb), [S-LDSC analysis](./Analysis/03_2_run_sldp_analysis.py), [Variant effect to BigWig](./Analysis/03_5_variant_effect_to_bigwig.py), [Large-scale variant screening pipeline](./Analysis/03_variant_effect_screen/)
- **04 - Prediction Comparison**: [View prediction differences](./Analysis/04_view_prediction_differences.py)
- **05 - Transcript Performance**: [Gene/transcript-level performance evaluation](./Analysis/05_transcripts_performance.py)
- **06 - GTEx Preprocessing**: [GTEx eQTL data preprocessing](./Analysis/06_GTEx_preprocessing.py)
- **07 - eQTL Analysis**: [eQTL analysis](./Analysis/07_eQTL_analysis.py), [Borzoi eQTL](./Analysis/07_eQTL_borzoi_analysis.py), [ChromBPNet eQTL](./Analysis/07_eQTL_chrombpnet_analysis.py), [H3K27me3 eQTL](./Analysis/07_eQTL_me3_analysis.py)
- **08 - TraitGym**: [TraitGym preprocessing](./Analysis/08_TraitGym_preprocessing.py), [TraitGym analysis](./Analysis/08_TraitGym_analysis.py)
- **09 - ABC Loop Analysis**: [ABC data preparation](./Analysis/09_1_ABC_prepare.py), [ABC attribution screening](./Analysis/09_3_ABC_screen_significant_attributions.py), [ABC attribution plots](./Analysis/09_4_ABC_screen_significant_attributions_plot.py)
- **10 - Differential Peak Analysis**: [Link DiffExpress to DiffPeak](./Analysis/10_1_link_DiffExpress_DiffPeak.py), [Screen DiffPeak attributions](./Analysis/10_3_screen_DiffPeak_attributions.py), [DiffPeak attribution plots](./Analysis/10_4_screen_DiffPeak_attributions_plot.py)
- **11 - Differential Expression & TF Motif Discovery**: [Differential expression](./Analysis/11_1_DiffExpress.py), [TF-MoDISco motif discovery](./Analysis/11_4_DiffExpress_TFMoDisco.py), [TOMTOM validation](./Analysis/11_5_DiffExpress_TOMTOM.py), [cCRE annotation](./Analysis/11_6_DiffExpress_cCRE.py)
- **12 - LDSC (LD Score Regression)**: [Continuous S-LDSC](./Analysis/12_s_ldsc_continous/)