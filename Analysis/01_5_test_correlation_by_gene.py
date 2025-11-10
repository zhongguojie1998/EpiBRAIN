#!/usr/bin/env python
"""
01_5_test_correlation_by_gene.py

Calculate gene-level correlation metrics for model predictions.
Aggregates predictions over gene regions and computes correlation statistics.

Data Transformation Pipeline:
-----------------------------
Forward (during preprocessing - Model/data/data_utils.py):
    1. sum_three_quarter: y = x^(3/4)
    2. soft_clip (clip_soft=48.0): if y > 48: y = 47 + sqrt(y - 47)
    3. scale: y = scale * y

Reverse (this script - untransform_predictions):
    1. undo scale: y = y / scale
    2. undo soft_clip: if y > 48: y = 47 + (y - 47)^2
    3. undo three_quarter: y = y^(4/3)

The untransform brings predictions back to the original scale before computing
gene-level aggregations and correlations.
"""

import os
import gc
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
import torch
import pyranges as pr
import pybedtools
from intervaltree import IntervalTree
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import quantile_transform
from tqdm import tqdm
from omegaconf import OmegaConf

# Import pygene for GTF parsing
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pygene


def untransform_predictions(data, label_meta=None, scale=1.0, clip_soft=48.0, sum_stat="sum_three_quarter"):
    """
    Untransform model predictions back to original scale.

    Reverses the forward transformations applied during data preprocessing:
    1. Scale multiplication: y = scale * y
    2. Soft clipping: if y > clip_soft: y = (clip_soft - 1) + sqrt(y - clip_soft + 1)
    3. Three-quarter power: y = y^(3/4) for sum_three_quarter

    Args:
        data: numpy array of predictions to untransform
        scale: scale factor applied in forward transform (default: 1.0)
        clip_soft: soft clipping threshold (default: 48.0)
        sum_stat: summary statistic used (default: "sum_three_quarter")

    Returns:
        Untransformed data in original scale
    """
    data = data.copy()

    if label_meta is not None:
        # do it for each trial based on label_meta
        for i, row in label_meta.iterrows():
            trial_scale = row.get('scale', 1.0)
            trial_clip_soft = row.get('clip_soft', 48.0)
            trial_sum_stat = row.get('sum_stat', 'sum_three_quarter')

            # Step 1: Undo scale
            if trial_scale != 1.0:
                data[:, :, i] = data[:, :, i] / trial_scale

            # Step 2: Undo soft clip
            # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
            # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
            if trial_clip_soft is not None:
                clip_mask = data[:, :, i] > trial_clip_soft
                data[clip_mask, i] = (trial_clip_soft - 1) + (data[clip_mask, i] - (trial_clip_soft - 1)) ** 2

            # Step 3: Undo three-quarter power
            # Forward: x = x^(3/4)
            # Reverse: x = x^(4/3)
            if trial_sum_stat == "sum_three_quarter":
                data[:, :, i] = data[:, :, i] ** (4.0 / 3.0)
            elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
                data[:, :, i] = (data[:, :, i] + 1) ** 2 - 1
            elif trial_sum_stat in ['sum', 'mean', "avg"]:
                # no transformation applied
                pass
            else:
                raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")
    else:
        # Step 1: Undo scale
        if scale != 1.0:
            data = data / scale

        # Step 2: Undo soft clip
        # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
        # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
        if clip_soft is not None:
            clip_mask = data > clip_soft
            data[clip_mask] = (clip_soft - 1) + (data[clip_mask] - (clip_soft - 1)) ** 2

        # Step 3: Undo three-quarter power
        # Forward: x = x^(3/4)
        # Reverse: x = x^(4/3)
        if sum_stat == "sum_three_quarter":
            data = data ** (4.0 / 3.0)
        elif sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
            data = (data + 1) ** 2 - 1
        elif sum_stat in ['sum', 'mean', "avg"]:
            # no transformation applied
            pass
        else:
            raise ValueError(f"Unknown sum_stat: {sum_stat}")

    return data


def apply_transform(data, transform_type="none"):
    """
    Apply transformation to data.

    Args:
        data: numpy array of shape (n_genes, n_trials)
        transform_type: str, one of "none", "log", "quantile", "log_quantile"

    Returns:
        Transformed data
    """
    if transform_type == "none":
        return data

    data = data.copy()

    if transform_type == "log":
        # Apply log1p transformation (log(1 + x))
        data = np.log1p(np.maximum(data, 0))

    elif transform_type == "quantile":
        # Apply quantile normalization
        data = quantile_transform(
            data, output_distribution="normal", n_quantiles=min(1000, data.shape[0])
        )

    elif transform_type == "log_quantile":
        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(
            data, output_distribution="normal", n_quantiles=min(1000, data.shape[0])
        )

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return data


def make_genes_exon(genes_bed_file: str, genes_gtf_file: str, out_dir: str):
    """Make a BED file with each genes' exons, excluding exons overlapping
      across genes.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write gene exons
    agenes_bed_file = "%s/genes_all.bed" % out_dir
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        # collect exons
        gene_intervals = IntervalTree()
        for tx_id, tx in gene.transcripts.items():
            for exon in tx.exons:
                gene_intervals[exon.start - 1 : exon.end] = True

        # union
        gene_intervals.merge_overlaps()

        # write
        for interval in sorted(gene_intervals):
            cols = [
                gene.chrom,
                str(interval.begin),
                str(interval.end),
                gene_id,
                ".",
                gene.strand,
            ]
            print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping exons
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_exons = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=True, wo=True):
        gene1_id = overlap[3]
        gene1_start = int(overlap[1])
        gene1_end = int(overlap[2])
        overlapping_exons.add((gene1_id, gene1_start, gene1_end))

        gene2_id = overlap[9]
        gene2_start = int(overlap[7])
        gene2_end = int(overlap[8])
        overlapping_exons.add((gene2_id, gene2_start, gene2_end))

    # filter for nonoverlapping exons
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        a = line.split()
        start = int(a[1])
        end = int(a[2])
        gene_id = a[-1]
        if (gene_id, start, end) not in overlapping_exons:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()


def make_genes_span(
    genes_bed_file: str, genes_gtf_file: str, out_dir: str, stranded: bool = True
):
    """Make a BED file with the span of each gene.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
      stranded (bool): Perform stranded intersection.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write all gene spans
    agenes_bed_file = "%s/genes_all.bed" % out_dir
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        start, end = gene.span()
        cols = [gene.chrom, str(start - 1), str(end), gene_id, ".", gene.strand]
        print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping genes
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_genes = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=stranded, wo=True):
        gene1_id = overlap[3]
        gene2_id = overlap[7]
        if gene1_id != gene2_id:
            overlapping_genes.add(gene1_id)
            overlapping_genes.add(gene2_id)

    # filter for nonoverlapping genes
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        gene_id = line.split()[-1]
        if gene_id not in overlapping_genes:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()
    
    
def aggregate_genes_from_predictions(
    predictions, targets, label_meta, sequences_bed, genes_bed_file, split, pool_width, filter_to_full_length_gene=True
):
    """
    Aggregate predictions and targets by gene.

    Args:
        predictions: Tensor of predictions (n_sequences, n_bins, n_trials)
        targets: Tensor of targets (n_sequences, n_bins, n_trials)
        sequences_bed: Path to sequences BED file
        genes_bed_file: Path to genes BED file
        split: Dataset split name (e.g., 'test')
        pool_width: Width of each prediction bin in bp
        untransform: Whether to untransform predictions/targets (default: True)
        scale: Scale factor for untransform (default: 1.0)
        clip_soft: Soft clipping threshold for untransform (default: 48.0)
        sum_stat: Summary statistic used in preprocessing (default: "sum_three_quarter")

    Returns:
        gene_targets: Array of gene-level target values (n_genes, n_trials)
        gene_preds: Array of gene-level predictions (n_genes, n_trials)
        gene_ids: List of gene IDs
        gene_within: Array of within-gene correlations (n_genes, n_trials)
    """
    # Read sequences
    seqs_df = pd.read_csv(
        sequences_bed,
        sep="\t",
        names=["Chromosome", "Start", "End", "Name"],
    )
    seqs_df = seqs_df[seqs_df.Name == split.lower()].reset_index(drop=True)
    seqs_pr = pr.PyRanges(seqs_df)

    # Read genes
    genes_pr = pr.read_bed(genes_bed_file)

    # Count gene normalization lengths and get strand
    gene_lengths = {}
    gene_strand = {}
    for line in open(genes_bed_file):
        a = line.rstrip().split("\t")
        gene_id = a[3]
        gene_seg_len = int(a[2]) - int(a[1])
        gene_lengths[gene_id] = gene_lengths.get(gene_id, 0) + gene_seg_len
        gene_strand[gene_id] = a[5]

    # Intersect sequences with genes
    print("Intersecting sequences with genes...")
    seqs_genes_pr = seqs_pr.join(genes_pr)

    # Filter out genes whose exons are not all in one sequence
    if filter_to_full_length_gene:
        print("Filtering genes with all exons in one sequence...")
        seqs_genes_df = seqs_genes_pr.df
        # get pairs of sequence-gene
        seqs_genes_df['ID'] = seqs_genes_df['Chromosome'].astype(str) + ":" + seqs_genes_df['Start'].astype(str) + "-" + seqs_genes_df['End'].astype(str) + "_" + seqs_genes_df['Name_b']
        # count the number of sequence-gene pairs
        valid_ids = []
        for seq_gene_id, count in seqs_genes_df['ID'].value_counts().items():
            gene = seq_gene_id.split("_")[1]
            # check the count matches the number of exons for that gene
            if count == len(genes_pr[genes_pr.Name == gene].df):
                valid_ids.append(seq_gene_id)
        print(f"Kept {len(valid_ids)} gene-sequences with all exons in one sequence (filtered {len(seqs_genes_df['ID'].unique()) - len(valid_ids)} genes-sequeces)")
        # filter seqs_genes_pr to only valid ids
        seqs_genes_pr = seqs_genes_pr[seqs_genes_df['ID'].isin(valid_ids)]
    # Hash predictions/targets by gene_id
    gene_preds_dict = defaultdict(list)
    gene_targets_dict = defaultdict(list)

    n_sequences = predictions.shape[0]
    # get trials for two strands
    trials_plus_strand = label_meta[label_meta['modality'] != 'RNAplus'].index.to_list()
    trials_minus_strand = label_meta[label_meta['modality'] != 'RNAminus'].index.to_list()
    # create a new label_meta with which modality is 'RNA'
    label_meta_rna = label_meta.iloc[trials_plus_strand].copy()
    label_meta_rna.loc[label_meta_rna['modality'] == 'RNAminus', 'modality'] = 'RNA'
    # rename the trials to 'RNA'
    label_meta_rna.loc[label_meta_rna['modality'] == 'RNAminus', 'trial'] = label_meta_rna.loc[label_meta_rna['modality'] == 'RNAminus', 'trial'].str.replace('RNAminus', 'RNA')
    n_trials = len(label_meta_rna)

    for si in tqdm(range(n_sequences), desc="Processing sequences"):
        seq = seqs_df.iloc[si]

        # Get genes overlapping this sequence
        cseqs_genes_df = seqs_genes_pr[seq.Chromosome].df
        if cseqs_genes_df.shape[0] == 0:
            continue

        seq_genes_df = cseqs_genes_df[cseqs_genes_df.Start == seq.Start]

        for _, seq_gene in seq_genes_df.iterrows():
            gene_id = seq_gene.Name_b
            gene_start = seq_gene.Start_b
            gene_end = seq_gene.End_b
            seq_start = seq_gene.Start

            # Clip boundaries
            gene_seq_start = max(0, gene_start - seq_start)
            gene_seq_end = max(0, gene_end - seq_start)

            # Convert to bin coordinates
            bin_start = int(np.round(gene_seq_start / pool_width))
            bin_end = int(np.round(gene_seq_end / pool_width))

            # Slice gene region, note for RNAminus or RNAplus tracks, we only select one strand based on gene_strand
            if gene_strand[gene_id] == '+':
                trials_to_use = trials_plus_strand
            else:
                trials_to_use = trials_minus_strand
            yhb = predictions[si, bin_start:bin_end][:, trials_to_use].astype("float32")
            yb = targets[si, bin_start:bin_end][:, trials_to_use].astype("float32")

            if len(yb) > 0:
                gene_preds_dict[gene_id].append(yhb)
                gene_targets_dict[gene_id].append(yb)

        if (si + 1) % 128 == 0:
            gc.collect()

    # Aggregate gene bin values into arrays
    gene_targets = []
    gene_preds = []
    gene_ids = sorted(gene_targets_dict.keys())
    gene_within = []

    print(f"Aggregating {len(gene_ids)} genes...")
    for gene_id in tqdm(gene_ids, desc="Aggregating genes"):
        gene_preds_gi = np.concatenate(gene_preds_dict[gene_id], axis=0).astype("float32")
        gene_targets_gi = np.concatenate(gene_targets_dict[gene_id], axis=0).astype("float32")

        if gene_targets_gi.shape[0] == 0:
            print(f"Warning: {gene_id} has no data, skipping")
            continue

        # Compute within-gene correlation before dropping length axis
        gene_corr_gi = np.zeros(n_trials)
        for ti in range(n_trials):
            if (
                gene_preds_gi[:, ti].var() > 1e-6
                and gene_targets_gi[:, ti].var() > 1e-6
            ):
                preds_log = np.log2(gene_preds_gi[:, ti] + 1)
                targets_log = np.log2(gene_targets_gi[:, ti] + 1)
                gene_corr_gi[ti] = pearsonr(preds_log, targets_log)[0]
            else:
                gene_corr_gi[ti] = np.nan
        gene_within.append(gene_corr_gi)

        # Mean coverage per base pair
        gene_preds_gi = gene_preds_gi.mean(axis=0) / float(pool_width)
        gene_targets_gi = gene_targets_gi.mean(axis=0) / float(pool_width)

        # Scale by gene length (is it really necessary in borzoi script?)
        # gene_preds_gi *= gene_lengths[gene_id]
        # gene_targets_gi *= gene_lengths[gene_id]
        # scale to RPKM
        gene_preds_gi *= 1e3
        gene_targets_gi *= 1e3

        gene_preds.append(gene_preds_gi)
        gene_targets.append(gene_targets_gi)

    gene_targets = np.array(gene_targets)
    gene_preds = np.array(gene_preds)
    gene_within = np.array(gene_within)

    return gene_targets, gene_preds, gene_ids, gene_within, label_meta_rna


@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint name")
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"], help="Dataset splits to process")
@click.option("--res_base", required=True, default="./Res", help="Results base directory")
@click.option("--log_base", required=True, default="./logs", help="Logs base directory")
@click.option("--data_base", required=True, default="./Data", help="Data base directory")
@click.option("--genes_gtf", type=str, default="Data/source/gencode.v48.annotation.gtf.gz", help="Path to genes GTF file")
@click.option("--use_span", is_flag=True, default=False, help="Use gene span instead of exons")
@click.option("--pool_width", type=int, default=32, help="Prediction bin width in bp")
@click.option("-t", "--transform", multiple=True,
              type=click.Choice(['none', 'log', 'quantile', 'log_quantile']),
              default=['none', 'log'],
              help="Data transformation(s) to apply before calculating correlation")
@click.option("--no_untransform", is_flag=True, default=False, help="Disable untransforming predictions back to original scale")
def main(exp_name, chk, splits, res_base, log_base, data_base, genes_gtf, use_span, pool_width, transform, no_untransform):
    """Calculate gene-level correlation metrics for model predictions."""

    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    # Create output directories
    gene_out_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/gene_level"
    os.makedirs(f"{gene_out_dir}/raw_data", exist_ok=True)

    # Load config
    config = OmegaConf.load(f"{LOG_BASE}/overall_setting.yaml")
    sequences_bed_path = f"{config.data.preprocess.storage_path}/sequences.bed"

    # Load label metadata
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    # Convert transform tuple to list
    transform_list = list(transform)
    print(f"Using transformations: {transform_list}")

    # Create gene BED file
    print("Creating gene BED file...")
    genes_bed_file = f"{gene_out_dir}/genes.bed"
    if use_span:
        make_genes_span(genes_bed_file, genes_gtf, gene_out_dir)
    else:
        make_genes_exon(genes_bed_file, genes_gtf, gene_out_dir)
    
    # Process each split
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Load predictions
        print(f"Loading predictions from {RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
        test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")

        # Get predictions and targets
        # Shape: (n_sequences, n_bins, n_trials)
        predictions = test_res["pred"]['regression'][:, :, label_meta['dim']].cpu().numpy()
        targets = test_res["label"]['regression'].cpu().numpy()
        # we should reorder the predictions and targets according to test_res['index']
        index_order = np.argsort(test_res['index'])
        predictions = predictions[index_order]
        targets = targets[index_order]
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # untransform the predictions and targets if needed
        if not no_untransform:
            predictions = untransform_predictions(predictions, label_meta=label_meta)
            targets = untransform_predictions(targets, label_meta=label_meta)

        # Aggregate by gene
        gene_targets, gene_preds, gene_ids, gene_within, label_meta_rna = aggregate_genes_from_predictions(
            predictions, targets, label_meta, sequences_bed_path, genes_bed_file, split, pool_width
        )

        print(f"Found {len(gene_ids)} genes with predictions")

        # Save raw gene values (before log transform)
        genes_targets_df = pd.DataFrame(
            gene_targets, index=gene_ids, columns=label_meta_rna["trial"]
        )
        genes_targets_df.to_csv(f"{gene_out_dir}/raw_data/{split}_gene_targets_raw.tsv", sep="\t")

        genes_preds_df = pd.DataFrame(
            gene_preds, index=gene_ids, columns=label_meta_rna["trial"]
        )
        genes_preds_df.to_csv(f"{gene_out_dir}/raw_data/{split}_gene_preds_raw.tsv", sep="\t")

        genes_within_df = pd.DataFrame(
            gene_within, index=gene_ids, columns=label_meta_rna["trial"]
        )
        genes_within_df.to_csv(f"{gene_out_dir}/raw_data/{split}_gene_within.tsv", sep="\t")

        # Calculate metrics for each transformation
        all_metrics = []

        for trans in transform_list:
            print(f"\nProcessing transform: {trans}")

            # Apply transformation
            gene_targets_trans = apply_transform(gene_targets, trans)
            gene_preds_trans = apply_transform(gene_preds, trans)

            # Calculate correlations for each trial
            acc_pearsonr = []
            acc_r2 = []
            acc_wpearsonr = []

            for ti in tqdm(range(len(label_meta_rna)), desc=f"  Calculating metrics ({trans})"):
                # Overall correlation
                r_ti = pearsonr(gene_targets_trans[:, ti], gene_preds_trans[:, ti])[0]
                acc_pearsonr.append(r_ti)

                # R2 score
                r2_ti = explained_variance_score(gene_targets_trans[:, ti], gene_preds_trans[:, ti])
                acc_r2.append(r2_ti)

                # Within-gene correlation (use original within-gene values)
                valid_mask = ~np.isnan(gene_within[:, ti])
                if valid_mask.sum() > 0:
                    wr_ti = gene_within[valid_mask, ti].mean()
                else:
                    wr_ti = np.nan
                acc_wpearsonr.append(wr_ti)

            # Create metrics dataframe
            col_suffix = f"_{trans}" if trans != "none" else ""
            metrics_df = pd.DataFrame({
                "trial": label_meta_rna["trial"],
                "cell_type": label_meta_rna["trial"].str.rsplit("_", n=1).str[0],
                "modality": label_meta_rna["trial"].str.rsplit("_", n=1).str[-1],
                f"pearsonr{col_suffix}": acc_pearsonr,
                f"r2{col_suffix}": acc_r2,
                f"pearsonr_within{col_suffix}": acc_wpearsonr,
            })

            all_metrics.append(metrics_df)

            print(f"  Overall PearsonR: {np.mean(acc_pearsonr):.4f}")
            print(f"  Overall R2: {np.mean(acc_r2):.4f}")
            print(f"  Within-gene PearsonR: {np.nanmean(acc_wpearsonr):.4f}")

        # Merge all metrics
        final_metrics = all_metrics[0]
        for i in range(1, len(all_metrics)):
            final_metrics = final_metrics.merge(
                all_metrics[i][["trial"] + [c for c in all_metrics[i].columns if c != "trial" and c not in final_metrics.columns]],
                on="trial"
            )

        # Save metrics
        metric_file = f"{gene_out_dir}/raw_data/{split}_gene_metrics.csv"
        final_metrics.to_csv(metric_file, index=False)
        print(f"\nSaved metrics to: {metric_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
