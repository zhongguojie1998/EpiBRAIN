#!/usr/bin/env python3
"""
Gene-based DeepLift interpretation script.

This script performs DeepLift interpretation on gene exon regions.
Similar to 02_motif_bed_diff_interpretation_DeepLift.py, but instead of reading
regions from a BED file, it extracts exon regions from a gene specified by name.

Usage:
    python 02_motif_gene_diff_interpretation_DeepLift.py \
        --gene_name GENE_NAME \
        --trial_pos POSITIVE_TRACK \
        --trial_neg NEGATIVE_TRACK1 [NEGATIVE_TRACK2 ...] \
        --exp_name EXP_NAME \
        --chk CHECKPOINT \
        [other options]
"""

import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift
from matplotlib.patches import Rectangle
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
sys.path.append(str(ROOT / "Analysis"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.data_utils import STD_CHR, ModelSeq, annotate_unmap, get_labels
from data.tokenizer import FastaInterval, str_to_one_hot
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger

# Import pygene for GTF parsing
try:
    from pygene import GTF
except ImportError:
    sys.path.append(str(ROOT / "Analysis"))
    from pygene import GTF


def parse_bin_range(bin_range_str):
    """
    Parse bin range string (e.g., "0-2;5-7")
    Returns numpy array of bin indices.
    """
    bins = []
    for bin_str in bin_range_str.split(';'):
        if '-' in bin_str:
            start, end = map(int, bin_str.split('-'))
            bins.extend(range(start, end))
    return np.array(bins)


def calculate_bin_range(start, end, real_start, window_size, n_window):
    """
    Calculate bin range for a genomic region.

    Args:
        start: Region start position
        end: Region end position
        real_start: Start position of the context window
        window_size: Size of each bin
        n_window: Total number of bins

    Returns:
        numpy array of bin indices
    """
    # Calculate which bins overlap the region
    bin_starts = np.arange(real_start, real_start + n_window * window_size, window_size)
    bin_ends = bin_starts + window_size

    # Find bins that overlap with [start, end)
    overlaps = (bin_starts < end) & (bin_ends > start)
    bin_indices = np.where(overlaps)[0]

    return bin_indices


def format_bin_range(bin_indices):
    """
    Format bin indices as a string (e.g., "0-3;5-7").
    """
    if len(bin_indices) == 0:
        return ""

    # Group consecutive bins
    ranges = []
    start = bin_indices[0]
    prev = start

    for i in range(1, len(bin_indices)):
        if bin_indices[i] != prev + 1:
            # End of a consecutive range
            ranges.append(f"{start}-{prev+1}")
            start = bin_indices[i]
        prev = bin_indices[i]

    # Add the last range
    ranges.append(f"{start}-{prev+1}")

    return ';'.join(ranges)


def get_gene_exon_regions(gene_name, gtf_file, window_size, n_window, context_length):
    """
    Extract exon regions for a gene from GTF file and aggregate them into a single region.

    Args:
        gene_name: Gene name to search for
        gtf_file: Path to GTF file
        window_size: Size of each bin
        n_window: Total number of bins
        context_length: Total context length

    Returns:
        List with single tuple: (chr, start, end, strand, bin_range_str, gene_name)
        where bin_range_str contains all bins that overlap with any exon
    """
    logger = BaseLogger(name="GTF Parser", level=logging.INFO)
    logger.info(f"Loading GTF file: {gtf_file}")

    gtf = GTF(gtf_file, trim_dot=False)

    # Find gene by name
    gene_obj = None
    gene_id = None
    for gid, gene in gtf.genes.items():
        if gene.name == gene_name:
            gene_obj = gene
            gene_id = gid
            break

    if gene_obj is None:
        raise ValueError(f"Gene '{gene_name}' not found in GTF file")

    logger.info(f"Found gene: {gene_name} ({gene_id})")
    logger.info(f"  Chromosome: {gene_obj.chrom}")
    logger.info(f"  Strand: {gene_obj.strand}")
    logger.info(f"  Transcripts: {len(gene_obj.transcripts)}")

    # Collect all unique exons across all transcripts
    exon_set = set()
    for tx_id, tx in gene_obj.transcripts.items():
        for exon in tx.exons:
            exon_set.add((exon.start, exon.end))

    # Sort exons by position
    exons = sorted(list(exon_set))
    logger.info(f"  Total unique exons: {len(exons)}")

    if len(exons) == 0:
        raise ValueError(f"No exons found for gene {gene_name}")

    # Find the overall gene span (from first exon start to last exon end)
    gene_start = exons[0][0]
    gene_end = exons[-1][1]
    gene_center = (gene_start + gene_end) // 2

    logger.info(f"  Gene span: {gene_obj.chrom}:{gene_start}-{gene_end}")

    # Create a context window centered on the gene
    region_start = max(0, gene_center - context_length // 2)
    region_end = region_start + context_length

    logger.info(f"  Context window: {gene_obj.chrom}:{region_start}-{region_end}")

    # Calculate bin ranges for all exons within this context window
    all_bin_indices = set()
    for i, (exon_start, exon_end) in enumerate(exons):
        bin_indices = calculate_bin_range(exon_start, exon_end, region_start, window_size, n_window)
        all_bin_indices.update(bin_indices)
        logger.info(f"    Exon {i+1}: {gene_obj.chrom}:{exon_start}-{exon_end} "
                   f"-> bins {list(bin_indices)}")

    # Convert to sorted array and format as string
    all_bin_indices = np.array(sorted(all_bin_indices))
    bin_range_str = format_bin_range(all_bin_indices)

    logger.info(f"  Aggregated bin range: {bin_range_str} ({len(all_bin_indices)} bins)")

    # Return single region with aggregated bin ranges
    regions = [(
        gene_obj.chrom,
        region_start,
        region_end,
        gene_obj.strand,
        bin_range_str,
        gene_name
    )]

    return regions


def untransform_predictions_numpy(preds, label_meta_row):
    """
    Untransform predictions back to original scale (numpy version).
    """
    preds = preds.copy()

    trial_scale = label_meta_row.get('scale', 1.0)
    trial_clip_soft = label_meta_row.get('clip_soft', 48.0)
    trial_sum_stat = label_meta_row.get('sum_stat', 'sum_three_quarter')

    # Step 1: Undo scale
    if trial_scale != 1.0:
        preds = preds / trial_scale

    # Step 2: Undo soft clip
    if trial_clip_soft is not None:
        clip_mask = preds > trial_clip_soft
        preds[clip_mask] = (trial_clip_soft - 1) + (preds[clip_mask] - (trial_clip_soft - 1)) ** 2

    # Step 3: Undo power transform
    if trial_sum_stat == "sum_three_quarter":
        preds = preds ** (4.0 / 3.0)
    elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
        preds = (preds + 1) ** 2 - 1
    elif trial_sum_stat in ['sum', 'mean', "avg"]:
        pass
    else:
        raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

    return preds


def clean_trial_name(trial_str, keep_suffix=True):
    """
    Clean trial name(s) for shorter filenames.
    Removes 'MiniAtlas-' prefix and 'RNA' substring.
    """
    trials = trial_str.split(';')
    cleaned_trials = []

    for trial in trials:
        cleaned = trial.replace('MiniAtlas-', '')
        cleaned = cleaned.replace('RNA', '')

        if not keep_suffix:
            cleaned = cleaned.replace('_plus', '').replace('_minus', '')

        cleaned_trials.append(cleaned)

    return '-'.join(cleaned_trials)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, output_key, target_pos_dims, target_neg_dims, bin_range,
                 no_untransform=False, label_meta_rows_pos=None, label_meta_rows_neg=None):
        super().__init__()
        self.model = model
        self.output_key = output_key
        self.no_untransform = no_untransform
        self.label_meta_rows_pos = label_meta_rows_pos
        self.label_meta_rows_neg = label_meta_rows_neg

        # Convert arrays to tensors
        if isinstance(target_pos_dims, np.ndarray):
            self.target_pos_dims = torch.from_numpy(target_pos_dims).long()
        else:
            self.target_pos_dims = torch.tensor(target_pos_dims, dtype=torch.long)

        if isinstance(target_neg_dims, np.ndarray):
            self.target_neg_dims = torch.from_numpy(target_neg_dims).long()
        else:
            self.target_neg_dims = torch.tensor(target_neg_dims, dtype=torch.long)

        if isinstance(bin_range, np.ndarray):
            self.bin_range = torch.from_numpy(bin_range).long()
        else:
            self.bin_range = torch.tensor(bin_range, dtype=torch.long)

    def forward(self, x):
        output_dict = self.model(x)
        output = output_dict[self.output_key]  # [batch, N, dim]

        # Untransform predictions if requested
        if not self.no_untransform:
            # Untransform positive trials
            if self.label_meta_rows_pos is not None:
                device = output.device
                target_pos_dims_np = self.target_pos_dims.cpu().numpy()
                for i, pos_dim in enumerate(target_pos_dims_np):
                    output_pos_raw = output[:, :, pos_dim]
                    output_pos_raw = self._untransform_single(output_pos_raw, self.label_meta_rows_pos.iloc[i])
                    output[:, :, pos_dim] = output_pos_raw

            # Untransform negative trials
            if self.label_meta_rows_neg is not None:
                device = output.device
                target_neg_dims_np = self.target_neg_dims.cpu().numpy()
                for i, neg_dim in enumerate(target_neg_dims_np):
                    output_neg_raw = output[:, :, neg_dim]
                    output_neg_raw = self._untransform_single(output_neg_raw, self.label_meta_rows_neg.iloc[i])
                    output[:, :, neg_dim] = output_neg_raw

        # Move index tensors to same device as output
        device = output.device
        bin_range = self.bin_range.to(device)
        target_pos_dims = self.target_pos_dims.to(device)
        target_neg_dims = self.target_neg_dims.to(device)

        # Calculate difference: mean of positive trials - mean of negative trials
        output_pos = output[:, bin_range, :][:, :, target_pos_dims].mean(dim=(1, 2))
        output_neg = output[:, bin_range, :][:, :, target_neg_dims].mean(dim=(1, 2))
        return output_pos - output_neg

    def _untransform_single(self, preds, label_meta_row):
        """Untransform predictions for a single trial."""
        trial_scale = label_meta_row.get('scale', 1.0)
        trial_clip_soft = label_meta_row.get('clip_soft', 48.0)
        trial_sum_stat = label_meta_row.get('sum_stat', 'sum_three_quarter')

        if trial_scale != 1.0:
            preds = preds / trial_scale

        if trial_clip_soft is not None:
            clip_mask = preds > trial_clip_soft
            preds = torch.where(
                clip_mask,
                (trial_clip_soft - 1) + (preds - (trial_clip_soft - 1)) ** 2,
                preds
            )

        if trial_sum_stat == "sum_three_quarter":
            preds = preds ** (4.0 / 3.0)
        elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
            preds = (preds + 1) ** 2 - 1
        elif trial_sum_stat in ['sum', 'mean', "avg"]:
            pass
        else:
            raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

        return preds


def process_region_chunk(args):
    """Process a chunk of regions for interpretation."""
    (
        exp_name,
        chk,
        region_chunk_data,
        baseline_types,
        config_path,
        checkpoint_path,
        label_meta_path,
        res_base,
        device,
        force_restart,
        save_raw,
        prefix,
        use_head,
        num_threads,
        no_untransform,
        trial_pos,
        trial_neg,
    ) = args

    # Set torch threads for CPU
    if device == "cpu" and num_threads is not None:
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

    save_base = f"{res_base}/{exp_name}/analysis_{chk}/raw_data"

    # Load config
    myconfig = load_config(config_name=config_path, skip_validation=True)
    logger = BaseLogger(name=f"Interpretation-{device}", level=logging.INFO)

    # Get label information
    label_meta = pd.read_csv(label_meta_path, index_col=None)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if myconfig.model.get("use_compile", False):
        logger.info("Disabling model compilation for interpretation")
        myconfig.model.use_compile = False

    model = setup_model(myconfig, logger)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Get positive trial dims and metadata with grep-style substring matching
    trial_pos_list = trial_pos.split(';')
    matched_pos_trials = []
    for pattern in trial_pos_list:
        # Use substring matching (grep-style)
        matches = label_meta[label_meta['trial'].str.contains(pattern, na=False, regex=False)]
        matched_trials = matches['trial'].tolist()
        matched_pos_trials.extend(matched_trials)
        if len(matched_trials) > 0:
            logger.info(f"Pattern '{pattern}' matched {len(matched_trials)} positive trials: {', '.join(matched_trials[:5])}{'...' if len(matched_trials) > 5 else ''}")
        else:
            logger.warning(f"Pattern '{pattern}' did not match any positive trials")

    # Remove duplicates while preserving order
    matched_pos_trials = list(dict.fromkeys(matched_pos_trials))
    label_meta_rows_pos = label_meta[label_meta['trial'].isin(matched_pos_trials)]
    trial_pos_dims = label_meta_rows_pos.dim.values

    if len(trial_pos_dims) == 0:
        logger.error(f"No valid positive trials found matching patterns: {trial_pos}")
        return

    logger.info(f"Selected {len(matched_pos_trials)} positive trial(s): {', '.join(matched_pos_trials[:5])}{'...' if len(matched_pos_trials) > 5 else ''}")

    # Get negative trial dims and metadata with grep-style substring matching
    trial_neg_list = trial_neg.split(';')
    matched_neg_trials = []
    for pattern in trial_neg_list:
        # Use substring matching (grep-style)
        matches = label_meta[label_meta['trial'].str.contains(pattern, na=False, regex=False)]
        matched_trials = matches['trial'].tolist()
        matched_neg_trials.extend(matched_trials)
        if len(matched_trials) > 0:
            logger.info(f"Pattern '{pattern}' matched {len(matched_trials)} negative trials: {', '.join(matched_trials[:5])}{'...' if len(matched_trials) > 5 else ''}")
        else:
            logger.warning(f"Pattern '{pattern}' did not match any negative trials")

    # Remove duplicates while preserving order
    matched_neg_trials = list(dict.fromkeys(matched_neg_trials))
    label_meta_rows_neg = label_meta[label_meta['trial'].isin(matched_neg_trials)]
    trial_neg_dims = label_meta_rows_neg.dim.values

    if len(trial_neg_dims) == 0:
        logger.error(f"No valid negative trials found matching patterns: {trial_neg}")
        return

    logger.info(f"Selected {len(matched_neg_trials)} negative trial(s): {', '.join(matched_neg_trials[:5])}{'...' if len(matched_neg_trials) > 5 else ''}")

    # Update trial_pos_list and trial_neg_list with matched trials
    trial_pos_list = matched_pos_trials
    trial_neg_list = matched_neg_trials

    # Setup baseline
    baseline_seq_onehots = []
    for baseline_type in baseline_types:
        if baseline_type == "random":
            np.random.seed(myconfig.training.seed)
            baseline_seq = "".join(
                np.random.choice(["A", "T", "C", "G"], size=myconfig.data.context_length, p=[0.3, 0.3, 0.2, 0.2])
            )
        elif baseline_type == "all_zero":
            baseline_seq = "N" * myconfig.data.context_length
        elif baseline_type == "pad":
            baseline_seq = "." * myconfig.data.context_length
        baseline_seq_onehot = str_to_one_hot(baseline_seq)
        baseline_seq_onehots.append((baseline_type, baseline_seq_onehot))

    # Setup tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(myconfig.data.refer_genom),
        context_length=myconfig.data.context_length
    )

    # Process each region
    for idx in range(len(region_chunk_data)):
        chr_name, start, end, strand, bin_range_str, region_name = region_chunk_data[idx]

        # Clean trial names for filenames
        trial_pos_clean = clean_trial_name(trial_pos, keep_suffix=True)
        neg_trial_count = len(trial_neg_list)
        trial_neg_clean = f"other-{neg_trial_count}"

        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{region_name}_{trial_pos_clean}_{trial_neg_clean}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{region_name}_{trial_pos_clean}_{trial_neg_clean}"
        )

        if chr_name not in STD_CHR:
            logger.warning(f"Skipping {chr_name} (not in standard chromosomes)")
            continue

        token_dict = dna_tokenizer(
            chr_name=chr_name, start=start, end=end, return_augs=False, return_rela_idx=True
        )

        test_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]
        test_seq_onehot.requires_grad = True

        # Parse bin_range
        bin_range = parse_bin_range(bin_range_str)

        if len(bin_range) == 0:
            logger.warning(f"Empty bin range for {name_base}, skip")
            continue

        if bin_range.min() < 0 or bin_range.max() >= myconfig.data.preprocess.n_window:
            logger.warning(f"Bin range out of bounds for {name_base}, skip")
            continue

        # Generate predictions for all positive and negative trials
        with torch.no_grad():
            pred_res = model(test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), use_head)
            pred_res_np = pred_res.detach().cpu().numpy()[0]  # [N, dim]

            # Untransform predictions for positive trials
            pred_res_pos_trials = []
            if not no_untransform:
                for i, pos_dim in enumerate(trial_pos_dims):
                    pred_trial = pred_res_np[:, pos_dim].copy()
                    pred_trial = untransform_predictions_numpy(pred_trial, label_meta_rows_pos.iloc[i])
                    pred_res_pos_trials.append(pred_trial)
            else:
                for pos_dim in trial_pos_dims:
                    pred_res_pos_trials.append(pred_res_np[:, pos_dim])

            # Average predictions across positive trials
            pred_res_pos_avg = np.mean(pred_res_pos_trials, axis=0)

            del pred_res

        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Generate labels for all positive and negative trials
        mseqs = [ModelSeq(chr_name, real_start, real_end, "test")]

        unmap_npy = f"{save_base}/label/{name_base}_mseqs_unmap.npy"
        mseqs_unmap = annotate_unmap(
            mseqs,
            myconfig.data.preprocess.unmap_bed,
            myconfig.data.preprocess.context_length,
            myconfig.data.preprocess.window_size,
        )
        np.save(unmap_npy, mseqs_unmap)

        # Get labels for all positive trials and average them
        label_pos_trials = []
        for trial_pos_single in trial_pos_list:
            label_h5_trial = f"{save_base}/label/{name_base}_{trial_pos_single}_label.h5"
            try:
                get_labels(
                    mseqs,
                    blacklist_bed=myconfig.data.preprocess.blacklist_bed,
                    pool_width=myconfig.data.preprocess.window_size,
                    kept_num_after_crop=myconfig.data.preprocess.n_window,
                    seqs_cov_file=label_h5_trial,
                    genome_cov_file=data_config.loc[trial_pos_single, "file"],
                    umap_npy_path=unmap_npy,
                    **data_config.loc[trial_pos_single, ["sum_stat", "baseline_pct", "umap_pct", "scale", "clip", "clip_soft"]].to_dict(),
                )
                with h5py.File(label_h5_trial, "r") as f:
                    label_pos_trials.append(f["targets"][0][:])
            except (ValueError, RuntimeError, IndexError) as e:
                logger.warning(f"Failed to get labels for {trial_pos_single}: {str(e)}")

        if len(label_pos_trials) == 0:
            logger.warning(f"No valid labels for positive trials in {name_base}, skip")
            continue

        # Average labels across positive trials
        label_pos_avg = np.mean(label_pos_trials, axis=0)

        plot_data = [label_pos_avg, pred_res_pos_avg]
        plot_title = [f"{'&'.join(trial_pos_list[:2])}... Target ({len(trial_pos_list)} tracks)",
                      f"{'&'.join(trial_pos_list[:2])}... Pred ({len(trial_pos_list)} tracks)"]

        # Prepare model wrapper
        model.zero_grad()
        model_wrapper = ModelWrapper(
            model, use_head, trial_pos_dims, trial_neg_dims, bin_range,
            no_untransform=no_untransform,
            label_meta_rows_pos=label_meta_rows_pos if not no_untransform else None,
            label_meta_rows_neg=label_meta_rows_neg if not no_untransform else None
        )

        dl_model = DeepLift(model_wrapper, multiply_by_inputs=False, eps=1e-7)

        sample_start_time = time.time()

        for baseline_type, baseline_seq_onehot in baseline_seq_onehots:
            identifier = f"{name_base}_{baseline_type}"
            if not os.path.exists(f"{save_base}/interp_diff/{identifier}.pt") or force_restart:
                dl_model.model.zero_grad()

                input_tensor = test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)
                baseline_tensor = baseline_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)

                attribution = dl_model.attribute(
                    inputs=input_tensor,
                    baselines=baseline_tensor,
                )

                if not torch.isfinite(attribution).all():
                    logger.warning(f"NAN occur in {identifier}")

                attribution_cpu = attribution.detach().cpu().permute(0, 2, 1)
                del attribution

                if device.startswith('cuda'):
                    torch.cuda.empty_cache()

                del input_tensor, baseline_tensor
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()

                if save_raw:
                    torch.save(attribution_cpu, f"{save_base}/interp_diff/{identifier}.pt")
            else:
                attribution_cpu = torch.load(f"{save_base}/interp_diff/{identifier}.pt")

            if not torch.isfinite(attribution_cpu).all():
                logger.warning(f"NAN in {identifier}, skip plotting")
                continue

            # Calculate importance scores
            with torch.no_grad():
                trim = (
                    myconfig.data.context_length // myconfig.data.preprocess.window_size
                    - myconfig.data.preprocess.n_window
                ) // 2

                signal = (attribution_cpu * test_seq_onehot).sum(dim=-1).mean(dim=0)
                signal = signal.reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim]
                signal = signal.mean(dim=-1).detach()

            plot_data.append(signal)
            plot_title.append(f"Importance Score ({baseline_type} baseline)")

        # Plot
        n = len(plot_data) + 1
        height_ratios = [1] * len(plot_data) + [0.1]
        fig, axes = plt.subplots(
            nrows=n, ncols=1, figsize=(8, n * 1.5), sharex=True,
            gridspec_kw={"height_ratios": height_ratios}
        )

        trim = (
            myconfig.data.context_length // myconfig.data.preprocess.window_size
            - myconfig.data.preprocess.n_window
        ) // 2
        x = np.arange(real_start, real_end).reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim, 0]

        for i, ax in enumerate(axes[:-1]):
            ax.plot(x, plot_data[i])
            ax.set_title(plot_title[i])
            ax.set_ylabel(None)
            ax.set_xlabel(None)

        ax = axes[-1]
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(None)
        ax.set_xlabel(None)

        linewidth = 1.5
        rec_width = 1
        ax.hlines(0.5, x.min(), x.max(), color="black", linewidth=linewidth)
        ax.set_title(f"Chromosome {chr_name[3:]}")

        rect = Rectangle(
            (x[bin_range[0]], 0.5 - 0.5 * rec_width),
            x[bin_range[-1]] - x[bin_range[0]],
            rec_width,
            facecolor="lightblue",
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(rect)

        plt.tight_layout()
        plt.savefig(
            f"{res_base}/{exp_name}/analysis_{chk}/plot/interp_diff/{name_base}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        sample_total_time = time.time() - sample_start_time
        logger.info(f"Sample {name_base} completed in {sample_total_time:.2f}s")

        del test_seq_onehot, plot_data, plot_title
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


@click.command()
@click.option("--gene_name", "-g", required=True, type=str, help="Gene name to analyze")
@click.option("--gtf_file", required=False, type=str,
              default="Data/source/gencode.v48.annotation.gtf.gz",
              help="Path to GTF annotation file")
@click.option("--trial_pos", required=True, multiple=True, type=str,
              help="Positive track names (can be specified multiple times)")
@click.option("--trial_neg", required=True, multiple=True, type=str,
              help="Negative track names (can be specified multiple times)")
@click.option("--exp_name", "-e", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option(
    "--baseline",
    "-b",
    required=True,
    multiple=True,
    type=click.Choice(["random", "all_zero", "pad"], case_sensitive=False),
    default=["random"],
)
@click.option("--prefix", required=False, type=str, help="Name prefix for saving files")
@click.option("--log_base", required=True, type=str, default="./logs")
@click.option("--chk_base", required=True, type=str, default="./Chk")
@click.option("--res_base", required=True, type=str, default="./Res")
@click.option("--force_restart", is_flag=True)
@click.option("--save_raw", is_flag=True)
@click.option(
    "--processor",
    required=True,
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
)
@click.option("--num_processes", type=int, default=4, help="Number of processes for parallel processing")
@click.option("--num_threads", type=int, default=None, help="Number of threads per process for CPU mode")
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
@click.option("--no_untransform", is_flag=True, help="Skip untransform")
def main(
    gene_name,
    gtf_file,
    trial_pos,
    trial_neg,
    exp_name,
    chk,
    baseline,
    prefix,
    log_base,
    chk_base,
    res_base,
    force_restart,
    save_raw,
    processor,
    num_processes,
    num_threads,
    use_head,
    no_untransform,
):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_diff", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label", exist_ok=True)

    logger = BaseLogger(name="Interpretation", level=logging.INFO)

    # Load config to get window size and other parameters
    config_path = f"{LOG_BASE}/{exp_name}/overall_setting.yaml"
    myconfig = load_config(config_name=config_path, skip_validation=True)

    # Convert trial_pos and trial_neg tuples to semicolon-separated strings
    trial_pos_str = ';'.join(trial_pos)
    trial_neg_str = ';'.join(trial_neg)

    # Validate we have at least one positive and negative track
    if len(trial_pos) == 0:
        logger.error("At least one positive track must be specified with --trial_pos")
        return
    if len(trial_neg) == 0:
        logger.error("At least one negative track must be specified with --trial_neg")
        return

    logger.info(f"Processing {len(trial_pos)} positive track(s): {', '.join(trial_pos)}")
    logger.info(f"Against {len(trial_neg)} negative track(s): {', '.join(trial_neg)}")

    # Get gene exon regions
    logger.info(f"Extracting exon regions for gene: {gene_name}")
    regions = get_gene_exon_regions(
        gene_name,
        gtf_file,
        myconfig.data.preprocess.window_size,
        myconfig.data.preprocess.n_window,
        myconfig.data.context_length
    )

    if len(regions) == 0:
        logger.error(f"No valid exon regions found for gene {gene_name}")
        return

    logger.info(f"Total regions to process: {len(regions)}")

    # Check device availability
    if processor == "gpu":
        available_devices = torch.cuda.device_count()
        if available_devices == 0:
            logger.warning("No GPU found. Using CPU")
            processor = "cpu"

    if processor == "cpu":
        available_devices = mp.cpu_count()

    if num_processes > available_devices:
        logger.warning(f"Requested {num_processes} {processor} but only {available_devices} available")
        num_processes = available_devices

    # Determine thread count for CPU mode
    if processor == "cpu":
        if num_threads is None:
            num_threads_per_process = 1 if num_processes > 1 else None
        else:
            num_threads_per_process = num_threads
    else:
        num_threads_per_process = None

    # Split regions into chunks
    chunks = []
    n = len(regions)
    base = n // num_processes
    extra = n % num_processes
    start = 0
    for i in range(num_processes):
        size = base + (1 if i < extra else 0)
        end = start + size
        if start < end:
            chunk = regions[start:end]
            chunks.append(chunk)
        start = end

    logger.info(f"Split {len(regions)} regions into {len(chunks)} chunks")

    # Prepare arguments for each process
    process_args = []
    for process_id, chunk in enumerate(chunks):
        args = (
            exp_name,
            chk,
            chunk,
            baseline,
            config_path,
            f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt",
            f"{LOG_BASE}/{exp_name}/regression_label_meta.csv",
            RES_BASE,
            f"cuda:{process_id}" if processor == "gpu" else "cpu",
            force_restart,
            save_raw,
            prefix,
            use_head,
            num_threads_per_process,
            no_untransform,
            trial_pos_str,
            trial_neg_str,
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if num_processes > 1:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_region_chunk, process_args)
    else:
        for arg in tqdm(process_args):
            process_region_chunk(arg)

    logger.info("Interpretation complete!")


if __name__ == "__main__":
    main()
