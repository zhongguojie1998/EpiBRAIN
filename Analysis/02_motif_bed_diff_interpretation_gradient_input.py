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
from matplotlib.patches import Rectangle
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.data_utils import STD_CHR, ModelSeq, annotate_unmap, get_labels
from data.tokenizer import FastaInterval, str_to_one_hot, one_hot_reverse_complement
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger


def parse_bin_range(bin_range_str):
    """
    Parse bin range string from BED file (e.g., "0-2;5-7")
    Returns numpy array of bin indices.
    """
    bins = []
    for bin_str in bin_range_str.split(';'):
        if '-' in bin_str:
            start, end = map(int, bin_str.split('-'))
            # Use Python's half-open interval [start, end)
            bins.extend(range(start, end))
    return np.array(bins)


def clean_trial_name(trial_str, keep_suffix=True):
    """
    Clean trial name(s) for shorter filenames.
    Removes 'MiniAtlas-' prefix and 'RNA' substring.

    Args:
        trial_str: Trial name(s), potentially semicolon-separated
        keep_suffix: If False, removes '_plus' and '_minus' suffixes

    Examples:
        'MiniAtlas-ACBGM_RNAplus' -> 'ACBGM_plus' (keep_suffix=True)
        'MiniAtlas-ACBGM_RNAplus' -> 'ACBGM' (keep_suffix=False)
        'MiniAtlas-AST_RNAplus;MiniAtlas-L23IT_RNAplus' -> 'AST_plus-L23IT_plus' (keep_suffix=True)
        'MiniAtlas-AST_RNAplus;MiniAtlas-L23IT_RNAplus' -> 'AST-L23IT' (keep_suffix=False)
    """
    # Split on semicolon if multiple trials
    trials = trial_str.split(';')
    cleaned_trials = []

    for trial in trials:
        # Remove MiniAtlas- prefix
        cleaned = trial.replace('MiniAtlas-', '')
        # Remove RNA substring
        cleaned = cleaned.replace('RNA', '')

        # Optionally remove plus/minus suffixes
        if not keep_suffix:
            cleaned = cleaned.replace('_plus', '').replace('_minus', '')

        cleaned_trials.append(cleaned)

    # Join with dash
    return '-'.join(cleaned_trials)


def gradients_input_attribution_diff(
    model,
    seq_input,
    output_key,
    target_pos_dim,
    target_neg_dims,
    bin_range,
    label_meta_row_pos=None,
    label_meta_rows_neg=None,
    pseudo_count=0.0,
    no_untransform=False,
    use_mean=True,
    subtract_avg=True,
    input_gate=False,
):
    """
    Compute gradient×input attribution for differential expression.

    This computes attribution for the difference between a positive track
    and the mean of negative tracks: log(pos_track) - log(mean(neg_tracks))

    Args:
        model: PyTorch model
        seq_input: Input sequence tensor [batch, channels, length]
        output_key: Which output head to use
        target_pos_dim: Positive target dimension/track
        target_neg_dims: Negative target dimensions/tracks (numpy array or tensor)
        bin_range: Which bins to aggregate over (numpy array or tensor)
        label_meta_row_pos: Row from label_meta for positive trial
        label_meta_rows_neg: List of rows from label_meta for negative trials
        pseudo_count: Small constant added before taking log
        no_untransform: Skip untransforming (if predictions are already in count space)
        use_mean: Use mean aggregation (vs sum) over bins
        subtract_avg: Subtract mean across nucleotides at each position
        input_gate: Multiply gradients by input (gradient × input), default False

    Returns:
        Gradients tensor [batch, length, channels]
    """
    # Convert arrays to tensors if needed
    if isinstance(target_neg_dims, np.ndarray):
        target_neg_dims = torch.from_numpy(target_neg_dims).long()
    elif not isinstance(target_neg_dims, torch.Tensor):
        target_neg_dims = torch.tensor(target_neg_dims, dtype=torch.long)

    if isinstance(bin_range, np.ndarray):
        bin_range = torch.from_numpy(bin_range).long()
    elif not isinstance(bin_range, torch.Tensor):
        bin_range = torch.tensor(bin_range, dtype=torch.long)

    # Move to same device as input
    device = seq_input.device
    target_neg_dims = target_neg_dims.to(device)
    bin_range = bin_range.to(device)

    # Enable gradient computation for input
    seq_input.requires_grad_(True)

    # Forward pass
    output_dict = model(seq_input)
    preds = output_dict[output_key]  # [batch, N_bins, dim]

    
    # Select bins of interest
    preds_slice = preds[:, bin_range, :]  # [batch, len(bin_range), dim]

    # Get positive and negative predictions
    preds_pos = preds_slice[:, :, target_pos_dim]  # [batch, len(bin_range)]
    preds_neg = preds_slice[:, :, target_neg_dims]  # [batch, len(bin_range), len(neg_dims)]
    
    # Untransform predictions using label_meta (reverse of training preprocessing)
    # Following the logic from 01_5_test_correlation_by_gene.py
    if not no_untransform and label_meta_row_pos is not None:
        # Get transformation parameters
        trial_scale = label_meta_row_pos.get('scale', 1.0)
        trial_clip_soft = label_meta_row_pos.get('clip_soft', 48.0)
        trial_sum_stat = label_meta_row_pos.get('sum_stat', 'sum_three_quarter')

        # Step 1: Undo scale (vectorized across all dimensions)
        if trial_scale != 1.0:
            preds_pos = preds_pos / trial_scale
            preds_neg = preds_neg / trial_scale

        # Step 2: Undo soft clip (vectorized across all dimensions)
        if trial_clip_soft is not None:
            clip_mask = preds_pos > trial_clip_soft
            preds_pos = torch.where(
                clip_mask,
                (trial_clip_soft - 1) + (preds_pos - (trial_clip_soft - 1)) ** 2,
                preds_pos
            )
            clip_mask = preds_neg > trial_clip_soft
            preds_neg = torch.where(
                clip_mask,
                (trial_clip_soft - 1) + (preds_neg - (trial_clip_soft - 1)) ** 2,
                preds_neg
            )

        # Step 3: Undo power transform based on sum_stat (vectorized across all dimensions)
        if trial_sum_stat == "sum_three_quarter":
            preds_pos = preds_pos ** (4.0 / 3.0)
            preds_neg = preds_neg ** (4.0 / 3.0)
        elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
            preds_pos = (preds_pos + 1) ** 2 - 1
            preds_neg = (preds_neg + 1) ** 2 - 1
        elif trial_sum_stat not in ['sum', 'mean', "avg"]:
            raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

    # Aggregate over bins
    if use_mean:
        preds_pos_agg = preds_pos.mean(dim=-1)  # [batch]
        preds_neg_agg = preds_neg.mean(dim=(1, 2))  # [batch]
    else:
        preds_pos_agg = preds_pos.sum(dim=-1)  # [batch]
        preds_neg_agg = preds_neg.mean(dim=-1).sum(dim=-1)  # [batch]

    # Compute differential score to differentiate
    # log(pos) - log(neg) = log(pos/neg)
    score = torch.log(preds_pos_agg + pseudo_count + 1e-6) - torch.log(preds_neg_agg + pseudo_count + 1e-6)

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=score,
        inputs=seq_input,
        grad_outputs=torch.ones_like(score),
        create_graph=False,
        retain_graph=False,
    )[0]

    # grads shape: [batch, channels, length]
    # Permute to [batch, length, channels] for processing
    grads = grads.permute(0, 2, 1)

    # Subtract mean across nucleotides at each position
    if subtract_avg:
        grads = grads - grads.mean(dim=-1, keepdim=True)

    # Multiply by input (gradient × input)
    if input_gate:
        seq_input_permuted = seq_input.permute(0, 2, 1)  # [batch, length, channels]
        grads = grads * seq_input_permuted

    return grads


def swap_rna_strand(trial_name):
    """
    Swap RNA strand in trial name (RNAplus <-> RNAminus).

    Args:
        trial_name: Trial name string (e.g., 'MiniAtlas-ACBGM_RNAplus')

    Returns:
        Trial name with swapped strand (e.g., 'MiniAtlas-ACBGM_RNAminus')
    """
    if 'RNAplus' in trial_name:
        return trial_name.replace('RNAplus', 'RNAminus')
    elif 'RNAminus' in trial_name:
        return trial_name.replace('RNAminus', 'RNAplus')
    else:
        # No strand information, return as is
        return trial_name


def unaugment_grads(grads, fwdrc=True):
    """
    Undo sequence augmentation for reverse complement.

    Args:
        grads: Gradient tensor [batch, length, channels]
        fwdrc: If False, gradients are from reverse complement and need to be transformed back

    Returns:
        Transformed gradients [batch, length, channels]
    """
    if not fwdrc:
        # Reverse complement: need to reverse and swap nucleotides

        # Reverse the sequence (along length dimension)
        grads = torch.flip(grads, dims=[1])

        # Swap A and T (indices 0 and 3)
        grads_copy = grads.clone()
        grads[:, :, 0] = grads_copy[:, :, 3]
        grads[:, :, 3] = grads_copy[:, :, 0]

        # Swap C and G (indices 1 and 2)
        grads[:, :, 1] = grads_copy[:, :, 2]
        grads[:, :, 2] = grads_copy[:, :, 1]

    return grads


def process_region_chunk(args):
    (
        exp_name,
        chk,
        region_chunk_data,
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
        pseudo_count,
        no_untransform,
        use_mean,
        subtract_avg,
        input_gate,
        rc,
        no_plot,
    ) = args

    # Set torch threads for CPU
    if device == "cpu" and num_threads is not None:
        torch.set_num_threads(num_threads)
        # Also set environment variables for BLAS/LAPACK libraries
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

    save_base = f"{res_base}/{exp_name}/analysis_{chk}/raw_data"

    # Load config
    myconfig = load_config(config_name=config_path, skip_validation=True)
    logger = BaseLogger(name=f"GradInputDiff-{device}", level=logging.INFO)

    # Get label information
    label_meta = pd.read_csv(label_meta_path, index_col=None)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Disable compilation for interpretation tasks
    if myconfig.model.get("use_compile", False):
        logger.info("Disabling model compilation for interpretation")
        myconfig.model.use_compile = False

    model = setup_model(myconfig, logger)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Setup tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
    )

    # Process each region in the chunk
    for idx in range(len(region_chunk_data)):
        # Read from BED file: chr, start, end, strand, bin_range_str, gene_name, trial_pos, trial_neg
        chr_name, start, end, strand, bin_range_str, gene_name, trial_pos, trial_neg = region_chunk_data.iloc[idx, [0, 1, 2, 3, 4, 5, 6, 7]]

        # Clean trial names for shorter filenames
        trial_pos_clean = clean_trial_name(trial_pos, keep_suffix=True)
        neg_trial_count = len(trial_neg.split(';'))
        trial_neg_clean = f"other-{neg_trial_count}"

        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{gene_name}_{trial_pos_clean}_{trial_neg_clean}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{gene_name}_{trial_pos_clean}_{trial_neg_clean}"
        )

        try:
            trial_pos_dim = int(label_meta.dim[label_meta['trial'] == trial_pos].values[0])
            # Get the label_meta row for positive trial (for untransform parameters)
            label_meta_row_pos = label_meta[label_meta['trial'] == trial_pos].iloc[0]
        except:
            logger.warning(f"{trial_pos} cannot be found in label meta, skip")
            continue

        # get negative trial dims and label_meta rows
        neg_trials = trial_neg.split(';')
        trial_neg_dims = label_meta.dim[label_meta['trial'].isin(neg_trials)].values
        # Get label_meta rows for negative trials (for untransform parameters)
        label_meta_rows_neg = [label_meta[label_meta['trial'] == t].iloc[0] for t in neg_trials if len(label_meta[label_meta['trial'] == t]) > 0]
        if len(trial_neg_dims) == 0:
            logger.warning(f"No valid negative trials found in {trial_neg}, skip")
            continue

        # Prepare reverse complement trial dims (swap RNAplus <-> RNAminus)
        if rc:
            # Swap strand for positive trial
            trial_pos_rev = swap_rna_strand(trial_pos)
            try:
                trial_pos_dim_rev = int(label_meta.dim[label_meta['trial'] == trial_pos_rev].values[0])
                label_meta_row_pos_rev = label_meta[label_meta['trial'] == trial_pos_rev].iloc[0]
            except:
                logger.warning(f"{trial_pos_rev} (RC of {trial_pos}) cannot be found in label meta, skip")
                continue

            # Swap strand for negative trials
            neg_trials_rev = [swap_rna_strand(t) for t in neg_trials]
            trial_neg_dims_rev = label_meta.dim[label_meta['trial'].isin(neg_trials_rev)].values
            label_meta_rows_neg_rev = [label_meta[label_meta['trial'] == t].iloc[0] for t in neg_trials_rev if len(label_meta[label_meta['trial'] == t]) > 0]
            if len(trial_neg_dims_rev) == 0:
                logger.warning(f"No valid negative trials found in {neg_trials_rev} (RC of {trial_neg}), skip")
                continue
        else:
            # No RC, set to None
            trial_pos_dim_rev = None
            trial_neg_dims_rev = None
            label_meta_row_pos_rev = None
            label_meta_rows_neg_rev = None

        if not chr_name in STD_CHR:
            continue

        token_dict = dna_tokenizer(
            chr_name=chr_name, start=start, end=end, return_augs=False, return_rela_idx=True
        )

        s_idx, e_idx = token_dict["rela_idx"]
        test_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]

        # Parse bin_range from BED file
        bin_range = parse_bin_range(bin_range_str)

        # Validate bin range
        if len(bin_range) == 0:
            logger.warning(f"Empty bin range for {name_base}, skip")
            continue

        if bin_range.min() < 0 or bin_range.max() >= myconfig.data.preprocess.n_window:
            logger.warning(f"Bin range out of bounds for {name_base}: [{bin_range.min()}, {bin_range.max()}], skip")
            continue

        # generate label and pred
        ## pred
        with torch.no_grad():
            pred_res = model(
                test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), use_head
            )
            pred_res_trial = pred_res.detach().cpu().numpy()[0, :, trial_pos_dim]
            del pred_res

        # Clear GPU cache after model prediction
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        ## label
        unmap_npy = f"{save_base}/label/{name_base}_mseqs_unmap.npy"
        label_h5 = f"{save_base}/label/{name_base}_label.h5"

        mseqs = [ModelSeq(chr_name, real_start, real_end, "test")]

        mseqs_unmap = annotate_unmap(
            mseqs,
            myconfig.data.preprocess.unmap_bed,
            myconfig.data.preprocess.context_length,
            myconfig.data.preprocess.window_size,
        )
        np.save(unmap_npy, mseqs_unmap)

        try:
            get_labels(
                mseqs,
                blacklist_bed=myconfig.data.preprocess.blacklist_bed,
                pool_width=myconfig.data.preprocess.window_size,
                kept_num_after_crop=myconfig.data.preprocess.n_window,
                seqs_cov_file=label_h5,
                genome_cov_file=data_config.loc[trial_pos, "file"],
                umap_npy_path=unmap_npy,
                **data_config.loc[trial_pos, ["sum_stat", "baseline_pct", "umap_pct", "scale", "clip", "clip_soft"]].to_dict(),
            )
            with h5py.File(label_h5, "r") as f:
                label_trial = f["targets"][0]
        except (ValueError, RuntimeError, IndexError) as e:
            logger.warning(f"Failed to get labels for {name_base}: {str(e)}. Skipping this region.")
            continue

        plot_data = [label_trial, pred_res_trial]
        plot_title = [f"{trial_pos} Target", f"{trial_pos} Pred"]

        # Start timer for this sample
        sample_start_time = time.time()

        identifier = f"{name_base}_grad_input"
        if not os.path.exists(f"{save_base}/interp_diff_gradient_input/{identifier}.pt") or force_restart:
            nan_occur = False

            # Prepare input tensor
            input_tensor = test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)

            # Accumulate gradients across forward and (optionally) reverse complement
            attribution_accumulated = None
            num_augmentations = 0

            # Loop over forward and reverse complement
            for rev_comp in [False, True] if rc else [False]:
                # Prepare input (reverse complement if needed)
                if rev_comp:
                    # one_hot_reverse_complement expects [..., length, 4] format
                    # input_tensor is [batch, 4, length], so permute before and after
                    input_tensor_aug = one_hot_reverse_complement(input_tensor.permute(0, 2, 1)).permute(0, 2, 1)
                    # Use strand-swapped trial dimensions for RC
                    target_pos_dim_use = trial_pos_dim_rev
                    target_neg_dims_use = trial_neg_dims_rev
                    label_meta_row_pos_use = label_meta_row_pos_rev
                    label_meta_rows_neg_use = label_meta_rows_neg_rev
                else:
                    input_tensor_aug = input_tensor
                    # Use original trial dimensions
                    target_pos_dim_use = trial_pos_dim
                    target_neg_dims_use = trial_neg_dims
                    label_meta_row_pos_use = label_meta_row_pos
                    label_meta_rows_neg_use = label_meta_rows_neg

                # Calculate gradient attribution for differential expression
                attribution = gradients_input_attribution_diff(
                    model=model,
                    seq_input=input_tensor_aug,
                    output_key=use_head,
                    target_pos_dim=target_pos_dim_use,
                    target_neg_dims=target_neg_dims_use,
                    bin_range=bin_range,
                    label_meta_row_pos=label_meta_row_pos_use,
                    label_meta_rows_neg=label_meta_rows_neg_use,
                    pseudo_count=pseudo_count,
                    no_untransform=no_untransform,
                    use_mean=use_mean,
                    subtract_avg=subtract_avg,
                    input_gate=input_gate,
                )

                # Transform back if reverse complement
                attribution = unaugment_grads(attribution, fwdrc=(not rev_comp))

                # Accumulate
                if attribution_accumulated is None:
                    attribution_accumulated = attribution
                else:
                    attribution_accumulated = attribution_accumulated + attribution
                num_augmentations += 1

                # Clean up
                del attribution
                if rev_comp:
                    del input_tensor_aug

            # Average across augmentations
            attribution_accumulated = attribution_accumulated / num_augmentations

            if not torch.isfinite(attribution_accumulated).all():
                logger.warning(f"NAN occur in {identifier}")
                nan_occur = True

            # Move to CPU immediately and clear GPU memory
            attribution_cpu = attribution_accumulated.detach().cpu()
            del attribution_accumulated

            # Clear GPU cache
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            # Clean up input tensors
            del input_tensor
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            if save_raw:
                torch.save(attribution_cpu, f"{save_base}/interp_diff_gradient_input/{identifier}.pt")
        else:
            attribution_cpu = torch.load(f"{save_base}/interp_diff_gradient_input/{identifier}.pt")
            if not torch.isfinite(attribution_cpu).all():
                logger.warning(f"NAN occur in {identifier}")
                nan_occur = True
            else:
                nan_occur = False

        if not no_plot:
            if nan_occur:
                logger.warning(f"NAN occur in {identifier}. Skip plotting")
            else:
                # Sum over nucleotide dimension (already multiplied by input in gradient computation)
                # [batch, N, 4] -> [batch, N] -> [N] -> [bin_num, window_size] -> [bin_num]
                with torch.no_grad():
                    # Calculate trim based on the tokenizer's context extension
                    trim = (
                        myconfig.data.context_length // myconfig.data.preprocess.window_size
                        - myconfig.data.preprocess.n_window
                    ) // 2

                    signal = attribution_cpu.sum(dim=-1).mean(dim=0)
                    signal = signal.reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim]
                    signal = signal.mean(dim=-1).detach()

                plot_data.append(signal)
                plot_title.append(f"Importance Score (Gradient×Input Diff)")

            # plot
            n = len(plot_data) + 1
            height_ratios = [1] * len(plot_data) + [0.1]
            fig, axes = plt.subplots(
                nrows=n, ncols=1, figsize=(8, n * 1.5), sharex=True, gridspec_kw={"height_ratios": height_ratios}
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
                f"{res_base}/{exp_name}/analysis_{chk}/plot/interp_diff_gradient_input/{name_base}_grad_input.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Log total time for this sample
            sample_total_time = time.time() - sample_start_time
            logger.info(f"Sample {name_base} interpretation completed in {sample_total_time:.2f}s")

        # Clean up variables at the end of each region processing
        del test_seq_onehot, plot_data, plot_title
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


@click.command()
@click.option("--region_bed", "-f", required=True, type=str, help="Path to BED file with regions")
@click.option("--exp_name", "-e", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--prefix", required=False, type=str, help="Name prefix for all the saving files")
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
@click.option("--num_processes", type=int, default=4, help="Number of subprocess to use for parallel processing")
@click.option("--num_threads", type=int, default=None, help="Number of threads per process for CPU mode (default: 1 if num_processes>1, else use all available)")
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
# Gradient×input specific parameters
@click.option("--pseudo_count", type=float, default=0.0, help="Pseudo count added before log")
@click.option("--no_untransform", is_flag=True, help="Skip untransform (use if predictions already in count space)")
@click.option("--use_mean", is_flag=True, default=True, help="Use mean (vs sum) for bin aggregation")
@click.option("--no_subtract_avg", is_flag=True, help="Don't subtract mean across nucleotides")
@click.option("--input_gate", is_flag=True, help="Multiply gradients by input (gradient × input)")
@click.option("--rc", is_flag=True, help="Ensemble forward and reverse complement gradients")
@click.option("--no_plot", is_flag=True, help="Skip generating plots")
def main(
    region_bed,
    exp_name,
    chk,
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
    pseudo_count,
    no_untransform,
    use_mean,
    no_subtract_avg,
    input_gate,
    rc,
    no_plot,
):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_diff_gradient_input", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff_gradient_input", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label", exist_ok=True)

    logger = BaseLogger(name="GradInputDiff", level=logging.INFO)

    # read region data from BED file
    region_df = pd.read_csv(region_bed, header=None, sep="\t")
    region_df["todo"] = True
    for i in range(len(region_df)):
        # BED columns: chr, start, end, strand, bin_range, gene_name, trial_pos, trial_neg
        chr_name, start, end, strand, bin_range_str, gene_name, trial_pos, trial_neg = region_df.iloc[i, [0, 1, 2, 3, 4, 5, 6, 7]]

        # Clean trial names for shorter filenames
        trial_pos_clean = clean_trial_name(trial_pos, keep_suffix=True)
        neg_trial_count = len(trial_neg.split(';'))
        trial_neg_clean = f"other-{neg_trial_count}"

        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{gene_name}_{trial_pos_clean}_{trial_neg_clean}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{gene_name}_{trial_pos_clean}_{trial_neg_clean}"
        )
        # Check if output files exist
        if no_plot:
            # When plotting is skipped, check for raw data files including .pt file
            output_files_exist = (
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff_gradient_input/{name_base}_grad_input.pt") and
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label/{name_base}_mseqs_unmap.npy") and
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label/{name_base}_label.h5")
            )
        else:
            # When plotting is enabled, also check for plot file
            output_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_diff_gradient_input/{name_base}_grad_input.png"
            output_files_exist = (
                os.path.exists(output_file) and
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff_gradient_input/{name_base}_grad_input.pt") and
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label/{name_base}_mseqs_unmap.npy") and
                os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label/{name_base}_label.h5")
            )
        region_df.at[i, 'todo'] = not output_files_exist or force_restart
        if not region_df.at[i, 'todo']:
            logger.info(f"Skip {name_base}, already exists")
    region_df = region_df[region_df["todo"]].copy()

    logger.info(f"Total regions to process: {len(region_df)}")
    if len(region_df) == 0:
        exit(0)

    # Check device availability
    if processor == "gpu":
        available_devices = torch.cuda.device_count()
        if available_devices == 0:
            logger.warning("No GPU found. Using CPU")
            processor = "cpu"

    if processor == "cpu":
        available_devices = mp.cpu_count()

    if num_processes > available_devices:
        logger.warning(
            f"Requested {num_processes} {processor} but only {available_devices} available. Using {available_devices} {processor}."
        )
        num_processes = available_devices

    # Determine thread count for CPU mode
    if processor == "cpu":
        if num_threads is None:
            # If multiple processes, use 1 thread per process to avoid oversubscription
            # If single process, use all available CPUs
            if num_processes > 1:
                num_threads_per_process = 1
            else:
                num_threads_per_process = None  # Let PyTorch use all available
        else:
            num_threads_per_process = num_threads

        if num_threads_per_process is not None:
            logger.info(f"Using {num_threads_per_process} threads per process for CPU mode")
        else:
            logger.info(f"Using all available CPUs for single process mode")
    else:
        num_threads_per_process = None

    # Multi-Process parallel processing
    logger.info(f"Processing with {num_processes} processes")

    # Split regions into chunks for parallel processing
    chunks = []
    n = len(region_df)

    # Use the same chunking strategy for both CPU and GPU
    base = n // num_processes
    extra = n % num_processes
    start = 0
    for i in range(num_processes):
        size = base + (1 if i < extra else 0)
        end = start + size
        if start < end:
            chunk = region_df.iloc[start:end].copy()
            chunks.append(chunk)
        start = end

    logger.info(f"Split {len(region_df)} regions into {len(chunks)} chunks")
    for chunk in chunks:
        logger.debug(chunk)

    # Prepare arguments for each process
    process_args = []
    for process_id, chunk in enumerate(chunks):
        args = (
            exp_name,
            chk,
            chunk,
            f"{LOG_BASE}/{exp_name}/overall_setting.yaml",
            f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt",
            f"{LOG_BASE}/{exp_name}/regression_label_meta.csv",
            RES_BASE,
            f"cuda:{process_id}" if processor == "gpu" else "cpu",
            force_restart,
            save_raw,
            prefix,
            use_head,
            num_threads_per_process,
            pseudo_count,
            no_untransform,
            use_mean,
            not no_subtract_avg,  # Convert flag to boolean
            input_gate,           # Already a boolean
            rc,                   # Already a boolean
            no_plot,
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if num_processes > 1:
        # Use spawn to avoid issues with model weight sharing
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_region_chunk, process_args)
    else:
        for arg in tqdm(process_args):
            process_region_chunk(arg)  # for debugging


if __name__ == "__main__":
    main()
