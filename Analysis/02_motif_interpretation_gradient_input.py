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
from modisco.visualization import viz_sequence
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from data.data_utils import STD_CHR, ModelSeq, annotate_unmap, get_labels
from data.tokenizer import FastaInterval, str_to_one_hot
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger


def which_bins(s_idx: int, e_idx: int, window_size: int):

    bin_start = s_idx // window_size
    bin_end = (e_idx - 1) // window_size
    return np.array(range(bin_start, bin_end + 1))


def gradients_input_attribution(
    model,
    seq_input,
    output_key,
    target_dim,
    bin_range,
    label_meta_row=None,
    pseudo_count=0.0,
    no_untransform=False,
    use_mean=True,
    subtract_avg=True,
    input_gate=True,
):
    """
    Compute gradient×input attribution, equivalent to seqnn.gradients_func().

    This implements the gradient×input method which computes:
    1. Forward pass to get predictions
    2. Optionally untransform predictions (inverse of training preprocessing)
    3. Aggregate predictions over specified bins and tracks
    4. Compute log of aggregated prediction as score
    5. Compute gradients of score w.r.t. input
    6. Optionally subtract mean across nucleotides
    7. Multiply gradients by input (gradient × input)

    Args:
        model: PyTorch model
        seq_input: Input sequence tensor [batch, channels, length]
        output_key: Which output head to use
        target_dim: Which target dimension/track to use
        bin_range: Which bins to aggregate over
        label_meta_row: Row from label_meta DataFrame with 'scale', 'clip_soft', 'sum_stat'
        pseudo_count: Small constant added before taking log
        no_untransform: Skip untransforming (if predictions are already in count space)
        use_mean: Use mean aggregation (vs sum) over bins
        subtract_avg: Subtract mean across nucleotides at each position
        input_gate: Multiply gradients by input (gradient × input)

    Returns:
        Gradients tensor [batch, length, channels]
    """
    # Enable gradient computation for input
    seq_input.requires_grad_(True)

    # Forward pass
    output_dict = model(seq_input)
    preds = output_dict[output_key]  # [batch, N_bins, dim]

    # Select target dimension
    preds = preds[:, :, target_dim]  # [batch, N_bins]

    # Untransform predictions using label_meta (reverse of training preprocessing)
    # Following the logic from 01_5_test_correlation_by_gene.py
    if not no_untransform and label_meta_row is not None:
        trial_scale = label_meta_row.get('scale', 1.0)
        trial_clip_soft = label_meta_row.get('clip_soft', 48.0)
        trial_sum_stat = label_meta_row.get('sum_stat', 'sum_three_quarter')

        # Step 1: Undo scale
        if trial_scale != 1.0:
            preds = preds / trial_scale

        # Step 2: Undo soft clip
        # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
        # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
        if trial_clip_soft is not None:
            clip_mask = preds > trial_clip_soft
            preds = torch.where(
                clip_mask,
                (trial_clip_soft - 1) + (preds - (trial_clip_soft - 1)) ** 2,
                preds
            )

        # Step 3: Undo power transform based on sum_stat
        if trial_sum_stat == "sum_three_quarter":
            # Forward: x = x^(3/4)
            # Reverse: x = x^(4/3)
            preds = preds ** (4.0 / 3.0)
        elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
            # Forward: x = sqrt(x + 1) - 1 = (x + 1)^0.5 - 1
            # Reverse: x = (x + 1)^2 - 1
            preds = (preds + 1) ** 2 - 1
        elif trial_sum_stat in ['sum', 'mean', "avg"]:
            # No transformation applied
            pass
        else:
            raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

    # Select bins of interest
    preds_slice = preds[:, bin_range]  # [batch, len(bin_range)]

    # Aggregate over bins
    if use_mean:
        preds_agg = preds_slice.mean(dim=-1)  # [batch]
    else:
        preds_agg = preds_slice.sum(dim=-1)  # [batch]

    # Compute score to differentiate (log of predictions)
    score = torch.log(preds_agg + pseudo_count + 1e-6)

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
        pseudo_count,
        no_untransform,
        use_mean,
        subtract_avg,
        input_gate,
    ) = args

    save_base = f"{res_base}/{exp_name}/analysis_{chk}/raw_data"

    # Load config
    myconfig = load_config(config_name=config_path)
    logger = BaseLogger(name=f"GradInput-{device}", level=logging.INFO)

    # Get label information
    label_meta = pd.read_csv(label_meta_path, index_col=None)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = setup_model(myconfig, logger)
    # If model was wrapped by torch.compile, load into the original module
    load_target = model._orig_mod if hasattr(model, '_orig_mod') else model
    load_target.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Setup tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
    )

    # Process each region in the chunk
    for idx in range(len(region_chunk_data)):
        chr_name, start, end, _, trial = region_chunk_data.iloc[idx, [0, 1, 2, 3, 4]]
        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{trial}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{trial}"
        )

        try:
            trial_dim = int(label_meta.dim[label_meta['trial'] == trial].values[0])
            # Get the label_meta row for this trial (for untransform parameters)
            label_meta_row = label_meta[label_meta['trial'] == trial].iloc[0]
        except:
            logger.warning(f"{trial} cannot be found in label meta, skip")
            continue

        if not chr_name in STD_CHR:
            continue

        token_dict = dna_tokenizer(
            chr_name=chr_name, start=start, end=end, return_augs=False, return_rela_idx=True
        )

        s_idx, e_idx = token_dict["rela_idx"]
        test_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]

        # Get the interested bin, given the raw idx
        bin_range = which_bins(s_idx, e_idx, myconfig.data.preprocess.window_size)
        trim = (
            myconfig.data.context_length // myconfig.data.preprocess.window_size
            - myconfig.data.preprocess.n_window
        ) // 2
        bin_range = bin_range - trim

        assert bin_range.min() >= 0
        assert bin_range.max() <= myconfig.data.preprocess.n_window

        # generate label and pred
        ## pred
        with torch.no_grad():
            pred_res = model(
                test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), use_head
            )
            pred_res_trial = pred_res.detach().cpu().numpy()[0, :, trial_dim]
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
                genome_cov_file=data_config.loc[trial, "file"],
                umap_npy_path=unmap_npy,
                **data_config.loc[trial, ["sum_stat", "baseline_pct", "umap_pct", "scale", "clip", "clip_soft"]].to_dict(),
            )
            with h5py.File(label_h5, "r") as f:
                label_trial = f["targets"][0]
        except (ValueError, RuntimeError, IndexError) as e:
            logger.warning(f"Failed to get labels for {name_base}: {str(e)}. Skipping this region.")
            continue

        plot_data = [label_trial, pred_res_trial]
        plot_title = [f"{trial} Target", f"{trial} Pred"]

        # Start timer for this sample
        sample_start_time = time.time()

        identifier = f"{name_base}_grad_input"
        if not os.path.exists(f"{save_base}/interp_gradient_input/{identifier}.pt") or force_restart:
            nan_occur = False

            # Prepare input tensor
            input_tensor = test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)

            # Calculate gradient×input attribution
            attribution = gradients_input_attribution(
                model=model,
                seq_input=input_tensor,
                output_key=use_head,
                target_dim=trial_dim,
                bin_range=bin_range,
                label_meta_row=label_meta_row,
                pseudo_count=pseudo_count,
                no_untransform=no_untransform,
                use_mean=use_mean,
                subtract_avg=subtract_avg,
                input_gate=input_gate,
            )

            if not torch.isfinite(attribution).all():
                logger.warning(f"NAN occur in {identifier}")
                nan_occur = True

            # Move to CPU immediately and clear GPU memory
            attribution_cpu = attribution.detach().cpu()
            del attribution

            # Clear GPU cache
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            # Clean up input tensors
            del input_tensor
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

            if save_raw:
                torch.save(attribution_cpu, f"{save_base}/interp_gradient_input/{identifier}.pt")
        else:
            attribution_cpu = torch.load(f"{save_base}/interp_gradient_input/{identifier}.pt")
            if not torch.isfinite(attribution_cpu).all():
                logger.warning(f"NAN occur in {identifier}")
                nan_occur = True
            else:
                nan_occur = False

        if nan_occur:
            logger.warning(f"NAN occur in {identifier}. Skip plotting")
        else:
            # Sum over nucleotide dimension (already multiplied by input in gradient computation)
            # [batch, N, 4] -> [batch, N] -> [N] -> [bin_num, window_size] -> [bin_num]
            with torch.no_grad():
                signal = attribution_cpu.sum(dim=-1).mean(dim=0)
                signal = signal.reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim]
                signal = signal.mean(dim=-1).detach()

            plot_data.append(signal)
            plot_title.append(f"Importance Score (Gradient×Input)")

        # plot
        n = len(plot_data) + 1
        height_ratios = [1] * len(plot_data) + [0.1]
        fig, axes = plt.subplots(
            nrows=n, ncols=1, figsize=(8, n * 1.5), sharex=True, gridspec_kw={"height_ratios": height_ratios}
        )

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
            f"{res_base}/{exp_name}/analysis_{chk}/plot/interp_gradient_input/{name_base}_grad_input.png",
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
@click.option("--region_bed", "-f", required=True, type=str, help="Path to a file with all regions to test")
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
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
# Gradient×input specific parameters
@click.option("--pseudo_count", type=float, default=0.0, help="Pseudo count added before log")
@click.option("--no_untransform", is_flag=True, help="Skip untransform (use if predictions already in count space)")
@click.option("--use_mean", is_flag=True, default=True, help="Use mean (vs sum) for bin aggregation")
@click.option("--no_subtract_avg", is_flag=True, help="Don't subtract mean across nucleotides")
@click.option("--no_input_gate", is_flag=True, help="Don't multiply by input (just use gradients)")
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
    use_head,
    pseudo_count,
    no_untransform,
    use_mean,
    no_subtract_avg,
    no_input_gate,
):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_gradient_input", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_gradient_input", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label", exist_ok=True)

    logger = BaseLogger(name="GradInput", level=logging.INFO)

    # read region data
    region_df = pd.read_csv(region_bed, header=None, sep="\t")
    region_df["todo"] = True
    for i in range(len(region_df)):
        chr_name, start, end, _, trial = region_df.iloc[i, [0, 1, 2, 3, 4]]
        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{trial}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{trial}"
        )
        # Check if output files exist
        output_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_gradient_input/{name_base}_grad_input.png"
        output_files_exist = (
            os.path.exists(output_file) and
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

    # Multi-Process parallel processing
    logger.info(f"Processing with {num_processes} processes")

    # Split regions into chunks for parallel processing
    chunks = []
    n = len(region_df)

    if processor == "gpu":
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
    elif processor == "cpu":
        for i in range(n):
            chunk = region_df.iloc[i : i + 1].copy()
            chunks.append(chunk)

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
            pseudo_count,
            no_untransform,
            use_mean,
            not no_subtract_avg,  # Convert flag to boolean
            not no_input_gate,    # Convert flag to boolean
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if processor == "gpu" and num_processes > 1:
        mp.set_start_method("spawn", force=True)
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_region_chunk, process_args)
    else:
        for arg in tqdm(process_args):
            process_region_chunk(arg)  # for debugging


if __name__ == "__main__":
    main()
