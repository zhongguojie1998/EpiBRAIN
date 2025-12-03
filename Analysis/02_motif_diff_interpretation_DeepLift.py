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
from captum.attr import DeepLift, IntegratedGradients
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

def untransform_predictions_numpy(preds, label_meta_row):
    """
    Untransform predictions back to original scale (numpy version).

    Args:
        preds: numpy array of predictions
        label_meta_row: Series with 'scale', 'clip_soft', 'sum_stat'

    Returns:
        Untransformed predictions
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

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, output_key, target_pos_dim, target_neg_dims, bin_range, no_untransform=False, label_meta_row_pos=None, label_meta_rows_neg=None):
        super().__init__()
        self.model = model
        self.output_key = output_key
        self.target_pos_dim = target_pos_dim
        self.target_neg_dims = target_neg_dims
        self.bin_range = bin_range
        self.no_untransform = no_untransform
        self.label_meta_row_pos = label_meta_row_pos
        self.label_meta_rows_neg = label_meta_rows_neg

    def forward(self, x):
        output_dict = self.model(x)
        output = output_dict[self.output_key]  # [batch, N, dim]

        # Untransform predictions if requested
        if not self.no_untransform and self.label_meta_row_pos is not None:
            # Untransform positive trial
            output_pos_raw = output[:, :, self.target_pos_dim]  # [batch, N]
            output_pos_raw = self._untransform_single(output_pos_raw, self.label_meta_row_pos)
            output[:, :, self.target_pos_dim] = output_pos_raw

            # Untransform negative trials
            if self.label_meta_rows_neg is not None:
                for i, neg_dim in enumerate(self.target_neg_dims):
                    output_neg_raw = output[:, :, neg_dim]  # [batch, N]
                    output_neg_raw = self._untransform_single(output_neg_raw, self.label_meta_rows_neg.iloc[i])
                    output[:, :, neg_dim] = output_neg_raw

        # [batch, N, dim] -> [batch]
        output_pos = output[:, self.bin_range, self.target_pos_dim].mean(dim=1)
        output_neg = output[:, self.bin_range, :].mean(dim=1)[:, self.target_neg_dims].mean(dim=1)
        return output_pos - output_neg

    def _untransform_single(self, preds, label_meta_row):
        """Untransform predictions for a single trial."""
        trial_scale = label_meta_row.get('scale', 1.0)
        trial_clip_soft = label_meta_row.get('clip_soft', 48.0)
        trial_sum_stat = label_meta_row.get('sum_stat', 'sum_three_quarter')

        # Step 1: Undo scale
        if trial_scale != 1.0:
            preds = preds / trial_scale

        # Step 2: Undo soft clip
        if trial_clip_soft is not None:
            clip_mask = preds > trial_clip_soft
            preds = torch.where(
                clip_mask,
                (trial_clip_soft - 1) + (preds - (trial_clip_soft - 1)) ** 2,
                preds
            )

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

def process_region_chunk(args):
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
    ) = args

    # Set torch threads for CPU
    if device == "cpu" and num_threads is not None:
        torch.set_num_threads(num_threads)
        # Also set environment variables for BLAS/LAPACK libraries
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

    save_base = f"{res_base}/{exp_name}/analysis_{chk}/raw_data"

    # Load config
    myconfig = load_config(config_name=config_path)
    logger = BaseLogger(name=f"Interpretation-{device}", level=logging.INFO)

    # Get label information
    label_meta = pd.read_csv(label_meta_path, index_col=None)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = setup_model(myconfig, logger)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

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
        fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
    )

    # Process each region in the chunk
    for idx in range(len(region_chunk_data)):
        chr_name, start, end, _, trial_pos, trial_neg = region_chunk_data.iloc[idx, [0, 1, 2, 3, 4, 5]]
        if pd.isna(trial_neg):
            trial_neg = "all"
        # other wise the trial_neg should be a str like "trial1;trial2;trial3"
        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{trial_pos}_{trial_neg.replace(';', '-')}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{trial_pos}_{trial_neg.replace(';', '-')}"
        )

        try:
            trial_pos_dim = int(label_meta.dim[label_meta['trial'] == trial_pos].values[0])
            label_meta_row_pos = label_meta[label_meta['trial'] == trial_pos].iloc[0]
        except:
            logger.warning(f"{trial_pos} cannot be found in label meta, skip")
            continue

        # get negative trial dims and label_meta rows
        trial_pos_modality = label_meta.modality[label_meta['trial'] == trial_pos].values[0]
        if trial_neg == "all":
            label_meta_rows_neg = label_meta[(label_meta['trial'] != trial_pos) & (label_meta['modality'] == trial_pos_modality)]
            trial_neg_dims = label_meta_rows_neg.dim.values
        else:
            neg_trials = trial_neg.split(';')
            label_meta_rows_neg = label_meta[label_meta['trial'].isin(neg_trials)]
            trial_neg_dims = label_meta_rows_neg.dim.values
            if len(trial_neg_dims) == 0:
                logger.warning(f"No valid negative trials found in {trial_neg}, skip")
                continue

        if not chr_name in STD_CHR:
            continue

        token_dict = dna_tokenizer(
            chr_name=chr_name, start=start, end=end, return_augs=False, return_rela_idx=True
        )

        s_idx, e_idx = token_dict["rela_idx"]
        test_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]
        test_seq_onehot.requires_grad = True

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
            pred_res_trial = pred_res.detach().cpu().numpy()[0, :, trial_pos_dim]

            # Untransform predictions if requested
            if not no_untransform and label_meta_row_pos is not None:
                pred_res_trial = untransform_predictions_numpy(pred_res_trial, label_meta_row_pos)

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

        # prepare model wrapper, will sum the output over the interested bins
        model.zero_grad()
        model_wrapper = ModelWrapper(
            model, use_head, trial_pos_dim, trial_neg_dims, bin_range,
            no_untransform=no_untransform,
            label_meta_row_pos=label_meta_row_pos if not no_untransform else None,
            label_meta_rows_neg=label_meta_rows_neg if not no_untransform else None
        )
        # init deep lift with new model wrapper
        dl_model = DeepLift(model_wrapper, multiply_by_inputs=False, eps=1e-7)
        
        # Start timer for this sample
        sample_start_time = time.time()

        for baseline_type, baseline_seq_onehot in baseline_seq_onehots:
            identifier = f"{name_base}_{baseline_type}"
            if not os.path.exists(f"{save_base}/interp_diff/{identifier}.pt") or force_restart:
                # clean up before new attribution
                dl_model.model.zero_grad()
                nan_occur = False

                # Prepare inputs once to avoid repeated tensor operations
                input_tensor = test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)
                baseline_tensor = baseline_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)
                # calculate attribution for bin sum
                attribution = dl_model.attribute(
                    inputs=input_tensor,
                    baselines=baseline_tensor,
                )
                if not torch.isfinite(attribution).all():
                    logger.warning(f"NAN occur in {identifier}, bin num {bin}")
                    nan_occur = True

                # Move to CPU immediately and clear GPU memory
                attribution_cpu = attribution.detach().cpu().permute(0, 2, 1)
                del attribution

                # Clear GPU cache after every iteration to prevent OOM
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()

                # Clean up input tensors
                del input_tensor, baseline_tensor
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    
                if save_raw:
                    torch.save(attribution_cpu, f"{save_base}/interp_diff/{identifier}.pt")
            else:
                attribution_cpu = torch.load(f"{save_base}/interp_diff/{identifier}.pt")
                if not torch.isfinite(attribution_cpu).all():
                    logger.warning(f"NAN occur in {identifier}")
                    nan_occur = True
                else:
                    nan_occur = False

            if nan_occur:
                logger.warning(f"NAN occur in {identifier}. Skip plotting")
                continue

            # we only look at the contribution from the ref genome
            # [batch, N, 4] -> [batch, N] -> [N] -> [bin_num, window_size] -> [bin_num]
            with torch.no_grad():
                signal = (attribution_cpu * test_seq_onehot).sum(dim=-1).mean(dim=0)
                signal = signal.reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim]
                signal = signal.mean(dim=-1).detach()

            plot_data.append(signal)

            # Clean up attribution tensor
            plot_title.append(f"Importance Score ({baseline_type} baseline)")
            # # Plot
            # viz_sequence.plot_weights(all_attribution.mean(dim=0)[:, s_idx:e_idx].T, subticks_frequency=20)
            # plt.savefig(
            #     f"{res_base}/{exp_name}/analysis_{chk}/plot/interp/{identifier}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )
            # plt.close()

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
            f"{res_base}/{exp_name}/analysis_{chk}/plot/interp_diff/{name_base}.png",
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
@click.option(
    "--baseline",
    "-b",
    required=True,
    multiple=True,
    type=click.Choice(["random", "all_zero", "pad"], case_sensitive=False),
    default=["random"],
)
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
@click.option("--no_untransform", is_flag=True, help="Skip untransform (use if predictions already in count space)")
def main(
    region_bed,
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

    # read region data
    region_df = pd.read_csv(region_bed, header=None, sep="\t")
    region_df["todo"] = True
    for i in range(len(region_df)):
        # region df is chr, start, end, name, trial+, trial- (optional)
        chr_name, start, end, _, trial_pos, trial_neg = region_df.iloc[i, [0, 1, 2, 3, 4, 5]]
        # if trial_neg is nan, default is use all trials as negative
        if pd.isna(trial_neg):
            trial_neg = "all"
        # other wise the trial_neg should be a str like "trial1;trial2;trial3"
        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{trial_pos}_{trial_neg.replace(';', '-')}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{trial_pos}_{trial_neg.replace(';', '-')}"
        )
        # Check if all output files exist
        output_files_exist = (
            os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp_diff/{name_base}.png") and
            os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff/{name_base}_mseqs_unmap.npy") and
            os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff/{name_base}_label.h5")
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
    # Divide regions evenly among num_processes
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
            baseline,
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
            no_untransform,
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
