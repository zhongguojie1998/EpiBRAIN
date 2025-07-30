import logging
import multiprocessing as mp
import os
import sys
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


def process_region_chunk(args):
    (
        exp_name,
        chk,
        region_chunk_data,
        baseline_types,
        config_path,
        checkpoint_path,
        res_base,
        device,
        force_restart,
        save_raw,
        prefix,
        use_head,
    ) = args

    save_base = f"{res_base}/{exp_name}/analysis_{chk}/raw_data"

    # Load config
    myconfig = load_config(config_name=config_path)
    logger = BaseLogger(name=f"Interpretation-{device}", level=logging.INFO)

    # Get label information
    label_meta = pd.read_csv(f"{myconfig.data.storage_path}/label_meta.csv", index_col=1)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = setup_model(myconfig, logger)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dl_model = DeepLift(model.to(device), multiply_by_inputs=False, eps=1e-7)

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
        chr_name, start, end, _, trial = region_chunk_data.iloc[idx, [0, 1, 2, 3, 4]]
        name_base = (
            f"{prefix}_{chr_name}_{start}_{end}_{trial}"
            if prefix is not None
            else f"{chr_name}_{start}_{end}_{trial}"
        )

        try:
            trial_dim = label_meta.loc[trial, "dim"]
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

        get_labels(
            mseqs,
            blacklist_bed=myconfig.data.preprocess.blacklist_bed,
            pool_width=myconfig.data.preprocess.window_size,
            kept_num_after_crop=myconfig.data.preprocess.n_window,
            seqs_cov_file=label_h5,
            genome_cov_file=data_config.loc[trial, "file"],
            umap_npy_path=unmap_npy,
            **data_config.loc[trial].drop(["Unnamed: 0", "file"]).to_dict(),
        )
        with h5py.File(label_h5, "r") as f:
            label_trial = f["targets"][0]

        plot_data = [label_trial, pred_res_trial]
        plot_title = [f"{trial} Target", f"{trial} Pred"]

        for baseline_type, baseline_seq_onehot in baseline_seq_onehots:
            identifier = f"{name_base}_{baseline_type}"
            if not os.path.exists(f"{save_base}/interp/{identifier}.pt") or force_restart:
                all_attribution = []
                nan_occur = False

                # Prepare inputs once to avoid repeated tensor operations
                input_tensor = test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)
                baseline_tensor = baseline_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device)

                for bin in bin_range:
                    attribution = dl_model.attribute(
                        inputs=input_tensor,
                        baselines=baseline_tensor,
                        target=(bin, trial_dim),
                    )
                    if not torch.isfinite(attribution).all():
                        logger.warning(f"NAN occur in {identifier}, bin num {bin}")
                        nan_occur = True

                    # Move to CPU immediately and clear GPU memory
                    attribution_cpu = attribution.squeeze(0).detach().cpu()
                    all_attribution.append(attribution_cpu)
                    del attribution

                    # Clear GPU cache periodically during attribution computation
                    if device.startswith('cuda') and len(all_attribution) % 10 == 0:
                        torch.cuda.empty_cache()

                # Clean up input tensors
                del input_tensor, baseline_tensor
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()

                all_attribution = torch.stack(all_attribution)

                if save_raw:
                    torch.save(all_attribution, f"{save_base}/interp/{identifier}.pt")
            else:
                all_attribution = torch.load(f"{save_base}/interp/{identifier}.pt")
                if not torch.isfinite(all_attribution).all():
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
                signal = (all_attribution * test_seq_onehot).sum(dim=-1).mean(dim=0)
                signal = signal.reshape(-1, myconfig.data.preprocess.window_size)[trim:-trim]
                signal = signal.mean(dim=-1).detach()

            plot_data.append(signal)

            # Clean up attribution tensor
            del all_attribution
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
            f"{res_base}/{exp_name}/analysis_{chk}/plot/interp/{name_base}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

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
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
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
    use_head,
):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/label", exist_ok=True)

    logger = BaseLogger(name="Interpretation", level=logging.INFO)

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
        region_df.iloc[i, -1] = (
            not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp/{name_base}.png")
            or force_restart
        )
        if not region_df.iloc[i, -1]:
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
            baseline,
            f"{LOG_BASE}/{exp_name}/overall_setting.yaml",
            f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt",
            RES_BASE,
            f"cuda:{process_id}" if processor == "gpu" else "cpu",
            force_restart,
            save_raw,
            prefix,
            use_head,
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if processor == "gpu":
        mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_region_chunk, process_args)


if __name__ == "__main__":
    main()
