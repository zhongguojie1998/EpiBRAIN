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


def onehot_to_str(seq_onehot):
    """Convert one-hot encoded tensor (L x 4) to DNA string"""
    mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "T", (0, 0, 0, 0): "N"}

    seq_str = ""
    for vec in seq_onehot:
        key = tuple((vec > 0.5).int().tolist())
        seq_str += mapping.get(key, "N")  # default to "N" if no match
    return seq_str


def process_vcf_chunk(args):
    chunk_data, config_path, checkpoint_path, save_base, device = args

    # Load config
    myconfig = load_config(config_name=config_path)
    logger = BaseLogger(name=f"Variant Effect-{device}", level=logging.INFO)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=0)
    label_meta = pd.read_csv(f"{myconfig.data.storage_path}/label_meta.csv", index_col=1)

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = setup_model(myconfig, logger)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    # Setup tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
    )

    for idx in range(len(chunk_data)):
        chr_name, pos, ref, alt = chunk_data.iloc[idx, [0, 1, 2, 3]]
        name_base = f"{chr_name}_{ref}{pos}{alt}"

        if not chr_name in STD_CHR:
            continue

        # get ref seq
        # tokenizer is python indexed (from 0) but the vcf is natural indexed (from 1)
        token_dict = dna_tokenizer(
            chr_name=chr_name, start=pos - 1, end=pos, return_augs=False, return_rela_idx=True
        )
        s_idx, e_idx = token_dict["rela_idx"]
        wt_seq_onehot = token_dict["one_hot"]
        real_start, real_end = token_dict["real_region"]

        # check ref
        wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
        if not ref == wt_nt_fetched:
            logger.warning(
                f"Ref info isn't consistent with genome. {chr_name}: {pos}, given {ref}, fetched {wt_nt_fetched}"
            )
            continue

        # get alt seq
        alt_nt_onehot = str_to_one_hot(alt)
        mut_seq_onehot = wt_seq_onehot.clone()
        mut_seq_onehot[s_idx:e_idx] = alt_nt_onehot

        # get pred result
        with torch.no_grad():
            pred_res_wt = (
                model(wt_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), myconfig.model.use_head, False)
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
            )
            pred_res_mut = (
                model(mut_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), myconfig.model.use_head, False)
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
            )

        # get label
        os.makedirs(f"{save_base}/tmp/", exist_ok=True)
        unmap_npy = f"{save_base}/tmp/{name_base}_mseqs_unmap.npy"
        label_h5 = f"{save_base}/tmp/{name_base}_label.h5"

        mseqs = [ModelSeq(chr_name, real_start, real_end, "test")]

        mseqs_unmap = annotate_unmap(
            mseqs,
            myconfig.data.preprocess.unmap_bed,
            myconfig.data.preprocess.context_length,
            myconfig.data.preprocess.window_size,
        )
        np.save(unmap_npy, mseqs_unmap)

        label_trial = {}
        for i in data_config.index:
            get_labels(
                mseqs,
                blacklist_bed=myconfig.data.preprocess.blacklist_bed,
                pool_width=myconfig.data.preprocess.window_size,
                kept_num_after_crop=myconfig.data.preprocess.n_window,
                seqs_cov_file=label_h5,
                genome_cov_file=data_config.loc[i, "file"],
                umap_npy_path=unmap_npy,
                **data_config.loc[i].drop(["exp", "file"]).to_dict(),
            )
            with h5py.File(label_h5, "r") as f:
                label_trial[data_config.loc[i, "exp"]] = f["targets"][0]
        labels = np.zeros((myconfig.data.preprocess.n_window, len(label_meta)))
        for k,v in label_trial.items():
            dim = label_meta.loc[k, "dim"]
            labels[:, dim] = v

        with h5py.File(f"{save_base}/{name_base}.h5", "w") as f:
            data_group = f.create_group("data")
            data_group.create_dataset("label", data=labels, compression="gzip")
            data_group.create_dataset("pred_wt", data=pred_res_wt, compression="gzip")
            data_group.create_dataset("pred_alt", data=pred_res_mut, compression="gzip")
            data_group.create_dataset("diff", data=pred_res_mut - pred_res_wt, compression="gzip")

            f.attrs["context_start"] = real_start
            f.attrs["context_end"] = real_end
            f.attrs["ref"] = ref
            f.attrs["alt"] = alt
            f.attrs["pos"] = pos
            f.attrs["rela_pos"] = (s_idx, e_idx)


@click.command()
@click.option("--vcf", "-f", required=True, type=str, help="Path to the vcf")
@click.option("--exp_name", "-e", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--log_base", required=True, type=str, default="./logs")
@click.option("--chk_base", required=True, type=str, default="./Chk")
@click.option("--res_base", required=True, type=str, default="./Res")
@click.option("--force_restart", is_flag=True)
@click.option(
    "--processor",
    required=True,
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
)
@click.option("--num_processes", type=int, default=4, help="Number of subprocess to use for parallel processing")
def main(vcf, exp_name, chk, log_base, chk_base, res_base, force_restart, processor, num_processes):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff", exist_ok=True)

    logger = BaseLogger(name="Variant Effect", level=logging.INFO)

    # read vsf data
    vcf_file = pd.read_csv(vcf, sep="\t")
    vcf_df = pd.DataFrame(columns=["chr", "pos", "ref", "alt", "todo"])
    for i in range(len(vcf_file)):
        chr_name, pos, ref, alt = vcf_file.iloc[i, [0, 1, 3, 4]]
        name_base = f"{chr_name}_{ref}{pos}{alt}"
        todo = (
            not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff/{name_base}.h5")
            or force_restart
        )
        vcf_df.loc[i] = [chr_name, pos, ref, alt, todo]

        if not todo:
            logger.info(f"Skip {name_base}, already exists")
    vcf_df = vcf_df[vcf_df["todo"]].copy()

    logger.info(f"Total regions to process: {len(vcf_df)}")
    if len(vcf_df) == 0:
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
    n = len(vcf_df)

    if processor == "gpu":
        base = n // num_processes
        extra = n % num_processes
        start = 0
        for i in range(num_processes):
            size = base + (1 if i < extra else 0)
            end = start + size
            if start < end:
                chunk = vcf_df.iloc[start:end].copy()
                chunks.append(chunk)
            start = end
    elif processor == "cpu":
        for i in range(n):
            chunk = vcf_df.iloc[i : i + 1].copy()
            chunks.append(chunk)

    logger.info(f"Split {len(vcf_df)} regions into {len(chunks)} chunks")
    for chunk in chunks:
        logger.debug(chunk)

    # Prepare arguments for each process
    process_args = []
    for process_id, chunk in enumerate(chunks):
        args = (
            chunk,
            f"{LOG_BASE}/{exp_name}/overall_setting.yaml",
            f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt",
            f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff",
            f"cuda:{process_id}" if processor == "gpu" else "cpu",
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if processor == "gpu":
        mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_vcf_chunk, process_args)


if __name__ == "__main__":
    main()
    # os.makedirs("./Res/250716_atac_finetune/analysis_best_valid_loss/raw_data/var_eff", exist_ok=True)
    # process_vcf_chunk(
    #     (
    #         pd.DataFrame({0: "chr21", 1: 5030200, 2: "A", 3: "C"}, index=[0]),
    #         "./logs/250716_atac_finetune/overall_setting.yaml",
    #         "./Chk/250716_atac_finetune/chk_epoch_best_valid_loss.pt",
    #         "./Res/250716_atac_finetune/analysis_best_valid_loss/raw_data/var_eff",
    #         "cpu",
    #     )
    # )
