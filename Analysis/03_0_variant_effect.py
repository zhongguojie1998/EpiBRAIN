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
from data.tokenizer import FastaInterval, str_to_one_hot, one_hot_reverse_complement
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


def create_coord_conversion_df(real_start, real_end, variant_pos, ref_len, alt_len, bin_size=32):
    """
    Create coordinate conversion dataframe for mapping genomic coordinates to bin indices.

    For MNVs (insertions/deletions), the mutant sequence has different length than WT,
    so we need to track how genomic coordinates map to bins in both sequences.

    Parameters:
    -----------
    real_start, real_end : int
        Genomic coordinates of the context sequence
    variant_pos : int
        1-indexed genomic position of the variant (VCF POS)
    ref_len, alt_len : int
        Length of reference and alternate alleles
    bin_size : int
        Bin size for model output (default 32 bp)

    Returns:
    --------
    pd.DataFrame with columns ['wt_bin_idx', 'mut_bin_idx'] indexed by genomic coordinates
    """
    length_diff = alt_len - ref_len  # positive = insertion, negative = deletion

    # Variant's 0-indexed position in the sequence
    var_seq_pos = variant_pos - 1 - real_start  # convert 1-indexed VCF pos to 0-indexed seq pos
    var_bin = var_seq_pos // bin_size

    coords = np.arange(real_start, real_end)
    n_coords = len(coords)

    # WT bin indices - straightforward
    wt_bin_idx = (coords - real_start) // bin_size

    # MUT bin indices - need to account for length difference
    mut_bin_idx = np.zeros(n_coords, dtype=np.int32)

    for i, coord in enumerate(coords):
        seq_pos = coord - real_start

        if coord < variant_pos:
            # Before variant - same as WT
            mut_bin_idx[i] = seq_pos // bin_size
        elif coord < variant_pos + ref_len:
            # Within ref region
            if length_diff < 0:  # deletion
                offset = coord - variant_pos
                if offset < alt_len:
                    # Position still exists in alt
                    mut_seq_pos = var_seq_pos + offset
                    mut_bin_idx[i] = mut_seq_pos // bin_size
                else:
                    # Position is deleted - map to variant bin
                    mut_bin_idx[i] = var_bin
            else:  # insertion or SNP
                offset = coord - variant_pos
                mut_seq_pos = var_seq_pos + offset
                mut_bin_idx[i] = mut_seq_pos // bin_size
        else:
            # After ref region - shifted by length_diff
            mut_seq_pos = seq_pos + length_diff
            mut_bin_idx[i] = mut_seq_pos // bin_size

    return pd.DataFrame({
        'wt_bin_idx': wt_bin_idx.astype(np.int32),
        'mut_bin_idx': mut_bin_idx
    }, index=coords)


def build_rc_swap_index(label_meta):
    """
    Build reverse complement swap index for RNAplus/RNAminus tracks.

    When doing reverse complement, RNAplus and RNAminus need to be swapped
    because the strand orientation changes.

    Returns:
        numpy array: Swap index where RNAplus and RNAminus are swapped
    """
    if label_meta is None:
        return None

    # Check if we have RNAplus and RNAminus tracks
    if 'modality' not in label_meta.columns:
        return None

    has_plus = 'RNAplus' in label_meta['modality'].values
    has_minus = 'RNAminus' in label_meta['modality'].values

    if not (has_plus and has_minus):
        return None, None

    # Build swap index using 'dim' column (track index in predictions)
    swap_index = []
    org_index = label_meta['dim'].tolist()
    for i, row in label_meta.iterrows():
        if row['modality'] == 'RNAplus':
            # Find the index of RNAminus for corresponding cell type
            if 'cell_type' in label_meta.columns:
                matching = label_meta[
                    (label_meta['modality'] == 'RNAminus') &
                    (label_meta['cell_type'] == row['cell_type'])
                ]
            else:
                # If no cell_type column, match by trial name pattern
                matching = label_meta[label_meta['modality'] == 'RNAminus']
            if len(matching) > 0:
                # Use the 'dim' column value (track index in pred)
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))
        elif row['modality'] == 'RNAminus':
            # Find the index of RNAplus for corresponding cell type
            if 'cell_type' in label_meta.columns:
                matching = label_meta[
                    (label_meta['modality'] == 'RNAplus') &
                    (label_meta['cell_type'] == row['cell_type'])
                ]
            else:
                # If no cell_type column, match by trial name pattern
                matching = label_meta[label_meta['modality'] == 'RNAplus']
            if len(matching) > 0:
                # Use the 'dim' column value (track index in pred)
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))
        else:
            # Don't change for other modalities
            swap_index.append(int(row['dim']))

    return np.array(org_index), np.array(swap_index)


def process_vcf_chunk(args):
    chunk_data, config_path, checkpoint_path, save_base, device, use_head = args

    # Load config
    myconfig = load_config(config_name=config_path, skip_validation=True)
    logger = BaseLogger(name=f"Variant Effect-{device}", level=logging.INFO)
    data_config = pd.read_csv(f"{myconfig.data.preprocess.trial_summary_path}", index_col=0)
    label_meta = pd.read_csv(f"{myconfig.logging.log_dir}/regression_label_meta.csv", index_col=1)
    # set index
    label_meta = label_meta.set_index("trial")

    # Build reverse complement swap index for RNA tracks
    rc_orig_index, rc_swap_index = build_rc_swap_index(label_meta)
    if rc_swap_index is not None:
        logger.info("Built reverse complement swap index for RNAplus/RNAminus tracks")

    # Setup model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # we should not compile the model for inference, override the config here
    myconfig.model.use_compile = False
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

        # Skip if output file already exists
        if os.path.exists(f"{save_base}/{name_base}.h5"):
            logger.info(f"Skipping {name_base}, output file already exists")
            continue

        try:
            if not chr_name in STD_CHR:
                continue

            # get ref seq
            # tokenizer is python indexed (from 0) but the vcf is natural indexed (from 1)
            # For MNVs, we need to fetch the full ref region
            ref_len = len(ref)
            alt_len = len(alt)
            token_dict = dna_tokenizer(
                chr_name=chr_name, start=pos - 1, end=pos - 1 + ref_len, return_augs=False, return_rela_idx=True
            )
            s_idx, e_idx = token_dict["rela_idx"]
            wt_seq_onehot = token_dict["one_hot"]
            real_start, real_end = token_dict["real_region"]

            # check ref
            wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
            if not ref == wt_nt_fetched:
                logger.warning(
                    f"Ref info isn't consistent with genome. {chr_name}: {pos}, given {ref}, fetched {wt_nt_fetched}. Replacing with ref."
                )
                # Replace wt_seq_onehot with ref nucleotide
                ref_nt_onehot = str_to_one_hot(ref)
                wt_seq_onehot[s_idx:e_idx] = ref_nt_onehot

            # get alt seq - use concatenation to handle insertions/deletions
            alt_nt_onehot = str_to_one_hot(alt)
            mut_seq_onehot = torch.cat([
                wt_seq_onehot[:s_idx],
                alt_nt_onehot,
                wt_seq_onehot[e_idx:]
            ], dim=0)

            # create reverse complement sequences for augmentation
            wt_seq_onehot_rev = one_hot_reverse_complement(wt_seq_onehot)
            mut_seq_onehot_rev = one_hot_reverse_complement(mut_seq_onehot)

            # get pred result with reverse complement augmentation
            with torch.no_grad():
                # Forward strand predictions
                pred_res_wt_fwd = (
                    model(wt_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), use_head)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                )
                pred_res_mut_fwd = (
                    model(mut_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), use_head)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                )

                # Reverse strand predictions
                pred_res_wt_rev = (
                    model(wt_seq_onehot_rev.unsqueeze(0).permute(0, 2, 1).to(device), use_head)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                )
                pred_res_mut_rev = (
                    model(mut_seq_onehot_rev.unsqueeze(0).permute(0, 2, 1).to(device), use_head)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(0)
                )

                # Flip predictions along sequence dimension (reverse order)
                pred_res_wt_rev = np.flip(pred_res_wt_rev, axis=0)
                pred_res_mut_rev = np.flip(pred_res_mut_rev, axis=0)

                # Swap RNAplus and RNAminus tracks if needed
                if rc_swap_index is not None:
                    pred_res_wt_rev[:, rc_orig_index] = pred_res_wt_rev[:, rc_swap_index]
                    pred_res_mut_rev[:, rc_orig_index] = pred_res_mut_rev[:, rc_swap_index]

                # Average forward and reverse predictions for strand-agnostic results
                pred_res_wt = (pred_res_wt_fwd + pred_res_wt_rev) / 2.0
                pred_res_mut = (pred_res_mut_fwd + pred_res_mut_rev) / 2.0

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
                    **data_config.loc[i].drop(["exp", "file", "celltype", "celltype_n", "modality", "atlas_name", "task"]).to_dict(),
                )
                with h5py.File(label_h5, "r") as f:
                    label_trial[data_config.loc[i, "exp"]] = f["targets"][0]
            labels = np.zeros((myconfig.data.preprocess.n_window, len(label_meta)))
            for i, (_, v) in enumerate(label_trial.items()):
                labels[:, i] = v

            # Create coordinate conversion dataframe for mapping genomic coords to bin indices
            coord_conv_df = create_coord_conversion_df(
                real_start, real_end, pos, ref_len, alt_len,
                bin_size=myconfig.data.preprocess.window_size
            )

            with h5py.File(f"{save_base}/{name_base}.h5", "w") as f:
                data_group = f.create_group("data")
                data_group.create_dataset("label", data=labels, compression="gzip")
                data_group.create_dataset("pred_wt", data=pred_res_wt, compression="gzip")
                data_group.create_dataset("pred_alt", data=pred_res_mut, compression="gzip")

                # Store coordinate conversion dataframe
                # Index: genomic coordinates (real_start to real_end)
                # wt_bin_idx: bin index in WT sequence predictions
                # mut_bin_idx: bin index in MUT sequence predictions (accounts for indels)
                coord_group = f.create_group("coord_conversion")
                coord_group.create_dataset("genomic_coords", data=coord_conv_df.index.values, compression="gzip")
                coord_group.create_dataset("wt_bin_idx", data=coord_conv_df['wt_bin_idx'].values, compression="gzip")
                coord_group.create_dataset("mut_bin_idx", data=coord_conv_df['mut_bin_idx'].values, compression="gzip")

                f.attrs["context_start"] = real_start
                f.attrs["context_end"] = real_end
                f.attrs["ref"] = ref
                f.attrs["alt"] = alt
                f.attrs["pos"] = pos
                f.attrs["rela_pos"] = (s_idx, e_idx)
                f.attrs["ref_len"] = ref_len
                f.attrs["alt_len"] = alt_len
                f.attrs["is_snp"] = (ref_len == 1 and alt_len == 1)
                f.attrs["is_insertion"] = (alt_len > ref_len)
                f.attrs["is_deletion"] = (ref_len > alt_len)

        except Exception as e:
            logger.error(
                f"Error processing variant {name_base} ({chr_name}:{pos} {ref}>{alt}): {type(e).__name__}: {str(e)}"
            )
            logger.info(f"Skipping variant {name_base}")
            continue


@click.command()
@click.option("--vcf", "-f", required=True, type=str, help="Path to the vcf")
@click.option("--exp_name", "-e", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--log_base", required=False, type=str, default="./logs")
@click.option("--chk_base", required=False, type=str, default="./Chk")
@click.option("--res_base", required=False, type=str, default="./Res")
@click.option("--force_restart", is_flag=True)
@click.option(
    "--processor",
    required=True,
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
)
@click.option("--num_processes", type=int, default=4, help="Number of subprocess to use for parallel processing")
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
def main(vcf, exp_name, chk, log_base, chk_base, res_base, force_restart, processor, num_processes, use_head):
    LOG_BASE = os.path.abspath(log_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/var_eff/raw_data/", exist_ok=True)

    logger = BaseLogger(name="Variant Effect", level=logging.INFO)

    # read vcf data (skip header lines starting with #)
    vcf_file = pd.read_csv(vcf, sep="\t", comment='#', header=None)
    vcf_df = pd.DataFrame(columns=["chr", "pos", "ref", "alt", "todo"])
    for i in range(len(vcf_file)):
        chr_name, pos, ref, alt = vcf_file.iloc[i, [0, 1, 3, 4]]
        name_base = f"{chr_name}_{ref}{pos}{alt}"
        todo = (
            not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/var_eff/raw_data/{name_base}.h5")
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
        else:
            if num_processes > available_devices:
                logger.warning(
                    f"Requested {num_processes} GPUs but only {available_devices} available. Using {available_devices} GPUs."
                )
                num_processes = available_devices
            if num_processes == 1:
                logger.info("Using single GPU mode: cuda:0")
            else:
                logger.info(f"Using multi-GPU mode: cuda:0 to cuda:{num_processes-1}")

    if processor == "cpu":
        available_devices = mp.cpu_count()
        if num_processes > available_devices:
            logger.warning(
                f"Requested {num_processes} CPUs but only {available_devices} available. Using {available_devices} CPUs."
            )
            num_processes = available_devices

    # Multi-Process parallel processing
    logger.info(f"Processing with {num_processes} processes")

    # Split regions into chunks for parallel processing
    chunks = []
    n = len(vcf_df)

    if processor == "gpu":
        if num_processes == 1:
            # Single GPU: process all variants in one chunk
            chunks.append(vcf_df.copy())
        else:
            # Multi-GPU: split variants across GPUs
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
        # CPU mode: one variant per chunk for parallel processing
        for i in range(n):
            chunk = vcf_df.iloc[i : i + 1].copy()
            chunks.append(chunk)

    logger.info(f"Split {len(vcf_df)} regions into {len(chunks)} chunks")
    for chunk in chunks:
        logger.debug(chunk)

    # Prepare arguments for each process
    process_args = []
    for chunk_id, chunk in enumerate(chunks):
        # Determine device
        if processor == "gpu":
            if num_processes == 1:
                device = "cuda:0"
            else:
                device = f"cuda:{chunk_id}"
        else:
            device = "cpu"

        args = (
            chunk,
            f"{LOG_BASE}/{exp_name}/overall_setting.yaml",
            f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt",
            f"{RES_BASE}/{exp_name}/analysis_{chk}/var_eff/raw_data",
            device,
            use_head,
        )
        process_args.append(args)

    # Run parallel processing
    logger.info("Starting processing...")
    if processor == "gpu":
        mp.set_start_method("spawn", force=True)
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            pool.map(process_vcf_chunk, process_args)
    else:
        for args in process_args:
            process_vcf_chunk(args)


if __name__ == "__main__":
    main()
