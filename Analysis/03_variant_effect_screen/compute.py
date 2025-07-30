import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))

from data.data_utils import STD_CHR
from data.tokenizer import FastaInterval, str_to_one_hot
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def onehot_to_str(seq_onehot):
    mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "T", (0, 0, 0, 0): "N"}
    seq_str = ""
    for vec in seq_onehot:
        key = tuple((vec > 0.5).int().tolist())
        seq_str += mapping.get(key, "N")
    return seq_str


class ModelPackage:
    def __init__(self, model, dna_tokenizer, config):
        self.model = model
        self.dna_tokenizer = dna_tokenizer
        self.config = config

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class VariantDataset(Dataset):
    def __init__(self, h5_path, task_indices, dna_tokenizer):
        self.task_indices = task_indices
        self.dna_tokenizer = dna_tokenizer

        with h5py.File(h5_path, "r") as f:
            variants = f["variants"]
            chrs = variants["chr"][task_indices]
            poses = variants["pos"][task_indices]
            refs = variants["ref"][task_indices]
            alts = variants["alt"][task_indices]

        if hasattr(chrs[0], "decode"):
            chrs = [c.decode("utf-8") for c in chrs]
        if hasattr(refs[0], "decode"):
            refs = [r.decode("utf-8") for r in refs]
        if hasattr(alts[0], "decode"):
            alts = [a.decode("utf-8") for a in alts]

        self.variants = list(zip(task_indices, chrs, poses, refs, alts))

    def __len__(self):
        return len(self.task_indices)

    def __getitem__(self, idx):
        task_index, chr_name, pos, ref, alt = self.variants[idx]
        if chr_name not in STD_CHR:
            return None, task_index, "invalid_chr"

        try:
            token_dict = self.dna_tokenizer(
                chr_name=chr_name, start=pos - 1, end=pos, return_augs=False, return_rela_idx=True
            )
            s_idx, e_idx = token_dict["rela_idx"]
            wt_seq_onehot = token_dict["one_hot"]

            wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
            if ref != wt_nt_fetched:
                return None, task_index, f"ref_mismatch(ref:{ref},get:{wt_nt_fetched})"

            alt_nt_onehot = str_to_one_hot(alt)
            mut_seq_onehot = wt_seq_onehot.clone()
            mut_seq_onehot[s_idx:e_idx] = alt_nt_onehot

            return (wt_seq_onehot, mut_seq_onehot), task_index, None
        except Exception as e:
            return None, task_index, str(e)


def collate_fn(batch):
    wt_list, mut_list, task_ids, msgs, masks = [], [], [], [], []
    for item, task_index, err in batch:
        if item is not None:
            wt_list.append(item[0])
            mut_list.append(item[1])
            masks.append(True)
        else:
            masks.append(False)
        task_ids.append(task_index)
        msgs.append(err)

    if len(wt_list) == 0:
        return None, None, np.array(task_ids), np.array(msgs), np.array(masks)
    return torch.stack(wt_list), torch.stack(mut_list), np.array(task_ids), np.array(msgs), np.array(masks)


def load_label_meta_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f.attrs["trial_dims"]


def save_intermediate_results(
    chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, save_count
):
    """Save current batch results independently"""

    # Concatenate current batch only
    successful_results = np.vstack(all_success_res).astype(np.float32)
    successful_indices = np.concatenate(all_success)
    error_indices = np.concatenate(all_error)
    error_info = np.concatenate(error_msgs)

    # Save to independent intermediate file
    intermediate_file = chunk_results_dir / f"chunk_{chunk_id}_part_{save_count}.h5"

    with h5py.File(intermediate_file, "w") as f:
        f.attrs["chunk_id"] = chunk_id
        f.attrs["part_idx"] = save_count
        f.attrs["part_nsample"] = len(successful_indices) + len(error_indices)
        f.attrs["saved_at"] = pd.Timestamp.now().isoformat()

        f.create_dataset("successful_indices", data=successful_indices, dtype="i8", compression="gzip")
        f.create_dataset("successful_results", data=successful_results, dtype="f4", compression="gzip")
        f.create_dataset("error_indices", data=error_indices, dtype="i8", compression="gzip")
        f.create_dataset("error_info", data=error_info, dtype=h5py.string_dtype(), compression="gzip")

    print(f"Part {save_count} saved: {successful_results.shape[0]} successful, {len(error_indices)} failed")


def get_model(checkpoint, device, config=None):
    if config is None:
        print("Using Prebuilt Model")
        with open(checkpoint, "rb") as f:
            model_package = pickle.load(f)

        model = model_package.model.eval().to(device)
        dna_tokenizer = model_package.dna_tokenizer
        myconfig = model_package.config

    else:
        print("Using Runtime Built Model")
        myconfig = load_config(config_name=config)
        logger = BaseLogger(name="Model packaging", level=logging.INFO)

        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model = setup_model(myconfig, logger=logger)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval().to(device)

        dna_tokenizer = FastaInterval(
            fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
        )

    return model, dna_tokenizer, myconfig


@click.command()
@click.option("-h5", "--hdf5_file", required=True)
@click.option("-c", "--chunk_indices", type=str, required=True)
@click.option("-m", "--model_path", required=True)
@click.option("--config_path", type=str, help="If provided, build the model in runtime, the model_path should be pointed to a chk")
@click.option("--device", default="cpu")
@click.option("--batch_size", type=int, default=32)
@click.option("--save_interval", type=int, default=20000, help="Save intermediate results every N samples")
@click.option("-p", "--precision", type=click.Choice(["float32", "float64"]), default="float32", help="Numerical precision (float32 for speed, float64 for accuracy)")
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
def main(hdf5_file, chunk_indices, model_path, config_path, device, batch_size, save_interval, precision, use_head):

    # Sort task indices for optimal HDF5 access pattern
    task_indices = np.load(chunk_indices)
    task_indices = np.sort(task_indices).tolist()
    chunk_id = int(Path(chunk_indices).stem.split("_")[1])
    if not task_indices:
        print("No tasks to process in this chunk")
        return

    model, dna_tokenizer, config = get_model(model_path, device, config_path)

    # Set model precision
    if precision == "float64":
        model = model.double()
        dtype_fn = lambda x: x.double()
        print("Using float64 precision for maximum accuracy")
    else:
        model = model.float()
        dtype_fn = lambda x: x.float()
        print("Using float32 precision for faster computation")

    dataset = VariantDataset(hdf5_file, task_indices, dna_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    trial_dims = load_label_meta_from_h5(hdf5_file)

    print(f"Computing variant effects for chunk {chunk_id}...")
    start_time = time.time()

    all_success_res = []
    all_success = []
    all_error = []
    error_msgs = []

    processed_count = 0
    part_count = 0
    last_save_time = start_time

    # Create intermediate results directory
    chunk_results_dir = Path(hdf5_file).parent / "chunk_results"
    chunk_results_dir.mkdir(exist_ok=True)

    for wt_batch, mut_batch, task_ids, msgs, masks in dataloader:
        if wt_batch is None:
            continue

        with torch.no_grad():
            # Apply selected precision to inputs
            wt_input = dtype_fn(wt_batch.permute(0, 2, 1).to(device))
            mut_input = dtype_fn(mut_batch.permute(0, 2, 1).to(device))

            pred_wt = model(wt_input, use_head).detach().cpu().numpy()[:,:, trial_dims, ...]
            pred_mut = model(mut_input, use_head).detach().cpu().numpy()[:, :, trial_dims, ...]
            diffs = pred_mut - pred_wt
            diffs = np.sum(diffs, axis=1)

        all_success_res.append(diffs)
        all_success.append(task_ids[masks])
        all_error.append(task_ids[~masks])
        error_msgs.append(msgs[~masks])

        processed_count += len(task_ids)

        # Save intermediate results every save_interval samples and reset accumulators
        if processed_count >= save_interval * (part_count + 1):
            part_count += 1
            current_time = time.time()
            time_since_last_save = current_time - last_save_time
            print(f"\nSaving part {part_count} at {processed_count} samples... (Time since last save: {time_since_last_save:.2f}s)")
            save_intermediate_results(
                chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, part_count
            )
            last_save_time = current_time
            # Reset accumulators for next part
            all_success_res = []
            all_success = []
            all_error = []
            error_msgs = []

    # Save any remaining data as the final part
    if all_success_res:
        part_count += 1
        current_time = time.time()
        time_since_last_save = current_time - last_save_time
        print(f"\nSaving final part {part_count}... (Time since last save: {time_since_last_save:.2f}s)")
        save_intermediate_results(
            chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, part_count
        )

    total_time = time.time() - start_time
    print(f"Finished in {total_time:.2f}s")

    # Create completion marker file
    completion_file = chunk_results_dir / f"chunk_{chunk_id}_summary.txt"
    with open(completion_file, "w") as f:
        f.write(f"Chunk {chunk_id} completed at {pd.Timestamp.now().isoformat()}\n")
        f.write(f"Total variants: {len(task_indices)}\n")
        f.write(f"Processing time: {total_time:.2f}s\n")
        f.write(f"Total parts saved: {part_count}\n")

    print(f"Chunk {chunk_id} completed with {part_count} parts saved")


if __name__ == "__main__":
    main()
