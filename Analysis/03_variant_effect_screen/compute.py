import os
import pickle
import sys
import time
from pathlib import Path
import warnings

import click
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))

from data.data_utils import STD_CHR
from data.tokenizer import str_to_one_hot


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
        return None, None, np.array(task_ids), np.array(msgs),  np.array(masks)
    return torch.stack(wt_list), torch.stack(mut_list), np.array(task_ids), np.array(msgs), np.array(masks)


def load_label_meta_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f.attrs["trial_names"]


@click.command()
@click.option("-h5", "--hdf5_file", required=True)
@click.option("-c", "--chunk_indices", type=str, required=True)
@click.option("-m", "--model_path", required=True)
@click.option("--device", default="cpu")
@click.option("--batch_size", type=int, default=32)
def main(hdf5_file, chunk_indices, model_path, device, batch_size):

    task_indices = np.load(chunk_indices).tolist()
    chunk_id = int(Path(chunk_indices).stem.split("_")[1])
    if not task_indices:
        print("No tasks to process in this chunk")
        return

    with open(model_path, "rb") as f:
        model_package = pickle.load(f)

    model = model_package.model.to(device)
    dna_tokenizer = model_package.dna_tokenizer
    config = model_package.config

    dataset = VariantDataset(hdf5_file, task_indices, dna_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

    print("Computing variant effects...")
    start_time = time.time()

    all_success_res = []
    all_success = []
    all_error = []
    error_msgs = []
    for wt_batch, mut_batch, task_ids, msgs, masks in tqdm(dataloader, desc="Computing effects"):
        if wt_batch is None:
            continue

        with torch.no_grad():
            pred_wt = model(wt_batch.permute(0, 2, 1).to(device), config.model.use_head, False).detach().cpu().numpy()
            pred_mut = model(mut_batch.permute(0, 2, 1).to(device), config.model.use_head, False).detach().cpu().numpy()
            diffs = pred_mut - pred_wt
            diffs = np.sum(diffs, axis=1)

        all_success_res.append(diffs)
        all_success.append(task_ids[masks])
        all_error.append(task_ids[~masks])
        error_msgs.append(msgs[~masks])

    successful_results = np.vstack(all_success_res)
    successful_indices = np.concatenate(all_success)
    error_indices = np.concatenate(all_error)
    error_info = np.concatenate(error_msgs)

    total_time = time.time() - start_time
    print(f"Finished in {total_time:.2f}s")

    chunk_results_dir = Path(hdf5_file).parent / "chunk_results"
    chunk_results_dir.mkdir(exist_ok=True)
    chunk_file = chunk_results_dir / f"chunk_{chunk_id}_results.h5"

    with h5py.File(chunk_file, "w") as f:
        f.attrs["chunk_id"] = chunk_id
        f.attrs["total_variants"] = len(task_indices)
        f.attrs["completed_at"] = pd.Timestamp.now().isoformat()
        f.attrs["forward_time_seconds"] = total_time

        f.create_dataset("successful_indices", data=successful_indices, dtype="i8", compression="gzip")
        f.create_dataset("successful_results", data=successful_results, dtype="f4", compression="gzip")
        f.create_dataset("error_indices", data=error_indices, dtype="i8", compression="gzip")
        f.create_dataset("error_info", data=error_info, dtype=h5py.string_dtype(), compression="gzip")

    print(f"Chunk results saved to: {chunk_file}")
    print(f"\nChunk {chunk_id} completed:")
    print(f"  Total variants: {len(task_indices)}")
    print(f"  Successfully computed: {len(successful_indices)}")
    print(f"  Failed: {len(error_indices)}")


if __name__ == "__main__":
    main()
