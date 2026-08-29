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
from data.tokenizer import FastaInterval, str_to_one_hot, one_hot_reverse_complement
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
            return None, None, task_index, "invalid_chr"

        try:
            token_dict = self.dna_tokenizer(
                chr_name=chr_name, start=pos - 1, end=pos, return_augs=False, return_rela_idx=True
            )
            s_idx, e_idx = token_dict["rela_idx"]
            wt_seq_onehot = token_dict["one_hot"]

            wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
            if ref != wt_nt_fetched:
                return None, None, task_index, f"ref_mismatch(ref:{ref},get:{wt_nt_fetched})"

            alt_nt_onehot = str_to_one_hot(alt)
            mut_seq_onehot = wt_seq_onehot.clone()
            mut_seq_onehot[s_idx:e_idx] = alt_nt_onehot
            # also output reverse complement sequence
            wt_seq_onehot_rev = one_hot_reverse_complement(wt_seq_onehot)
            mut_seq_onehot_rev = one_hot_reverse_complement(mut_seq_onehot)

            return (wt_seq_onehot, mut_seq_onehot), (wt_seq_onehot_rev, mut_seq_onehot_rev), task_index, None
        except Exception as e:
            return None, None, task_index, str(e)


def collate_fn(batch):
    wt_list, mut_list, wt_rev_list, mut_rev_list, task_ids, msgs, masks = [], [], [], [], [], [], []
    for item1, item2, task_index, err in batch:
        if item1 is not None and item2 is not None:
            wt_list.append(item1[0])
            mut_list.append(item1[1])
            wt_rev_list.append(item2[0])
            mut_rev_list.append(item2[1])
            masks.append(True)
        else:
            masks.append(False)
        task_ids.append(task_index)
        msgs.append(err)

    if len(wt_list) == 0:
        return None, None, None, None, np.array(task_ids), np.array(msgs), np.array(masks)
    return torch.stack(wt_list), torch.stack(mut_list), torch.stack(wt_rev_list), torch.stack(mut_rev_list), np.array(task_ids), np.array(msgs), np.array(masks)


def load_label_meta_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f["model_meta/trial_dims"][:]


def build_rc_swap_index(label_meta_df):
    """
    Build reverse complement swap index for stranded tracks (+/-).

    When doing reverse complement, + and - strand tracks need to be swapped
    because the strand orientation changes.

    Args:
        label_meta_df: DataFrame with label metadata containing 'dim' and 'trial' columns

    Returns:
        tuple: (org_index, swap_index) where both are numpy arrays, or (None, None) if no swapping needed
    """
    if label_meta_df is None:
        return None, None

    # Check if we have trial column
    if 'trial' not in label_meta_df.columns:
        return None, None

    # Build swap index using 'dim' column (track index in predictions)
    swap_index = []
    org_index = label_meta_df['dim'].tolist()

    for i, row in label_meta_df.iterrows():
        trial_name = row['trial']

        if trial_name.endswith('+'):
            # Find corresponding minus track
            minus_name = trial_name[:-1] + '-'
            matching = label_meta_df[label_meta_df['trial'] == minus_name]

            if len(matching) > 0:
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))

        elif trial_name.endswith('-'):
            # Find corresponding plus track
            plus_name = trial_name[:-1] + '+'
            matching = label_meta_df[label_meta_df['trial'] == plus_name]

            if len(matching) > 0:
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))
        else:
            # Don't change for other tracks
            swap_index.append(int(row['dim']))

    # Check if any swapping is needed
    if swap_index == org_index:
        return None, None

    print(f"Built reverse complement swap index for stranded tracks")
    return np.array(org_index), np.array(swap_index)


def _as_track_index(mask):
    """Turn a boolean track mask into the cheapest equivalent index.

    Returns None when nothing is selected, a slice when the selected tracks are
    contiguous (Borzoi's RNA tracks are, at 6068:7611), else the mask itself. A
    slice indexes a view; a boolean mask forces a copy in and a scatter back.
    """
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return None
    if hits[-1] - hits[0] + 1 == hits.size:
        return slice(int(hits[0]), int(hits[-1]) + 1)
    return mask


def untransform_predictions(data, label_meta=None, scale=1.0, clip_soft=48.0, sum_stat="sum_three_quarter"):
    """
    Untransform model predictions back to original scale.

    Reverses the forward transformations applied during data preprocessing:
    1. Scale multiplication: y = scale * y
    2. Soft clipping: if y > clip_soft: y = (clip_soft - 1) + sqrt(y - clip_soft + 1)
    3. Three-quarter power: y = y^(3/4) for sum_three_quarter

    Args:
        data: numpy array of predictions to untransform
        label_meta: DataFrame with transformation parameters per trial (scale, clip_soft, sum_stat)
        scale: scale factor applied in forward transform (default: 1.0)
        clip_soft: soft clipping threshold (default: 48.0)
        sum_stat: summary statistic used (default: "sum_three_quarter")

    Returns:
        Untransformed data in original scale
    """
    data = data.copy()

    if label_meta is not None:
        # Vectorised across tracks. This previously ran one Python iteration per
        # trial (7,611 of them) over a strided data[:, :, i] view, twice per batch.
        # The maths below is the same, applied to whole (B, L, T) arrays at once.
        n_tracks = data.shape[2]
        if len(label_meta) != n_tracks:
            raise ValueError(
                f"label_meta has {len(label_meta)} rows but predictions have {n_tracks} tracks"
            )

        def _column(name, default):
            if name in label_meta.columns:
                col = label_meta[name]
                return col.to_numpy() if name == "sum_stat" else col.to_numpy(dtype=np.float64)
            return np.full(n_tracks, default)

        trial_scale = _column("scale", 1.0)
        trial_clip_soft = _column("clip_soft", 48.0)
        trial_sum_stat = _column("sum_stat", "sum_three_quarter").astype(str)

        KNOWN = {"sum_three_quarter", "sum_sqrt", "mean_sqrt", "avg_sqrt", "sum", "mean", "avg"}
        unknown = sorted(set(trial_sum_stat) - KNOWN)
        if unknown:
            raise ValueError(f"Unknown sum_stat: {unknown[0]}")

        # Step 1: Undo scale
        np.divide(data, trial_scale.astype(data.dtype, copy=False), out=data)

        # Step 2: Undo soft clip
        # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
        # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
        clip_row = trial_clip_soft.astype(data.dtype, copy=False)
        # Cheap guard first: predictions rarely exceed clip_soft, and this avoids
        # allocating a (B, L, T) boolean mask on every batch. NaN compares False.
        if np.nanmax(data) > np.nanmin(clip_row):
            clip_mask = data > clip_row
            if clip_mask.any():
                edge = np.broadcast_to(clip_row - 1, data.shape)[clip_mask]
                data[clip_mask] = edge + (data[clip_mask] - edge) ** 2

        # Step 3: Undo the summary-statistic transform ('sum'/'mean'/'avg' are no-ops)
        # Forward: x = x^(3/4)  ->  Reverse: x = x^(4/3)
        three_quarter = _as_track_index(trial_sum_stat == "sum_three_quarter")
        sqrt_like = _as_track_index(np.isin(trial_sum_stat, ["sum_sqrt", "mean_sqrt", "avg_sqrt"]))
        if three_quarter is not None:
            data[:, :, three_quarter] = data[:, :, three_quarter] ** (4.0 / 3.0)
        if sqrt_like is not None:
            data[:, :, sqrt_like] = (data[:, :, sqrt_like] + 1) ** 2 - 1
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


def save_intermediate_results(
    chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, save_count, score_names
):
    """Save current batch results independently"""

    # Concatenate current batch only
    successful_results = {}
    for score_name in score_names:
        successful_results[score_name] = np.vstack(all_success_res[score_name]).astype(np.float32)
    
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
        f.attrs["score_names"] = score_names

        f.create_dataset("successful_indices", data=successful_indices, dtype="i8", compression="gzip")
        
        # Save results for each score type
        results_grp = f.create_group("successful_results")
        for score_name in score_names:
            results_grp.create_dataset(score_name, data=successful_results[score_name], dtype="f4", compression="gzip")
        
        f.create_dataset("error_indices", data=error_indices, dtype="i8", compression="gzip")
        f.create_dataset("error_info", data=error_info, dtype=h5py.string_dtype(), compression="gzip")

    print(f"Part {save_count} saved: {len(successful_indices)} successful, {len(error_indices)} failed")


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


def validate_score_names(score_names):
    """Validate and filter score names based on implemented scores."""
    IMPLEMENTED_SCORES = {"raw_diff", "raw_log_diff", "l1_sum", "l2_sum", "log_square", 
                          "local_raw_diff", "local_raw_log_diff", "local_l1_sum", "local_l2_sum", "local_log_square"}  # Add more score names as you implement them
    original_score_names = set(score_names)
    score_names = [name for name in score_names if name in IMPLEMENTED_SCORES]

    removed_scores = original_score_names - set(score_names)
    if removed_scores:
        for score in removed_scores:
            print(f"Warning: Score '{score}' not implemented, removing from computation")

    if not score_names:
        raise ValueError("No implemented scores found in the requested score names")

    return score_names


LOCAL_HALF_WIDTH = 15  # centre bin +/- 15 -> 31 bins of 32 bp = 992 bp


def crop_local(data):
    """Centre-crop (B, L, T) predictions to the 31 bins the local_* scores use.

    The local_* scores previously ran their elementwise maths (log2, squaring)
    across all L bins and only then discarded 99.5% of the result. Every one of
    those operations is elementwise, so cropping first is bit-identical and about
    two orders of magnitude cheaper.
    """
    mid = data.shape[1] // 2
    return data[:, mid - LOCAL_HALF_WIDTH: mid + LOCAL_HALF_WIDTH + 1, :]


# calculate score
def vep_score(pred_mut, pred_wt, score_name):
    if score_name == "raw_diff":
        diffs = pred_mut - pred_wt
        scores = np.sum(diffs, axis=1)
    elif score_name == "raw_log_diff":
        log_alt = np.log2(1 + pred_mut)
        log_ref = np.log2(1 + pred_wt)
        diffs = log_alt - log_ref
        scores = np.sum(diffs, axis=1)
    elif score_name == "l1_sum":
        diffs = np.abs(pred_mut - pred_wt)  # shape: (B, L, T)
        scores = np.sum(diffs, axis=1)  # shape: (B, T)
    elif score_name == "l2_sum":
        diff_squared = (pred_mut - pred_wt) ** 2  # shape: (B, L, T)
        sum_diff = np.sum(diff_squared, axis=1)  # shape: (B, T)
        scores = np.sqrt(sum_diff)  # shape: (B, T)
    elif score_name == "log_square":
        log_alt = np.log2(1 + pred_mut)
        log_ref = np.log2(1 + pred_wt)
        diff_squared = (log_alt - log_ref) ** 2  # shape: (B, L, T)
        sum_diff = np.sum(diff_squared, axis=1)  # shape: (B, T)
        scores = np.sqrt(sum_diff)  # shape: (B, T)
    elif score_name == "local_l1_sum":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        mut_local, wt_local = crop_local(pred_mut), crop_local(pred_wt)  # (B, 31, T)
        scores = np.sum(np.abs(mut_local - wt_local), axis=1)  # shape: (B, T)
    elif score_name == "local_l2_sum":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        mut_local, wt_local = crop_local(pred_mut), crop_local(pred_wt)  # (B, 31, T)
        sum_diff = np.sum((mut_local - wt_local) ** 2, axis=1)  # shape: (B, T)
        scores = np.sqrt(sum_diff)  # shape: (B, T)
    elif score_name == "local_raw_diff":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        mut_local, wt_local = crop_local(pred_mut), crop_local(pred_wt)  # (B, 31, T)
        scores = np.sum(mut_local - wt_local, axis=1)  # shape: (B, T)
    elif score_name == "local_raw_log_diff":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        mut_local, wt_local = crop_local(pred_mut), crop_local(pred_wt)  # (B, 31, T)
        diffs = np.log2(1 + mut_local) - np.log2(1 + wt_local)  # shape: (B, 31, T)
        scores = np.sum(diffs, axis=1)  # shape: (B, T)
    elif score_name == "local_log_square":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        mut_local, wt_local = crop_local(pred_mut), crop_local(pred_wt)  # (B, 31, T)
        diffs = np.log2(1 + mut_local) - np.log2(1 + wt_local)  # shape: (B, 31, T)
        sum_diff = np.sum(diffs ** 2, axis=1)  # shape: (B, T)
        scores = np.sqrt(sum_diff)  # shape: (B, T)
    else:
        raise ValueError(f"Score '{score_name}' not implemented")
    return scores
            
@click.command()
@click.option("-h5", "--hdf5_file", required=True)
@click.option("-c", "--chunk_indices", type=str, required=True)
@click.option("-m", "--model_path", required=True)
@click.option("--config_path", type=str, help="If provided, build the model in runtime, the model_path should be pointed to a chk")
@click.option("--device", default="cpu")
@click.option("--batch_size", type=int, default=32)
@click.option("--save_interval", type=int, default=2000, help="Save intermediate results every N samples")
@click.option("-p", "--precision", type=click.Choice(["float32", "float64"]), default="float32", help="Numerical precision (float32 for speed, float64 for accuracy)")
@click.option("--use_head", type=str, default="regression", help="Which prediction head to use")
@click.option("-s", "--score_names", multiple=True, help="Score names to compute (will read from HDF5 if not provided)")
@click.option("--untransform", is_flag=True, default=False, help="Untransform predictions back to original scale")
@click.option("--label_meta", type=str, default="Data/source/GWAS/borzoi_label_meta.csv", help="Path to label metadata CSV file")
def main(hdf5_file, chunk_indices, model_path, config_path, device, batch_size, save_interval, precision, use_head, score_names, untransform, label_meta):

    # Sort task indices for optimal HDF5 access pattern
    task_indices = np.load(chunk_indices)
    task_indices = np.sort(task_indices).tolist()
    chunk_id = int(Path(chunk_indices).stem.split("_")[1])
    if not task_indices:
        print("No tasks to process in this chunk")
        return

    # Load label metadata
    label_meta_df = None
    rc_orig_index = None
    rc_swap_index = None
    if label_meta:
        label_meta_path = Path(label_meta)
        if not label_meta_path.is_absolute():
            # If relative path, resolve from ROOT
            label_meta_path = ROOT / label_meta

        if label_meta_path.exists():
            print(f"Loading label metadata from {label_meta_path}")
            label_meta_df = pd.read_csv(label_meta_path, index_col=None)

            if untransform:
                print(f"Untransform enabled: will reverse transformations using label metadata")

            # Build reverse complement swap index for stranded tracks
            rc_orig_index, rc_swap_index = build_rc_swap_index(label_meta_df)
        else:
            print(f"Warning: Label metadata file not found at {label_meta_path}")
            if untransform:
                raise ValueError(f"--untransform requires valid --label_meta file")

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

    # Get score names from HDF5 or use provided ones
    if not score_names:
        with h5py.File(hdf5_file, "r") as f:
            score_names = f.attrs["score_names"]
    else:
        score_names = score_names

    score_names = validate_score_names(score_names)

    print(f"Score names: {score_names}")

    dataset = VariantDataset(hdf5_file, task_indices, dna_tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    trial_dims = load_label_meta_from_h5(hdf5_file)

    # Precompute GPU-side indices for the forward/reverse combination.
    trial_idx = torch.as_tensor(np.asarray(trial_dims), dtype=torch.long, device=device)
    rc_swap_t = None
    if rc_swap_index is not None:
        if not np.array_equal(np.asarray(rc_orig_index), np.arange(len(rc_orig_index))):
            raise ValueError("rc_orig_index must enumerate every track in order")
        rc_swap_t = torch.as_tensor(np.asarray(rc_swap_index), dtype=torch.long, device=device)
    # Every local_* score reads only the centre 31 bins, so the other 6113 never
    # need to leave the GPU. Mixed score sets keep the full window.
    local_only = all(name.startswith("local_") for name in score_names)
    if local_only:
        print(f"All scores are local_*: cropping to {2 * LOCAL_HALF_WIDTH + 1} bins on device")

    print(f"Computing variant effects for chunk {chunk_id}...")
    start_time = time.time()

    # Initialize result storage for each score type
    all_success_res = {score_name: [] for score_name in score_names}
    all_success = []
    all_error = []
    error_msgs = []

    processed_count = 0
    part_count = 0
    last_save_time = start_time

    # Create intermediate results directory
    h5_name = Path(hdf5_file).stem  # Get filename without .h5 extension
    chunk_results_dir = Path(hdf5_file).parent / f"{h5_name}_chunk_results"
    chunk_results_dir.mkdir(exist_ok=True)

    for wt_batch, mut_batch, wt_rev_batch, mut_rev_batch, task_ids, msgs, masks in dataloader:
        if wt_batch is None:
            continue

        with torch.no_grad():
            # Apply selected precision to inputs
            wt_input = dtype_fn(wt_batch.permute(0, 2, 1).to(device))
            mut_input = dtype_fn(mut_batch.permute(0, 2, 1).to(device))
            wt_rev_input = dtype_fn(wt_rev_batch.permute(0, 2, 1).to(device))
            mut_rev_input = dtype_fn(mut_rev_batch.permute(0, 2, 1).to(device))

            # Combine forward and reverse-complement on the GPU, then copy once.
            # The old code copied all four raw (B, 6144, T) predictions to the host
            # (~0.43 s/variant of pageable transfer) and combined them in numpy.
            # flip/swap are pure data movement and (a+b)/2 is exact in float32, so
            # this is bit-identical. When every requested score is a local_* one,
            # crop to 31 bins before the copy as well -- but only AFTER the flip:
            # L is even, so a centre crop taken before flipping is off by one bin.
            def _combine(fwd_out, rev_out):
                fwd = fwd_out.index_select(2, trial_idx)
                rev = rev_out.index_select(2, trial_idx).flip(1)
                if rc_swap_t is not None:
                    rev = rev.index_select(2, rc_swap_t)
                combined = (fwd + rev) / 2.0
                if local_only:
                    combined = crop_local(combined)
                return combined.detach().cpu().numpy()

            pred_wt = _combine(model(wt_input, use_head), model(wt_rev_input, use_head))
            pred_mut = _combine(model(mut_input, use_head), model(mut_rev_input, use_head))

            # Untransform predictions if requested
            if untransform:
                pred_wt = untransform_predictions(pred_wt, label_meta=label_meta_df)
                pred_mut = untransform_predictions(pred_mut, label_meta=label_meta_df)

            # Calculate different scores based on score_names
            for score_name in score_names:
                scores = vep_score(pred_mut, pred_wt, score_name)
                all_success_res[score_name].append(scores)
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
                chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, part_count, score_names
            )
            last_save_time = current_time
            # Reset accumulators for next part
            all_success_res = {score_name: [] for score_name in score_names}
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
            chunk_results_dir, chunk_id, all_success_res, all_success, all_error, error_msgs, part_count, score_names
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
