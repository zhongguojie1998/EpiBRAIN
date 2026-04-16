import gzip
import logging
import os
import pickle
import re
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
    Build reverse complement swap index for RNAplus/RNAminus tracks.

    When doing reverse complement, RNAplus and RNAminus need to be swapped
    because the strand orientation changes.

    Returns:
        tuple: (org_index, swap_index) where both are numpy arrays, or (None, None) if no swapping needed
    """
    if label_meta_df is None:
        return None, None

    # Check if we have modality column and RNAplus/RNAminus tracks
    if 'modality' not in label_meta_df.columns:
        return None, None

    has_plus = 'RNAplus' in label_meta_df['modality'].values
    has_minus = 'RNAminus' in label_meta_df['modality'].values

    if not (has_plus and has_minus):
        return None, None

    # Build swap index using 'dim' column (track index in predictions)
    swap_index = []
    org_index = label_meta_df['dim'].tolist()

    for i, row in label_meta_df.iterrows():
        if row['modality'] == 'RNAplus':
            # Find the index of RNAminus for corresponding cell type
            if 'cell_type' in label_meta_df.columns:
                matching = label_meta_df[
                    (label_meta_df['modality'] == 'RNAminus') &
                    (label_meta_df['cell_type'] == row['cell_type'])
                ]
            else:
                # Match by trial name pattern
                matching = label_meta_df[label_meta_df['modality'] == 'RNAminus']

            if len(matching) > 0:
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))

        elif row['modality'] == 'RNAminus':
            # Find the index of RNAplus for corresponding cell type
            if 'cell_type' in label_meta_df.columns:
                matching = label_meta_df[
                    (label_meta_df['modality'] == 'RNAplus') &
                    (label_meta_df['cell_type'] == row['cell_type'])
                ]
            else:
                # Match by trial name pattern
                matching = label_meta_df[label_meta_df['modality'] == 'RNAplus']

            if len(matching) > 0:
                swap_index.append(int(matching.iloc[0]['dim']))
            else:
                swap_index.append(int(row['dim']))
        else:
            # Don't change for other modalities
            swap_index.append(int(row['dim']))

    print(f"Built reverse complement swap index for RNAplus/RNAminus tracks")
    return np.array(org_index), np.array(swap_index)


def untransform_predictions(data, label_meta=None, scale=1.0, clip_soft=48.0, sum_stat="sum_three_quarter"):
    """
    Untransform model predictions back to original scale.

    Reverses the forward transformations applied during data preprocessing:
    1. Scale multiplication: y = scale * y
    2. Soft clipping: if y > clip_soft: y = (clip_soft - 1) + sqrt(y - clip_soft + 1)
    3. Three-quarter power: y = y^(3/4) for sum_three_quarter
    4. BasalGanglia ATAC correction: multiply by 100 for BasalGanglia-*_ATAC tracks

    Args:
        data: numpy array of predictions to untransform
        label_meta: DataFrame with transformation parameters per trial (scale, clip_soft, sum_stat, trial)
        scale: scale factor applied in forward transform (default: 1.0)
        clip_soft: soft clipping threshold (default: 48.0)
        sum_stat: summary statistic used (default: "sum_three_quarter")

    Returns:
        Untransformed data in original scale
    """
    data = data.copy()

    if label_meta is not None:
        # do it for each trial based on label_meta
        for i, row in label_meta.iterrows():
            trial_scale = row.get('scale', 1.0)
            trial_clip_soft = row.get('clip_soft', 48.0)
            trial_sum_stat = row.get('sum_stat', 'sum_three_quarter')
            trial_name = row.get('exp', '')

            # Step 1: Undo scale
            if trial_scale != 1.0:
                data[:, :, i] = data[:, :, i] / trial_scale

            # Step 2: Undo soft clip
            # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
            # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
            if trial_clip_soft is not None:
                clip_mask = data[:, :, i] > trial_clip_soft
                data[clip_mask, i] = (trial_clip_soft - 1) + (data[clip_mask, i] - (trial_clip_soft - 1)) ** 2

            # Step 3: Undo three-quarter power
            # Forward: x = x^(3/4)
            # Reverse: x = x^(4/3)
            if trial_sum_stat == "sum_three_quarter":
                data[:, :, i] = data[:, :, i] ** (4.0 / 3.0)
            elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
                data[:, :, i] = (data[:, :, i] + 1) ** 2 - 1
            elif trial_sum_stat in ['sum', 'mean', "avg"]:
                # no transformation applied
                pass
            else:
                raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

            # Step 4: BasalGanglia ATAC correction - multiply by 100
            if trial_name.startswith('BasalGanglia-') and trial_name.endswith('_ATAC'):
                data[:, :, i] = data[:, :, i] * 100
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


def parse_vcf_gene_map(vcf_path):
    """Parse VCF INFO column for gene_ID / gene_name.

    Returns dict mapping (chr, pos, ref, alt) -> gene identifier (Ensembl version stripped).
    Prefers gene_ID over gene_name.
    """
    open_func = gzip.open if vcf_path.endswith(".gz") else open
    mode = "rt" if vcf_path.endswith(".gz") else "r"
    gene_map = {}
    with open_func(vcf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _id, ref, alt, _qual, _filter, info = parts[:8]
            try:
                pos_i = int(pos)
            except ValueError:
                continue
            ref = ref.strip().upper()
            alt = alt.strip().upper()
            gene_id = None
            gene_name = None
            for kv in info.split(";"):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                if k == "gene_ID":
                    gene_id = v.strip()
                elif k == "gene_name":
                    gene_name = v.strip()
            gene = gene_id if gene_id else gene_name
            if gene:
                gene = gene.split(".")[0]
                gene_map[(chrom, pos_i, ref, alt)] = gene
    return gene_map


def parse_gtf_exons(gtf_path, gene_keys):
    """Parse GTF for exon features; keep only genes whose id or name is in gene_keys.

    Returns dict: gene_key -> list of (chrom, start_1based, end_1based) tuples (unique exons).
    Keys are indexed by both stripped gene_id and gene_name to allow either lookup.
    """
    open_func = gzip.open if gtf_path.endswith(".gz") else open
    mode = "rt" if gtf_path.endswith(".gz") else "r"
    gid_re = re.compile(r'gene_id "([^"]+)"')
    gname_re = re.compile(r'gene_name "([^"]+)"')
    gene_exons = {}
    with open_func(gtf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            attrs = parts[8]
            m_gid = gid_re.search(attrs)
            m_gname = gname_re.search(attrs)
            gid = m_gid.group(1).split(".")[0] if m_gid else None
            gname = m_gname.group(1) if m_gname else None
            for key in (gid, gname):
                if key is not None and key in gene_keys:
                    gene_exons.setdefault(key, set()).add((chrom, start, end))
    return {k: list(v) for k, v in gene_exons.items()}


def compute_gene_lfc_scores(pred_mut, pred_wt, task_ids, variant_info, gene_map, gene_exons, context_length, bin_size):
    """Compute gene_lfc = log(mean_exon_bins(ALT)+eps) - log(mean_exon_bins(REF)+eps).

    pred_mut, pred_wt: np.ndarray shape (B, L, T).
    task_ids: shape (B,), task indices corresponding to rows of pred_*.
    variant_info: dict task_index -> (chrom, pos, ref, alt).
    gene_map: dict (chrom, pos, ref, alt) -> gene_id.
    gene_exons: dict gene_id -> list of (chrom, start_1based, end_1based).
    context_length: int, model input context length in bp.
    bin_size: int, bp per output bin (from config.data.preprocess.window_size).

    The L output bins are the center-cropped portion of the context window
    (L*bin_size bp centered on the variant).

    Returns (B, T) float32 array; rows for variants without exon overlap are NaN.
    """
    B, L, T = pred_wt.shape
    covered = L * bin_size
    crop_bp = (context_length - covered) // 2
    eps = 0.001

    scores = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        tid = int(task_ids[b])
        info = variant_info.get(tid)
        if info is None:
            continue
        chrom, pos, ref, alt = info
        gene = gene_map.get((chrom, int(pos), ref, alt))
        if gene is None:
            continue
        exons = gene_exons.get(gene)
        if not exons:
            continue

        # 0-based half-open model context window centered on the variant
        context_start = (int(pos) - 1) - ((context_length - 1) // 2)
        # Output bins cover [bins_start, bins_end), a center-crop of the context
        bins_start = context_start + crop_bp
        bins_end = bins_start + covered

        mask = np.zeros(L, dtype=bool)
        for ec, es, ee in exons:
            if ec != chrom:
                continue
            es0 = es - 1
            ee0 = ee
            if ee0 <= bins_start or es0 >= bins_end:
                continue
            lo = max(0, (es0 - bins_start) // bin_size)
            hi = min(L, (ee0 - bins_start + bin_size - 1) // bin_size)
            if hi > lo:
                mask[lo:hi] = True
        if not mask.any():
            continue
        mean_alt = pred_mut[b, mask, :].mean(axis=0)
        mean_ref = pred_wt[b, mask, :].mean(axis=0)
        scores[b] = np.log(mean_alt + eps) - np.log(mean_ref + eps)
    return scores


def validate_score_names(score_names):
    """Validate and filter score names based on implemented scores."""
    IMPLEMENTED_SCORES = {"raw_diff", "raw_log_diff", "l1_sum", "l2_sum", "log_square",
                          "local_raw_diff", "local_raw_log_diff", "local_l1_sum", "local_l2_sum", "local_log_square",
                          "gene_lfc"}  # Add more score names as you implement them
    original_score_names = set(score_names)
    score_names = [name for name in score_names if name in IMPLEMENTED_SCORES]

    removed_scores = original_score_names - set(score_names)
    if removed_scores:
        for score in removed_scores:
            print(f"Warning: Score '{score}' not implemented, removing from computation")

    if not score_names:
        raise ValueError("No implemented scores found in the requested score names")

    return score_names

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
        diffs = np.abs(pred_mut - pred_wt)  # shape: (B, L, T)
        diffs_local = diffs[:, diffs.shape[1] // 2 - 15: diffs.shape[1] // 2 + 16, :]  # shape: (B, 31, T)
        scores = np.sum(diffs_local, axis=1)  # shape: (B, T)
    elif score_name == "local_l2_sum":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        diff_squared = (pred_mut - pred_wt) ** 2  # shape: (B, L, T)
        diff_squared_local = diff_squared[:, diff_squared.shape[1] // 2 - 15: diff_squared.shape[1] // 2 + 16, :]  # shape: (B, 31, T)
        sum_diff = np.sum(diff_squared_local, axis=1)  # shape: (B, T)
        scores = np.sqrt(sum_diff)  # shape: (B, T)
    elif score_name == "local_raw_diff":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        diffs = pred_mut - pred_wt
        diffs_local = diffs[:, diffs.shape[1] // 2 - 15: diffs.shape[1] // 2 + 16, :]  # shape: (B, 31, T)
        scores = np.sum(diffs_local, axis=1)  # shape: (B, T)
    elif score_name == "local_raw_log_diff":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        log_alt = np.log2(1 + pred_mut)
        log_ref = np.log2(1 + pred_wt)
        diffs = log_alt - log_ref
        diffs_local = diffs[:, diffs.shape[1] // 2 - 15: diffs.shape[1] // 2 + 16, :]  # shape: (B, 31, T)
        scores = np.sum(diffs_local, axis=1)  # shape: (B, T)
    elif score_name == "local_log_square":
        # only consider the center position ± 15 bins, i.e., 31 bins in total, giving 32*31=992bp
        log_alt = np.log2(1 + pred_mut)
        log_ref = np.log2(1 + pred_wt)
        diff_squared = (log_alt - log_ref) ** 2  # shape: (B, L, T)
        diff_squared_local = diff_squared[:, diff_squared.shape[1] // 2 - 15: diff_squared.shape[1] // 2 + 16, :]  # shape: (B, 31, T)
        sum_diff = np.sum(diff_squared_local, axis=1)  # shape: (B, T)
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
@click.option("--label_meta", type=str, help="Path to label metadata CSV file (required if --untransform is used)")
@click.option("--gtf", type=str, help="Path to GENCODE GTF (required if 'gene_lfc' is in score_names)")
@click.option("--vcf", type=str, help="Path to VCF with gene_ID/gene_name in INFO (required if 'gene_lfc' is in score_names)")
def main(hdf5_file, chunk_indices, model_path, config_path, device, batch_size, save_interval, precision, use_head, score_names, untransform, label_meta, gtf, vcf):

    # Sort task indices for optimal HDF5 access pattern
    task_indices = np.load(chunk_indices)
    task_indices = np.sort(task_indices).tolist()
    chunk_id = int(Path(chunk_indices).stem.split("_")[1])
    if not task_indices:
        print("No tasks to process in this chunk")
        return

    # Validate untransform option
    if untransform and not label_meta:
        raise ValueError("--label_meta is required when --untransform is enabled")

    # Load label metadata if untransform is enabled
    label_meta_df = None
    rc_orig_index = None
    rc_swap_index = None
    if untransform:
        print(f"Loading label metadata from {label_meta}")
        label_meta_df = pd.read_csv(label_meta, index_col=None)
        print(f"Untransform enabled: will reverse transformations using label metadata")

        # Build reverse complement swap index for RNAplus/RNAminus tracks
        rc_orig_index, rc_swap_index = build_rc_swap_index(label_meta_df)
        if rc_swap_index is not None:
            print("Built reverse complement swap index for RNAplus/RNAminus tracks")
    else:
        # Even if not untransforming, we might still need swap index if label_meta is provided
        if label_meta:
            label_meta_df = pd.read_csv(label_meta, index_col=None)
            rc_orig_index, rc_swap_index = build_rc_swap_index(label_meta_df)
            if rc_swap_index is not None:
                print("Built reverse complement swap index for RNAplus/RNAminus tracks")

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

    # Set up gene_lfc auxiliary data (gene map from VCF, exon spans from GTF)
    need_gene_lfc = "gene_lfc" in score_names
    gene_map = None
    gene_exons = None
    variant_info = None
    context_length_bp = None
    bin_size_bp = None
    if need_gene_lfc:
        if not gtf or not vcf:
            raise ValueError("--gtf and --vcf are required when 'gene_lfc' is in score_names")
        print(f"Loading gene map from VCF: {vcf}")
        gene_map = parse_vcf_gene_map(vcf)
        print(f"  Loaded {len(gene_map)} variant->gene entries")
        unique_genes = set(gene_map.values())
        print(f"Loading exons for {len(unique_genes)} unique genes from GTF: {gtf}")
        gene_exons = parse_gtf_exons(gtf, unique_genes)
        print(f"  Loaded exons for {len(gene_exons)} genes")
        context_length_bp = int(config.data.context_length)
        bin_size_bp = int(config.data.preprocess.window_size)
        print(f"  Using context_length={context_length_bp} bp, bin_size={bin_size_bp} bp")

    dataset = VariantDataset(hdf5_file, task_indices, dna_tokenizer)

    # Build task_index -> (chr, pos, ref, alt) lookup for gene_lfc
    if need_gene_lfc:
        variant_info = {int(tid): (c, int(p), r, a) for (tid, c, p, r, a) in dataset.variants}
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    trial_dims = load_label_meta_from_h5(hdf5_file)

    # Remap rc_orig/rc_swap from full-dim space to positions within trial_dims,
    # since predictions are sliced by trial_dims below (axis 2 size = len(trial_dims)).
    if rc_swap_index is not None:
        trial_dims_arr = np.asarray(trial_dims).astype(int)
        dim_to_pos = {int(d): i for i, d in enumerate(trial_dims_arr)}
        remapped_orig, remapped_swap = [], []
        for o, s in zip(rc_orig_index, rc_swap_index):
            o, s = int(o), int(s)
            if o in dim_to_pos and s in dim_to_pos:
                remapped_orig.append(dim_to_pos[o])
                remapped_swap.append(dim_to_pos[s])
        if remapped_orig:
            rc_orig_index = np.array(remapped_orig, dtype=np.int64)
            rc_swap_index = np.array(remapped_swap, dtype=np.int64)
            print(f"Remapped rc swap index to trial_dims space: {len(remapped_orig)} pairs")
        else:
            rc_orig_index = rc_swap_index = None
            print("No rc swap pairs survived trial_dims filter; disabling swap")

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

            pred_wt_fwd = model(wt_input, use_head).detach().cpu().numpy()[:,:, trial_dims, ...]
            pred_mut_fwd = model(mut_input, use_head).detach().cpu().numpy()[:, :, trial_dims, ...]
            pred_wt_rev = model(wt_rev_input, use_head).detach().cpu().numpy()[:,:, trial_dims, ...]
            pred_mut_rev = model(mut_rev_input, use_head).detach().cpu().numpy()[:, :, trial_dims, ...]

            # Flip predictions along sequence dimension (reverse order)
            pred_wt_rev = np.flip(pred_wt_rev, axis=1)
            pred_mut_rev = np.flip(pred_mut_rev, axis=1)

            # Swap RNAplus and RNAminus tracks if needed
            if rc_swap_index is not None:
                pred_wt_rev[:, :, rc_orig_index] = pred_wt_rev[:, :, rc_swap_index]
                pred_mut_rev[:, :, rc_orig_index] = pred_mut_rev[:, :, rc_swap_index]

            # Average forward and reverse predictions
            pred_wt = (pred_wt_fwd + pred_wt_rev) / 2.0
            pred_mut = (pred_mut_fwd + pred_mut_rev) / 2.0

            # Untransform predictions if requested
            if untransform:
                pred_wt = untransform_predictions(pred_wt, label_meta=label_meta_df)
                pred_mut = untransform_predictions(pred_mut, label_meta=label_meta_df)

            # Calculate different scores based on score_names
            valid_task_ids = task_ids[masks]
            for score_name in score_names:
                if score_name == "gene_lfc":
                    scores = compute_gene_lfc_scores(
                        pred_mut, pred_wt, valid_task_ids, variant_info, gene_map, gene_exons, context_length_bp, bin_size_bp
                    )
                else:
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
    if all_success or all_error:
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
