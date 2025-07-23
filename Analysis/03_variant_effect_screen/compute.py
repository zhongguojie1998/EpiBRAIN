import json
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
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from data.data_utils import STD_CHR
from data.tokenizer import str_to_one_hot


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


def onehot_to_str(seq_onehot):
    """Convert one-hot encoded tensor (L x 4) to DNA string"""
    mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "T", (0, 0, 0, 0): "N"}

    seq_str = ""
    for vec in seq_onehot:
        key = tuple((vec > 0.5).int().tolist())
        seq_str += mapping.get(key, "N")
    return seq_str


def compute_variant_effect(model, dna_tokenizer, config, chr_name, pos, ref, alt, device="cpu"):
    """Compute variant effect using the model"""

    if chr_name not in STD_CHR:
        return None

    try:
        # Get reference sequence (tokenizer is 0-indexed, VCF is 1-indexed)
        token_dict = dna_tokenizer(
            chr_name=chr_name, start=pos - 1, end=pos, return_augs=False, return_rela_idx=True
        )

        s_idx, e_idx = token_dict["rela_idx"]
        wt_seq_onehot = token_dict["one_hot"]

        # Verify reference allele
        wt_nt_fetched = onehot_to_str(wt_seq_onehot[s_idx:e_idx])
        if ref != wt_nt_fetched:
            print(f"Warning: Ref mismatch at {chr_name}:{pos}, expected {ref}, got {wt_nt_fetched}")
            return None

        # Create alternate sequence
        alt_nt_onehot = str_to_one_hot(alt)
        mut_seq_onehot = wt_seq_onehot.clone()
        mut_seq_onehot[s_idx:e_idx] = alt_nt_onehot

        # Get predictions
        with torch.no_grad():
            # Wild-type prediction
            pred_res_wt = (
                model(wt_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), config.model.use_head, False)
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
            )

            # Mutant prediction
            pred_res_mut = (
                model(mut_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(device), config.model.use_head, False)
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
            )

        # Compute difference and sum across genomic positions (dim 0)
        diff = pred_res_mut - pred_res_wt
        variant_effect = np.sum(diff, axis=0)  # Sum across genomic positions

        return variant_effect.astype(np.float32)

    except Exception as e:
        print(f"Error processing variant {chr_name}:{pos} {ref}>{alt}: {e}")
        return None


def load_label_meta_from_h5(h5_path):
    """Load label metadata from HDF5 file"""
    with h5py.File(h5_path, "r") as f:
        return f.attrs["trial_names"]


@click.command()
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 file")
@click.option("-c", "--chunk_file", type=str, required=True, help="Path to the chunk info json")
@click.option("-m", "--model_path", required=True, help="Path to packaged model file")
@click.option("--device", default="cpu", help="Computing device (cpu/cuda)")
def main(hdf5_file, chunk_file, model_path, device):
    """Compute variant effects for a chunk of tasks"""

    # Load chunk information
    chunk_id = int(Path(chunk_file).stem.split("_")[-1])
    with open(chunk_file, "r") as f:
        chunk_info = json.load(f)

    task_indices = chunk_info["task_indices"]

    if not task_indices:
        print("No tasks to process in this chunk")
        return
    else:
        print(f"Processing {len(task_indices)} variants")

    # Load model package and initialize once
    with open(model_path, "rb") as f:
        model_package = pickle.load(f)
    
    # Initialize model components once
    model = model_package.model.to(device)
    dna_tokenizer = model_package.dna_tokenizer
    config = model_package.config

    # Load label metadata from HDF5 file
    trials = load_label_meta_from_h5(hdf5_file)
    n_dims = len(trials)

    # Process variants
    results = np.full((len(task_indices), n_dims), np.nan, dtype=np.float32)
    successful_computations = 0

    # First read all variant data (read-only, safe for concurrent access)
    with h5py.File(hdf5_file, "r") as f:
        variants_grp = f["variants"]

        # Batch read variant data for better performance
        task_indices = np.array(task_indices)
        chrs = variants_grp["chr"][task_indices]  
        positions = variants_grp["pos"][task_indices]
        refs = variants_grp["ref"][task_indices]
        alts = variants_grp["alt"][task_indices]
        
        # Convert bytes to strings if needed
        if hasattr(chrs[0], 'decode'):
            chrs = [c.decode("utf-8") for c in chrs]
        if hasattr(refs[0], 'decode'):
            refs = [r.decode("utf-8") for r in refs]
        if hasattr(alts[0], 'decode'):
            alts = [a.decode("utf-8") for a in alts]

    # Compute all variants (no file access)
    print("Computing variant effects...")
    forward_start_time = time.time()
    
    for i, (chr_name, pos, ref, alt) in enumerate(
        tqdm(zip(chrs, positions, refs, alts), total=len(task_indices), desc="Computing effects")
    ):
        variant_effect = compute_variant_effect(model, dna_tokenizer, config, chr_name, pos, ref, alt, device)

        if variant_effect is not None:
            results[i, :] = variant_effect
            successful_computations += 1
    
    forward_total_time = time.time() - forward_start_time
    print(f"Forward computation time: {forward_total_time:.2f} seconds")
    print(f"Average time per variant: {forward_total_time/len(task_indices):.4f} seconds")

    # Write results to individual chunk file (no concurrency issues)
    print("Storing results to chunk file...")
    chunk_results_dir = Path(hdf5_file).parent / "chunk_results"
    chunk_results_dir.mkdir(exist_ok=True)

    chunk_file = chunk_results_dir / f"chunk_{chunk_id}_results.h5"

    # Store chunk results safely
    with h5py.File(chunk_file, "w") as f:
        # Store metadata
        f.attrs["chunk_id"] = chunk_id
        f.attrs["total_variants"] = len(task_indices)
        f.attrs["successful_computations"] = successful_computations
        f.attrs["n_dims"] = n_dims
        f.attrs["completed_at"] = pd.Timestamp.now().isoformat()
        f.attrs["forward_time_seconds"] = forward_total_time
        f.attrs["avg_time_per_variant"] = forward_total_time / len(task_indices)

        # Store task indices and results
        f.create_dataset("task_indices", data=np.array(task_indices, dtype="i8"))
        f.create_dataset("results", data=results, compression="gzip")

        # Store successful indices for quick filtering
        successful_mask = ~np.all(np.isnan(results), axis=1)
        successful_indices = np.array(task_indices)[successful_mask]
        successful_results = results[successful_mask]

        f.create_dataset("successful_indices", data=successful_indices, dtype="i8")
        f.create_dataset("successful_results", data=successful_results, compression="gzip")

    print(f"Chunk results saved to: {chunk_file}")

    print(f"\nChunk {chunk_id} completed:")
    print(f"  Total variants: {len(task_indices)}")
    print(f"  Successfully computed: {successful_computations}")
    print(f"  Failed: {len(task_indices) - successful_computations}")


if __name__ == "__main__":
    main()
