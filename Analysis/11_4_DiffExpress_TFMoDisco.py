#!/usr/bin/env python3
"""
Apply TF-MoDISco to differential expression gradients.

This script:
1. Identifies cell types from gradient files
2. For each cell type, processes gradient files in parallel using joblib:
   - Loads gradient scores from differential expression analysis
   - Loads corresponding DNA sequences from reference genome using FastaInterval
   - Re-weights gradients to down-weight promoters and up-weight enhancers
3. Saves gradients and sequences as NPZ files ({celltype}.gradient.npz, {celltype}.onehot.npz)
4. Runs TF-MoDISco via command-line interface to identify transcription factor motifs

Re-weighting approach:
- Compute standard deviation at each position across the four nucleotides
- Apply Gaussian filter (s.d. = 1,280; truncate = 2) to the vector of standard deviations
- Divide the gradient scores by this smoothed vector
- This down-weights long contiguous stretches (promoters) and up-weights sparse regions (enhancers)

Command-line usage:
    # Run MoDISco directly with parallel processing (using borzoi-like defaults):
    python 11_4_DiffExpress_TFMoDisco.py \\
        --exp_name <experiment> \\
        --chk <checkpoint> \\
        --log_base ./logs \\
        --res_base ./Res \\
        --baseline random \\
        --n_jobs -1

    # Generate a bash script to run MoDISco later:
    python 11_4_DiffExpress_TFMoDisco.py \\
        --exp_name <experiment> \\
        --chk <checkpoint> \\
        --log_base ./logs \\
        --res_base ./Res \\
        --baseline random \\
        --n_jobs -1 \\
        --generate_script

    # Custom MoDISco parameters (defaults shown):
    python 11_4_DiffExpress_TFMoDisco.py \\
        --exp_name <experiment> \\
        --chk <checkpoint> \\
        --log_base ./logs \\
        --res_base ./Res \\
        --num_seqlets 40000 \\
        --trim_to_window_size 24 \\
        --initial_flank_to_add 8 \\
        --sliding_window_size 18 \\
        --flank_size 8 \\
        --center_bp 524288 \\
        --generate_script

Note:
    - Window size (-w) is automatically set to context_length from config (e.g., 524288)
    - Array shapes are (batch_size, 4, seq_len) as required by modisco v2.x
    - Default parameters are based on borzoi's TFMoDISco usage
    - Center bp (-c/--center_bp): Extract only center bp from sequences (default: None, use full sequence)
    - Reference: https://github.com/calico/borzoi

MoDISco v2.x parameter mapping (modisco defaults -> borzoi values):
    - Max seqlets (-n): default=None -> borzoi=40000
    - Window (-w): default=400 -> auto-set to context_length (524288)
    - Trim size (-t): default=30 -> borzoi=24
    - Initial flank (-g): default=10 -> borzoi=8
    - Seqlet core size (-z): default=20 -> borzoi=18
    - Seqlet flank size (-f): default=5 -> borzoi=8
"""

import logging
import os
import sys
from pathlib import Path
import warnings

import click
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from utils.logging import BaseLogger
from utils.config import load_config
from data.tokenizer import FastaInterval


def parse_genomic_coords_from_filename(filename):
    """
    Parse genomic coordinates from gradient filename.

    Format: {chr}_{start}_{end}_{gene}_{celltype}_{strand}_{background}_{baseline}.pt
    Example: chr10_100047062_100571350_PKD2L1_L45IT_plus_other-22_random.pt

    Returns:
        tuple: (chr_name, start, end, celltype) or (None, None, None, None) if parsing fails
    """
    # Remove .pt extension
    stem = filename.replace('.pt', '')

    # Split by underscore
    parts = stem.split('_')

    # Check we have enough parts
    if len(parts) < 8:
        return None, None, None, None

    chr_name = parts[0]
    start = int(parts[1])
    end = int(parts[2])
    celltype = parts[4]

    return chr_name, start, end, celltype


def parse_celltype_from_filename(filename):
    """
    Parse cell type from gradient filename.

    Format: {chr}_{start}_{end}_{gene}_{celltype}_{strand}_{background}_{baseline}.pt
    Example: chr10_100047062_100571350_PKD2L1_L45IT_plus_other-22_random.pt

    The structure is:
    - Index 0: chromosome (e.g., "chr10")
    - Index 1: start position (e.g., "100047062")
    - Index 2: end position (e.g., "100571350")
    - Index 3: gene name (e.g., "PKD2L1")
    - Index 4: cell type (e.g., "L45IT") ← THIS IS WHAT WE WANT
    - Index 5: strand (e.g., "plus" or "minus")
    - Index 6: background (e.g., "other-22")
    - Index 7: baseline (e.g., "random.pt")

    Returns celltype (e.g., "L45IT")
    """
    # Remove .pt extension
    stem = filename.replace('.pt', '')

    # Split by underscore
    parts = stem.split('_')

    # Check we have enough parts
    # Minimum: chr, start, end, gene, celltype, strand, background, baseline = 8 parts
    if len(parts) < 8:
        return None

    # Cell type is at index 4
    celltype = parts[4]

    return celltype


def process_single_gradient_file(grad_file, fasta_file, context_length, gaussian_sd, truncate, center_bp=None):
    """
    Process a single gradient file: load gradient, load sequence, and reweight.

    Args:
        grad_file: Path to gradient file
        fasta_file: Path to reference genome FASTA file
        context_length: Sequence context length
        gaussian_sd: Gaussian filter standard deviation
        truncate: Gaussian filter truncate parameter
        center_bp: If specified, extract only center bp (default: None, use full sequence)

    Returns:
        tuple: (reweighted_gradient, sequence, metadata) or None if failed
    """
    try:
        # Parse genomic coordinates from filename
        chr_name, start, end, celltype = parse_genomic_coords_from_filename(grad_file.name)

        if chr_name is None:
            return None

        # Load gradient tensor [1, N, 4]
        grad = torch.load(grad_file, map_location='cpu', weights_only=True)

        # Remove batch dimension: [1, N, 4] -> [N, 4]
        if grad.dim() == 3:
            grad = grad.squeeze(0)

        grad_numpy = grad.numpy()

        # Load sequence using tokenizer
        from data.tokenizer import FastaInterval
        dna_tokenizer = FastaInterval(
            fasta_file=fasta_file,
            context_length=context_length
        )

        token_dict = dna_tokenizer(
            chr_name=chr_name,
            start=start,
            end=end,
            return_augs=False,
            return_rela_idx=False
        )

        seq_onehot = token_dict["one_hot"].numpy()  # [N, 4]

        # Extract center region if center_bp is specified
        if center_bp is not None:
            seq_len = grad_numpy.shape[0]
            pos_start = seq_len // 2 - center_bp // 2
            pos_end = pos_start + center_bp

            # Extract center region
            grad_numpy = grad_numpy[pos_start:pos_end]
            seq_onehot = seq_onehot[pos_start:pos_end]

        # Reweight gradient
        # Step 1: Compute standard deviation across nucleotides at each position
        std_vec = np.std(grad_numpy, axis=1)

        # Step 2: Apply Gaussian filter to smooth the standard deviation vector
        smoothed_std = gaussian_filter1d(
            std_vec,
            sigma=gaussian_sd,
            truncate=truncate,
            mode='nearest'
        )

        # Step 3: Avoid division by zero
        epsilon = 1e-7
        smoothed_std = np.maximum(smoothed_std, epsilon)

        # Step 4: Divide gradient scores by smoothed standard deviations
        reweighted_grad = grad_numpy / smoothed_std[:, np.newaxis]

        # Create metadata
        parts = grad_file.stem.rsplit('_', 1)[0]  # Remove baseline suffix
        metadata = {
            'filename': grad_file.name,
            'path': str(grad_file),
            'identifier': parts,
            'celltype': celltype
        }

        return (reweighted_grad, seq_onehot, metadata)

    except Exception as e:
        print(f"Failed to process {grad_file}: {e}")
        return None


def load_gradients_from_directory(gradient_dir, baseline_type="random", logger=None):
    """
    Load all gradient files from the specified directory, grouped by cell type.

    Args:
        gradient_dir: Directory containing .pt gradient files
        baseline_type: Type of baseline to load (default: "random")
        logger: Logger instance

    Returns:
        gradients_by_celltype: Dict mapping celltype to list of gradient arrays
        metadata_by_celltype: Dict mapping celltype to list of metadata dicts
    """
    if logger:
        logger.info(f"Loading gradients from {gradient_dir}")

    gradient_files = sorted(Path(gradient_dir).glob(f"*_{baseline_type}.pt"))

    if not gradient_files:
        if logger:
            logger.warning(f"No gradient files found matching pattern *_{baseline_type}.pt")
        return {}, {}

    if logger:
        logger.info(f"Found {len(gradient_files)} gradient files")

    # Group by cell type
    gradients_by_celltype = {}
    metadata_by_celltype = {}

    for grad_file in tqdm(gradient_files, desc="Loading gradients"):
        try:
            # Parse cell type from filename
            celltype = parse_celltype_from_filename(grad_file.name)

            if celltype is None:
                if logger:
                    logger.warning(f"Could not parse celltype from {grad_file.name}")
                continue

            # Load gradient tensor [1, N, 4]
            grad = torch.load(grad_file, map_location='cpu')

            # Remove batch dimension: [1, N, 4] -> [N, 4]
            if grad.dim() == 3:
                grad = grad.squeeze(0)

            # Initialize lists for this celltype if needed
            if celltype not in gradients_by_celltype:
                gradients_by_celltype[celltype] = []
                metadata_by_celltype[celltype] = []

            gradients_by_celltype[celltype].append(grad.numpy())

            # Parse metadata from filename
            parts = grad_file.stem.rsplit('_', 1)[0]  # Remove baseline suffix

            metadata_by_celltype[celltype].append({
                'filename': grad_file.name,
                'path': str(grad_file),
                'identifier': parts,
                'celltype': celltype
            })

        except Exception as e:
            if logger:
                logger.warning(f"Failed to load {grad_file}: {e}")
            continue

    if logger:
        logger.info(f"Successfully loaded gradients for {len(gradients_by_celltype)} cell types")
        for celltype, grads in gradients_by_celltype.items():
            logger.info(f"  {celltype}: {len(grads)} gradients")

    return gradients_by_celltype, metadata_by_celltype


def process_gradients_parallel(gradient_files, fasta_file, context_length, gaussian_sd, truncate, center_bp=None, n_jobs=-1, logger=None):
    """
    Process all gradient files in parallel: load gradients, sequences, and reweight.

    Args:
        gradient_files: List of paths to gradient files
        fasta_file: Path to reference genome FASTA file
        context_length: Sequence context length
        gaussian_sd: Gaussian filter standard deviation
        truncate: Gaussian filter truncate parameter
        center_bp: If specified, extract only center bp (default: None, use full sequence)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        logger: Logger instance

    Returns:
        tuple: (reweighted_gradients, sequences, metadata_list)
    """
    if logger:
        logger.info(f"Processing {len(gradient_files)} files in parallel with {n_jobs if n_jobs > 0 else 'all'} CPUs")

    # Process all files in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_gradient_file)(
            grad_file, fasta_file, context_length, gaussian_sd, truncate, center_bp
        ) for grad_file in gradient_files
    )

    # Filter out failed results and separate into components
    reweighted_gradients = []
    sequences = []
    metadata_list = []

    for result in results:
        if result is not None:
            reweighted_grad, seq_onehot, metadata = result
            reweighted_gradients.append(reweighted_grad)
            sequences.append(seq_onehot)
            metadata_list.append(metadata)

    if logger:
        logger.info(f"Successfully processed {len(reweighted_gradients)}/{len(gradient_files)} files")

    return reweighted_gradients, sequences, metadata_list


def load_sequences_for_gradients(gradient_files, fasta_file, context_length, logger=None):
    """
    Load one-hot encoded sequences for each gradient file.

    Args:
        gradient_files: List of paths to gradient files
        fasta_file: Path to reference genome FASTA file
        context_length: Sequence context length
        logger: Logger instance

    Returns:
        sequences: List of one-hot encoded sequences [N, 4]
    """
    if logger:
        logger.info(f"Loading sequences from reference genome: {fasta_file}")

    # Initialize tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=fasta_file,
        context_length=context_length
    )

    sequences = []

    for grad_file in tqdm(gradient_files, desc="Loading sequences"):
        # Parse genomic coordinates from filename
        chr_name, start, end, _ = parse_genomic_coords_from_filename(grad_file.name)

        if chr_name is None:
            if logger:
                logger.warning(f"Could not parse coordinates from {grad_file.name}")
            continue

        # Load sequence using tokenizer
        token_dict = dna_tokenizer(
            chr_name=chr_name,
            start=start,
            end=end,
            return_augs=False,
            return_rela_idx=False
        )

        # Get one-hot encoded sequence
        seq_onehot = token_dict["one_hot"].numpy()  # [N, 4]
        sequences.append(seq_onehot)

    if logger:
        logger.info(f"Loaded {len(sequences)} sequences")

    return sequences


def reweight_gradients(gradients, gaussian_sd=1280, truncate=2, logger=None):
    """
    Re-weight gradients to down-weight promoters and up-weight enhancers.

    This re-weighting scheme:
    1. Computes standard deviation at each position across the four nucleotides
    2. Applies a Gaussian filter to smooth the standard deviations
    3. Divides gradient scores by the smoothed standard deviations

    This down-weights regulatory regions with long contiguous stretches of large magnitude
    (often promoter regions) and up-weights sparser regulatory regions (transcriptional enhancers).

    Args:
        gradients: List of gradient arrays, each with shape [N, 4]
        gaussian_sd: Standard deviation for Gaussian filter (default: 1280)
        truncate: Truncate parameter for Gaussian filter (default: 2)
        logger: Logger instance

    Returns:
        reweighted_gradients: List of re-weighted gradient arrays
    """
    if logger:
        logger.info(f"Re-weighting {len(gradients)} gradients")
        logger.info(f"Gaussian filter parameters: s.d.={gaussian_sd}, truncate={truncate}")

    reweighted_gradients = []

    for grad in tqdm(gradients, desc="Re-weighting gradients"):
        # grad shape: [N, 4]

        # Step 1: Compute standard deviation across nucleotides at each position
        # std_vec shape: [N]
        std_vec = np.std(grad, axis=1)

        # Step 2: Apply Gaussian filter to smooth the standard deviation vector
        # sigma is the standard deviation in units of array indices
        smoothed_std = gaussian_filter1d(
            std_vec,
            sigma=gaussian_sd,
            truncate=truncate,
            mode='nearest'
        )

        # Step 3: Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        smoothed_std = np.maximum(smoothed_std, epsilon)

        # Step 4: Divide gradient scores by smoothed standard deviations
        # Expand smoothed_std to [N, 1] for broadcasting
        reweighted_grad = grad / smoothed_std[:, np.newaxis]

        reweighted_gradients.append(reweighted_grad)

    if logger:
        logger.info("Re-weighting completed")

    return reweighted_gradients


def prepare_modisco_npz_files(gradients, sequences, output_gradient_npz, output_onehot_npz, logger=None):
    """
    Prepare input NPZ files for TF-MoDISco command-line interface.

    Args:
        gradients: List of gradient arrays [N, 4]
        sequences: List of one-hot sequence arrays [N, 4]
        output_gradient_npz: Path to output gradient NPZ file
        output_onehot_npz: Path to output one-hot NPZ file
        logger: Logger instance
    """
    if logger:
        logger.info(f"Preparing MoDISco NPZ files")

    # Stack all gradients and transpose to (batch_size, 4, seq_len)
    all_gradients = np.array(gradients)  # [num_sequences, N, 4]
    all_gradients = np.transpose(all_gradients, (0, 2, 1))  # -> [num_sequences, 4, N]

    if logger:
        logger.info(f"Stacked gradients shape (after transpose): {all_gradients.shape}")

    # Stack all sequences and transpose to (batch_size, 4, seq_len)
    all_sequences = np.array(sequences)  # [num_sequences, N, 4]
    all_sequences = np.transpose(all_sequences, (0, 2, 1))  # -> [num_sequences, 4, N]

    if logger:
        logger.info(f"Stacked sequences shape (after transpose): {all_sequences.shape}")

    # Verify shapes match
    if all_gradients.shape != all_sequences.shape:
        raise ValueError(f"Shape mismatch: gradients {all_gradients.shape} vs sequences {all_sequences.shape}")

    # Save gradients
    np.savez_compressed(output_gradient_npz, arr_0=all_gradients)
    if logger:
        logger.info(f"Saved gradients to {output_gradient_npz}")

    # Save sequences
    np.savez_compressed(output_onehot_npz, arr_0=all_sequences)
    if logger:
        logger.info(f"Saved one-hot sequences to {output_onehot_npz}")


def run_tfmodisco_cli(gradient_npz, onehot_npz, output_h5, num_seqlets=40000, window_size=None,
                      trim_to_window_size=24, initial_flank_to_add=8,
                      sliding_window_size=18, flank_size=8, logger=None):
    """
    Run TF-MoDISco using the command-line interface.

    Args:
        gradient_npz: Path to gradient NPZ file
        onehot_npz: Path to one-hot NPZ file
        output_h5: Path to output HDF5 file
        num_seqlets: Max seqlets per metacluster (default: 40000, like borzoi)
        window_size: Window size for MoDISco (default: None, uses modisco default of 400)
        trim_to_window_size: Trim to window size (default: 24, like borzoi)
        initial_flank_to_add: Initial flank to add (default: 8, like borzoi)
        sliding_window_size: Sliding window size (default: 18, like borzoi)
        flank_size: Flank size (default: 8, like borzoi)
        logger: Logger instance

    Returns:
        returncode: 0 if successful, non-zero otherwise
    """
    if logger:
        logger.info("Running TF-MoDISco via command-line interface")
        logger.info(f"  Gradient file: {gradient_npz}")
        logger.info(f"  One-hot file: {onehot_npz}")
        logger.info(f"  Output file: {output_h5}")
        logger.info(f"  Max seqlets per metacluster: {num_seqlets}")
        logger.info(f"  Window size: {window_size if window_size else 'default (400)'}")
        logger.info(f"  Trim to window size: {trim_to_window_size}")
        logger.info(f"  Initial flank to add: {initial_flank_to_add}")
        logger.info(f"  Sliding window size: {sliding_window_size}")
        logger.info(f"  Flank size: {flank_size}")

    # Build command
    cmd = [
        "modisco", "motifs",
        "-s", onehot_npz,
        "-a", gradient_npz,
        "-n", str(num_seqlets),
        "-o", output_h5
    ]

    # Add window size if specified
    if window_size is not None:
        cmd.extend(["-w", str(window_size)])

    # Add borzoi-like parameters using correct modisco v2.x flags
    cmd.extend(["-t", str(trim_to_window_size)])  # --trim_size
    cmd.extend(["-g", str(initial_flank_to_add)])  # --initial_flank_to_add
    cmd.extend(["-z", str(sliding_window_size)])  # --size (seqlet core size)
    cmd.extend(["-f", str(flank_size)])  # --seqlet_flank_size

    if logger:
        logger.info(f"Running command: {' '.join(cmd)}")

    # Run command
    import subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    # Log output
    if result.stdout:
        if logger:
            logger.info(f"MoDISco stdout:\n{result.stdout}")
        else:
            print(result.stdout)

    if result.stderr:
        if logger:
            logger.info(f"MoDISco stderr:\n{result.stderr}")
        else:
            print(result.stderr)

    if result.returncode == 0:
        if logger:
            logger.info("TF-MoDISco completed successfully")
    else:
        if logger:
            logger.error(f"TF-MoDISco failed with return code {result.returncode}")

    return result.returncode


@click.command()
@click.option("--exp_name", "-e", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint number")
@click.option("--baseline", "-b", type=str, default="random", help="Baseline type (default: random)")
@click.option("--log_base", required=True, type=str, default="./logs", help="Logs base directory")
@click.option("--res_base", required=True, type=str, default="./Res", help="Results base directory")
@click.option("--gaussian_sd", type=float, default=1280, help="Gaussian filter std dev (default: 1280)")
@click.option("--truncate", type=float, default=2, help="Gaussian filter truncate parameter (default: 2)")
@click.option("--num_seqlets", type=int, default=40000, help="Max seqlets per metacluster (default: 40000, like borzoi)")
@click.option("--trim_to_window_size", type=int, default=24, help="Trim to window size (default: 24, like borzoi)")
@click.option("--initial_flank_to_add", type=int, default=8, help="Initial flank to add (default: 8, like borzoi)")
@click.option("--sliding_window_size", type=int, default=18, help="Sliding window size (default: 18, like borzoi)")
@click.option("--flank_size", type=int, default=8, help="Flank size (default: 8, like borzoi)")
@click.option("--center_bp", "-c", type=int, default=None, help="Extract only center bp (default: None, use full sequence)")
@click.option("--n_jobs", type=int, default=-1, help="Number of parallel jobs for processing (-1 = all CPUs)")
@click.option("--force_restart", is_flag=True, help="Force restart even if intermediate files exist")
@click.option("--generate_script", is_flag=True, help="Generate bash script instead of running MoDISco")
def main(exp_name, chk, baseline, log_base, res_base, gaussian_sd, truncate, num_seqlets,
         trim_to_window_size, initial_flank_to_add, sliding_window_size, flank_size, center_bp, n_jobs, force_restart, generate_script):
    """
    Apply TF-MoDISco to differential expression gradients, processing each cell type separately.
    """
    # Setup paths
    LOG_BASE = os.path.abspath(log_base)
    RES_BASE = os.path.abspath(res_base)
    gradient_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp_diff_gradient_input"
    output_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/modisco"
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = BaseLogger(name="TFMoDISco", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("TF-MoDISco Analysis for Differential Expression")
    logger.info("=" * 80)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Checkpoint: {chk}")
    logger.info(f"Baseline: {baseline}")
    logger.info(f"Gradient directory: {gradient_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Gaussian filter: s.d.={gaussian_sd}, truncate={truncate}")
    logger.info(f"Max seqlets per metacluster: {num_seqlets}")
    logger.info(f"Trim to window size: {trim_to_window_size}")
    logger.info(f"Initial flank to add: {initial_flank_to_add}")
    logger.info(f"Sliding window size: {sliding_window_size}")
    logger.info(f"Flank size: {flank_size}")
    logger.info(f"Center bp: {center_bp if center_bp else 'None (use full sequence)'}")
    logger.info(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all CPUs'}")

    # Load config to get reference genome path and context length
    config_path = f"{LOG_BASE}/{exp_name}/overall_setting.yaml"
    logger.info(f"Loading config from: {config_path}")
    myconfig = load_config(config_name=config_path, skip_validation=True)

    fasta_file = os.path.abspath(myconfig.data.refer_genom)
    context_length = myconfig.data.context_length

    logger.info(f"Reference genome: {fasta_file}")
    logger.info(f"Context length: {context_length}")

    if generate_script:
        logger.info("Script generation mode: Commands will be written to bash script")

    # List to collect MoDISco commands for script generation
    modisco_commands = []

    # Check if gradient directory exists
    if not os.path.exists(gradient_dir):
        logger.error(f"Gradient directory not found: {gradient_dir}")
        logger.error("Please run 11_3_DiffExpress_run.slurm first to generate gradients")
        sys.exit(1)

    # Step 1: Identify all cell types from gradient files
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Identifying cell types from gradient files")
    logger.info("=" * 80)

    gradient_files = sorted(Path(gradient_dir).glob(f"*_{baseline}.pt"))

    if not gradient_files:
        logger.error(f"No gradient files found matching pattern *_{baseline}.pt")
        sys.exit(1)

    logger.info(f"Found {len(gradient_files)} gradient files")

    # Extract unique cell types
    celltypes_set = set()
    for grad_file in gradient_files:
        celltype = parse_celltype_from_filename(grad_file.name)
        if celltype:
            celltypes_set.add(celltype)

    celltypes = sorted(celltypes_set)
    logger.info(f"\nIdentified {len(celltypes)} cell types: {', '.join(celltypes)}")

    # Track successful and failed cell types
    successful_celltypes = []
    failed_celltypes = []

    for celltype in celltypes:
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing cell type: {celltype}")
        logger.info("=" * 80)

        try:
            # Get gradient files for this celltype
            gradient_files = sorted(Path(gradient_dir).glob(f"*_{celltype}_*_{baseline}.pt"))

            logger.info(f"Number of gradient files for {celltype}: {len(gradient_files)}")

            gradient_npz = f"{output_dir}/{celltype}.gradient.npz"
            onehot_npz = f"{output_dir}/{celltype}.onehot.npz"

            # Step 2: Process all files in parallel (load sequences, load gradients, reweight)
            if not os.path.exists(gradient_npz) or not os.path.exists(onehot_npz) or force_restart:
                logger.info("\n" + "-" * 80)
                logger.info(f"Step 2: Processing files in parallel for {celltype}")
                logger.info("-" * 80)

                reweighted_gradients, sequences, metadata_list = process_gradients_parallel(
                    gradient_files,
                    fasta_file,
                    context_length,
                    gaussian_sd=gaussian_sd,
                    truncate=truncate,
                    center_bp=center_bp,
                    n_jobs=n_jobs,
                    logger=logger
                )

                if len(reweighted_gradients) == 0:
                    logger.error(f"No gradients successfully processed for {celltype}")
                    raise ValueError("No gradients processed successfully")

                # Save as NPZ files
                logger.info("\n" + "-" * 80)
                logger.info(f"Step 3: Saving NPZ files for {celltype}")
                logger.info("-" * 80)

                prepare_modisco_npz_files(
                    reweighted_gradients,
                    sequences,
                    gradient_npz,
                    onehot_npz,
                    logger=logger
                )
            else:
                logger.info(f"NPZ files already exist:")
                logger.info(f"  {gradient_npz}")
                logger.info(f"  {onehot_npz}")
                logger.info("Skipping preparation step (use --force_restart to re-run)")

            # Step 4: Run TF-MoDISco for this cell type (or collect command)
            logger.info("\n" + "-" * 80)
            logger.info(f"Step 4: Running TF-MoDISco for {celltype}")
            logger.info("-" * 80)

            modisco_output_file = f"{output_dir}/{celltype}_modisco_results.h5"

            if not os.path.exists(modisco_output_file) or force_restart:
                if generate_script:
                    # Generate command for bash script (using correct modisco v2.x flags)
                    cmd = (f"modisco motifs -s {onehot_npz} -a {gradient_npz} -n {num_seqlets} "
                           f"-w {context_length} -t {trim_to_window_size} "
                           f"-g {initial_flank_to_add} -z {sliding_window_size} "
                           f"-f {flank_size} -o {modisco_output_file}")
                    modisco_commands.append(cmd)
                    logger.info(f"Added command to script: {cmd}")
                else:
                    # Run MoDISco directly
                    returncode = run_tfmodisco_cli(
                        gradient_npz,
                        onehot_npz,
                        modisco_output_file,
                        num_seqlets=num_seqlets,
                        window_size=context_length,
                        trim_to_window_size=trim_to_window_size,
                        initial_flank_to_add=initial_flank_to_add,
                        sliding_window_size=sliding_window_size,
                        flank_size=flank_size,
                        logger=logger
                    )

                    if returncode != 0:
                        raise RuntimeError(f"MoDISco failed with return code {returncode}")
            else:
                logger.info(f"MoDISco results already exist: {modisco_output_file}")
                logger.info("Skipping MoDISco step (use --force_restart to re-run)")

            logger.info(f"\nCompleted processing for cell type: {celltype}")
            successful_celltypes.append(celltype)

        except ValueError as e:
            logger.error(f"\nValueError encountered for cell type {celltype}: {e}")
            logger.error(f"This typically occurs when all seqlets are filtered out during MoDISco processing.")
            logger.error(f"Skipping {celltype} and continuing with next cell type...")
            failed_celltypes.append((celltype, "ValueError: " + str(e)))
            continue

        except Exception as e:
            logger.error(f"\nUnexpected error encountered for cell type {celltype}: {e}")
            logger.error(f"Skipping {celltype} and continuing with next cell type...")
            failed_celltypes.append((celltype, str(e)))
            continue

    logger.info("\n" + "=" * 80)
    logger.info("Analysis completed!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nSuccessfully processed {len(successful_celltypes)}/{len(celltypes)} cell types:")
    if successful_celltypes:
        logger.info(f"  SUCCESS: {', '.join(successful_celltypes)}")

    if failed_celltypes:
        logger.info(f"\nFailed to process {len(failed_celltypes)} cell types:")
        for celltype, error in failed_celltypes:
            logger.info(f"  FAILED: {celltype}")
            logger.info(f"    Error: {error[:100]}...")  # Truncate long errors
        logger.warning(f"\nConsider adjusting parameters for failed cell types:")
        logger.warning(f"  - Try smaller --gaussian_sd (e.g., 32, 64, 128)")
        logger.warning(f"  - Current gaussian_sd: {gaussian_sd}")
    else:
        logger.info("\nAll cell types processed successfully!")

    # Write bash script if in generate_script mode
    if generate_script and modisco_commands:
        script_file = f"{output_dir}/run_modisco_{baseline}.sh"
        logger.info("\n" + "=" * 80)
        logger.info(f"Writing MoDISco commands to: {script_file}")
        logger.info("=" * 80)

        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated script to run TF-MoDISco v2.x\n")
            f.write(f"# Experiment: {exp_name}\n")
            f.write(f"# Checkpoint: {chk}\n")
            f.write(f"# Baseline: {baseline}\n")
            f.write("#\n")
            f.write("# MoDISco parameters (based on borzoi):\n")
            f.write(f"#   -n (max seqlets): {num_seqlets}\n")
            f.write(f"#   -w (window size): {context_length}\n")
            f.write(f"#   -t (trim size): {trim_to_window_size}\n")
            f.write(f"#   -g (initial flank): {initial_flank_to_add}\n")
            f.write(f"#   -z (seqlet core size): {sliding_window_size}\n")
            f.write(f"#   -f (seqlet flank size): {flank_size}\n")
            f.write("#\n")
            f.write("# Reference: https://github.com/calico/borzoi\n")
            f.write("\n")
            f.write("set -e  # Exit on error\n\n")

            for i, cmd in enumerate(modisco_commands, 1):
                f.write(f"# Command {i}/{len(modisco_commands)}\n")
                f.write(f"echo 'Running command {i}/{len(modisco_commands)}...'\n")
                f.write(cmd + "\n\n")

            f.write("echo 'All MoDISco runs completed!'\n")

        # Make script executable
        import stat
        os.chmod(script_file, os.stat(script_file).st_mode | stat.S_IEXEC)

        logger.info(f"Generated script with {len(modisco_commands)} commands")
        logger.info(f"Run with: bash {script_file}")
        logger.info("Or make it executable and run: " + script_file)
    elif generate_script and not modisco_commands:
        logger.info("\n" + "=" * 80)
        logger.info("No new MoDISco commands to generate (all results already exist)")
        logger.info("Use --force_restart to regenerate all commands")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
