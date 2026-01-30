#!/usr/bin/env python3
"""
Borzoi-style TF-MoDISco analysis for differential expression gradients.

This script replicates the functionality of borzoi_tfmodisco.py to run TF-MoDISco
on gradient scores stored in PyTorch (.pt) files within a directory.

Unlike the original script that takes a single h5 file, this script:
1. Takes a directory containing multiple .pt gradient files
2. Aggregates all gradients from the .pt files
3. Loads sequences from reference genome using FastaInterval
4. Runs TF-MoDISco analysis

The .pt files should contain:
- Gradient tensor (loaded with torch.load)
- Genomic coordinates in the filename (chr_start_end_gene_celltype_strand_background_baseline.pt)

Usage:
    # List available cell types:
    python 11_4_DiffExpress_TFMoDisco_borzoi.py <gradient_dir>

    # Run analysis for a specific cell type:
    python 11_4_DiffExpress_TFMoDisco_borzoi.py <gradient_dir> --celltype <celltype> [options]

Example:
    # List cell types
    python 11_4_DiffExpress_TFMoDisco_borzoi.py /path/to/gradient_files

    # Run analysis
    python 11_4_DiffExpress_TFMoDisco_borzoi.py /path/to/gradient_files \
        --celltype Astro \
        --fasta /path/to/genome.fa \
        --context_length 524288 \
        -c 524288 \
        -o tfm_out \
        --modisco_max_seqlets 40000 \
        --baseline random
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from optparse import OptionParser

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from joblib import Parallel, delayed

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))

from data.tokenizer import FastaInterval

# Try to import modisco
try:
    import modisco
    # Monkey-patch sklearn's TSNE to use init='random' by default to avoid PCA initialization issues with sparse matrices
    import sklearn.manifold
    import scipy.sparse
    import inspect

    # Store the original TSNE __init__ and fit_transform
    _original_tsne_init = sklearn.manifold.TSNE.__init__
    _original_tsne_fit_transform = sklearn.manifold.TSNE.fit_transform

    def _patched_tsne_init(self, n_components=2, perplexity=30.0, early_exaggeration=12.0,
                          learning_rate='auto', n_iter=1000, n_iter_without_progress=300,
                          min_grad_norm=1e-7, metric="euclidean", init="random",  # Changed default from "pca" to "random"
                          verbose=0, random_state=None, method='barnes_hut', angle=0.5,
                          n_jobs=None, **kwargs):
        """Patched TSNE __init__ that defaults to init='random' instead of 'pca'."""
        # Force init='random' to avoid PCA issues with sparse matrices
        if init == 'pca' or init == 'warn':
            init = 'random'
            if verbose:
                print(f"TSNE patch: Changed init from 'pca' to 'random'")

        # Get the signature of the original __init__ to handle version differences
        sig = inspect.signature(_original_tsne_init)
        params = {
            'n_components': n_components,
            'perplexity': perplexity,
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'n_iter_without_progress': n_iter_without_progress,
            'min_grad_norm': min_grad_norm,
            'metric': metric,
            'init': init,
            'verbose': verbose,
            'random_state': random_state,
            'method': method,
            'angle': angle,
            'n_jobs': n_jobs
        }

        # Filter out parameters not in the original signature
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        valid_params.update(kwargs)

        # Call original __init__ with valid parameters
        _original_tsne_init(self, **valid_params)

    def _patched_tsne_fit_transform(self, X, y=None):
        """Patched fit_transform that converts sparse matrices to dense and adjusts perplexity."""
        # Convert sparse matrix to dense if needed
        if scipy.sparse.issparse(X):
            if self.verbose:
                print(f"TSNE patch: Converting sparse matrix of shape {X.shape} to dense")
            X = X.toarray()

        # Dynamically adjust perplexity based on number of samples
        # perplexity must be less than n_samples
        n_samples = X.shape[0]
        adjusted_perplexity = min(self.perplexity, n_samples - 1)
        if adjusted_perplexity != self.perplexity:
            if self.verbose or adjusted_perplexity < 5:
                print(f"TSNE patch: Adjusted perplexity from {self.perplexity} to {adjusted_perplexity} (n_samples={n_samples})")
                if adjusted_perplexity < 5:
                    print(f"TSNE patch: Warning - perplexity < 5 may produce suboptimal results")
            self.perplexity = adjusted_perplexity

        # Call original fit_transform
        return _original_tsne_fit_transform(self, X, y)

    # Apply the patches to sklearn's TSNE
    sklearn.manifold.TSNE.__init__ = _patched_tsne_init
    sklearn.manifold.TSNE.fit_transform = _patched_tsne_fit_transform

    print("Applied TSNE patch for sparse matrix compatibility")

except ImportError:
    print("Warning: modisco not found. Please install with: pip install modisco")
    modisco = None


def parse_genomic_coords_from_filename(pt_file):
    """
    Parse genomic coordinates from gradient filename.

    Format: {chr}_{start}_{end}_{gene}_{celltype}_{strand}_{background}_{baseline}.pt
    Example: chr10_100047062_100571350_PKD2L1_L45IT_plus_other-22_random.pt

    Args:
        pt_file: Path to .pt file

    Returns:
        tuple: (chr_name, start, end) or (None, None, None) if parsing fails
    """
    filename = pt_file.name if isinstance(pt_file, Path) else Path(pt_file).name

    # Remove .pt extension
    stem = filename.replace('.pt', '')

    # Split by underscore
    parts = stem.split('_')

    # Check we have enough parts
    # Minimum: chr, start, end, gene, celltype, strand, background, baseline = 8 parts
    if len(parts) < 8:
        return None, None, None

    try:
        chr_name = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        return chr_name, start, end
    except (ValueError, IndexError):
        return None, None, None


def parse_celltype_from_filename(pt_file):
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
    - Index 7: baseline (e.g., "random")

    Args:
        pt_file: Path to .pt file

    Returns:
        celltype (e.g., "L45IT") or None if parsing fails
    """
    filename = pt_file.name if isinstance(pt_file, Path) else Path(pt_file).name

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


def identify_celltypes(gradient_dir, baseline="random"):
    """
    Identify all unique cell types from gradient files in directory.

    Args:
        gradient_dir: Directory containing .pt gradient files
        baseline: Baseline type to filter (default: "random")

    Returns:
        list: Sorted list of unique cell types
    """
    gradient_dir = Path(gradient_dir)
    pattern = f"*_{baseline}.pt"
    pt_files = sorted(gradient_dir.glob(pattern))

    if not pt_files:
        raise ValueError(f"No .pt files found in {gradient_dir} matching pattern {pattern}")

    print(f"Found {len(pt_files)} gradient files")

    # Extract unique cell types
    celltypes_set = set()
    for pt_file in pt_files:
        celltype = parse_celltype_from_filename(pt_file)
        if celltype:
            celltypes_set.add(celltype)

    celltypes = sorted(celltypes_set)
    print(f"Identified {len(celltypes)} cell types: {', '.join(celltypes)}")

    return celltypes


def load_pt_gradients_for_celltype(gradient_dir, celltype, baseline="random"):
    """
    Load gradient .pt files for a specific cell type from directory.

    Args:
        gradient_dir: Directory containing .pt gradient files
        celltype: Cell type to filter for
        baseline: Baseline type to load (default: "random")

    Returns:
        list: List of tuples (pt_file_path, gradient_array, chr, start, end)
    """
    gradient_dir = Path(gradient_dir)
    # Pattern to match: *_{celltype}_*_{baseline}.pt
    pattern = f"*_{celltype}_*_{baseline}.pt"
    pt_files = sorted(gradient_dir.glob(pattern))

    if not pt_files:
        print(f"Warning: No .pt files found for cell type {celltype}")
        return []

    print(f"Found {len(pt_files)} gradient files for {celltype}")

    gradients_data = []

    for pt_file in tqdm(pt_files, desc=f"Loading {celltype} gradients"):
        try:
            # Parse coordinates from filename
            chr_name, start, end = parse_genomic_coords_from_filename(pt_file)

            if chr_name is None:
                print(f"Warning: Could not parse coordinates from {pt_file.name}, skipping")
                continue

            # Load gradient tensor [1, N, 4] or [N, 4]
            grad = torch.load(pt_file, map_location='cpu', weights_only=True)

            # Convert to numpy
            if isinstance(grad, torch.Tensor):
                gradient = grad.numpy()
            else:
                gradient = np.array(grad)

            gradients_data.append((pt_file, gradient, chr_name, start, end))

        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue

    print(f"Successfully loaded {len(gradients_data)} gradients for {celltype}")
    return gradients_data


def load_sequences_from_fasta(gradients_data, fasta_file, context_length):
    """
    Load one-hot encoded sequences from reference genome using FastaInterval.

    Args:
        gradients_data: List of tuples (h5_file, gradient, chr, start, end)
        fasta_file: Path to reference genome FASTA file
        context_length: Context length for tokenizer

    Returns:
        list: List of one-hot encoded sequences
    """
    print(f"Loading sequences from reference genome: {fasta_file}")

    # Initialize tokenizer
    dna_tokenizer = FastaInterval(
        fasta_file=fasta_file,
        context_length=context_length
    )

    sequences = []

    for h5_file, gradient, chr_name, start, end in tqdm(gradients_data, desc="Loading sequences"):
        try:
            # Load sequence using tokenizer
            token_dict = dna_tokenizer(
                chr_name=chr_name,
                start=start,
                end=end,
                return_augs=False,
                return_rela_idx=False
            )

            # Get one-hot encoded sequence [N, 4]
            seq_onehot = token_dict["one_hot"].numpy()
            sequences.append(seq_onehot)

        except Exception as e:
            print(f"Error loading sequence for {h5_file.name}: {e}")
            sequences.append(None)
            continue

    print(f"Loaded {sum(s is not None for s in sequences)} sequences")
    return sequences


def run_tfmodisco_for_celltype(gradient_dir,
                                celltype,
                                fasta_file,
                                context_length,
                                baseline="random",
                                center_bp=None,
                                meme_db='meme-5.4.1/motif_databases/CIS-BP_2.00/Homo_sapiens.meme',
                                gc_content=0.41,
                                force_fwd=0,
                                modisco_window_size=24,
                                modisco_flank=8,
                                modisco_sliding_window_size=18,
                                modisco_sliding_window_flank=8,
                                modisco_max_seqlets=20000,
                                modisco_n_cores=1,
                                ic_t=0.1,
                                norm_type='max',
                                out_dir='tfm_out',
                                clip_perc=25,
                                kmer_len=None,
                                num_gaps=None,
                                num_mismatches=None,
                                use_cache=True):
    """
    Run TF-MoDISco analysis for a specific cell type.

    This function replicates the main() functionality from borzoi_tfmodisco.py
    but works with .pt gradient files for a single cell type.

    Args:
        gradient_dir: Directory containing .pt gradient files
        celltype: Cell type to analyze
        fasta_file: Path to reference genome FASTA file
        context_length: Context length for sequence loading
        baseline: Baseline type to load (default: "random")
        center_bp: Extract only center bp (default: None, use full sequence)
        meme_db: Path to MEME database for TOMTOM
        gc_content: Genome GC content for background model (default: 0.41)
        force_fwd: Do not use rev-comp in modisco (default: 0)
        modisco_window_size: Modisco window size (default: 24)
        modisco_flank: Modisco flanks to add (default: 8)
        modisco_sliding_window_size: Modisco sliding window size (default: 18)
        modisco_sliding_window_flank: Modisco sliding window flanks (default: 8)
        modisco_max_seqlets: Max seqlets per metacluster (default: 20000)
        modisco_n_cores: Number of cores for modisco internal parallelism (default: 1)
        ic_t: Information content threshold (default: 0.1)
        norm_type: Normalization type: 'max' or 'gaussian' (default: 'max')
        out_dir: Output directory (default: 'tfm_out')
        clip_perc: Percentile of max deviations to clip by (default: 25)
        kmer_len: K-mer length for pattern matching (optional)
        num_gaps: Number of gaps for pattern matching (optional)
        num_mismatches: Number of mismatches for pattern matching (optional)
        use_cache: Whether to use cached intermediate results (default: True)

    Returns:
        tfm_results: TF-MoDISco results object
        tfm_pwms: Dictionary of pattern PWMs
    """

    # Check if modisco is available
    if modisco is None:
        raise ImportError("modisco is required but not installed. Install with: pip install modisco")

    # Setup output dir
    os.makedirs(out_dir, exist_ok=True)

    # Define cache file paths
    cache_dir = os.path.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{celltype}_preprocessed_data.npz')

    # Initialize cache loaded flag
    cache_loaded = False

    # Try to load from cache first
    if use_cache and os.path.exists(cache_file):
        print(f'\n{"=" * 80}')
        print(f'Loading cached data for cell type: {celltype}')
        print(f'Cache file: {cache_file}')
        print(f'{"=" * 80}')
        try:
            t0 = time.time()
            cached_data = np.load(cache_file, allow_pickle=True)
            hyp_scores = cached_data['hyp_scores']
            seqs_1hot = cached_data['seqs_1hot']
            print(f'Loaded normalized data from cache: {hyp_scores.shape} in {time.time() - t0:.1f}s')
            print('Skipping gradient loading, sequence loading, and normalization...')
            # Skip to TF-MoDISco section
            cache_loaded = True
        except Exception as e:
            print(f'Error loading cache: {e}')
            print('Falling back to loading from scratch...')
            use_cache = False
            cache_loaded = False
    else:
        cache_loaded = False

    if not cache_loaded:
        # Load gradients from .pt files for this cell type
        t0 = time.time()
        print(f'\n{"=" * 80}')
        print(f'Processing cell type: {celltype}')
        print(f'{"=" * 80}')
        gradients_data = load_pt_gradients_for_celltype(gradient_dir, celltype, baseline=baseline)

        if not gradients_data:
            raise ValueError(f"No valid gradient data loaded for {celltype}")

        print(f'Loaded {len(gradients_data)} gradients in {time.time() - t0:.1f}s')

        # Load sequences from reference genome
        t0 = time.time()
        print('Loading sequences from reference genome...', flush=True)
        sequences = load_sequences_from_fasta(gradients_data, fasta_file, context_length)
        print(f'Loaded {len(sequences)} sequences in {time.time() - t0:.1f}s')

        # Filter out failed loads
        valid_data = []
        for (h5_file, gradient, chr_name, start, end), seq in zip(gradients_data, sequences):
            if seq is not None:
                valid_data.append((gradient, seq))

        if not valid_data:
            raise ValueError("No valid gradient/sequence pairs after loading")

        print(f"Valid gradient/sequence pairs: {len(valid_data)}")

        # Extract gradients and sequences
        all_gradients = [grad for grad, seq in valid_data]
        all_sequences = [seq for grad, seq in valid_data]

        print(f"Processing {len(all_gradients)} gradient/sequence pairs...")

        # Process gradients - handle different shapes
        processed_gradients = []
        for grad in all_gradients:
            # Remove batch dimension if present
            if grad.ndim == 4:
                grad = grad.squeeze(0)  # [1, N, 4, T] -> [N, 4, T]
            elif grad.ndim == 3 and grad.shape[0] == 1:
                grad = grad.squeeze(0)  # [1, N, 4] -> [N, 4]

            # Transpose if needed to get [N, 4] or [N, 4, T]
            if grad.ndim == 3 and grad.shape[1] == 4:
                # [N, 4, T] is correct
                pass
            elif grad.ndim == 2 and grad.shape[1] == 4:
                # [N, 4] is correct
                pass
            elif grad.ndim == 3 and grad.shape[0] == 4:
                # [4, N, T] -> [N, 4, T]
                grad = grad.transpose(1, 0, 2)
            elif grad.ndim == 2 and grad.shape[0] == 4:
                # [4, N] -> [N, 4]
                grad = grad.T

            processed_gradients.append(grad)

        # Stack gradients and sequences
        hyp_scores = np.array(processed_gradients)
        seqs_1hot = np.array(all_sequences)

        # Extract center region if specified
        if center_bp is not None:
            seq_len = seqs_1hot.shape[1]
            pos_start = seq_len // 2 - center_bp // 2
            pos_end = pos_start + center_bp
            hyp_scores = hyp_scores[:, pos_start:pos_end, ...]
            seqs_1hot = seqs_1hot[:, pos_start:pos_end, :]

        num_seqs, seq_len, _ = seqs_1hot.shape
        print(f'Final data shape: {num_seqs} sequences, {seq_len} bp')

        # Average across targets if present
        if hyp_scores.ndim == 4:
            print('Averaging across targets...')
            hyp_scores = hyp_scores.mean(axis=-1, dtype='float32')

        # Ensure shape is [num_seqs, seq_len, 4]
        if hyp_scores.shape != seqs_1hot.shape:
            raise ValueError(f"Shape mismatch: gradients {hyp_scores.shape} vs sequences {seqs_1hot.shape}")

        # Normalize scores by sequence
        t0 = time.time()
        print('Normalizing scores...', flush=True, end='')
        if norm_type == 'max':
            scores_max = hyp_scores.std(axis=-1).max(axis=-1)
            max_clip = np.percentile(scores_max, clip_perc)
            scores_max = np.clip(scores_max, max_clip, np.inf)
            hyp_scores /= np.reshape(scores_max, (num_seqs, 1, 1))
        elif norm_type == 'gaussian':
            scores_std = hyp_scores.std(axis=-1)
            scores_std_wide = gaussian_filter1d(scores_std, sigma=1280, truncate=2)
            wide_clip = np.percentile(scores_std_wide, clip_perc)
            scores_std_wide = np.clip(scores_std_wide, wide_clip, np.inf)
            hyp_scores /= np.expand_dims(scores_std_wide, axis=-1)
        else:
            print('Unrecognized normalization %s' % norm_type)
        print('DONE in %ds.' % (time.time() - t0))

        # Save normalized data to cache
        print(f'Saving normalized data to cache: {cache_file}')
        try:
            np.savez_compressed(
                cache_file,
                hyp_scores=hyp_scores,
                seqs_1hot=seqs_1hot
            )
            print(f'Cache saved successfully')
        except Exception as e:
            print(f'Warning: Could not save cache: {e}')

    ################################################
    # TF-MoDISco

    # Compute contribution scores
    num_seqs = seqs_1hot.shape[0]
    contrib_scores = np.multiply(hyp_scores, seqs_1hot)

    # Make seqlets to patterns factory
    if kmer_len is not None and num_gaps is not None and num_mismatches is not None:
        tfm_seqlets = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            n_cores=modisco_n_cores,
            trim_to_window_size=modisco_window_size,
            initial_flank_to_add=modisco_flank,
            kmer_len=kmer_len,
            num_gaps=num_gaps,
            num_mismatches=num_mismatches,
            final_min_cluster_size=20)
    else:
        tfm_seqlets = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            n_cores=modisco_n_cores,
            trim_to_window_size=modisco_window_size,
            initial_flank_to_add=modisco_flank,
            final_min_cluster_size=20)

    # Make modisco workflow
    tfm_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        sliding_window_size=modisco_sliding_window_size,
        flank_size=modisco_sliding_window_flank,
        max_seqlets_per_metacluster=modisco_max_seqlets,
        seqlets_to_patterns_factory=tfm_seqlets)

    # Run modisco workflow
    print('Running TF-MoDISco...', flush=True)
    task_label = 'out0'
    tfm_results = tfm_workflow(
        task_names=[task_label],
        contrib_scores={task_label: contrib_scores},
        hypothetical_contribs={task_label: hyp_scores},
        revcomp=False if force_fwd == 1 else True,
        one_hot=seqs_1hot)

    # Save results
    tfm_h5_file = '%s/tfm.h5' % out_dir
    print(f'Saving TF-MoDISco results to {tfm_h5_file}...', flush=True)
    with h5py.File(tfm_h5_file, 'w') as tfm_h5:
        tfm_results.save_hdf5(tfm_h5)

    ################################################
    # Extract motif PWMs

    print('Extracting motif PWMs...', flush=True)
    at_pct = (1 - gc_content) / 2
    gc_pct = gc_content / 2
    background = np.array([at_pct, gc_pct, gc_pct, at_pct])

    tfm_pwms = {}

    with h5py.File(tfm_h5_file, 'r') as tfm_h5:
        metacluster_names = [mcr.decode("utf-8") if isinstance(mcr, bytes) else mcr
                            for mcr in list(tfm_h5["metaclustering_results"]["all_metacluster_names"][:])]
        for metacluster_name in metacluster_names:
            metacluster_grp = tfm_h5["metacluster_idx_to_submetacluster_results"][metacluster_name]
            all_patterns = metacluster_grp["seqlets_to_patterns_result"]["patterns"]["all_pattern_names"][:]
            all_pattern_names = [x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in list(all_patterns)]
            for pattern_name in all_pattern_names:
                pattern_id = (metacluster_name + '_' + pattern_name)
                pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
                fwd = np.array(pattern["sequence"]["fwd"])
                clip_pwm = ic_clip(fwd, ic_t, background)
                if clip_pwm is None:
                    print('pattern_id: %s is skipped because no bp pass threshold.' % pattern_id)
                else:
                    tfm_pwms[pattern_id] = clip_pwm
                    print('pattern_id: %s is converted to pwm.' % pattern_id)

    ################################################
    # TOMTOM

    print('Running TOMTOM...', flush=True)
    # Initialize MEME
    modisco_meme_file = out_dir + '/modisco_' + out_dir.replace("/", "_") + '.meme'
    modisco_meme_open = open(modisco_meme_file, 'w')

    # Header
    modisco_meme_open.write('MEME version 4\n\n')
    modisco_meme_open.write('ALPHABET= ACGT\n\n')
    modisco_meme_open.write('strands: + -\n\n')
    modisco_meme_open.write('Background letter frequencies\n')
    modisco_meme_open.write('A %f C %f G %f T %f\n\n' % tuple(background))

    # PWMs
    for key in tfm_pwms.keys():
        modisco_meme_open.write('MOTIF ' + key + '\n')
        modisco_meme_open.write('letter-probability matrix: alength= 4 w= ' + str(tfm_pwms[key].shape[0]) + '\n')
        np.savetxt(modisco_meme_open, tfm_pwms[key])
        modisco_meme_open.write('\n')

    modisco_meme_open.close()

    # Run tomtom
    if os.path.exists(meme_db):
        tomtom_cmd = 'tomtom -dist pearson -thresh 0.1 -oc %s %s %s' % \
            (out_dir, modisco_meme_file, meme_db)
        subprocess.call(tomtom_cmd, shell=True)
    else:
        print(f'MEME database not found at {meme_db}, skipping TOMTOM')

    print('TF-MoDISco analysis completed!')
    print(f'Results saved to: {out_dir}')

    return tfm_results, tfm_pwms


def _process_single_celltype(celltype, gradient_dir, fasta_file, context_length,
                             baseline, center_bp, meme_db, gc_content, force_fwd,
                             modisco_window_size, modisco_flank, modisco_sliding_window_size,
                             modisco_sliding_window_flank, modisco_max_seqlets, modisco_n_cores, ic_t,
                             norm_type, out_dir, clip_perc, kmer_len, num_gaps, num_mismatches,
                             use_cache):
    """
    Wrapper function to process a single cell type (for joblib parallelization).

    Returns:
        tuple: (celltype, success, result_or_error)
            - success=True: result_or_error is (tfm_results, tfm_pwms)
            - success=False: result_or_error is error message string
    """
    try:
        # Create cell type-specific output directory
        celltype_out_dir = os.path.join(out_dir, celltype)

        # Run TF-MoDISco for this cell type
        tfm_results, tfm_pwms = run_tfmodisco_for_celltype(
            gradient_dir=gradient_dir,
            celltype=celltype,
            fasta_file=fasta_file,
            context_length=context_length,
            baseline=baseline,
            center_bp=center_bp,
            meme_db=meme_db,
            gc_content=gc_content,
            force_fwd=force_fwd,
            modisco_window_size=modisco_window_size,
            modisco_flank=modisco_flank,
            modisco_sliding_window_size=modisco_sliding_window_size,
            modisco_sliding_window_flank=modisco_sliding_window_flank,
            modisco_max_seqlets=modisco_max_seqlets,
            modisco_n_cores=modisco_n_cores,
            ic_t=ic_t,
            norm_type=norm_type,
            out_dir=celltype_out_dir,
            clip_perc=clip_perc,
            kmer_len=kmer_len,
            num_gaps=num_gaps,
            num_mismatches=num_mismatches,
            use_cache=use_cache
        )

        return (celltype, True, (tfm_results, tfm_pwms))

    except Exception as e:
        return (celltype, False, str(e))


def run_tfmodisco_analysis(gradient_dir,
                           fasta_file,
                           context_length,
                           baseline="random",
                           center_bp=None,
                           meme_db='meme-5.4.1/motif_databases/CIS-BP_2.00/Homo_sapiens.meme',
                           gc_content=0.41,
                           force_fwd=0,
                           modisco_window_size=24,
                           modisco_flank=8,
                           modisco_sliding_window_size=18,
                           modisco_sliding_window_flank=8,
                           modisco_max_seqlets=20000,
                           modisco_n_cores=1,
                           ic_t=0.1,
                           norm_type='max',
                           out_dir='tfm_out',
                           clip_perc=25,
                           kmer_len=None,
                           num_gaps=None,
                           num_mismatches=None,
                           specific_celltype=None,
                           n_jobs=-1,
                           use_cache=True):
    """
    Run TF-MoDISco analysis on all cell types in gradient directory.

    This function identifies all cell types from the gradient files and runs
    TF-MoDISco analysis separately for each cell type in parallel using joblib,
    similar to 11_4_DiffExpress_TFMoDisco.py.

    Args:
        gradient_dir: Directory containing .pt gradient files
        fasta_file: Path to reference genome FASTA file
        context_length: Context length for sequence loading
        baseline: Baseline type to load (default: "random")
        center_bp: Extract only center bp (default: None, use full sequence)
        meme_db: Path to MEME database for TOMTOM
        gc_content: Genome GC content for background model (default: 0.41)
        force_fwd: Do not use rev-comp in modisco (default: 0)
        modisco_window_size: Modisco window size (default: 24)
        modisco_flank: Modisco flanks to add (default: 8)
        modisco_sliding_window_size: Modisco sliding window size (default: 18)
        modisco_sliding_window_flank: Modisco sliding window flanks (default: 8)
        modisco_max_seqlets: Max seqlets per metacluster (default: 20000)
        modisco_n_cores: Number of cores for modisco internal parallelism (default: 1)
        ic_t: Information content threshold (default: 0.1)
        norm_type: Normalization type: 'max' or 'gaussian' (default: 'max')
        out_dir: Output directory (default: 'tfm_out')
        clip_perc: Percentile of max deviations to clip by (default: 25)
        kmer_len: K-mer length for pattern matching (optional)
        num_gaps: Number of gaps for pattern matching (optional)
        num_mismatches: Number of mismatches for pattern matching (optional)
        specific_celltype: If specified, only run for this cell type (default: None, run all)
        n_jobs: Number of parallel jobs (-1 = all CPUs) (default: -1)
        use_cache: Whether to use cached intermediate results (default: True)

    Returns:
        dict: Dictionary mapping celltype to (tfm_results, tfm_pwms) tuples
    """
    print(f'\n{"=" * 80}')
    print('TF-MoDISco Analysis for Differential Expression (Borzoi-style)')
    print(f'{"=" * 80}')
    print(f'Gradient directory: {gradient_dir}')
    print(f'Reference genome: {fasta_file}')
    print(f'Baseline: {baseline}')
    print(f'Output directory: {out_dir}')
    print(f'Parallel jobs: {n_jobs if n_jobs > 0 else "all CPUs"}')
    print(f'{"=" * 80}\n')

    # Identify all cell types
    if specific_celltype is None:
        print('\nStep 1: Identifying cell types from gradient files')
        print('-' * 80)
        celltypes = identify_celltypes(gradient_dir, baseline=baseline)
    else:
        celltypes = [specific_celltype]
        print(f'\nAnalyzing specific cell type: {specific_celltype}')

    # Track results and errors
    results_by_celltype = {}
    successful_celltypes = []
    failed_celltypes = []

    # Process cell types
    if n_jobs == 1:
        print(f'\nStep 2: Processing {len(celltypes)} cell types sequentially')
        print('-' * 80)
        results = []
        for celltype in celltypes:
            result = _process_single_celltype(
                celltype, gradient_dir, fasta_file, context_length,
                baseline, center_bp, meme_db, gc_content, force_fwd,
                modisco_window_size, modisco_flank, modisco_sliding_window_size,
                modisco_sliding_window_flank, modisco_max_seqlets, modisco_n_cores, ic_t,
                norm_type, out_dir, clip_perc, kmer_len, num_gaps, num_mismatches,
                use_cache
            )
            results.append(result)
    else:
        print(f'\nStep 2: Processing {len(celltypes)} cell types in parallel')
        print('-' * 80)
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_celltype)(
                celltype, gradient_dir, fasta_file, context_length,
                baseline, center_bp, meme_db, gc_content, force_fwd,
                modisco_window_size, modisco_flank, modisco_sliding_window_size,
                modisco_sliding_window_flank, modisco_max_seqlets, modisco_n_cores, ic_t,
                norm_type, out_dir, clip_perc, kmer_len, num_gaps, num_mismatches,
                use_cache
            ) for celltype in celltypes
        )

    # Process results
    for celltype, success, result_or_error in results:
        if success:
            results_by_celltype[celltype] = result_or_error
            successful_celltypes.append(celltype)
        else:
            failed_celltypes.append((celltype, result_or_error))

    # Print summary
    print(f'\n{"=" * 80}')
    print('Analysis Completed!')
    print(f'{"=" * 80}')
    print(f'Successfully processed {len(successful_celltypes)}/{len(celltypes)} cell types')

    if successful_celltypes:
        print(f'\nSUCCESS: {", ".join(successful_celltypes)}')

    if failed_celltypes:
        print(f'\nFailed to process {len(failed_celltypes)} cell types:')
        for celltype, error in failed_celltypes:
            print(f'  FAILED: {celltype}')
            print(f'    Error: {error[:100]}...')

    print(f'\nResults saved to: {out_dir}')
    print(f'{"=" * 80}\n')

    return results_by_celltype


def ic_clip(pwm, threshold, background=[0.25]*4):
    """
    Clip PWM sides with an information content threshold.

    This function is copied from borzoi_tfmodisco.py.

    Args:
        pwm: Position weight matrix
        threshold: IC threshold
        background: Background nucleotide frequencies

    Returns:
        Clipped PWM or None if no bp pass threshold
    """
    pc = 0.001
    _odds_ratio = ((pwm + pc) / (1 + 4 * pc)) / (background[None, :])  # Computed but not used
    ic = (np.log((pwm + pc) / (1 + 4 * pc)) / np.log(2)) * pwm
    ic -= (np.log(background) * background / np.log(2))[None, :]
    ic_total = np.sum(ic, axis=1)[:, None]

    # No bp pass threshold
    if ~np.any(ic_total.flatten() > threshold):
        return None
    else:
        left = np.where(ic_total > threshold)[0][0]
        right = np.where(ic_total > threshold)[0][-1]
        return pwm[left:(right + 1)]


def main():
    """
    Main function for command-line usage.

    Modified to accept a directory of .pt gradient files instead of a single h5 file.
    """
    usage = 'usage: %prog [options] <gradient_dir>'
    parser = OptionParser(usage)
    parser.add_option(
        '--fasta',
        dest='fasta_file',
        default=None,
        type='str',
        help='Path to reference genome FASTA file [REQUIRED]',
    )
    parser.add_option(
        '--context_length',
        dest='context_length',
        default=524288,
        type='int',
        help='Context length for sequence loading [Default: %default]',
    )
    parser.add_option(
        '-b',
        '--baseline',
        dest='baseline',
        default='random',
        type='str',
        help='Baseline type to load (e.g., random, zeros) [Default: %default]',
    )
    parser.add_option(
        '--celltype',
        dest='specific_celltype',
        default=None,
        type='str',
        help='Cell type to analyze (REQUIRED). If not provided, script will list available cell types and exit [Default: %default]',
    )
    parser.add_option(
        '-c',
        dest='center_bp',
        default=None,
        type='int',
        help='Extract only center bp [Default: %default]',
    )
    parser.add_option(
        '-d',
        dest='meme_db',
        default='meme-5.4.1/motif_databases/CIS-BP_2.00/Homo_sapiens.meme',
        help='Meme database [Default: %default]',
    )
    parser.add_option(
        '--gc',
        dest='gc_content',
        default=0.41,
        type='float',
        help='Genome GC content [Default: %default]',
    )
    parser.add_option(
        '--fwd',
        dest='force_fwd',
        default=0,
        type='int',
        help='Do not use rev-comp in modisco [Default: %default]',
    )
    parser.add_option(
        '--modisco_window_size',
        dest='modisco_window_size',
        default=24,
        type='int',
        help='Modisco window size [Default: %default]',
    )
    parser.add_option(
        '--modisco_flank',
        dest='modisco_flank',
        default=8,
        type='int',
        help='Modisco flanks to add [Default: %default]',
    )
    parser.add_option(
        '--modisco_sliding_window_size',
        dest='modisco_sliding_window_size',
        default=18,
        type='int',
        help='Modisco sliding window size [Default: %default]',
    )
    parser.add_option(
        '--modisco_sliding_window_flank',
        dest='modisco_sliding_window_flank',
        default=8,
        type='int',
        help='Modisco sliding window flanks [Default: %default]',
    )
    parser.add_option(
        '--modisco_max_seqlets',
        dest='modisco_max_seqlets',
        default=20000,
        type='int',
        help='Modisco max seqlets [Default: %default]',
    )
    parser.add_option(
        '--modisco_n_cores',
        dest='modisco_n_cores',
        default=1,
        type='int',
        help='Number of cores for modisco internal parallelism [Default: %default]',
    )
    parser.add_option(
        '-i',
        dest='ic_t',
        default=0.1,
        type='float',
        help='Information content threshold [Default: %default]',
    )
    parser.add_option(
        '-n',
        dest='norm_type',
        default='gaussian',
        help='Normalization type: max or gaussian [Default: %default]',
    )
    parser.add_option(
        '-o',
        dest='out_dir',
        default='tfm_out',
        help='Output directory [Default: %default]',
    )
    parser.add_option(
        '--clip_perc',
        dest='clip_perc',
        default=25,
        type='int',
        help='Percentile of max deviations to clip by [Default: %default]',
    )
    parser.add_option(
        '--kmer_len',
        dest='kmer_len',
        default=None,
        type='int',
        help='K-mer length for pattern matching [Default: %default]',
    )
    parser.add_option(
        '--num_gaps',
        dest='num_gaps',
        default=None,
        type='int',
        help='Number of gaps for pattern matching [Default: %default]',
    )
    parser.add_option(
        '--num_mismatches',
        dest='num_mismatches',
        default=None,
        type='int',
        help='Number of mismatches for pattern matching [Default: %default]',
    )
    parser.add_option(
        '--n_jobs',
        dest='n_jobs',
        default=1,
        type='int',
        help='[DEPRECATED - IGNORED] Always uses n_jobs=1 (sequential) to avoid threading issues [Default: %default]',
    )
    parser.add_option(
        '--use_cache',
        dest='use_cache',
        default=True,
        action='store_true',
        help='Use cached intermediate results (gradients and sequences) [Default: %default]',
    )
    parser.add_option(
        '--no_cache',
        dest='use_cache',
        action='store_false',
        help='Do not use cached intermediate results, reload from scratch',
    )

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide gradient directory.')
    else:
        gradient_dir = args[0]

    # If celltype is not specified, list available cell types and exit
    if options.specific_celltype is None:
        print(f'\n{"=" * 80}')
        print('TF-MoDISco Analysis - Cell Type Discovery Mode')
        print(f'{"=" * 80}')
        print(f'Gradient directory: {gradient_dir}')
        print(f'Baseline: {options.baseline}')
        print(f'{"=" * 80}\n')

        print('Scanning for available cell types...\n')
        celltypes = identify_celltypes(gradient_dir, baseline=options.baseline)

        print(f'\n{"=" * 80}')
        print(f'Found {len(celltypes)} cell types:')
        print(f'{"=" * 80}')
        for i, celltype in enumerate(celltypes, 1):
            print(f'{i:3d}. {celltype}')
        print(f'{"=" * 80}\n')

        print('To run analysis for a specific cell type, use:')
        print(f'  python {os.path.basename(__file__)} {gradient_dir} --celltype <celltype_name>')
        print('\nExiting.')
        return

    if options.fasta_file is None:
        parser.error('Must provide reference genome FASTA file with --fasta')

    # Force n_jobs=1 (always run sequentially)
    print('Note: Running with n_jobs=1 (sequential execution) to avoid threading issues')

    # Run analysis
    run_tfmodisco_analysis(
        gradient_dir=gradient_dir,
        fasta_file=options.fasta_file,
        context_length=options.context_length,
        baseline=options.baseline,
        center_bp=options.center_bp,
        meme_db=options.meme_db,
        gc_content=options.gc_content,
        force_fwd=options.force_fwd,
        modisco_window_size=options.modisco_window_size,
        modisco_flank=options.modisco_flank,
        modisco_sliding_window_size=options.modisco_sliding_window_size,
        modisco_sliding_window_flank=options.modisco_sliding_window_flank,
        modisco_max_seqlets=options.modisco_max_seqlets,
        modisco_n_cores=options.modisco_n_cores,
        ic_t=options.ic_t,
        norm_type=options.norm_type,
        out_dir=options.out_dir,
        clip_perc=options.clip_perc,
        kmer_len=options.kmer_len,
        num_gaps=options.num_gaps,
        num_mismatches=options.num_mismatches,
        specific_celltype=options.specific_celltype,
        n_jobs=1,  # Always use sequential execution
        use_cache=options.use_cache
    )


if __name__ == '__main__':
    main()
