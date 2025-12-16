#!/usr/bin/env python3
"""
Extract PWMs from motif interpretation results for a specific region and run TOMTOM.

This script:
1. Loads attribution and sequence data from motif interpretation output
2. Extracts the specified genomic region
3. Converts attribution scores to PWM format
4. Writes PWM in MEME format
5. Runs TOMTOM to compare against known motif databases

Usage:
    # Extract motif from specific region and run TOMTOM
    python 02_motif_region_tomtom.py \
        --data_dir ./Res/exp/analysis_20/raw_data/interp_diff \
        --name_base chr12_40144409_40668697_LRRK2_STR-D1-MSN_minus \
        --baseline random \
        --region chr12:40208950-40208975 \
        --output_dir ./tomtom_results \
        --meme_db /path/to/motif_database.meme

    # Extract multiple regions with custom background frequencies
    python 02_motif_region_tomtom.py \
        --data_dir ./Res/exp/analysis_20/raw_data/interp_diff \
        --name_base chr12_40144409_40668697_LRRK2_STR-D1-MSN_minus \
        --baseline random \
        --region chr12:40208950-40208975,chr12:40210000-40210025 \
        --background-region chr12:40000000-41000000 \
        --output_dir ./tomtom_results \
        --meme_db /path/to/motif_database.meme
"""

import logging
import os
import sys
import subprocess
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from utils.logging import BaseLogger


def parse_region(region_str):
    """
    Parse genomic region string.

    Args:
        region_str: Region string like "chr12:40208950-40208975"

    Returns:
        tuple: (chr, start, end)
    """
    chr_part, coord_part = region_str.split(':')
    start, end = map(int, coord_part.split('-'))
    return chr_part, start, end


def extract_pwm_from_attribution(attribution, sequence, region_start, region_end,
                                  region_start_bp, normalize=True,
                                  clip_threshold=0.0):
    """
    Extract PWM from attribution scores for a specific region.

    Following the borzoi/TF-MoDISco approach:
    1. Compute contribution scores = attribution × sequence (element-wise)
    2. Take absolute value to get magnitude of contribution
    3. Normalize to create probability matrix

    This captures how much each nucleotide at each position contributes
    to the prediction, which is the essence of a motif.

    Args:
        attribution: Attribution scores [length, 4]
        sequence: Sequence one-hot [length, 4]
        region_start: Start position of region to extract (genomic coordinate)
        region_end: End position of region to extract (genomic coordinate)
        region_start_bp: Start position of the trimmed attribution data (genomic coordinate)
        normalize: Whether to normalize to probability matrix
        clip_threshold: Minimum absolute contribution value to keep

    Returns:
        PWM [region_length, 4] or None if region is outside data
    """
    # Calculate bp indices into the data arrays (0-indexed from region_start_bp)
    # Same logic as in 02_motif_interpretation_plot.py
    start_bp_idx = region_start - region_start_bp
    end_bp_idx = region_end - region_start_bp

    # Check if indices are valid
    if start_bp_idx < 0 or end_bp_idx > len(attribution):
        return None
    if start_bp_idx >= end_bp_idx:
        return None

    # Extract attribution and sequence for region
    region_attr = attribution[start_bp_idx:end_bp_idx].copy()
    region_seq = sequence[start_bp_idx:end_bp_idx].copy()

    # Step 1: Get the attribution value for the actual reference nucleotide at each position
    # Multiply attribution by sequence to get contribution of reference base
    ref_contrib = np.sum(region_attr * region_seq, axis=1)  # [length]

    # Step 2: Check the average sign of reference contributions
    avg_ref_contrib = np.mean(ref_contrib)

    # Step 3: Apply sign-based transformation
    # If the average reference contribution is negative, negate all attributions
    # This ensures we're measuring importance in the right direction
    if avg_ref_contrib < 0:
        region_attr = -region_attr

    # Step 4: Apply exponential transformation 2^x with numerical stability
    # Use the softmax trick: subtract max value per row to prevent overflow
    # Since 2^(x - max) / sum(2^(x - max)) = 2^x / sum(2^x), this is mathematically equivalent
    # but numerically stable
    max_per_row = np.max(region_attr, axis=1, keepdims=True)
    region_attr_stable = region_attr - max_per_row

    # Now apply 2^x - this won't overflow since max value per row is now 0
    pwm = np.power(2, region_attr_stable)

    # Step 5: Normalize to probability matrix
    if normalize:
        row_sums = pwm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        pwm = pwm / row_sums

    # Clip low values after normalization
    if clip_threshold > 0:
        pwm[pwm < clip_threshold] = 0
        # Re-normalize after clipping
        if normalize:
            row_sums = pwm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            pwm = pwm / row_sums

    return pwm


def calculate_background_frequencies(sequence, region_start, region_end, region_start_bp, logger=None):
    """
    Calculate background nucleotide frequencies from a genomic region.

    Args:
        sequence: Sequence one-hot [length, 4] where columns are [A, C, G, T]
        region_start: Start position of background region (genomic coordinate)
        region_end: End position of background region (genomic coordinate)
        region_start_bp: Start position of the data (genomic coordinate)
        logger: Logger instance

    Returns:
        tuple: Background frequencies (A, C, G, T) or None if region is invalid
    """
    # Calculate bp indices into the data array
    start_bp_idx = region_start - region_start_bp
    end_bp_idx = region_end - region_start_bp

    # Check if indices are valid
    if start_bp_idx < 0 or end_bp_idx > len(sequence):
        if logger:
            logger.warning(f"Background region {region_start}-{region_end} is outside data range")
        return None
    if start_bp_idx >= end_bp_idx:
        if logger:
            logger.warning(f"Invalid background region: start >= end")
        return None

    # Extract sequence for region
    region_seq = sequence[start_bp_idx:end_bp_idx]

    # Calculate frequencies (columns are A, C, G, T)
    freq_A = region_seq[:, 0].mean()
    freq_C = region_seq[:, 1].mean()
    freq_G = region_seq[:, 2].mean()
    freq_T = region_seq[:, 3].mean()

    if logger:
        logger.info(f"  Calculated background frequencies from {region_end - region_start} bp:")
        logger.info(f"    A: {freq_A:.4f}, C: {freq_C:.4f}, G: {freq_G:.4f}, T: {freq_T:.4f}")

    return (freq_A, freq_C, freq_G, freq_T)


def clip_pwm_by_information_content(pwm, threshold=0.2, background=(0.25, 0.25, 0.25, 0.25)):
    """
    Clip PWM edges based on information content.

    Args:
        pwm: Position weight matrix [length, 4]
        threshold: Information content threshold (default: 0.2)
        background: Background nucleotide frequencies (default: uniform)

    Returns:
        Clipped PWM or None if no position passes threshold
    """
    pc = 0.001  # Pseudocount
    background = np.array(background)

    # Calculate information content
    ic = (np.log((pwm + pc) / (1 + 4 * pc)) / np.log(2)) * pwm
    ic -= (np.log(background) * background / np.log(2))[None, :]
    ic_total = np.sum(ic, axis=1)

    # No bp passes threshold
    if not np.any(ic_total > threshold):
        return None

    # Find left and right boundaries
    left = np.where(ic_total > threshold)[0][0]
    right = np.where(ic_total > threshold)[0][-1]

    return pwm[left:(right + 1)]


def write_meme_format(pwms, output_meme_file, background=(0.25, 0.25, 0.25, 0.25), logger=None):
    """
    Write PWMs in MEME format.

    Args:
        pwms: Dictionary of PWMs keyed by motif ID
        output_meme_file: Path to output MEME file
        background: Background nucleotide frequencies (default: uniform)
        logger: Logger instance
    """
    if logger:
        logger.info(f"Writing {len(pwms)} motifs to MEME format: {output_meme_file}")

    with open(output_meme_file, 'w') as meme_file:
        # Write header
        meme_file.write('MEME version 4\n\n')
        meme_file.write('ALPHABET= ACGT\n\n')
        meme_file.write('strands: + -\n\n')
        meme_file.write('Background letter frequencies\n')
        meme_file.write('A %f C %f G %f T %f\n\n' % background)

        # Write each motif
        for motif_id, pwm in pwms.items():
            meme_file.write(f'MOTIF {motif_id}\n')
            meme_file.write(f'letter-probability matrix: alength= 4 w= {pwm.shape[0]}\n')
            np.savetxt(meme_file, pwm, fmt='%.6f')
            meme_file.write('\n')

    if logger:
        logger.info(f"MEME file written successfully")


def run_tomtom(meme_file, output_dir, meme_db, dist='pearson', thresh=1.0, use_evalue=True, logger=None):
    """
    Run TOMTOM to compare motifs against a database.

    Args:
        meme_file: Path to MEME format file with query motifs
        output_dir: Output directory for TOMTOM results
        meme_db: Path to MEME format database file
        dist: Distance metric (default: pearson)
        thresh: Significance threshold (default: 1.0 for e-value)
        use_evalue: Use e-value threshold instead of q-value (default: True, uses e-value)
        logger: Logger instance

    Returns:
        returncode: 0 if successful, non-zero otherwise
    """
    if logger:
        logger.info("Running TOMTOM")
        logger.info(f"  Query motifs: {meme_file}")
        logger.info(f"  Database: {meme_db}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Distance metric: {dist}")
        logger.info(f"  Threshold type: {'e-value' if use_evalue else 'q-value'}")
        logger.info(f"  Threshold: {thresh}")

    # Build TOMTOM command
    # -evalue is a flag (no value) that switches from q-value to e-value mode
    # -thresh <value> sets the threshold (works for both modes)
    evalue_flag = "-evalue" if use_evalue else ""
    cmd = f"tomtom -dist {dist} {evalue_flag} -thresh {thresh} -oc {output_dir} {meme_file} {meme_db}".replace("  ", " ")

    if logger:
        logger.info(f"Running command: {cmd}")

    # Run TOMTOM
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Log output
    if result.stdout:
        if logger:
            logger.info(f"TOMTOM stdout:\n{result.stdout}")

    if result.stderr:
        if logger:
            logger.warning(f"TOMTOM stderr:\n{result.stderr}")

    if result.returncode == 0:
        if logger:
            logger.info("TOMTOM completed successfully")
    else:
        if logger:
            logger.error(f"TOMTOM failed with return code {result.returncode}")

    return result.returncode


@click.command()
@click.option("--data_dir", "-d", required=True, type=str,
              help="Directory containing motif interpretation data (raw_data/interp_diff)")
@click.option("--name_base", "-n", required=True, type=str,
              help="Name base for interpretation files (e.g., chr12_40144409_40668697_LRRK2_STR-D1-MSN_minus)")
@click.option("--baseline", "-b", required=True, type=str,
              help="Baseline type (e.g., random, shuffle)")
@click.option("--region", "-r", required=True, type=str,
              help="Genomic region(s) to extract (e.g., chr12:40208950-40208975 or multiple: region1,region2,...)")
@click.option("--background-region", type=str, default=None,
              help="Genomic region to use for calculating background nucleotide frequencies (e.g., chr12:40000000-41000000). If not provided, uniform frequencies (0.25 each) are used.")
@click.option("--output_dir", "-o", required=True, type=str,
              help="Output directory for MEME file and TOMTOM results")
@click.option("--meme_db", required=True, type=str,
              help="Path to MEME format motif database for TOMTOM")
@click.option("--normalize", is_flag=True, default=True,
              help="Normalize PWM to probability matrix (default: True)")
@click.option("--clip_threshold", type=float, default=0.0,
              help="Minimum absolute attribution value to keep (default: 0.0)")
@click.option("--ic_threshold", type=float, default=0.2,
              help="Information content threshold for clipping PWM edges (default: 0.2)")
@click.option("--tomtom_dist", type=str, default="pearson",
              help="TOMTOM distance metric (default: pearson)")
@click.option("--tomtom_thresh", type=float, default=1.0,
              help="TOMTOM significance threshold (default: 1.0 for e-value)")
@click.option("--tomtom_use_qvalue", is_flag=True, default=False,
              help="Use q-value threshold instead of e-value (default: use e-value)")
def main(data_dir, name_base, baseline, region, background_region, output_dir, meme_db, normalize,
         clip_threshold, ic_threshold, tomtom_dist, tomtom_thresh, tomtom_use_qvalue):
    """
    Extract PWMs from motif interpretation for specific regions and run TOMTOM.
    """
    # Setup logger
    logger = BaseLogger(name="RegionTOMTOM", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("Region-specific Motif TOMTOM Analysis")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Name base: {name_base}")
    logger.info(f"Baseline: {baseline}")
    logger.info(f"Region(s): {region}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if meme_db exists
    if not os.path.exists(meme_db):
        logger.error(f"Error: MEME database not found: {meme_db}")
        sys.exit(1)

    # Load data
    data_dir = Path(data_dir)
    identifier = f"{name_base}_{baseline}"

    # Load attribution and sequence
    attr_file = data_dir / f"{identifier}_attribution.npy"
    seq_file = data_dir / f"{identifier}_sequence.npy"
    metadata_file = data_dir / f"{name_base}_metadata.npy"

    if not attr_file.exists():
        logger.error(f"Attribution file not found: {attr_file}")
        sys.exit(1)

    if not seq_file.exists():
        logger.error(f"Sequence file not found: {seq_file}")
        sys.exit(1)

    logger.info("Loading attribution and sequence data...")
    attribution = np.load(attr_file)
    sequence = np.load(seq_file)

    logger.info(f"  Attribution shape: {attribution.shape}")
    logger.info(f"  Sequence shape: {sequence.shape}")

    # Load metadata to get window coordinates
    if metadata_file.exists():
        metadata = np.load(metadata_file, allow_pickle=True).item()
        window_chr = metadata['chr_name']

        # Calculate the trimmed region that the attribution data actually covers
        # (same logic as in 02_motif_interpretation_plot.py)
        window_size = metadata['window_size']
        real_start = metadata['real_start']
        real_end = metadata['real_end']

        trim = (
            metadata['context_length'] // window_size
            - metadata['n_window']
        ) // 2

        total_bp = metadata['n_window'] * window_size
        window_start = real_start + trim * window_size
        window_end = window_start + total_bp

        logger.info(f"  Full region: {window_chr}:{real_start}-{real_end}")
        logger.info(f"  Trimmed data region: {window_chr}:{window_start}-{window_end} (covers {total_bp} bp)")
    else:
        logger.error(f"Metadata file not found: {metadata_file}")
        logger.error("Cannot determine window coordinates")
        sys.exit(1)

    # Calculate background frequencies if background region is provided
    background = (0.25, 0.25, 0.25, 0.25)  # Default uniform frequencies
    if background_region:
        logger.info(f"\nCalculating background frequencies from region: {background_region}")
        try:
            bg_chr, bg_start, bg_end = parse_region(background_region)

            # Check chromosome match
            if bg_chr != window_chr:
                logger.warning(f"Background region chromosome ({bg_chr}) does not match data chromosome ({window_chr})")
                logger.warning("Using default uniform background frequencies")
            else:
                # Calculate frequencies
                bg_freqs = calculate_background_frequencies(
                    sequence, bg_start, bg_end, window_start, logger=logger
                )

                if bg_freqs is not None:
                    background = bg_freqs
                    logger.info("Using calculated background frequencies")
                else:
                    logger.warning("Failed to calculate background frequencies, using default uniform frequencies")
        except Exception as e:
            logger.warning(f"Error processing background region: {e}")
            logger.warning("Using default uniform background frequencies")
    else:
        logger.info("\nUsing default uniform background frequencies: A=0.25, C=0.25, G=0.25, T=0.25")

    # Parse regions
    regions = region.split(',')
    logger.info(f"Extracting {len(regions)} region(s):")

    pwms = {}

    for region_str in regions:
        region_str = region_str.strip()
        logger.info(f"\n  Processing region: {region_str}")

        try:
            region_chr, region_start, region_end = parse_region(region_str)
        except Exception as e:
            logger.error(f"    Error parsing region: {e}")
            continue

        # Check chromosome match
        if region_chr != window_chr:
            logger.warning(f"    Chromosome mismatch: region is on {region_chr}, but data is for {window_chr}")
            continue

        # Extract PWM
        pwm = extract_pwm_from_attribution(
            attribution, sequence,
            region_start, region_end,
            window_start,  # region_start_bp
            normalize=normalize,
            clip_threshold=clip_threshold
        )

        if pwm is None:
            logger.warning(f"    Region not in window, skipping")
            continue

        logger.info(f"    Extracted PWM: {pwm.shape[0]} bp")

        # Clip by information content (always use uniform background for clipping)
        if ic_threshold > 0:
            pwm_clipped = clip_pwm_by_information_content(pwm, threshold=ic_threshold)
            if pwm_clipped is None:
                logger.warning(f"    No positions pass IC threshold {ic_threshold}, using full PWM")
                pwm_clipped = pwm
            else:
                logger.info(f"    Clipped by IC: {pwm_clipped.shape[0]} bp (from {pwm.shape[0]} bp)")
                pwm = pwm_clipped

        # Create motif ID
        motif_id = f"{region_chr}:{region_start}-{region_end}"
        pwms[motif_id] = pwm

        logger.info(f"    Final PWM: {pwm.shape[0]} bp")
        logger.info(f"    PWM value range: [{pwm.min():.6f}, {pwm.max():.6f}]")
        logger.info(f"    PWM row sums: min={pwm.sum(axis=1).min():.6f}, max={pwm.sum(axis=1).max():.6f}, mean={pwm.sum(axis=1).mean():.6f}")

    if not pwms:
        logger.error("No PWMs extracted, exiting")
        sys.exit(1)

    logger.info(f"\nExtracted {len(pwms)} PWMs total")

    # Write MEME file
    meme_file = os.path.join(output_dir, f"{name_base}_{baseline}_regions.meme")
    write_meme_format(pwms, meme_file, background=background, logger=logger)

    # Run TOMTOM
    logger.info("")
    returncode = run_tomtom(
        meme_file,
        output_dir,
        meme_db,
        dist=tomtom_dist,
        thresh=tomtom_thresh,
        use_evalue=not tomtom_use_qvalue,  # Default is e-value, flag switches to q-value
        logger=logger
    )

    if returncode == 0:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 80)
        logger.info(f"MEME file: {meme_file}")
        logger.info(f"TOMTOM results: {output_dir}/tomtom.html")
    else:
        logger.error("TOMTOM failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
