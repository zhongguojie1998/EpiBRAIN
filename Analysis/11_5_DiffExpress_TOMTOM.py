#!/usr/bin/env python3
"""
Extract motifs from TF-MoDISco results and run TOMTOM comparison.

This script:
1. Loads TF-MoDISco results from HDF5 file
2. Extracts position weight matrices (PWMs) from metaclusters and patterns
3. Clips PWMs based on information content threshold
4. Writes motifs in MEME format
5. Runs TOMTOM to compare against known motif databases

Based on borzoi's approach:
https://github.com/calico/borzoi/blob/main/src/scripts/borzoi_tfmodisco.py

Command-line usage:
    # Process single modisco results file:
    python 11_5_DiffExpress_TOMTOM.py \\
        --modisco_h5 ACBGM_modisco_results.h5 \\
        --output_dir ./tomtom_results \\
        --meme_db /path/to/motif_database.meme

    # Process all modisco results in a directory:
    python 11_5_DiffExpress_TOMTOM.py \\
        --modisco_dir ./Res/exp/analysis_20/modisco \\
        --output_dir ./Res/exp/analysis_20/tomtom \\
        --meme_db /path/to/motif_database.meme
"""

import logging
import os
import sys
import subprocess
from pathlib import Path

import click
import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from utils.logging import BaseLogger


def ic_clip(pwm, threshold=0.3, background=(0.25, 0.25, 0.25, 0.25)):
    """
    Clip PWM sides with an information content threshold.

    Args:
        pwm: Position weight matrix [length, 4]
        threshold: Information content threshold (default: 0.3)
        background: Background nucleotide frequencies (default: uniform)

    Returns:
        Clipped PWM or None if no position passes threshold
    """
    pc = 0.001  # Pseudocount
    background = np.array(background)

    # Calculate odds ratio
    odds_ratio = ((pwm + pc) / (1 + 4 * pc)) / background[None, :]

    # Calculate information content
    ic = (np.log((pwm + pc) / (1 + 4 * pc)) / np.log(2)) * pwm
    ic -= (np.log(background) * background / np.log(2))[None, :]
    ic_total = np.sum(ic, axis=1)[:, None]

    # No bp passes threshold
    if ~np.any(ic_total.flatten() > threshold):
        return None
    else:
        left = np.where(ic_total > threshold)[0][0]
        right = np.where(ic_total > threshold)[0][-1]
        return pwm[left:(right + 1)]


def extract_pwms_from_modisco(modisco_h5_file, ic_threshold=0.3, logger=None):
    """
    Extract PWMs from TF-MoDISco results HDF5 file.

    Args:
        modisco_h5_file: Path to modisco results HDF5 file
        ic_threshold: Information content threshold for clipping (default: 0.3)
        logger: Logger instance

    Returns:
        dict: PWMs keyed by pattern ID (metacluster_pattern)
    """
    if logger:
        logger.info(f"Extracting PWMs from {modisco_h5_file}")

    pwms = {}

    with h5py.File(modisco_h5_file, 'r') as tfm_h5:
        # Get metacluster names
        metacluster_names = [
            mcr.decode("utf-8") for mcr in
            list(tfm_h5["metaclustering_results"]["all_metacluster_names"][:])
        ]

        if logger:
            logger.info(f"Found {len(metacluster_names)} metaclusters: {metacluster_names}")

        for metacluster_name in metacluster_names:
            metacluster_grp = tfm_h5["metacluster_idx_to_submetacluster_results"][metacluster_name]

            # Get pattern names
            all_patterns = metacluster_grp["seqlets_to_patterns_result"]["patterns"]["all_pattern_names"][:]
            all_pattern_names = [x.decode("utf-8") for x in list(all_patterns)]

            if logger:
                logger.info(f"  {metacluster_name}: {len(all_pattern_names)} patterns")

            for pattern_name in all_pattern_names:
                pattern_id = metacluster_name + '_' + pattern_name
                pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]

                # Extract forward PWM
                fwd = np.array(pattern["sequence"]["fwd"])

                # Clip based on information content
                fwd_clipped = ic_clip(fwd, threshold=ic_threshold)

                if fwd_clipped is not None:
                    pwms[pattern_id] = fwd_clipped
                    if logger:
                        logger.debug(f"    {pattern_id}: {fwd_clipped.shape[0]} bp (clipped from {fwd.shape[0]} bp)")
                else:
                    if logger:
                        logger.debug(f"    {pattern_id}: filtered (no position > IC threshold)")

    if logger:
        logger.info(f"Extracted {len(pwms)} PWMs after IC filtering")

    return pwms


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


def run_tomtom(meme_file, output_dir, meme_db, dist='pearson', thresh=0.1, logger=None):
    """
    Run TOMTOM to compare motifs against a database.

    Args:
        meme_file: Path to MEME format file with query motifs
        output_dir: Output directory for TOMTOM results
        meme_db: Path to MEME format database file
        dist: Distance metric (default: pearson)
        thresh: Significance threshold (default: 0.1)
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
        logger.info(f"  Threshold: {thresh}")

    # Build TOMTOM command
    cmd = f"tomtom -dist {dist} -thresh {thresh} -oc {output_dir} {meme_file} {meme_db}"

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
@click.option("--modisco_h5", type=str, help="Path to single modisco results HDF5 file")
@click.option("--modisco_dir", type=str, help="Directory containing multiple modisco results files")
@click.option("--output_dir", "-o", required=True, type=str, help="Output directory for results")
@click.option("--meme_db", required=True, type=str, help="Path to MEME format motif database")
@click.option("--ic_threshold", type=float, default=0.3, help="Information content threshold (default: 0.3)")
@click.option("--tomtom_dist", type=str, default="pearson", help="TOMTOM distance metric (default: pearson)")
@click.option("--tomtom_thresh", type=float, default=0.1, help="TOMTOM significance threshold (default: 0.1)")
def main(modisco_h5, modisco_dir, output_dir, meme_db, ic_threshold, tomtom_dist, tomtom_thresh):
    """
    Extract motifs from TF-MoDISco results and run TOMTOM comparison.
    """
    # Setup logger
    logger = BaseLogger(name="TOMTOM", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("TF-MoDISco to TOMTOM Pipeline")
    logger.info("=" * 80)

    # Validate input
    if not modisco_h5 and not modisco_dir:
        logger.error("Error: Either --modisco_h5 or --modisco_dir must be provided")
        sys.exit(1)

    if modisco_h5 and modisco_dir:
        logger.error("Error: Cannot use both --modisco_h5 and --modisco_dir at the same time")
        sys.exit(1)

    # Check if meme_db exists
    if not os.path.exists(meme_db):
        logger.error(f"Error: MEME database not found: {meme_db}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of modisco files to process
    if modisco_h5:
        modisco_files = [Path(modisco_h5)]
    else:
        modisco_files = sorted(Path(modisco_dir).glob("*_modisco_results.h5"))

    if not modisco_files:
        logger.error(f"No modisco results files found")
        sys.exit(1)

    logger.info(f"Found {len(modisco_files)} modisco results files to process")

    # Process each file
    for modisco_file in tqdm(modisco_files, desc="Processing modisco files"):
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing: {modisco_file.name}")
        logger.info("=" * 80)

        # Extract cell type name from filename
        # Format: {celltype}_modisco_results.h5
        celltype = modisco_file.stem.replace("_modisco_results", "")

        # Create subdirectory for this cell type
        celltype_output_dir = os.path.join(output_dir, celltype)
        os.makedirs(celltype_output_dir, exist_ok=True)

        try:
            # Extract PWMs
            pwms = extract_pwms_from_modisco(
                str(modisco_file),
                ic_threshold=ic_threshold,
                logger=logger
            )

            if not pwms:
                logger.warning(f"No PWMs extracted for {celltype}, skipping TOMTOM")
                continue

            # Write MEME file
            meme_file = os.path.join(celltype_output_dir, f"{celltype}_motifs.meme")
            write_meme_format(pwms, meme_file, logger=logger)

            # Run TOMTOM
            returncode = run_tomtom(
                meme_file,
                celltype_output_dir,
                meme_db,
                dist=tomtom_dist,
                thresh=tomtom_thresh,
                logger=logger
            )

            if returncode == 0:
                logger.info(f"Successfully completed TOMTOM for {celltype}")
                logger.info(f"Results: {celltype_output_dir}/tomtom.html")
            else:
                logger.error(f"TOMTOM failed for {celltype}")

        except Exception as e:
            logger.error(f"Error processing {celltype}: {e}")
            continue

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
