#!/usr/bin/env python3
"""
Verify TensorFlow to PyTorch conversion by comparing model outputs.

Usage:
    python 01_verify_tf_to_pytorch_conversion.py \
        --tf_checkpoint /path/to/model_best.h5 \
        --tf_config /path/to/params_train.json \
        --pt_checkpoint /path/to/converted_model.pt \
        --pt_config /path/to/overall_setting.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import yaml
import json
import logging

# Set working directory
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../../')
sys.path.append(str(Path(PWD).parent.parent / "Model"))
sys.path.append('/gpfs/commons/groups/ren_lab/guojiezhong/baskerville_me/src')


def setup_logger():
    """Setup a basic logger."""
    logger = logging.getLogger('verify_conversion')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_tf_model(checkpoint_path, config_path, logger):
    """Load TensorFlow model."""
    import tensorflow as tf

    logger.info("Loading TensorFlow model...")

    # Load the saved model directly
    # The .h5 file contains the full model architecture and weights
    try:
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        logger.info(f"TensorFlow model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model directly: {e}")
        logger.info("Trying alternative loading method...")

        # Alternative: Try loading with Baskerville SeqNN
        from baskerville import seqnn

        # Load config
        with open(config_path, 'r') as f:
            params = json.load(f)

        # Create model
        seqnn_model = seqnn.SeqNN(params['model'])

        # Load weights
        seqnn_model.restore(checkpoint_path)

        logger.info(f"TensorFlow model loaded from {checkpoint_path}")

        return seqnn_model


def load_pt_model(checkpoint_path, config_path, logger):
    """Load PyTorch model."""
    import torch
    from types import SimpleNamespace
    from model.model_utils import setup_model

    logger.info("Loading PyTorch model...")

    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert to SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
        return d

    config = dict_to_namespace(config_dict)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Setup model
    config.training.load_checkpoint = None
    model = setup_model(config, logger, checkpoint=None)

    # Load converted weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"PyTorch model loaded from {checkpoint_path}")

    return model


def create_random_dna_sequence(seq_length, batch_size=1, seed=42):
    """Create random one-hot encoded DNA sequence."""
    np.random.seed(seed)

    # Create random DNA sequence (one-hot encoded)
    # Shape: (batch_size, seq_length, 4) for TF
    seq_1hot_tf = np.zeros((batch_size, seq_length, 4), dtype=np.float32)

    for b in range(batch_size):
        for i in range(seq_length):
            base_idx = np.random.randint(0, 4)
            seq_1hot_tf[b, i, base_idx] = 1.0

    return seq_1hot_tf


def compare_outputs(tf_output, pt_output, logger, tolerance=1e-4):
    """Compare TensorFlow and PyTorch outputs."""
    logger.info("\n" + "="*80)
    logger.info("Output Comparison")
    logger.info("="*80)

    # Check shapes
    logger.info(f"TF output shape: {tf_output.shape}")
    logger.info(f"PT output shape: {pt_output.shape}")

    if tf_output.shape != pt_output.shape:
        logger.error("❌ Output shapes don't match!")
        return False

    # Compute statistics
    abs_diff = np.abs(tf_output - pt_output)
    rel_diff = abs_diff / (np.abs(tf_output) + 1e-8)

    logger.info(f"\nAbsolute difference statistics:")
    logger.info(f"  Mean: {abs_diff.mean():.6e}")
    logger.info(f"  Std:  {abs_diff.std():.6e}")
    logger.info(f"  Max:  {abs_diff.max():.6e}")
    logger.info(f"  Min:  {abs_diff.min():.6e}")

    logger.info(f"\nRelative difference statistics:")
    logger.info(f"  Mean: {rel_diff.mean():.6e}")
    logger.info(f"  Std:  {rel_diff.std():.6e}")
    logger.info(f"  Max:  {rel_diff.max():.6e}")

    # Check if outputs are close
    close_mask = abs_diff < tolerance
    close_pct = close_mask.sum() / close_mask.size * 100

    logger.info(f"\nPercentage of outputs within tolerance ({tolerance}):")
    logger.info(f"  {close_pct:.2f}%")

    # Correlation
    correlation = np.corrcoef(tf_output.flatten(), pt_output.flatten())[0, 1]
    logger.info(f"\nPearson correlation: {correlation:.6f}")

    # Summary
    logger.info(f"\n{'='*80}")
    if close_pct > 99.9 and correlation > 0.999:
        logger.info("✓ PASSED: Models produce very similar outputs!")
        logger.info(f"{'='*80}\n")
        return True
    elif close_pct > 95 and correlation > 0.99:
        logger.warning("⚠ WARNING: Models produce similar but not identical outputs")
        logger.info(f"{'='*80}\n")
        return True
    else:
        logger.error("❌ FAILED: Models produce significantly different outputs")
        logger.info(f"{'='*80}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify TF to PyTorch conversion')
    parser.add_argument('--tf_checkpoint', type=str, required=True,
                        help='Path to TensorFlow .h5 checkpoint')
    parser.add_argument('--tf_config', type=str, required=True,
                        help='Path to TensorFlow config (params_train.json)')
    parser.add_argument('--pt_checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--pt_config', type=str, required=True,
                        help='Path to PyTorch config (overall_setting.yaml)')
    parser.add_argument('--seq_length', type=int, default=524288,
                        help='Input sequence length (default: 524288)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for output comparison (default: 1e-4)')
    parser.add_argument('--num_tests', type=int, default=3,
                        help='Number of random sequences to test (default: 3)')

    args = parser.parse_args()

    logger = setup_logger()

    logger.info("="*80)
    logger.info("TensorFlow to PyTorch Conversion Verification")
    logger.info("="*80)

    # Verify files exist
    for path, name in [(args.tf_checkpoint, 'TensorFlow checkpoint'),
                       (args.tf_config, 'TensorFlow config'),
                       (args.pt_checkpoint, 'PyTorch checkpoint'),
                       (args.pt_config, 'PyTorch config')]:
        if not Path(path).exists():
            logger.error(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load models
    try:
        tf_model = load_tf_model(args.tf_checkpoint, args.tf_config, logger)
    except Exception as e:
        logger.error(f"Failed to load TensorFlow model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        pt_model = load_pt_model(args.pt_checkpoint, args.pt_config, logger)
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run verification tests
    import torch

    all_passed = True

    for test_idx in range(args.num_tests):
        logger.info(f"\n{'='*80}")
        logger.info(f"Test {test_idx + 1}/{args.num_tests}")
        logger.info(f"{'='*80}")

        # Create random input
        seed = 42 + test_idx
        seq_1hot_tf = create_random_dna_sequence(args.seq_length, args.batch_size, seed=seed)

        logger.info(f"Created random DNA sequence (seed={seed})")
        logger.info(f"  Shape: {seq_1hot_tf.shape}")

        # Run TensorFlow model
        logger.info("\nRunning TensorFlow model...")
        try:
            tf_output = tf_model(seq_1hot_tf, training=False)
            logger.info(f"  TF output shape: {tf_output.shape}")
        except Exception as e:
            logger.error(f"TensorFlow model failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue

        # Convert input for PyTorch (N, L, 4) -> (N, 4, L)
        seq_1hot_pt = torch.from_numpy(seq_1hot_tf).permute(0, 2, 1).float()

        # Run PyTorch model
        logger.info("Running PyTorch model...")
        try:
            with torch.no_grad():
                pt_output_dict = pt_model(seq_1hot_pt, use_head='regression')
                # Convert (N, L, C) -> (N, L, C) for comparison
                pt_output = pt_output_dict.cpu().numpy()
            logger.info(f"  PT output shape: {pt_output.shape}")
        except Exception as e:
            logger.error(f"PyTorch model failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue

        # Compare outputs
        passed = compare_outputs(tf_output, pt_output, logger, tolerance=args.tolerance)

        if not passed:
            all_passed = False

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("The conversion was successful! PyTorch model produces similar outputs to TensorFlow model.")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("The conversion may have issues. Please review the output comparison above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
