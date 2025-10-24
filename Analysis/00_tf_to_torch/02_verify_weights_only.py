#!/usr/bin/env python3
"""
Verify TensorFlow to PyTorch conversion by comparing weights directly.

Usage:
    python 02_verify_weights_only.py \
        --tf_checkpoint /path/to/model_best.h5 \
        --pt_checkpoint /path/to/converted_model.pt
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import h5py
import logging

PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../../')


def setup_logger():
    """Setup a basic logger."""
    logger = logging.getLogger('verify_weights')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_tf_weights(h5_path):
    """Load all TensorFlow weights from HDF5 file."""
    tf_weights = {}

    with h5py.File(h5_path, 'r') as f:
        def extract_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                tf_weights[name] = np.array(obj)

        f.visititems(extract_weights)

    return tf_weights


def transpose_conv_weights(tf_weights):
    """Convert TensorFlow conv1d weights to PyTorch format."""
    return np.transpose(tf_weights, (2, 1, 0))


def verify_layer(tf_key, pt_key, tf_weights, pt_state_dict, transform_fn, logger, tolerance=1e-6):
    """Verify a single layer's weights match."""

    if tf_key not in tf_weights:
        logger.error(f"  ❌ TF key not found: {tf_key}")
        return False

    if pt_key not in pt_state_dict:
        logger.error(f"  ❌ PT key not found: {pt_key}")
        return False

    # Get weights
    tf_weight = tf_weights[tf_key]
    pt_weight = pt_state_dict[pt_key].cpu().numpy()

    # Transform TF weight to PT format
    tf_weight_transformed = transform_fn(tf_weight)

    # Check shapes
    if tf_weight_transformed.shape != pt_weight.shape:
        logger.error(f"  ❌ Shape mismatch: TF {tf_weight_transformed.shape} vs PT {pt_weight.shape}")
        return False

    # Compare values
    abs_diff = np.abs(tf_weight_transformed - pt_weight)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    if max_diff < tolerance:
        logger.info(f"  ✓ MATCH (max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e})")
        return True
    else:
        # Check if they're close enough (might be small numerical differences)
        if np.allclose(tf_weight_transformed, pt_weight, rtol=1e-5, atol=1e-5):
            logger.warning(f"  ⚠ CLOSE (max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e})")
            return True
        else:
            logger.error(f"  ❌ MISMATCH (max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e})")
            # Show some sample values
            logger.error(f"     TF sample: {tf_weight_transformed.flatten()[:5]}")
            logger.error(f"     PT sample: {pt_weight.flatten()[:5]}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Verify TF to PyTorch weight conversion')
    parser.add_argument('--tf_checkpoint', type=str, required=True,
                        help='Path to TensorFlow .h5 checkpoint')
    parser.add_argument('--pt_checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Tolerance for weight comparison (default: 1e-6)')

    args = parser.parse_args()

    logger = setup_logger()

    logger.info("="*80)
    logger.info("TensorFlow to PyTorch Weight Verification")
    logger.info("="*80)

    # Verify files exist
    for path, name in [(args.tf_checkpoint, 'TensorFlow checkpoint'),
                       (args.pt_checkpoint, 'PyTorch checkpoint')]:
        if not Path(path).exists():
            logger.error(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load weights
    logger.info("\nLoading TensorFlow weights...")
    tf_weights = load_tf_weights(args.tf_checkpoint)
    logger.info(f"Loaded {len(tf_weights)} TensorFlow weight tensors")

    logger.info("\nLoading PyTorch weights...")
    import torch
    pt_checkpoint = torch.load(args.pt_checkpoint, map_location='cpu', weights_only=False)
    pt_state_dict = pt_checkpoint['model_state_dict']
    logger.info(f"Loaded {len(pt_state_dict)} PyTorch weight tensors")

    # Verify sample layers
    logger.info("\n" + "="*80)
    logger.info("Verifying Sample Layers")
    logger.info("="*80)

    test_cases = [
        # (TF key, PT key, transform function, description)
        (
            'model_weights/conv1d/conv1d/kernel:0',
            'res_tower.resol_1_conv.conv_layer.weight',
            transpose_conv_weights,
            "Initial DNA convolution"
        ),
        (
            'model_weights/batch_normalization/batch_normalization/gamma:0',
            'res_tower.resol_2_conv.block.0.weight',
            lambda x: x,
            "First ResNet BatchNorm gamma"
        ),
        (
            'model_weights/conv1d_1/conv1d_1/kernel:0',
            'res_tower.resol_2_conv.block.2.weight',
            transpose_conv_weights,
            "First ResNet convolution"
        ),
        (
            'model_weights/multihead_attention/multihead_attention/q_layer/kernel:0',
            'transformer.0.0.fn.1.to_q.weight',
            lambda x: np.transpose(x, (1, 0)),
            "First Transformer Q projection"
        ),
        (
            'model_weights/dense/dense/kernel:0',
            'transformer.0.1.fn.1.weight',
            lambda x: np.transpose(x, (1, 0)),
            "First Transformer FFN layer 1"
        ),
        (
            'model_weights/separable_conv1d/separable_conv1d/depthwise_kernel:0',
            'upsample_tower.resol_64_separable.block.2.0.weight',
            lambda x: np.transpose(x, (1, 2, 0)),
            "First upsample separable depthwise"
        ),
        (
            'model_weights/separable_conv1d/separable_conv1d/pointwise_kernel:0',
            'upsample_tower.resol_64_separable.block.2.1.weight',
            lambda x: np.transpose(x, (2, 1, 0)),
            "First upsample separable pointwise"
        ),
        (
            'model_weights/dense_16/dense_16/kernel:0',
            'upsample_tower.resol_64_x.0.block.2.weight',
            lambda x: np.transpose(x, (1, 0))[:, :, np.newaxis],
            "Upsample resol_64 x path"
        ),
        (
            'model_weights/dense_17/dense_17/kernel:0',
            'upsample_tower.resol_64_horizon.block.2.weight',
            lambda x: np.transpose(x, (1, 0))[:, :, np.newaxis],
            "Upsample resol_64 horizon path"
        ),
        (
            'model_weights/conv1d_7/conv1d_7/kernel:0',
            'final_joined_convs.0.block.2.weight',
            transpose_conv_weights,
            "Final joined convolution"
        ),
        (
            'model_weights/dense_20/dense_20/kernel:0',
            'prediction_head.heads.regression.weight',
            lambda x: np.transpose(x, (1, 0)),
            "Prediction head"
        ),
    ]

    passed = 0
    failed = 0

    for tf_key, pt_key, transform_fn, description in test_cases:
        logger.info(f"\n{description}:")
        logger.info(f"  TF: {tf_key}")
        logger.info(f"  PT: {pt_key}")

        if verify_layer(tf_key, pt_key, tf_weights, pt_state_dict, transform_fn, logger, args.tolerance):
            passed += 1
        else:
            failed += 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Passed: {passed}/{len(test_cases)}")
    logger.info(f"Failed: {failed}/{len(test_cases)}")

    # Check conversion info
    if 'conversion_info' in pt_checkpoint:
        info = pt_checkpoint['conversion_info']
        logger.info(f"\nConversion info:")
        logger.info(f"  Converted tensors: {info.get('converted_tensors', 'N/A')}")
        logger.info(f"  Missing in TF: {info.get('missing_in_tf', 'N/A')}")
        logger.info(f"  Missing in PT: {info.get('missing_in_pt', 'N/A')}")
        logger.info(f"  Shape mismatches: {info.get('shape_mismatches', 'N/A')}")

    logger.info("\n" + "="*80)

    if failed == 0:
        logger.info("✓ ALL WEIGHTS VERIFIED SUCCESSFULLY")
        logger.info("The conversion was successful!")
        sys.exit(0)
    else:
        logger.error("❌ SOME WEIGHTS DO NOT MATCH")
        logger.error("The conversion may have issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
