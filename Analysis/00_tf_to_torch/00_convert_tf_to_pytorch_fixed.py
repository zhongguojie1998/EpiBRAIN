#!/usr/bin/env python3
"""
Convert TensorFlow model weights (.h5) to PyTorch model weights (.pt)

Usage:
    python convert_tf_to_pytorch_fixed.py \
        --tf_checkpoint /path/to/model_best.h5 \
        --config /path/to/overall_setting.yaml \
        --output /path/to/converted_checkpoint.pt
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
import sys
import yaml
from collections import OrderedDict
import logging
import os

# Set working directory
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../../')
sys.path.append(str(Path(PWD).parent.parent / "Model"))


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def setup_logger():
    """Setup a basic logger for the conversion process."""
    logger = logging.getLogger('convert_tf_to_pytorch')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def transpose_conv_weights(tf_weights):
    """
    Convert TensorFlow conv1d weights to PyTorch format.
    TF: (kernel_size, in_channels, out_channels)
    PT: (out_channels, in_channels, kernel_size)
    """
    return np.transpose(tf_weights, (2, 1, 0))


def load_tf_weights(h5_path):
    """Load all TensorFlow weights from HDF5 file."""
    tf_weights = {}

    with h5py.File(h5_path, 'r') as f:
        def extract_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                tf_weights[name] = np.array(obj)

        f.visititems(extract_weights)

    return tf_weights


def create_weight_mapping(config, tf_weights, logger):
    """
    Create mapping between TensorFlow and PyTorch weights.

    This mapping is based on the actual model architecture from overall_setting.yaml
    and the TensorFlow checkpoint structure.
    """
    mapping = {}

    # Extract model architecture parameters
    res_tower_depth = config['model']['res_tower_param']['depth']
    transformer_depth = config['model']['transformer_param']['depth']
    upsample_layer_num = config['model']['upsample_param']['upsample_layer_num']

    logger.info(f"Model architecture: ResNet depth={res_tower_depth}, Transformer depth={transformer_depth}, Upsample layers={upsample_layer_num}")

    # Calculate resolution levels based on pool_size=2
    pool_size = config['model']['res_tower_param']['pool_size']
    resolutions = [pool_size**i for i in range(res_tower_depth + 1)]  # [1, 2, 4, 8, 16, 32, 64]

    logger.info(f"Resolution levels: {resolutions}")

    # 1. Initial DNA convolution layer (resol_1)
    tf_prefix = 'model_weights/conv1d/conv1d'
    pt_prefix = 'res_tower.resol_1_conv.conv_layer'

    if f'{tf_prefix}/kernel:0' in tf_weights:
        mapping[f'{pt_prefix}.weight'] = {
            'tf_key': f'{tf_prefix}/kernel:0',
            'transform': transpose_conv_weights
        }
        mapping[f'{pt_prefix}.bias'] = {
            'tf_key': f'{tf_prefix}/bias:0',
            'transform': lambda x: x
        }
        logger.info(f"Mapped initial DNA conv: {tf_prefix} -> {pt_prefix}")

    # 2. ResNet tower layers (resol_2, resol_4, resol_8, resol_16, resol_32, resol_64)
    # TF uses conv1d_1 to conv1d_6 and batch_normalization to batch_normalization_5
    for idx in range(res_tower_depth):
        resol = resolutions[idx + 1]  # Skip resol_1 (initial conv)

        # BatchNorm comes BEFORE conv in the block
        # PyTorch: res_tower.resol_{resol}_conv.block.0 = BatchNorm
        #          res_tower.resol_{resol}_conv.block.2 = Conv1d

        # TF BatchNorm naming: batch_normalization, batch_normalization_1, ..., batch_normalization_5
        if idx == 0:
            tf_bn_prefix = 'model_weights/batch_normalization/batch_normalization'
        else:
            tf_bn_prefix = f'model_weights/batch_normalization_{idx}/batch_normalization_{idx}'

        pt_bn_prefix = f'res_tower.resol_{resol}_conv.block.0'

        mapping[f'{pt_bn_prefix}.weight'] = {
            'tf_key': f'{tf_bn_prefix}/gamma:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_bn_prefix}.bias'] = {
            'tf_key': f'{tf_bn_prefix}/beta:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_bn_prefix}.running_mean'] = {
            'tf_key': f'{tf_bn_prefix}/moving_mean:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_bn_prefix}.running_var'] = {
            'tf_key': f'{tf_bn_prefix}/moving_variance:0',
            'transform': lambda x: x
        }

        # Conv parameters
        # TF conv naming: conv1d_1, conv1d_2, ..., conv1d_6
        tf_conv_idx = idx + 1
        tf_conv_prefix = f'model_weights/conv1d_{tf_conv_idx}/conv1d_{tf_conv_idx}'
        pt_conv_prefix = f'res_tower.resol_{resol}_conv.block.2'

        mapping[f'{pt_conv_prefix}.weight'] = {
            'tf_key': f'{tf_conv_prefix}/kernel:0',
            'transform': transpose_conv_weights
        }
        mapping[f'{pt_conv_prefix}.bias'] = {
            'tf_key': f'{tf_conv_prefix}/bias:0',
            'transform': lambda x: x
        }

        logger.info(f"Mapped ResNet block {idx}: resol_{resol} (TF: conv1d_{tf_conv_idx}, bn_{idx})")

    # 3. Transformer layers
    for i in range(transformer_depth):
        # LayerNorm before attention
        # TF naming: layer_normalization, layer_normalization_1, ..., layer_normalization_15
        # For 8 transformer blocks, we have 16 layer norms (2 per block)
        tf_ln_attn_idx = i * 2
        if tf_ln_attn_idx == 0:
            tf_ln_attn_prefix = 'model_weights/layer_normalization/layer_normalization'
        else:
            tf_ln_attn_prefix = f'model_weights/layer_normalization_{tf_ln_attn_idx}/layer_normalization_{tf_ln_attn_idx}'
        pt_ln_attn_prefix = f'transformer.{i}.0.fn.0'

        mapping[f'{pt_ln_attn_prefix}.weight'] = {
            'tf_key': f'{tf_ln_attn_prefix}/gamma:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_ln_attn_prefix}.bias'] = {
            'tf_key': f'{tf_ln_attn_prefix}/beta:0',
            'transform': lambda x: x
        }

        # Attention layer
        # TF naming: multihead_attention, multihead_attention_1, ..., multihead_attention_7
        if i == 0:
            tf_attn_prefix = 'model_weights/multihead_attention/multihead_attention'
        else:
            tf_attn_prefix = f'model_weights/multihead_attention_{i}/multihead_attention_{i}'
        pt_attn_prefix = f'transformer.{i}.0.fn.1'

        # Q, K, V projections (transpose for Linear layers: TF is (in, out), PT is (out, in))
        mapping[f'{pt_attn_prefix}.to_q.weight'] = {
            'tf_key': f'{tf_attn_prefix}/q_layer/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_attn_prefix}.to_k.weight'] = {
            'tf_key': f'{tf_attn_prefix}/k_layer/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_attn_prefix}.to_v.weight'] = {
            'tf_key': f'{tf_attn_prefix}/v_layer/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }

        # Relative position embeddings
        mapping[f'{pt_attn_prefix}.to_rel_k.weight'] = {
            'tf_key': f'{tf_attn_prefix}/r_k_layer/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }

        # Biases
        mapping[f'{pt_attn_prefix}.rel_content_bias'] = {
            'tf_key': f'{tf_attn_prefix}/r_w_bias:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_attn_prefix}.rel_pos_bias'] = {
            'tf_key': f'{tf_attn_prefix}/r_r_bias:0',
            'transform': lambda x: x
        }

        # Output projection
        mapping[f'{pt_attn_prefix}.to_out.weight'] = {
            'tf_key': f'{tf_attn_prefix}/embedding_layer/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_attn_prefix}.to_out.bias'] = {
            'tf_key': f'{tf_attn_prefix}/embedding_layer/bias:0',
            'transform': lambda x: x
        }

        # LayerNorm before FFN
        tf_ln_ffn_idx = i * 2 + 1
        if tf_ln_ffn_idx == 1:
            tf_ln_ffn_prefix = 'model_weights/layer_normalization_1/layer_normalization_1'
        else:
            tf_ln_ffn_prefix = f'model_weights/layer_normalization_{tf_ln_ffn_idx}/layer_normalization_{tf_ln_ffn_idx}'
        pt_ln_ffn_prefix = f'transformer.{i}.1.fn.0'

        mapping[f'{pt_ln_ffn_prefix}.weight'] = {
            'tf_key': f'{tf_ln_ffn_prefix}/gamma:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_ln_ffn_prefix}.bias'] = {
            'tf_key': f'{tf_ln_ffn_prefix}/beta:0',
            'transform': lambda x: x
        }

        # FFN layers
        # TF naming: dense, dense_1, dense_2, ... for first linear in each FFN
        # dense_1, dense_2, dense_3, ... for second linear in each FFN
        tf_dense1_idx = i * 2
        tf_dense2_idx = i * 2 + 1

        if tf_dense1_idx == 0:
            tf_dense1_prefix = 'model_weights/dense/dense'
        else:
            tf_dense1_prefix = f'model_weights/dense_{tf_dense1_idx}/dense_{tf_dense1_idx}'

        tf_dense2_prefix = f'model_weights/dense_{tf_dense2_idx}/dense_{tf_dense2_idx}'

        pt_dense1_prefix = f'transformer.{i}.1.fn.1'
        pt_dense2_prefix = f'transformer.{i}.1.fn.4'

        mapping[f'{pt_dense1_prefix}.weight'] = {
            'tf_key': f'{tf_dense1_prefix}/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_dense1_prefix}.bias'] = {
            'tf_key': f'{tf_dense1_prefix}/bias:0',
            'transform': lambda x: x
        }

        mapping[f'{pt_dense2_prefix}.weight'] = {
            'tf_key': f'{tf_dense2_prefix}/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_dense2_prefix}.bias'] = {
            'tf_key': f'{tf_dense2_prefix}/bias:0',
            'transform': lambda x: x
        }

        logger.info(f"Mapped Transformer block {i}")

    # 4. Upsample tower
    # The upsample tower uses separable convolutions
    # PyTorch structure:
    #   upsample_tower.resol_{resol}_horizon.block.0 = BatchNorm
    #   upsample_tower.resol_{resol}_horizon.block.2 = Conv1d (1x1)
    #   upsample_tower.resol_{resol}_x.0.block.0 = BatchNorm
    #   upsample_tower.resol_{resol}_x.0.block.2 = Conv1d (1x1)
    #   upsample_tower.resol_{resol}_separable.block.0 = depthwise conv
    #   upsample_tower.resol_{resol}_separable.block.1 = pointwise conv

    # The upsample resolution is the reverse of the last `upsample_layer_num` resolutions
    # For upsample_layer_num=2: [64, 32]
    upsample_resolutions = resolutions[::-1][:upsample_layer_num]

    logger.info(f"Upsample resolutions: {upsample_resolutions}")

    # Need to identify which TF layers correspond to upsample tower
    # Based on the checkpoint inspection:
    # - separable_conv1d and separable_conv1d_1 are the separable convolutions
    # - Need to map the associated 1x1 convs and batch norms

    # After the ResNet (6 blocks) and before final conv, we have:
    # - batch_normalization_6 to batch_normalization_10 (5 total, but only some for upsample)
    # Let's inspect the TF checkpoint to understand the upsample structure better

    # For now, let's map the separable convolutions we know exist
    for sep_idx in range(upsample_layer_num):
        resol = upsample_resolutions[sep_idx]

        # Separable conv
        # PyTorch structure: block.2.0 (depthwise), block.2.1 (pointwise)
        if sep_idx == 0:
            tf_sep_prefix = 'model_weights/separable_conv1d/separable_conv1d'
        else:
            tf_sep_prefix = f'model_weights/separable_conv1d_{sep_idx}/separable_conv1d_{sep_idx}'

        pt_sep_prefix = f'upsample_tower.resol_{resol}_separable.block.2'

        # Depthwise kernel
        if f'{tf_sep_prefix}/depthwise_kernel:0' in tf_weights:
            # TF depthwise: (kernel_size, in_channels, multiplier)
            # PT depthwise: (out_channels, 1, kernel_size) where out_channels = in_channels * multiplier
            # For groups=in_channels separable conv, multiplier should be 1
            # So: (kernel_size, in_channels, 1) -> (in_channels, 1, kernel_size)
            mapping[f'{pt_sep_prefix}.0.weight'] = {
                'tf_key': f'{tf_sep_prefix}/depthwise_kernel:0',
                'transform': lambda x: np.transpose(x, (1, 2, 0))  # (K, C, 1) -> (C, 1, K)
            }

        # Pointwise kernel
        if f'{tf_sep_prefix}/pointwise_kernel:0' in tf_weights:
            # TF pointwise: (1, in_channels, out_channels)
            # PT pointwise: (out_channels, in_channels, 1)
            mapping[f'{pt_sep_prefix}.1.weight'] = {
                'tf_key': f'{tf_sep_prefix}/pointwise_kernel:0',
                'transform': lambda x: np.transpose(x, (2, 1, 0))
            }
            mapping[f'{pt_sep_prefix}.1.bias'] = {
                'tf_key': f'{tf_sep_prefix}/bias:0',
                'transform': lambda x: x
            }

        logger.info(f"Mapped upsample separable conv {sep_idx} for resol_{resol}")

    # Map the horizon and x 1x1 convolutions for upsample
    # These are harder to identify without inspecting the full checkpoint structure
    # For now, we'll need to manually identify which conv1d and batch_norm correspond
    # Based on the layer count, after conv1d_6 (last ResNet), we might have more convs

    # Looking at the checkpoint:
    # - batch_normalization_6, _7, _8, _9, _10 are after ResNet
    # - We need to map these to horizon and x convolutions

    # Let's map based on expected structure:
    # For each upsample layer, we have:
    # 1. horizon: 1x1 conv with BatchNorm
    # 2. x: 1x1 conv with BatchNorm + Upsample
    # 3. separable: separable conv (already mapped)

    # This requires more investigation of the TF checkpoint structure
    # For now, we'll leave this partially mapped

    # 5. Final joined convolution
    # This is conv1d_7 based on our checkpoint inspection
    tf_final_conv_idx = 7

    # The final_joined_convs has BatchNorm + GELU + Conv
    # PyTorch: final_joined_convs.0.block.0 = BatchNorm
    #          final_joined_convs.0.block.2 = Conv1d

    # Need to identify which BatchNorm corresponds to this
    # Based on the checkpoint, we have batch_normalization_6 to _10
    # The final one before dense layers is likely for conv1d_7

    # Let's try batch_normalization_10 (the last one)
    tf_final_bn_idx = 10
    tf_final_bn_prefix = f'model_weights/batch_normalization_{tf_final_bn_idx}/batch_normalization_{tf_final_bn_idx}'
    pt_final_bn_prefix = 'final_joined_convs.0.block.0'

    if f'{tf_final_bn_prefix}/gamma:0' in tf_weights:
        mapping[f'{pt_final_bn_prefix}.weight'] = {
            'tf_key': f'{tf_final_bn_prefix}/gamma:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_final_bn_prefix}.bias'] = {
            'tf_key': f'{tf_final_bn_prefix}/beta:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_final_bn_prefix}.running_mean'] = {
            'tf_key': f'{tf_final_bn_prefix}/moving_mean:0',
            'transform': lambda x: x
        }
        mapping[f'{pt_final_bn_prefix}.running_var'] = {
            'tf_key': f'{tf_final_bn_prefix}/moving_variance:0',
            'transform': lambda x: x
        }
        logger.info(f"Mapped final joined conv BatchNorm")

    # Final conv
    tf_final_conv_prefix = f'model_weights/conv1d_{tf_final_conv_idx}/conv1d_{tf_final_conv_idx}'
    pt_final_conv_prefix = 'final_joined_convs.0.block.2'

    if f'{tf_final_conv_prefix}/kernel:0' in tf_weights:
        mapping[f'{pt_final_conv_prefix}.weight'] = {
            'tf_key': f'{tf_final_conv_prefix}/kernel:0',
            'transform': transpose_conv_weights
        }
        mapping[f'{pt_final_conv_prefix}.bias'] = {
            'tf_key': f'{tf_final_conv_prefix}/bias:0',
            'transform': lambda x: x
        }
        logger.info(f"Mapped final joined conv")

    # 6. Upsample tower horizon and x convolutions
    # Based on TF unet_conv block structure (from blocks.py):
    # Each unet_conv has:
    #   1. BatchNorm on current1 (upsampled path from previous layer)
    #   2. BatchNorm on current2 (skip connection from encoder)
    #   3. Dense on current1 (if upsample_conv=True)
    #   4. Dense on current2 (skip connection)
    #   5. Upsample current1
    #   6. Add
    #   7. SeparableConv1D
    #
    # For 2 unet_conv blocks, the order is:
    #   First unet_conv (resol_64):
    #     - BatchNorm: batch_normalization_6 (current1/x)
    #     - BatchNorm: batch_normalization_7 (current2/horizon)
    #     - Dense: dense_16 (current1/x)
    #     - Dense: dense_17 (current2/horizon)
    #   Second unet_conv (resol_32):
    #     - BatchNorm: batch_normalization_8 (current1/x)
    #     - BatchNorm: batch_normalization_9 (current2/horizon)
    #     - Dense: dense_18 (current1/x)
    #     - Dense: dense_19 (current2/horizon)
    #
    # Map upsample tower layers
    upsample_mapping = [
        # resol_64 (first upsample block)
        {
            'resol': 64,
            'x_bn_idx': 6,
            'horizon_bn_idx': 7,
            'x_dense_idx': 16,
            'horizon_dense_idx': 17,
        },
        # resol_32 (second upsample block)
        {
            'resol': 32,
            'x_bn_idx': 8,
            'horizon_bn_idx': 9,
            'x_dense_idx': 18,
            'horizon_dense_idx': 19,
        },
    ]

    for upsample_info in upsample_mapping:
        resol = upsample_info['resol']

        # Map x path BatchNorm
        tf_x_bn_idx = upsample_info['x_bn_idx']
        tf_x_bn_prefix = f'model_weights/batch_normalization_{tf_x_bn_idx}/batch_normalization_{tf_x_bn_idx}'
        pt_x_bn_prefix = f'upsample_tower.resol_{resol}_x.0.block.0'

        if f'{tf_x_bn_prefix}/gamma:0' in tf_weights:
            mapping[f'{pt_x_bn_prefix}.weight'] = {
                'tf_key': f'{tf_x_bn_prefix}/gamma:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_x_bn_prefix}.bias'] = {
                'tf_key': f'{tf_x_bn_prefix}/beta:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_x_bn_prefix}.running_mean'] = {
                'tf_key': f'{tf_x_bn_prefix}/moving_mean:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_x_bn_prefix}.running_var'] = {
                'tf_key': f'{tf_x_bn_prefix}/moving_variance:0',
                'transform': lambda x: x
            }

        # Map horizon path BatchNorm
        tf_horizon_bn_idx = upsample_info['horizon_bn_idx']
        tf_horizon_bn_prefix = f'model_weights/batch_normalization_{tf_horizon_bn_idx}/batch_normalization_{tf_horizon_bn_idx}'
        pt_horizon_bn_prefix = f'upsample_tower.resol_{resol}_horizon.block.0'

        if f'{tf_horizon_bn_prefix}/gamma:0' in tf_weights:
            mapping[f'{pt_horizon_bn_prefix}.weight'] = {
                'tf_key': f'{tf_horizon_bn_prefix}/gamma:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_horizon_bn_prefix}.bias'] = {
                'tf_key': f'{tf_horizon_bn_prefix}/beta:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_horizon_bn_prefix}.running_mean'] = {
                'tf_key': f'{tf_horizon_bn_prefix}/moving_mean:0',
                'transform': lambda x: x
            }
            mapping[f'{pt_horizon_bn_prefix}.running_var'] = {
                'tf_key': f'{tf_horizon_bn_prefix}/moving_variance:0',
                'transform': lambda x: x
            }

        # Map x path Dense (implemented as 1x1 Conv in PyTorch)
        tf_x_dense_idx = upsample_info['x_dense_idx']
        tf_x_dense_prefix = f'model_weights/dense_{tf_x_dense_idx}/dense_{tf_x_dense_idx}'
        pt_x_conv_prefix = f'upsample_tower.resol_{resol}_x.0.block.2'

        if f'{tf_x_dense_prefix}/kernel:0' in tf_weights:
            # TF Dense: (in_features, out_features)
            # PT Conv1d with kernel_size=1: (out_channels, in_channels, 1)
            # We need to add an extra dimension for kernel_size
            mapping[f'{pt_x_conv_prefix}.weight'] = {
                'tf_key': f'{tf_x_dense_prefix}/kernel:0',
                'transform': lambda x: np.transpose(x, (1, 0))[:, :, np.newaxis]
            }
            mapping[f'{pt_x_conv_prefix}.bias'] = {
                'tf_key': f'{tf_x_dense_prefix}/bias:0',
                'transform': lambda x: x
            }

        # Map horizon path Dense (implemented as 1x1 Conv in PyTorch)
        tf_horizon_dense_idx = upsample_info['horizon_dense_idx']
        tf_horizon_dense_prefix = f'model_weights/dense_{tf_horizon_dense_idx}/dense_{tf_horizon_dense_idx}'
        pt_horizon_conv_prefix = f'upsample_tower.resol_{resol}_horizon.block.2'

        if f'{tf_horizon_dense_prefix}/kernel:0' in tf_weights:
            mapping[f'{pt_horizon_conv_prefix}.weight'] = {
                'tf_key': f'{tf_horizon_dense_prefix}/kernel:0',
                'transform': lambda x: np.transpose(x, (1, 0))[:, :, np.newaxis]
            }
            mapping[f'{pt_horizon_conv_prefix}.bias'] = {
                'tf_key': f'{tf_horizon_dense_prefix}/bias:0',
                'transform': lambda x: x
            }

        logger.info(f"Mapped upsample tower resol_{resol} (x: bn_{tf_x_bn_idx}/dense_{tf_x_dense_idx}, horizon: bn_{tf_horizon_bn_idx}/dense_{tf_horizon_dense_idx})")

    # 7. Prediction head
    # Based on checkpoint inspection: dense_20 has shape (1920, 199)
    # This matches prediction_head.heads.regression: Linear(1920, 199)

    tf_head_dense_idx = 20
    tf_head_prefix = f'model_weights/dense_{tf_head_dense_idx}/dense_{tf_head_dense_idx}'
    pt_head_prefix = 'prediction_head.heads.regression'

    if f'{tf_head_prefix}/kernel:0' in tf_weights:
        mapping[f'{pt_head_prefix}.weight'] = {
            'tf_key': f'{tf_head_prefix}/kernel:0',
            'transform': lambda x: np.transpose(x, (1, 0))
        }
        mapping[f'{pt_head_prefix}.bias'] = {
            'tf_key': f'{tf_head_prefix}/bias:0',
            'transform': lambda x: x
        }
        logger.info(f"Mapped prediction head (dense_{tf_head_dense_idx})")

    return mapping


def convert_weights(tf_checkpoint_path, config_path, output_path):
    """Convert TensorFlow weights to PyTorch format."""

    logger = setup_logger()
    logger.info("="*80)
    logger.info("TensorFlow to PyTorch Conversion")
    logger.info("="*80)

    logger.info(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)

    logger.info(f"\nLoading TensorFlow weights from {tf_checkpoint_path}...")
    tf_weights = load_tf_weights(tf_checkpoint_path)
    logger.info(f"Loaded {len(tf_weights)} TensorFlow weight tensors")

    logger.info("\nCreating PyTorch model from config...")
    from model.model_utils import setup_model
    from types import SimpleNamespace

    # Convert config to SimpleNamespace for compatibility
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
        return d

    config_ns = dict_to_namespace(config)

    # Temporarily disable checkpoint loading for model creation
    original_load_checkpoint = config_ns.training.load_checkpoint
    config_ns.training.load_checkpoint = None

    # Create model from config
    model = setup_model(config_ns, logger, checkpoint=None)

    # Restore original checkpoint setting
    config_ns.training.load_checkpoint = original_load_checkpoint

    # Get model state dict as template
    pt_state_dict = model.state_dict()
    logger.info(f"Created PyTorch model with {len(pt_state_dict)} parameters")

    logger.info("\nCreating weight mapping...")
    mapping = create_weight_mapping(config, tf_weights, logger)

    logger.info(f"\n{'='*80}")
    logger.info(f"Converting {len(mapping)} weight tensors...")
    logger.info(f"{'='*80}\n")

    converted_count = 0
    missing_in_tf = []
    missing_in_pt = []
    shape_mismatches = []

    new_state_dict = OrderedDict()

    # Copy all existing PyTorch weights first
    for k, v in pt_state_dict.items():
        if torch.is_tensor(v):
            new_state_dict[k] = v.clone()

    # Convert mapped weights
    for pt_key, mapping_info in mapping.items():
        tf_key = mapping_info['tf_key']
        transform_fn = mapping_info['transform']

        if tf_key not in tf_weights:
            missing_in_tf.append((pt_key, tf_key))
            continue

        if pt_key not in pt_state_dict:
            missing_in_pt.append((pt_key, tf_key))
            continue

        # Get TF weight and transform
        tf_weight = tf_weights[tf_key]
        converted_weight = transform_fn(tf_weight)

        # Convert to PyTorch tensor
        pt_tensor = torch.from_numpy(converted_weight).float()

        # Check shape compatibility
        expected_shape = pt_state_dict[pt_key].shape
        if pt_tensor.shape != expected_shape:
            shape_mismatches.append({
                'pt_key': pt_key,
                'tf_key': tf_key,
                'expected': expected_shape,
                'got': pt_tensor.shape
            })
            continue

        # Update state dict
        new_state_dict[pt_key] = pt_tensor
        converted_count += 1
        logger.debug(f"✓ {pt_key}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Conversion Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully converted: {converted_count} tensors")

    if missing_in_tf:
        logger.warning(f"\n⚠ {len(missing_in_tf)} weights not found in TensorFlow checkpoint:")
        for pt_key, tf_key in missing_in_tf[:10]:
            logger.warning(f"  PT: {pt_key}")
            logger.warning(f"  TF: {tf_key}")
        if len(missing_in_tf) > 10:
            logger.warning(f"  ... and {len(missing_in_tf) - 10} more")

    if missing_in_pt:
        logger.warning(f"\n⚠ {len(missing_in_pt)} weights not found in PyTorch model:")
        for pt_key, tf_key in missing_in_pt[:10]:
            logger.warning(f"  PT: {pt_key}")
            logger.warning(f"  TF: {tf_key}")
        if len(missing_in_pt) > 10:
            logger.warning(f"  ... and {len(missing_in_pt) - 10} more")

    if shape_mismatches:
        logger.error(f"\n❌ {len(shape_mismatches)} shape mismatches:")
        for mismatch in shape_mismatches[:10]:
            logger.error(f"  {mismatch['pt_key']}:")
            logger.error(f"    TF: {mismatch['tf_key']}")
            logger.error(f"    Expected: {mismatch['expected']}, Got: {mismatch['got']}")
        if len(shape_mismatches) > 10:
            logger.error(f"  ... and {len(shape_mismatches) - 10} more")

    # Save converted checkpoint
    logger.info(f"\nSaving converted checkpoint to {output_path}...")
    checkpoint = {
        'model_state_dict': new_state_dict,
        'epoch': 0,
        'step': 0,
        'best_valid_loss': float('inf'),
        'conversion_info': {
            'converted_from': 'tensorflow',
            'tf_checkpoint': str(tf_checkpoint_path),
            'config': str(config_path),
            'converted_tensors': converted_count,
            'missing_in_tf': len(missing_in_tf),
            'missing_in_pt': len(missing_in_pt),
            'shape_mismatches': len(shape_mismatches),
        }
    }
    torch.save(checkpoint, output_path)

    logger.info(f"\n{'='*80}")
    logger.info("✓ Conversion complete!")
    logger.info(f"{'='*80}\n")

    return {
        'converted': converted_count,
        'missing_in_tf': len(missing_in_tf),
        'missing_in_pt': len(missing_in_pt),
        'shape_mismatches': len(shape_mismatches)
    }


def main():
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to PyTorch')
    parser.add_argument('--tf_checkpoint', type=str, required=True,
                        help='Path to TensorFlow .h5 checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML file (overall_setting.yaml)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save converted PyTorch checkpoint')

    args = parser.parse_args()

    # Verify input files exist
    for path, name in [(args.tf_checkpoint, 'TensorFlow checkpoint'),
                       (args.config, 'Config file')]:
        if not Path(path).exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Run conversion
    stats = convert_weights(
        args.tf_checkpoint,
        args.config,
        args.output
    )

    print(f"\nFinal Statistics:")
    print(f"  Converted: {stats['converted']}")
    print(f"  Missing in TF: {stats['missing_in_tf']}")
    print(f"  Missing in PT: {stats['missing_in_pt']}")
    print(f"  Shape mismatches: {stats['shape_mismatches']}")

    if stats['shape_mismatches'] > 0:
        print(f"\n⚠ WARNING: There were shape mismatches. Please review the conversion.")
        sys.exit(1)

    if stats['missing_in_tf'] > 0 or stats['missing_in_pt'] > 0:
        print(f"\n⚠ WARNING: Some weights were not converted. This may be expected for partial conversion.")


if __name__ == "__main__":
    main()
