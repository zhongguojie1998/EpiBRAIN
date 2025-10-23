#!/usr/bin/env python3
"""
Convert TensorFlow model weights (.h5) to PyTorch model weights (.pt)

Usage:
    python convert_tf_to_pytorch.py \
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
from types import SimpleNamespace


def load_config(config_path):
    """Load YAML config file and convert to nested namespace."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dict to nested namespace for compatibility with setup_model
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(config_dict)


def namespace_to_dict(obj):
    """Convert nested SimpleNamespace back to dictionary."""
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj


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


def create_weight_mapping(transformer_depth, tf_weights, pt_model):
    """Create mapping between TensorFlow and PyTorch weights."""
    mapping = {}

    # Use transformer depth directly (passed as parameter)

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

    # 2. ResNet tower layers (resol_2, resol_4, resol_8, resol_16, resol_32, resol_64)
    resolutions = [2, 4, 8, 16, 32, 64]
    for idx, resol in enumerate(resolutions):
        # BatchNorm comes BEFORE conv in the block, so it normalizes the INPUT
        # For resol_2: input is 512 (from conv1d output), so BatchNorm is batch_normalization (no suffix)
        # For resol_4: input is 608 (from conv1d_1 output), so BatchNorm is batch_normalization_1
        # TF: first BatchNorm has no number, subsequent ones are _1, _2, etc.
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
        # Conv indices are _1, _2, _3, etc. (no conv without number for resblocks)
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

    # 3. Transformer layers
    for i in range(transformer_depth):
        # LayerNorm before attention
        # TF naming: layer_normalization, layer_normalization_1, layer_normalization_2, ...
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
        # TF naming: multihead_attention, multihead_attention_1, multihead_attention_2, ...
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

        # Biases (reshape from (1, heads, 1, dim) to (1, heads, 1, dim))
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

    # 4. Upsample tower
    # Note: The upsample tower uses additional convolutions and separable convolutions
    # These would need to be mapped if they exist in the TensorFlow model
    # For now, we skip this as the mapping is complex and model-specific
    # The core architecture (ResNet + Transformer) has been successfully mapped

    # 5. Final joined convolution
    tf_final_conv_idx = 7  # Typically conv1d_7 based on the inspection

    # BatchNorm before final conv
    tf_final_bn_idx = 7  # batch_normalization_7
    tf_final_bn_prefix = f'model_weights/batch_normalization_{tf_final_bn_idx}/batch_normalization_{tf_final_bn_idx}'
    pt_final_bn_prefix = 'final_joined_convs.0.block.0'

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

    # Final conv
    tf_final_conv_prefix = f'model_weights/conv1d_{tf_final_conv_idx}/conv1d_{tf_final_conv_idx}'
    pt_final_conv_prefix = 'final_joined_convs.0.block.2'

    mapping[f'{pt_final_conv_prefix}.weight'] = {
        'tf_key': f'{tf_final_conv_prefix}/kernel:0',
        'transform': transpose_conv_weights
    }
    mapping[f'{pt_final_conv_prefix}.bias'] = {
        'tf_key': f'{tf_final_conv_prefix}/bias:0',
        'transform': lambda x: x
    }

    return mapping


def convert_weights(tf_checkpoint_path, config_path, output_path):
    """Convert TensorFlow weights to PyTorch format."""

    print("Loading configuration...")
    config = load_config(config_path)

    print("Setting up logger...")
    logger = setup_logger()

    print("Creating PyTorch model from config...")
    # Add Model directory to Python path to import model modules
    sys.path.insert(0, str(Path(__file__).parent / "Model"))
    from model.model_utils import setup_model

    # Temporarily disable checkpoint loading for model creation
    original_load_checkpoint = config.training.load_checkpoint
    config.training.load_checkpoint = None

    # Create model from config
    model = setup_model(config, logger, checkpoint=None)

    # Restore original checkpoint setting
    config.training.load_checkpoint = original_load_checkpoint

    # Get model state dict as template
    pt_state_dict = model.state_dict()
    print(f"Created model with {len(pt_state_dict)} parameters")

    print("\nLoading TensorFlow weights...")
    tf_weights = load_tf_weights(tf_checkpoint_path)
    print(f"Loaded {len(tf_weights)} TensorFlow weight tensors")

    print("Creating weight mapping...")
    # Extract transformer depth from config for mapping
    transformer_depth = config.model.transformer_param.depth
    mapping = create_weight_mapping(transformer_depth, tf_weights, model)

    print(f"\nConverting {len(mapping)} weight tensors...")
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

    print(f"\nSuccessfully converted: {converted_count} tensors")

    if missing_in_tf:
        print(f"\nWarning: {len(missing_in_tf)} weights not found in TensorFlow checkpoint:")
        for pt_key, tf_key in missing_in_tf[:5]:
            print(f"  PT: {pt_key}")
            print(f"  TF: {tf_key}")
        if len(missing_in_tf) > 5:
            print(f"  ... and {len(missing_in_tf) - 5} more")

    if missing_in_pt:
        print(f"\nWarning: {len(missing_in_pt)} weights not found in PyTorch checkpoint:")
        for pt_key, tf_key in missing_in_pt[:5]:
            print(f"  PT: {pt_key}")
            print(f"  TF: {tf_key}")
        if len(missing_in_pt) > 5:
            print(f"  ... and {len(missing_in_pt) - 5} more")

    if shape_mismatches:
        print(f"\nError: {len(shape_mismatches)} shape mismatches:")
        for mismatch in shape_mismatches[:5]:
            print(f"  {mismatch['pt_key']}:")
            print(f"    TF: {mismatch['tf_key']}")
            print(f"    Expected: {mismatch['expected']}, Got: {mismatch['got']}")
        if len(shape_mismatches) > 5:
            print(f"  ... and {len(shape_mismatches) - 5} more")

    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to {output_path}...")
    checkpoint = {
        'model_state_dict': new_state_dict,
        'epoch': 0,
        'conversion_info': {
            'converted_from': 'tensorflow',
            'tf_checkpoint': str(tf_checkpoint_path),
            'config': str(config_path),
        }
    }
    torch.save(checkpoint, output_path)

    print("Conversion complete!")

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
                        help='Path to model config YAML file')
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

    print(f"\nConversion statistics:")
    print(f"  Converted: {stats['converted']}")
    print(f"  Missing in TF: {stats['missing_in_tf']}")
    print(f"  Missing in PT: {stats['missing_in_pt']}")
    print(f"  Shape mismatches: {stats['shape_mismatches']}")


if __name__ == "__main__":
    main()
