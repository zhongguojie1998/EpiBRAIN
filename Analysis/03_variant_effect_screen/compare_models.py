#!/usr/bin/env python3
"""
Model Comparison Script

Compares the differences between:
1. compute.py model loading approach
2. vep_enformer_borzoi.py grelu model loading approach

This script analyzes model architectures, parameters, and prediction capabilities.
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from gpn.data import Genome

# Add paths for compute.py imports
os.chdir("/gpfs/commons/groups/ren_lab/guojiezhong/BICAN")
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))

# Suppress warnings
warnings.filterwarnings("ignore")

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

def load_compute_model(model_path, config_path=None, device='cpu'):
    """Load model using compute.py approach"""
    # Import compute.py functions
    from data.tokenizer import FastaInterval
    from model.model_utils import setup_model
    from utils.config import load_config
    from utils.logging import BaseLogger
    import logging
    import os

    if config_path is None:
        print("Loading Prebuilt Model (compute.py style)")
        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
        model = model_package.model.eval().to(device)
        dna_tokenizer = model_package.dna_tokenizer
        config = model_package.config
    else:
        print("Loading Runtime Built Model (compute.py style)")
        config = load_config(config_name=config_path)
        logger = BaseLogger(name="Model packaging", level=logging.INFO)
        checkpoint_data = torch.load(model_path, map_location="cpu")
        model = setup_model(config, logger=logger)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval().to(device)
        dna_tokenizer = FastaInterval(
            fasta_file=os.path.abspath(config.data.refer_genom),
            context_length=config.data.context_length
        )

    return model, dna_tokenizer, config

def load_grelu_model(project, model_name, device='cpu'):
    """Load model using grelu approach"""
    try:
        import grelu.resources
        from grelu.sequence.format import strings_to_one_hot

        print(f"Loading GRELU Model: {project}/{model_name}")
        model = grelu.resources.load_model(project=project, model_name=model_name)

        # Wrap in VEPModel class (from vep_enformer_borzoi.py)
        class VEPModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def get_scores(self, x_ref, x_alt):
                y_ref = self.model(x_ref)
                y_alt = self.model(x_alt)
                lfc = torch.log2(1 + y_alt) - torch.log2(1 + y_ref)
                l2 = torch.linalg.norm(lfc, dim=2)
                return l2

            def forward(self, x_ref_fwd=None, x_alt_fwd=None, x_ref_rev=None, x_alt_rev=None):
                fwd = self.get_scores(x_ref_fwd, x_alt_fwd)
                rev = self.get_scores(x_ref_rev, x_alt_rev)
                return (fwd + rev) / 2

        columns = model.data_params['tasks']["name"]
        window_size = model.data_params["train_seq_len"]
        wrapped_model = VEPModel(model.model).eval().to(device)

        return wrapped_model, model, columns, window_size, strings_to_one_hot

    except ImportError as e:
        print(f"GRELU import failed: {e}")
        return None, None, None, None, None

def compare_model_architectures(compute_model, grelu_model_obj):
    """Compare model architectures"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*60)

    # Compute model analysis
    print("\n--- COMPUTE.PY MODEL ---")
    print(f"Model type: {type(compute_model)}")
    print(f"Total parameters: {sum(p.numel() for p in compute_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in compute_model.parameters() if p.requires_grad):,}")

    # Model structure
    print("\nModel structure:")
    for name, module in compute_model.named_children():
        print(f"  {name}: {type(module)}")

    # GRELU model analysis
    if grelu_model_obj is not None:
        print("\n--- GRELU MODEL ---")
        print(f"Model type: {type(grelu_model_obj.model)}")
        print(f"Total parameters: {sum(p.numel() for p in grelu_model_obj.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in grelu_model_obj.model.parameters() if p.requires_grad):,}")

        print(f"Data params keys: {list(grelu_model_obj.data_params.keys())}")
        print(f"Tasks: {grelu_model_obj.data_params.get('tasks', {}).get('name', 'Not found')}")
        print(f"Sequence length: {grelu_model_obj.data_params.get('train', {}).get('seq_len', 'Not found')}")

        print("\nModel structure:")
        for name, module in grelu_model_obj.model.named_children():
            print(f"  {name}: {type(module)}")

def compare_tokenization(compute_tokenizer, grelu_strings_to_one_hot, genome=None, window_size=131072):
    """Compare tokenization approaches and return tokenized sequences"""
    print("\n" + "="*60)
    print("TOKENIZATION COMPARISON")
    print("="*60)

    # Test parameters similar to what's used in compute.py VariantDataset
    test_chr_name = "chr1"
    test_pos = 1425822  # 1-based position
    test_start = int(test_pos - 524288/2)  # Convert to 0-based for compute.py
    test_end = int(test_pos + 524288/2)

    # Store tokenization results
    compute_tokens = None
    grelu_tokens = None

    print("\n--- COMPUTE.PY TOKENIZER ---")
    print(f"Tokenizer type: {type(compute_tokenizer)}")
    if hasattr(compute_tokenizer, 'context_length'):
        print(f"Context length: {compute_tokenizer.context_length}")

    # Test with chr_name, start, end parameters (like in compute.py)
    if hasattr(compute_tokenizer, '__call__'):
        try:
            print(f"Testing with chr_name={test_chr_name}, start={test_start}, end={test_end}")
            token_dict = compute_tokenizer(
                chr_name=test_chr_name,
                start=test_start,
                end=test_end,
                return_augs=False,
                return_rela_idx=True
            )
            print("Compute tokenization successful")
            print(f"Token dict keys: {list(token_dict.keys())}")
            if 'one_hot' in token_dict:
                print(f"One-hot tensor shape: {token_dict['one_hot'].shape}")
            if 'rela_idx' in token_dict:
                print(f"Relative index: {token_dict['rela_idx']}")
            compute_tokens = token_dict
        except Exception as e:
            print(f"Compute tokenization with chr_name/start/end failed: {e}")

    print("\n--- GRELU TOKENIZER ---")
    if grelu_strings_to_one_hot is not None:
        print(f"Function: {grelu_strings_to_one_hot}")

        # Try to fetch sequence using genome.get_seq_fwd_rev like in vep_enformer_borzoi.py
        if genome is not None:
            try:
                print(f"Testing with genome sequence fetching for chr{test_chr_name}:{test_pos}")
                # Use the same approach as vep_enformer_borzoi.py lines 51-79
                start_window = test_pos - window_size // 2
                end_window = test_pos + window_size // 2

                seq_fwd, seq_rev = genome.get_seq_fwd_rev(test_chr_name, start_window, end_window)
                print(f"Retrieved sequence length: {len(seq_fwd)}")
                print(f"Expected window size: {window_size}")

                # Convert to one-hot encoding
                grelu_encoded_fwd = grelu_strings_to_one_hot([seq_fwd.upper()])
                grelu_encoded_rev = grelu_strings_to_one_hot([seq_rev.upper()])
                print(f"GRELU forward sequence shape: {grelu_encoded_fwd.shape}")
                print(f"GRELU reverse sequence shape: {grelu_encoded_rev.shape}")

                grelu_tokens = {
                    'seq_fwd': seq_fwd,
                    'seq_rev': seq_rev,
                    'encoded_fwd': grelu_encoded_fwd,
                    'encoded_rev': grelu_encoded_rev
                }

            except Exception as e:
                print(f"GRELU genome sequence fetching failed: {e}")
        else:
            print(f"genome object not provided, cannot test sequence fetching")
    else:
        print("GRELU strings_to_one_hot function not available")

    return compute_tokens, grelu_tokens

def compare_predictions(compute_model, compute_tokens, grelu_wrapped_model, grelu_tokens):
    """Compare model predictions using pre-tokenized sequences"""
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)

    try:
        # Compute.py prediction
        print("\n--- COMPUTE.PY PREDICTION ---")
        if compute_model is not None and compute_tokens is not None:
            try:
                print("Using pre-tokenized compute sequences")
                print(f"Token dict keys: {list(compute_tokens.keys())}")

                if 'one_hot' in compute_tokens:
                    with torch.no_grad():
                        # Run model inference
                        input_tensor = compute_tokens['one_hot'].unsqueeze(0)  # Add batch dimension
                        prediction1 = compute_model(input_tensor.permute(0, 2, 1))  # Permute to (B, C, L) if needed
                        print(f"Compute model output shape: {prediction1.shape}")
                        print(f"Compute model output type: {type(prediction1)}")

            except Exception as e:
                print(f"Compute model prediction failed: {e}")
        else:
            print("Compute model or tokens not available")

        # GRELU prediction
        print("\n--- GRELU PREDICTION ---")
        if grelu_wrapped_model is not None and grelu_tokens is not None:
            try:
                print("Using pre-tokenized GRELU sequences")
                print(f"GRELU token keys: {list(grelu_tokens.keys())}")

                if 'encoded_fwd' in grelu_tokens and 'encoded_rev' in grelu_tokens:
                    with torch.no_grad():
                        # For variant effect prediction, we need ref and alt sequences
                        # For now, just test with the sequences we have
                        fwd_tensor = grelu_tokens['encoded_fwd']
                        rev_tensor = grelu_tokens['encoded_rev']

                        print(f"GRELU forward tensor shape: {fwd_tensor.shape}")
                        print(f"GRELU reverse tensor shape: {rev_tensor.shape}")

                        # Test model inference on forward sequence
                        prediction2 = grelu_wrapped_model.model(fwd_tensor)
                        print(f"GRELU model output shape: {prediction2.shape}")
                        print(f"GRELU model output type: {type(prediction2)}")

            except Exception as e:
                print(f"GRELU model prediction failed: {e}")
        else:
            print("GRELU model or tokens not available")

    except Exception as e:
        print(f"Prediction comparison failed: {e}")
    return prediction1, prediction2

def main():
    parser = argparse.ArgumentParser(description="Compare compute.py vs grelu model approaches")
    parser.add_argument("--compute_model", type=str, required=True,
                       help="Path to compute.py model (pickle file or checkpoint)")
    parser.add_argument("--compute_config", type=str, default=None,
                       help="Config for compute.py model (if using checkpoint)")
    parser.add_argument("--grelu_project", type=str, required=True,
                       help="GRELU project name")
    parser.add_argument("--grelu_model", type=str, required=True,
                       help="GRELU model name")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--genome_path", type=str,
                       default="/gpfs/commons/groups/ren_lab/guojiezhong/Data/Ref/hg38/hg38.fa",
                       help="Path to reference genome FASTA file")
    parser.add_argument("--output", type=str, default="model_comparison_report.txt",
                       help="Output report file")

    args = parser.parse_args()

    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    print(f"Compute model: {args.compute_model}")
    print(f"GRELU model: {args.grelu_project}/{args.grelu_model}")
    print(f"Device: {args.device}")

    # Load compute.py model
    try:
        compute_model, compute_tokenizer, compute_config = load_compute_model(
            args.compute_model, args.compute_config, args.device
        )
        print("✓ Compute model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load compute model: {e}")
        compute_model, compute_tokenizer, compute_config = None, None, None

    # Load GRELU model
    try:
        grelu_wrapped_model, grelu_model_obj, columns, window_size, strings_to_one_hot = load_grelu_model(
            args.grelu_project, args.grelu_model, args.device
        )
        if grelu_wrapped_model is not None:
            print("✓ GRELU model loaded successfully")
        else:
            print("✗ GRELU model loading failed")
    except Exception as e:
        print(f"✗ Failed to load GRELU model: {e}")
        grelu_wrapped_model, grelu_model_obj, columns, window_size, strings_to_one_hot = None, None, None, None, None

    # Perform comparisons
    if compute_model is not None or grelu_model_obj is not None:
        # Architecture comparison
        compare_model_architectures(compute_model, grelu_model_obj)

        # Tokenization comparison
        compute_tokens, grelu_tokens = None, None
        if compute_tokenizer is not None or strings_to_one_hot is not None:
            # Try to create genome object for sequence fetching
            genome = None
            try:
                if os.path.exists(args.genome_path):
                    genome = Genome(args.genome_path)
                    print(f"Loaded genome from: {args.genome_path}")
                else:
                    print(f"Genome file not found at: {args.genome_path}")
            except Exception as e:
                print(f"Failed to load genome: {e}")

            compute_tokens, grelu_tokens = compare_tokenization(compute_tokenizer, strings_to_one_hot, genome, window_size)

        # Prediction comparison
        if compute_model is not None and grelu_wrapped_model is not None:
            prediction1, prediction2 = compare_predictions(
                compute_model, compute_tokens,
                grelu_wrapped_model, grelu_tokens
            )
            # check if predictions are similar
            if prediction1 is not None and prediction2 is not None:
                try:
                    pred1_np = prediction1.permute(0, 2, 1).cpu().numpy()
                    pred2_np = prediction2.cpu().numpy()
                    if pred1_np.shape == pred2_np.shape:
                        diff = np.abs(pred1_np - pred2_np)
                        mean_diff = np.mean(diff)
                        max_diff = np.max(diff)
                        print(f"\nPrediction difference stats:")
                        print(f"  Mean absolute difference: {mean_diff:.6f}")
                        print(f"  Max absolute difference: {max_diff:.6f}")
                    else:
                        print("Prediction shapes differ, cannot compute difference")
                except Exception as e:
                    print(f"Failed to compare predictions: {e}")

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()