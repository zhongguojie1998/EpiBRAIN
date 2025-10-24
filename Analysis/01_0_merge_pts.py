#!/usr/bin/env python3
"""
Merge .pt files by split prefix in a folder.

This script merges .pt files that have a common split prefix (e.g., "Test_preds_rank_",
"Train_preds_rank_", "Valid_preds_rank_") into single merged files per split.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import torch
import re


def get_split_prefix(filename):
    """
    Extract the split prefix from a filename.

    For example:
        "Test_preds_rank_0_epoch_0_batch_0.pt" -> "Test_preds"
        "Train_preds_rank_1_epoch_0_batch_5.pt" -> "Train_preds"

    Args:
        filename: Name of the .pt file

    Returns:
        Split prefix string or None if pattern doesn't match
    """
    match = re.match(r'^([A-Za-z]+_preds)_rank_', filename)
    if match:
        return match.group(1)
    return None


def merge_pt_files(input_dir, output_dir=None):
    """
    Merge .pt files by split prefix.

    Args:
        input_dir: Directory containing .pt files to merge
        output_dir: Directory to save merged files (defaults to input_dir)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group files by split prefix
    files_by_split = defaultdict(list)

    for file in sorted(input_path.glob("*.pt")):
        split_prefix = get_split_prefix(file.name)
        if split_prefix:
            files_by_split[split_prefix].append(file)
        else:
            print(f"Skipping file (no matching prefix): {file.name}")

    # Merge files for each split
    for split_prefix, files in files_by_split.items():
        print(f"\nMerging {len(files)} files for split: {split_prefix}")

        merged_data = defaultdict(lambda: defaultdict(list))

        for file_path in sorted(files):
            print(f"  Loading: {file_path.name}")
            data = torch.load(file_path, weights_only=False)

            # Collect data from each key
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        # Handle nested dict (e.g., label/pred containing 'regression')
                        for sub_key, sub_value in value.items():
                            if torch.is_tensor(sub_value):
                                merged_data[key][sub_key].append(sub_value)
                            else:
                                merged_data[key][sub_key].append(torch.tensor(sub_value))
                    elif torch.is_tensor(value):
                        # Handle direct tensor (e.g., index)
                        merged_data[key]['_tensor'].append(value)
                    else:
                        merged_data[key]['_tensor'].append(torch.tensor(value))
            else:
                raise ValueError(f"Unexpected data type in {file_path.name}: {type(data)}")

        # Concatenate tensors for each key
        final_data = {}
        for key, sub_dict in merged_data.items():
            if '_tensor' in sub_dict:
                # Direct tensor, no nested dict
                print(f"  Concatenating {len(sub_dict['_tensor'])} tensors for key '{key}'")
                final_data[key] = torch.cat(sub_dict['_tensor'], dim=0)
                print(f"    Final shape for '{key}': {final_data[key].shape}")
            else:
                # Nested dict structure
                final_data[key] = {}
                for sub_key, tensors in sub_dict.items():
                    print(f"  Concatenating {len(tensors)} tensors for key '{key}.{sub_key}'")
                    final_data[key][sub_key] = torch.cat(tensors, dim=0)
                    print(f"    Final shape for '{key}.{sub_key}': {final_data[key][sub_key].shape}")

        # Save merged file
        output_file = output_path / f"{split_prefix}_merged.pt"
        print(f"  Saving to: {output_file}")
        torch.save(final_data, output_file)
        print(f"  Successfully saved: {output_file.name}")

    print(f"\nMerging complete! Processed {len(files_by_split)} splits.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge .pt files by split prefix in a folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .pt files to merge"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save merged files (defaults to input_dir)"
    )

    args = parser.parse_args()
    merge_pt_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
