#!/usr/bin/env python3
"""
Prediction Difference Viewer

This script loads checkpoint predicted results, maps them to genomic coordinates,
and visualizes prediction differences across the genome using Manhattan-style plots.
Uses lazy data generation and HDF5 storage for efficiency.
"""

import os
import warnings
from typing import Any, Dict, List, Optional
import multiprocessing as mp

warnings.filterwarnings("ignore")

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class PredictionDifferenceProcessor:
    """Process and store prediction results with genomic coordinates."""

    def __init__(
        self,
        data_base: str,
        res_base: str,
        exp_name: str,
        chk: str,
        context_length: int = 196608,
        central_bin_num: int = 896,
        window_size: int = 128,
    ):
        self.data_base = os.path.abspath(data_base)
        self.res_base = os.path.abspath(res_base)
        self.exp_name = exp_name
        self.chk = chk
        self.context_length = context_length
        self.central_bin_num = central_bin_num
        self.window_size = window_size

        # Load metadata
        self.label_meta = pd.read_csv(f"{self.data_base}/label_meta.csv", index_col=1)
        self.sequences = pd.read_csv(f"{self.data_base}/sequences.bed", sep="\t", header=None)
        self.sequences.columns = ["chrom", "start", "end", "split"]

        # Create output directories
        self.output_dir = f"{self.res_base}/{self.exp_name}/analysis_{self.chk}"
        os.makedirs(f"{self.output_dir}/plot/genome_view", exist_ok=True)
        os.makedirs(f"{self.output_dir}/raw_data", exist_ok=True)

    def load_predictions(self, split: str) -> Dict[str, torch.Tensor]:
        """Load prediction results for a given split."""
        pred_file = f"{self.res_base}/{self.exp_name}/{split}_preds_epoch_{self.chk}.pt"
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        print(f"Loading predictions from {pred_file}")
        return torch.load(pred_file, map_location="cpu")

    def calculate_genomic_coordinates(self, indices: torch.Tensor, split: str) -> pd.DataFrame:
        """Calculate genomic coordinates following the same logic as get_labels in data_utils."""
        split_sequences = self.sequences[self.sequences["split"] == split.lower()].reset_index(drop=True)

        coords_list = []
        for idx in tqdm(indices, desc="Calculating genomic coordinates"):
            seq_row = split_sequences.iloc[int(idx)]
            chrom, seq_start, seq_end = seq_row["chrom"], seq_row["start"], seq_row["end"]

            # Calculate sequence length and target length (following get_labels logic)
            seq_len_nt = seq_end - seq_start
            target_length = seq_len_nt // self.window_size

            # Calculate start position for central bins (following get_labels trimming logic)
            trim = (target_length - self.central_bin_num) // 2
            central_start = seq_start + trim * self.window_size

            # Generate coordinates for each bin
            for bin_idx in range(self.central_bin_num):
                bin_start = central_start + bin_idx * self.window_size
                bin_end = bin_start + self.window_size
                bin_midpoint = (bin_start + bin_end) / 2

                coords_list.append(
                    {
                        "seq_idx": int(idx),
                        "bin_idx": bin_idx,
                        "chrom": chrom,
                        "start": bin_start,
                        "end": bin_end,
                        "midpoint": bin_midpoint,
                    }
                )

        return pd.DataFrame(coords_list)

    def process_and_save_prediction_differences(self, split: str) -> str:
        """Process predictions and save prediction differences in HDF5 format."""
        output_file = f"{self.output_dir}/raw_data/{split}_prediction_differences.h5"

        if os.path.exists(output_file):
            print(f"Data file already exists: {output_file}")
            return output_file

        print(f"Processing prediction differences for {split}")

        # Load predictions
        pred_data = self.load_predictions(split)
        labels = pred_data["label"]  # shape: [test_size, central_bin_num, trial_num]
        preds = pred_data["pred"]  # shape: [test_size, central_bin_num, trial_num]
        indices = pred_data["index"]  # shape: [test_size]

        print(f"Data shapes - Labels: {labels.shape}, Preds: {preds.shape}, Indices: {indices.shape}")

        # Calculate absolute differences
        diff = preds - labels  # shape: [test_size, central_bin_num, trial_num]

        # Calculate genomic coordinates
        coord_df = self.calculate_genomic_coordinates(indices, split)

        # Reshape data to [test_size * central_bin_num, trial_num]
        test_size, central_bin_num, trial_num = labels.shape

        labels_reshaped = labels.reshape(-1, trial_num)
        preds_reshaped = preds.reshape(-1, trial_num)
        diff_reshaped = diff.reshape(-1, trial_num)

        # Save to HDF5 without redundant metadata
        with h5py.File(output_file, "w") as f:
            # Save genomic coordinates
            coord_group = f.create_group("coordinates")
            coord_group.create_dataset("chrom", data=coord_df["chrom"].astype("S"))
            coord_group.create_dataset("start", data=coord_df["start"].values)
            coord_group.create_dataset("end", data=coord_df["end"].values)
            coord_group.create_dataset("midpoint", data=coord_df["midpoint"].values)
            coord_group.create_dataset("seq_idx", data=coord_df["seq_idx"].values)
            coord_group.create_dataset("bin_idx", data=coord_df["bin_idx"].values)

            # Save prediction data
            data_group = f.create_group("data")
            data_group.create_dataset("labels", data=labels_reshaped.numpy(), compression="gzip")
            data_group.create_dataset("predictions", data=preds_reshaped.numpy(), compression="gzip")
            data_group.create_dataset("diff", data=diff_reshaped.numpy(), compression="gzip")

            # Save essential attributes only
            f.attrs["test_size"] = test_size
            f.attrs["trial_num"] = trial_num
            f.attrs["central_bin_num"] = central_bin_num
            f.attrs["window_size"] = self.window_size
            f.attrs["context_length"] = self.context_length

        print(f"Data saved to {output_file}")
        return output_file

    def load_data_for_trials(self, split: str, trial_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load processed data for specific trials only."""
        data_file = f"{self.output_dir}/raw_data/{split}_prediction_differences.h5"
        if not os.path.exists(data_file):
            data_file = self.process_and_save_prediction_differences(split)

        # Get trial indices from label_meta
        if trial_names is not None:
            trial_indices = []
            for trial_name in trial_names:
                if trial_name in self.label_meta.index:
                    trial_idx = self.label_meta.loc[trial_name, "dim"]
                    trial_indices.append(trial_idx)
                else:
                    print(f"Warning: Trial {trial_name} not found in label_meta")
            if not trial_indices:
                raise ValueError(f"No valid trials found from {trial_names}")
        else:
            trial_indices = list(range(len(self.label_meta)))

        with h5py.File(data_file, "r") as f:
            coords = {
                "chrom": f["coordinates/chrom"][:].astype(str),
                "start": f["coordinates/start"][:],
                "end": f["coordinates/end"][:],
                "midpoint": f["coordinates/midpoint"][:],
                "seq_idx": f["coordinates/seq_idx"][:],
                "bin_idx": f["coordinates/bin_idx"][:],
            }

            # Load only specified trials
            data = {
                "labels": f["data/labels"][:, trial_indices],
                "predictions": f["data/predictions"][:, trial_indices],
                "diff": f["data/diff"][:, trial_indices],
            }

            metadata = {
                "trial_names": self.label_meta[self.label_meta["dim"].isin(trial_indices)].index,
                "trial_indices": trial_indices,
                "test_size": f.attrs["test_size"],
                "central_bin_num": f.attrs["central_bin_num"],
                "trial_num": len(trial_indices),
                "window_size": f.attrs["window_size"],
                "context_length": f.attrs["context_length"],
            }

        return {"coordinates": coords, "data": data, "metadata": metadata}

    def filter_by_region(
        self,
        data: Dict[str, Any],
        chrom: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Filter data by genomic region."""
        coords = data["coordinates"]

        mask = np.ones(len(coords["chrom"]), dtype=bool)

        if chrom is not None:
            chrom_mask = coords["chrom"] == chrom
            mask = mask & chrom_mask

        if start is not None:
            start_mask = coords["start"] >= start
            mask = mask & start_mask

        if end is not None:
            end_mask = coords["end"] <= end
            mask = mask & end_mask

        # Apply mask to all data
        filtered_data = {
            "coordinates": {k: v[mask] for k, v in coords.items()},
            "data": {k: v[mask] for k, v in data["data"].items()},
            "metadata": data["metadata"],
        }

        return filtered_data


class GenomeViewer:
    """Create genome-wide visualization plots from processed prediction data."""

    def __init__(self, processor: PredictionDifferenceProcessor):
        self.processor = processor
        self.chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
        self.chrom_to_int = {chrom: i + 1 for i, chrom in enumerate(self.chrom_order)}
        self.colors = ["#1f77b4", "#ff7f0e"]

    def create_genome_view(
        self, data: Dict[str, Any], trial_idx: int, output_path: str, data_type: str = "diff"
    ):
        """Create genome-wide view for a specific trial."""
        coords = data["coordinates"]
        plot_data = (
            data["data"][data_type][:, trial_idx]
            if "abs" not in data_type
            else torch.abs(data["data"][data_type][:, trial_idx])
        )
        trial_name = data["metadata"]["trial_names"][trial_idx]

        # Create DataFrame for plotting
        df = pd.DataFrame({"chrom": coords["chrom"], "midpoint": coords["midpoint"], "value": plot_data})

        # Convert chromosome to numeric and filter
        df["chrom_clean"] = df["chrom"].str.replace("chr", "")
        df["chrom_idx"] = df["chrom_clean"].map(lambda x: self.chrom_to_int.get(x, np.nan))
        df = df.dropna(subset=["chrom_idx"]).astype({"chrom_idx": int})

        if len(df) == 0:
            print(f"No valid data for trial {trial_name}")
            return

        # Calculate cumulative positions
        max_mid = df.groupby("chrom_idx")["midpoint"].max().sort_index()
        offsets = max_mid.cumsum().shift(fill_value=0)
        df["cum_midpoint"] = df["midpoint"] + df["chrom_idx"].map(offsets)
        df["color"] = df["chrom_idx"].map(lambda idx: self.colors[idx % 2])

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(df["cum_midpoint"], df["value"], c=df["color"], s=1, edgecolor="none", alpha=0.7)

        # Set x-axis ticks
        chrom_centers = (offsets + max_mid / 2).to_dict()
        if chrom_centers:
            ticks = list(chrom_centers.values())
            labels = [self.chrom_order[idx - 1] for idx in chrom_centers.keys()]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=0, fontsize=10)

        # Customize plot
        ax.set_xlabel("Chromosome", fontsize=12)
        ylabel_map = {
            "diff": "Difference (Pred - Target)",
            "abs_diff": "Absolute Difference",
            "predictions": "Predicted Value",
            "labels": "Actual Value",
        }
        ax.set_ylabel(ylabel_map.get(data_type, "Value"), fontsize=12)
        ax.set_title(f'Genome-wide {ylabel_map.get(data_type, "Value")} — {trial_name}', fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Add summary statistics
        mean_val = np.mean(df["value"])
        median_val = np.median(df["value"])
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_val:.4f}\nMedian: {median_val:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_path}")


def plot_single_trial(args):
    """Helper function for parallel plotting of individual trials."""
    (processor_config, split, trial_name, output_path, data_type, region_filter) = args
    
    try:
        # Recreate processor and viewer objects (needed for multiprocessing)
        processor = PredictionDifferenceProcessor(**processor_config)
        viewer = GenomeViewer(processor)
        
        # Load data for only this specific trial
        data = processor.load_data_for_trials(split, [trial_name])
        
        # Apply region filtering if specified
        if region_filter:
            chrom, start, end = region_filter
            data = processor.filter_by_region(data, chrom, start, end)
            if len(data["coordinates"]["chrom"]) == 0:
                return f"No data in specified region for trial {trial_name}"
        
        # Create the plot (trial_idx is always 0 since we loaded only one trial)
        viewer.create_genome_view(data, 0, output_path, data_type)
        return f"Successfully created plot for {trial_name}"
    except Exception as e:
        return f"Error creating plot for {trial_name}: {str(e)}"


@click.command()
@click.option("-e", "--exp_name", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint identifier")
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"], help="Data splits to process")
@click.option("--res_base", required=True, default="./Res", help="Results base directory")
@click.option(
    "--data_base", required=True, default="./Data/enformer_style_split_data_v1", help="Data base directory"
)
@click.option("--trials", multiple=True, type=str, help="Specific trials to plot (default: all)")
@click.option(
    "--data_type",
    type=click.Choice(["diff", "abs_diff", "predictions", "labels"]),
    default="diff",
    help="Type of data to visualize",
)
@click.option("--chrom", type=str, help="Filter by chromosome (e.g., 'chr1')")
@click.option("--start", type=int, help="Filter by start position")
@click.option("--end", type=int, help="Filter by end position")
@click.option("--context_length", default=196608, help="Context length used in model")
@click.option("--central_bin_num", default=896, help="Central bins kept in model")
@click.option("--window_size", default=128, help="Window size in base pairs")
@click.option("--num_workers", default=4, help="Number of parallel workers for plotting")
def main(
    exp_name,
    chk,
    splits,
    res_base,
    data_base,
    trials,
    data_type,
    chrom,
    start,
    end,
    context_length,
    central_bin_num,
    window_size,
    num_workers,
):
    """
    Visualize prediction differences across the genome.

    This script processes prediction results, maps them to genomic coordinates,
    and creates genome-wide visualizations showing the specified data type.
    """

    # Initialize processor for data preprocessing only
    processor = PredictionDifferenceProcessor(
        data_base, res_base, exp_name, chk, context_length, central_bin_num, window_size
    )

    for split in splits:
        print(f"\nProcessing {split} split...")

        # Ensure data is processed and saved (but don't load it all into memory)
        try:
            processor.process_and_save_prediction_differences(split)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        # Determine which trials to process
        if trials:
            trial_names = []
            for trial_name in trials:
                if trial_name in processor.label_meta.index:
                    trial_names.append(trial_name)
                else:
                    print(f"Warning: Trial {trial_name} not found in label_meta")
            if not trial_names:
                print(f"No valid trials found from {trials}")
                continue
        else:
            trial_names = processor.label_meta.index.tolist()

        print(f"Creating genome views for {len(trial_names)} trials...")

        # Prepare arguments for parallel processing
        processor_config = {
            'data_base': data_base,
            'res_base': res_base,
            'exp_name': exp_name,
            'chk': chk,
            'context_length': context_length,
            'central_bin_num': central_bin_num,
            'window_size': window_size
        }

        # Prepare region filter
        region_filter = None
        if chrom or start or end:
            region_filter = (chrom, start, end)
            print(f"Will filter by region: {chrom}:{start}-{end}")

        plot_args = []
        for trial_name in trial_names:
            # Generate filename
            region_suffix = ""
            if region_filter:
                chrom_part, start_part, end_part = region_filter
                region_suffix = f"_{chrom_part or 'all'}_{start_part or 'start'}_{end_part or 'end'}"

            output_path = f"{processor.output_dir}/plot/genome_view/{split}_{trial_name}_{data_type}{region_suffix}.png"
            plot_args.append((processor_config, split, trial_name, output_path, data_type, region_filter))

        # Execute plotting in parallel
        if num_workers > 1 and len(plot_args) > 1:
            print(f"Using {num_workers} workers for parallel plotting...")
            with mp.Pool(processes=min(num_workers, len(plot_args))) as pool:
                results = pool.map(plot_single_trial, plot_args)
            
            # Print results
            for result in results:
                print(result)
        else:
            print("Using sequential plotting...")
            for args in tqdm(plot_args, desc="Creating plots"):
                result = plot_single_trial(args)
                print(result)

        print(f"Completed {split} split")


if __name__ == "__main__":
    main()
