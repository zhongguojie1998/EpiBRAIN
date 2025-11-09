# %%
import os
import warnings

warnings.filterwarnings("ignore")

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from sklearn.preprocessing import quantile_transform

base_cmap = plt.get_cmap("tab20")


def untransform_predictions(data, label_meta):
    """
    Untransform model predictions back to original scale.

    Reverses the forward transformations applied during data preprocessing:
    1. Scale multiplication: y = scale * y
    2. Soft clipping: if y > clip_soft: y = (clip_soft - 1) + sqrt(y - clip_soft + 1)
    3. Three-quarter power: y = y^(3/4) for sum_three_quarter

    Args:
        data: numpy array of predictions to untransform
        scale: scale factor applied in forward transform (default: 1.0)
        clip_soft: soft clipping threshold (default: 48.0)
        sum_stat: summary statistic used (default: "sum_three_quarter")

    Returns:
        Untransformed data in original scale
    """
    data = data.copy()

    if label_meta is not None:
        # do it for each trial based on label_meta
        for i, row in label_meta.iterrows():
            trial_scale = row.get('scale', 1.0)
            trial_clip_soft = row.get('clip_soft', 48.0)
            trial_sum_stat = row.get('sum_stat', 'sum_three_quarter')

            # Step 1: Undo scale
            if trial_scale != 1.0:
                data[:, i] = data[:, i] / trial_scale

            # Step 2: Undo soft clip
            # Forward: if x > clip_soft: x = (clip_soft - 1) + sqrt(x - clip_soft + 1)
            # Reverse: if x > clip_soft: x = clip_soft - 1 + (x - (clip_soft - 1))^2
            if trial_clip_soft is not None:
                clip_mask = data[:, i] > trial_clip_soft
                data[clip_mask, i] = (trial_clip_soft - 1) + (data[clip_mask, i] - (trial_clip_soft - 1)) ** 2

            # Step 3: Undo three-quarter power
            # Forward: x = x^(3/4)
            # Reverse: x = x^(4/3)
            if trial_sum_stat == "sum_three_quarter":
                data[:, i] = data[:, i] ** (4.0 / 3.0)
            elif trial_sum_stat in ["sum_sqrt", "mean_sqrt", "avg_sqrt"]:
                data[:, i] = (data[:, i] + 1) ** 2 - 1
            elif trial_sum_stat in ['sum', 'mean', "avg"]:
                # no transformation applied
                pass
            else:
                raise ValueError(f"Unknown sum_stat: {trial_sum_stat}")

    return data




def apply_transform(data, transform_type="none"):
    """
    Apply transformation to data.

    Args:
        data: numpy array of shape (n_samples, n_features)
        transform_type: str, one of "none", "log", "log_quantile"

    Returns:
        Transformed data
    """
    if transform_type == "none":
        return data

    data = data.copy()

    if transform_type == "log":
        # Apply log1p transformation (log(1 + x))
        data = np.log1p(np.maximum(data, 0))

    elif transform_type == "log_quantile":
        # Apply log transformation followed by quantile normalization
        data = np.log1p(np.maximum(data, 0))
        data = quantile_transform(data, output_distribution='normal', n_quantiles=min(1000, data.shape[0]))

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return data


def calculate_modality_metrics_batch(mod_name, label_data, pred_data):
    """
    Calculate Pearson correlation for all samples of one modality using vectorized operations
    label_data, pred_data: [n_samples, n_celltypes] arrays
    """
    # Center the data (subtract mean along celltype axis)
    label_centered = label_data - label_data.mean(axis=1, keepdims=True)
    pred_centered = pred_data - pred_data.mean(axis=1, keepdims=True)

    # Calculate standard deviations
    label_std = label_data.std(axis=1)
    pred_std = pred_data.std(axis=1)

    # Vectorized Pearson correlation: sum(centered_x * centered_y) / (n * std_x * std_y)
    n_celltypes = label_data.shape[1]
    numerator = (label_centered * pred_centered).sum(axis=1)
    denominator = n_celltypes * label_std * pred_std

    # Handle division by zero (constant arrays)
    with np.errstate(divide='ignore', invalid='ignore'):
        pearsonr_vals = numerator / denominator
        pearsonr_vals = np.where(np.isfinite(pearsonr_vals), pearsonr_vals, np.nan)

    # Calculate label statistics
    label_vars = label_data.var(axis=1)
    label_means = label_data.mean(axis=1)

    return mod_name, (pearsonr_vals, label_vars, label_means)

# %%
@click.command()
@click.option("-e", "--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("-s", "--splits", multiple=True, type=str, default=["Test"])
@click.option("--res_base", required=True, default="./Res")
@click.option("--log_base", required=True, default="./logs")
@click.option("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
@click.option("--transform", multiple=True, type=click.Choice(['none', 'log', 'log_quantile']),
              default=['none', 'log'], help="Data transformation(s) to apply before calculating correlation (can specify multiple)")
def main(exp_name, chk, splits, res_base, log_base, n_processes, transform):
    LOG_BASE = os.path.abspath(f"{log_base}/{exp_name}/")
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data", exist_ok=True)

    # Convert transform tuple to list
    transform_list = list(transform)
    print(f"Using transformations: {transform_list}")

    # load label meta info
    label_meta = pd.read_csv(f"{LOG_BASE}/regression_label_meta.csv", index_col=None)

    # splits = ["Train", "Valid", "Test"]

    # calculate correlation and other metrics
    for split in splits:
        for trans in transform_list:
            metric_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.csv"
            if os.path.exists(metric_file):
                print(f"Skipping {split} with transform {trans}: file already exists")
                continue
            print(f"Calculate metric for {split}")

            test_res = torch.load(f"{RES_BASE}/{exp_name}/{split}_preds_epoch_{chk}.pt")
            # Convert to numpy arrays to avoid deprecation warnings
            test_label_orig = test_res["label"]['regression'].reshape(-1, test_res["label"]['regression'].shape[-1]).cpu().numpy()
            test_pred_orig = test_res["pred"]['regression'].reshape(-1, test_res["pred"]['regression'].shape[-1]).cpu().numpy()
            # transform back to original scale
            test_label_orig = untransform_predictions(test_label_orig, label_meta)
            test_pred_orig = untransform_predictions(test_pred_orig, label_meta)

            # Apply transformation
            print(f"  Applying transform: {trans}")
            test_label = apply_transform(test_label_orig, trans)
            test_pred = apply_transform(test_pred_orig, trans)

            metric = pd.DataFrame(index=label_meta["modality"].unique(),
                                  columns=["PearsonR:mean", "PearsonR:std", "PearsonR:median", "PearsonR:25%", "PearsonR:75%"])

            # Precompute modality indices and prepare batched data
            modality_data = {}
            for mod in metric.index:
                mod_celltypes = label_meta[label_meta["modality"] == mod]
                label_indices = mod_celltypes.index.values
                pred_indices = mod_celltypes['dim'].values

                # Prepare batched data for this modality (all samples at once)
                modality_data[mod] = (
                    test_label[:, label_indices],
                    test_pred[:, pred_indices]
                )

            # Set number of processes
            n_proc = n_processes if n_processes else cpu_count()
            print(f"Using {n_proc} processes for parallel calculation")
            print(f"Processing {len(metric.index)} modalities with {test_label.shape[0]} samples each")

            # Calculate metrics in parallel, one task per modality (batched)
            results = Parallel(n_jobs=n_proc, backend='loky')(
                delayed(calculate_modality_metrics_batch)(
                    mod,
                    modality_data[mod][0],
                    modality_data[mod][1]
                )
                for mod in tqdm(metric.index, desc="Calculating metrics")
            )

            # Store results
            metric_dict = {mod: {"pearsonr": [], "label_mean": [], "label_var": []} for mod in metric.index}
            for mod_name, (pearsonr_vals, label_vars, label_means) in results:
                metric_dict[mod_name]["pearsonr"] = pearsonr_vals.tolist()
                metric_dict[mod_name]["label_mean"] = label_means.tolist()
                metric_dict[mod_name]["label_var"] = label_vars.tolist()

            pkl_file = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/{split}_metric_across_celltypes_{trans}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(metric_dict, f)

            for mod in metric.index:
                pearsonr_vals = np.array(metric_dict[mod]["pearsonr"])
                # remove nan values
                pearsonr_vals = pearsonr_vals[~np.isnan(pearsonr_vals)]
                if pearsonr_vals.size > 0:
                    metric.loc[mod, "PearsonR:mean"] = np.mean(pearsonr_vals)
                    metric.loc[mod, "PearsonR:std"] = np.std(pearsonr_vals)
                    metric.loc[mod, "PearsonR:median"] = np.median(pearsonr_vals)
                    metric.loc[mod, "PearsonR:25%"] = np.percentile(pearsonr_vals, 25)
                    metric.loc[mod, "PearsonR:75%"] = np.percentile(pearsonr_vals, 75)

            metric.to_csv(metric_file)
            print(f"  Saved metrics to {metric_file}")

# %%
if __name__ == "__main__":
    main()
