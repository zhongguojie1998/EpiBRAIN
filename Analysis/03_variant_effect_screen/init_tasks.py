import os
from pathlib import Path

import click
import h5py
import pandas as pd
from tqdm import tqdm


def load_enriched_sumstats(csv_path):
    """Load enriched summary statistics"""
    df = pd.read_csv(csv_path, sep="\t", compression="gzip" if csv_path.endswith(".gz") else None)

    # Validate required columns
    required_cols = ["rsid", "chr", "pos", "ref", "alt", "index_key", "reverse_map", "z_score", "n_sample"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in enriched file: {missing_cols}")

    return df


def init_hdf5_structure(h5_path, label_meta_path):
    """Initialize HDF5 with optimized structure"""
    label_meta = pd.read_csv(label_meta_path)
    label_meta = label_meta.sort_values("dim")
    trial_names = label_meta["trial"].tolist()

    with h5py.File(h5_path, "w") as f:
        # Metadata
        f.attrs["trial_names"] = trial_names
        f.attrs["n_dims"] = len(trial_names)

        # Main variants table (compressed)
        variants_grp = f.create_group("variants")
        variants_grp.create_dataset(
            "index_key", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip"
        )
        variants_grp.create_dataset("rsid", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip")
        variants_grp.create_dataset("chr", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip")
        variants_grp.create_dataset("pos", (0,), maxshape=(None,), dtype="i8", compression="gzip")
        variants_grp.create_dataset("ref", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip")
        variants_grp.create_dataset("alt", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip")
        variants_grp.create_dataset(
            "status", (0,), maxshape=(None,), dtype=h5py.string_dtype(), compression="gzip"
        )

        # Results table (compressed)
        results_grp = f.create_group("results")
        results_grp.create_dataset(
            "variant_effects",
            (0, len(trial_names)),
            maxshape=(None, len(trial_names)),
            dtype="f4",
            chunks=True,
            compression="gzip",
        )

        # Experiments group for Z/N storage
        f.create_group("experiments")

    return trial_names


def load_existing_index(h5_path):
    """Load existing variant index for fast lookup"""
    index_lookup = set()

    if not os.path.exists(h5_path):
        return index_lookup

    with h5py.File(h5_path, "r") as f:
        if "variants" not in f:
            return index_lookup

        variants_grp = f["variants"]
        index_lookup = set(variants_grp["index_key"])

    return index_lookup


@click.command()
@click.option(
    "-s", "--enriched_sumstats", required=True, help="Path to enriched summary statistics CSV/CSV.GZ file"
)
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 storage file")
@click.option("-l", "--label_meta", required=True, help="Path to label_meta.csv")
@click.option("-e", "--experiment_name", required=True, help="Unique experiment name")
@click.option("--force", is_flag=True, help="Force recreate HDF5 file")
def main(enriched_sumstats, hdf5_file, label_meta, experiment_name, force):
    """Initialize variant effect analysis from enriched summary statistics"""

    output_dir = Path(hdf5_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing experiment: {experiment_name}")
    print(f"Enriched sumstats: {enriched_sumstats}")
    print(f"HDF5 file: {hdf5_file}")

    # Load enriched summary statistics
    print("Loading enriched summary statistics...")
    enriched_df = load_enriched_sumstats(enriched_sumstats)
    print(f"Loaded {len(enriched_df)} enriched variants")

    # Remove existing file if force flag is set
    if force and os.path.exists(hdf5_file):
        os.remove(hdf5_file)
        print("Removed existing HDF5 file")

    # Initialize or load HDF5
    if not os.path.exists(hdf5_file):
        print("Initializing HDF5 structure...")
        trial_names = init_hdf5_structure(hdf5_file, label_meta)
        existing_index = set()
    else:
        print("Loading existing HDF5...")
        existing_index = load_existing_index(hdf5_file)
        # Load trial names
        with h5py.File(hdf5_file, "r") as f:
            trial_names = f.attrs["trial_names"]

    print(f"Existing variants in index: {len(existing_index)}")

    # Separate new and existing variants using lists for better performance
    new_variant_data = {"index_key": [], "rsid": [], "chr": [], "pos": [], "ref": [], "alt": [], "status": []}
    experiment_data = []

    for _, variant in tqdm(enriched_df.iterrows()):
        index_key = variant["index_key"]

        # Store experiment data regardless
        experiment_data.append(
            {
                "index_key": index_key,
                "reverse_map": variant["reverse_map"],
                "z_score": variant["z_score"],
                "n_sample": variant["n_sample"],
            }
        )

        # Only add to main table if new
        if index_key not in existing_index:
            new_variant_data["index_key"].append(index_key)
            new_variant_data["rsid"].append(variant["rsid"])
            new_variant_data["chr"].append(variant["chr"])
            new_variant_data["pos"].append(variant["pos"])
            new_variant_data["ref"].append(variant["ref"])
            new_variant_data["alt"].append(variant["alt"])
            new_variant_data["status"].append("pending")

    n_new_variants = len(new_variant_data["index_key"])
    print(f"New variants to add: {n_new_variants}")
    print(f"Experiment data entries: {len(experiment_data)}")

    # Update HDF5
    with h5py.File(hdf5_file, "a") as f:
        # Add new variants to main table
        if new_variant_data["index_key"]:
            print("Adding new variants...")
            variants_grp = f["variants"]
            results_grp = f["results"]

            current_size = len(variants_grp["index_key"])
            new_size = current_size + n_new_variants

            # Resize datasets
            for key in ["index_key", "rsid", "chr", "ref", "alt", "status"]:
                variants_grp[key].resize((new_size,))
            variants_grp["pos"].resize((new_size,))
            results_grp["variant_effects"].resize((new_size, len(trial_names)))

            # Add data in batch - directly use the prepared lists (much faster)
            start_idx = current_size
            end_idx = current_size + n_new_variants

            # Batch write all at once using pre-prepared lists
            variants_grp["index_key"][start_idx:end_idx] = new_variant_data["index_key"]
            variants_grp["rsid"][start_idx:end_idx] = new_variant_data["rsid"]
            variants_grp["chr"][start_idx:end_idx] = new_variant_data["chr"]
            variants_grp["pos"][start_idx:end_idx] = new_variant_data["pos"]
            variants_grp["ref"][start_idx:end_idx] = new_variant_data["ref"]
            variants_grp["alt"][start_idx:end_idx] = new_variant_data["alt"]
            variants_grp["status"][start_idx:end_idx] = new_variant_data["status"]

        # Add experiment data
        if experiment_data:
            print(f"Storing experiment data for: {experiment_name}")
            exp_grp = f["experiments"]

            if experiment_name in exp_grp:
                print(f"Experiment {experiment_name} already exists, overwriting...")
                del exp_grp[experiment_name]

            exp_data_grp = exp_grp.create_group(experiment_name)

            # Convert to arrays for efficient storage
            exp_df = pd.DataFrame(experiment_data)

            exp_data_grp.create_dataset(
                "index_key", data=exp_df["index_key"], dtype=h5py.string_dtype(), compression="gzip"
            )
            exp_data_grp.create_dataset(
                "reverse_map", data=exp_df["reverse_map"].values, dtype=bool, compression="gzip"
            )
            exp_data_grp.create_dataset("z_score", data=exp_df["z_score"].values, dtype="f4", compression="gzip")
            exp_data_grp.create_dataset("n_sample", data=exp_df["n_sample"].values, dtype="f4", compression="gzip")

    # Summary
    print("\n" + "=" * 60)
    print("INITIALIZATION SUMMARY")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Total enriched variants: {len(enriched_df)}")
    print(f"New variants added to main table: {n_new_variants}")
    print(f"Experiment data entries: {len(experiment_data)}")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Experiment data stored under: experiments/{experiment_name}")


if __name__ == "__main__":
    main()
