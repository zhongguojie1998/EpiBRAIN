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


def load_enriched_sumstats_from_filelist(filelist_path):
    """Load enriched summary statistics from multiple files listed in filelist TSV"""
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")
    
    # Read filelist as TSV: file_path, experiment_name (optional)
    filelist_df = pd.read_csv(filelist_path, sep='\t', header=None, names=['file_path', 'experiment_name'])
    
    # Fill missing experiment names with file stem
    filelist_df['experiment_name'] = filelist_df.apply(
        lambda row: row['experiment_name'] if pd.notna(row['experiment_name']) 
        else Path(row['file_path']).stem.replace('.tsv', '').replace('.csv', ''),
        axis=1
    )
    
    print(f"Loading {len(filelist_df)} files from filelist...")
    
    df_dict = {}
    for _, row in tqdm(filelist_df.iterrows(), desc="Loading files", total=len(filelist_df)):
        file_path = row['file_path']
        exp_name = row['experiment_name']
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        
        try:
            df = load_enriched_sumstats(file_path)
            df_dict[exp_name] = df
            print(f"  Loaded {len(df)} variants from {file_path} as experiment '{exp_name}'")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not df_dict:
        raise ValueError("No valid files could be loaded from filelist")
    
    return df_dict


def init_hdf5_structure(h5_path, label_meta_path, score_names):
    """Initialize HDF5 with optimized structure"""
    label_meta = pd.read_csv(label_meta_path)
    trial_names = label_meta["trial"].tolist()

    with h5py.File(h5_path, "w") as f:
        # Metadata
        f.attrs["trial_names"] = label_meta["trial"].tolist()
        f.attrs["trial_dims"] = label_meta["dim"].tolist()
        f.attrs["score_names"] = score_names

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
            "finished", (0,), maxshape=(None,), dtype="bool", compression="gzip"
        )

        # Results table (compressed) - create datasets for each score type
        results_grp = f.create_group("results")
        for score_name in score_names:
            results_grp.create_dataset(
                score_name,
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
        all_keys = variants_grp["index_key"]

        print("Hashing existing index keys for fast lookup...")
        index_lookup = set(all_keys)

    return index_lookup


@click.command()
@click.option("-s", "--enriched_sumstats", help="Path to enriched summary statistics CSV/CSV.GZ file")
@click.option("-e", "--experiment_name", help="Unique experiment name")
@click.option(
    "-f", "--filelist", help="Path to file containing list of enriched summary statistics files (one per line)"
)
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 storage file")
@click.option("-l", "--label_meta", required=True, help="Path to label_meta.csv")
@click.option("--force", is_flag=True, help="Force recreate HDF5 file")
@click.option("-s", "--score_names", multiple=True, default=["raw_diff"], help="Score names for results_grp (can be used multiple times)")
def main(enriched_sumstats, experiment_name, filelist, hdf5_file, label_meta, force, score_names):
    """Initialize variant effect analysis from enriched summary statistics"""

    # Validate input arguments
    if not enriched_sumstats and not filelist:
        raise click.UsageError("Either --enriched_sumstats or --filelist must be provided")
    if enriched_sumstats and filelist:
        raise click.UsageError("Cannot specify both --enriched_sumstats and --filelist")

    output_dir = Path(hdf5_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if filelist:
        print(f"Filelist: {filelist}")
    else:
        print(f"Enriched sumstats: {enriched_sumstats}")
    print(f"HDF5 file: {hdf5_file}")

    # Load enriched summary statistics
    print("Loading enriched summary statistics...")
    if filelist:
        enriched_df_dict = load_enriched_sumstats_from_filelist(filelist)
        total_variants = sum(len(df) for df in enriched_df_dict.values())
        print(f"Loaded {total_variants} total variants from {len(enriched_df_dict)} experiments")
    else:
        enriched_df = load_enriched_sumstats(enriched_sumstats)
        enriched_df_dict = {experiment_name: enriched_df}
        total_variants = len(enriched_df)
        print(f"Loaded {total_variants} enriched variants")

    # Remove existing file if force flag is set
    if force and os.path.exists(hdf5_file):
        os.remove(hdf5_file)
        print("Removed existing HDF5 file")

    # Initialize or load HDF5
    if not os.path.exists(hdf5_file):
        print("Initializing HDF5 structure...")
        trial_names = init_hdf5_structure(hdf5_file, label_meta, list(score_names))
        existing_index = set()
    else:
        print("Loading existing HDF5...")
        existing_index = load_existing_index(hdf5_file)
        # Load trial names and score names
        with h5py.File(hdf5_file, "r") as f:
            trial_names = f.attrs["trial_names"]
            # Get score names from existing file, or use default
            if "score_names" in f.attrs:
                score_names = f.attrs["score_names"]

    print(f"Existing variants in index: {len(existing_index)}")
    print(f"Score names: {score_names}")

    # Process each experiment separately to efficiently use existing_index
    new_variant_data = {"index_key": [], "rsid": [], "chr": [], "pos": [], "ref": [], "alt": [], "finished": []}
    experiment_data_dict = {}

    for exp_idx, (exp_name, enriched_df) in enumerate(enriched_df_dict.items()):
        print(f"Processing experiment {exp_idx + 1}/{len(enriched_df_dict)}: '{exp_name}' with {len(enriched_df)} variants...")

        # Sort by index_key for efficient processing
        enriched_df = enriched_df.sort_values('index_key')

        experiment_data = []
        for _, variant in tqdm(enriched_df.iterrows(), desc=f"Processing {exp_name}"):
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

            # Only add to main table if new (using existing_index set for O(1) lookup)
            if index_key not in existing_index:
                new_variant_data["index_key"].append(index_key)
                new_variant_data["rsid"].append(variant["rsid"])
                new_variant_data["chr"].append(variant["chr"])
                new_variant_data["pos"].append(variant["pos"])
                new_variant_data["ref"].append(variant["ref"])
                new_variant_data["alt"].append(variant["alt"])
                new_variant_data["finished"].append(False)
                # Add to existing_index to avoid duplicates in later experiments
                existing_index.add(index_key)

        experiment_data_dict[exp_name] = experiment_data

    n_new_variants = len(new_variant_data["index_key"])
    print(f"New variants to add: {n_new_variants}")
    print(f"Number of experiments: {len(experiment_data_dict)}")

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
            for key in ["index_key", "rsid", "chr", "ref", "alt", "finished", "pos"]:
                variants_grp[key].resize((new_size,))

            # Resize all score datasets
            for score_name in score_names:
                results_grp[score_name].resize((new_size, len(trial_names)))

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
            variants_grp["finished"][start_idx:end_idx] = new_variant_data["finished"]

        # Add experiment data for all experiments
        if experiment_data_dict:
            print(f"Storing experiment data for {len(experiment_data_dict)} experiments...")
            exp_grp = f["experiments"]

            for exp_name, experiment_data in experiment_data_dict.items():
                print(f"  Storing experiment: {exp_name}")

                if exp_name in exp_grp:
                    print(f"    Experiment {exp_name} already exists, overwriting...")
                    del exp_grp[exp_name]

                exp_data_grp = exp_grp.create_group(exp_name)

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

    print(f"Total enriched variants: {total_variants}")
    print(f"New variants added to main table: {n_new_variants}")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Experiment data stored under: experiments/[experiment_name]")


if __name__ == "__main__":
    main()
