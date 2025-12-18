import os
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

REQUIRED_COLS = ["rsid", "chr", "pos", "ref", "alt", "index_key"]
REQUIRED_COLS_ALT = {'rsid': 'variant_id'}


def load_enriched_sumstats(csv_path: str):
    """Load enriched summary statistics"""
    if csv_path.endswith("vcf") or csv_path.endswith("vcf.gz"):
        # Skip lines starting with ## but keep the #CHROM header line
        import gzip
        open_func = gzip.open if csv_path.endswith(".gz") else open
        mode = 'rt' if csv_path.endswith(".gz") else 'r'

        # Count header lines starting with ##
        skip_lines = []
        with open_func(csv_path, mode) as f:
            for i, line in enumerate(f):
                if line.startswith('##'):
                    skip_lines.append(i)
                else:
                    break

        df = pd.read_csv(
            csv_path,
            sep="\t",
            compression="gzip" if csv_path.endswith(".gz") else None,
            skiprows=skip_lines
        )
        df.rename({"#CHROM": "chr", "POS": "pos", "ID": "rsid", "REF": "ref", "ALT": "alt"}, axis=1, inplace=True)
        if "rsid" not in df.columns or df["rsid"].isna().all():
            df["rsid"] = "NA"

        # Normalize for consistent index_key generation
        # Strip whitespace and ensure uppercase for alleles
        df["chr"] = df["chr"].astype(str).str.strip()
        df["ref"] = df["ref"].astype(str).str.strip().str.upper()
        df["alt"] = df["alt"].astype(str).str.strip().str.upper()

        df["index_key"] = df["chr"] + ":" + df["pos"].astype("str") + ":" + df["ref"] + ":" + df["alt"]
    else:
        df = pd.read_csv(csv_path, sep="\t", compression="gzip" if csv_path.endswith(".gz") else None)

        # Normalize for consistent index_key generation
        if "chr" in df.columns and "pos" in df.columns and "ref" in df.columns and "alt" in df.columns:
            df["chr"] = df["chr"].astype(str).str.strip()
            df["ref"] = df["ref"].astype(str).str.strip().str.upper()
            df["alt"] = df["alt"].astype(str).str.strip().str.upper()

            # Recreate index_key if it doesn't exist or to ensure consistency
            df["index_key"] = df["chr"] + ":" + df["pos"].astype("str") + ":" + df["ref"] + ":" + df["alt"]

    # Validate required columns
    missing_cols = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            if col in REQUIRED_COLS_ALT and REQUIRED_COLS_ALT[col] in df.columns:
                df.rename({REQUIRED_COLS_ALT[col]: col}, axis=1, inplace=True)
            else:
                missing_cols.append(col)
    if missing_cols:
        raise ValueError(f"Missing required columns in enriched file: {missing_cols}")

    return df


def _load_single_file(file_path, exp_name):
    """Helper function to load a single file for parallel processing"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        return None, None, None

    try:
        df = load_enriched_sumstats(file_path)
        return exp_name, df, None
    except Exception as e:
        return exp_name, None, str(e)


def load_enriched_sumstats_from_filelist(filelist_path, n_jobs=36):
    """Load enriched summary statistics from multiple files listed in filelist TSV

    Args:
        filelist_path: Path to the filelist TSV file
        n_jobs: Number of parallel jobs (default: -1, uses all available cores)
    """
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")

    # Read filelist as TSV: file_path, experiment_name (optional)
    filelist_df = pd.read_csv(filelist_path, sep="\t", header=None, names=["file_path", "experiment_name"])

    # Fill missing experiment names with file stem
    filelist_df["experiment_name"] = filelist_df.apply(
        lambda row: (
            row["experiment_name"]
            if pd.notna(row["experiment_name"])
            else Path(row["file_path"]).stem.replace(".tsv", "").replace(".csv", "")
        ),
        axis=1,
    )

    print(f"Loading {len(filelist_df)} files from filelist using {n_jobs if n_jobs > 0 else 'all'} parallel jobs...")

    # Load files in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_load_single_file)(row["file_path"], row["experiment_name"])
        for _, row in filelist_df.iterrows()
    )

    # Process results
    df_dict = {}
    for exp_name, df, error in results:
        if error is not None:
            print(f"Error loading {exp_name}: {error}")
            continue
        if df is not None:
            df_dict[exp_name] = df
            print(f"  Loaded {len(df)} variants from experiment '{exp_name}'")

    if not df_dict:
        raise ValueError("No valid files could be loaded from filelist")

    return df_dict


def init_hdf5_structure(h5_path, label_meta_path, score_names):
    """Initialize HDF5 with optimized structure"""
    label_meta = pd.read_csv(label_meta_path)
    trial_names = label_meta["trial"].tolist()

    with h5py.File(h5_path, "w") as f:
        f.attrs["score_names"] = score_names

        # Metadata
        model_grp = f.create_group("model_meta")
        model_grp.create_dataset(
            "trial_names", data=label_meta["trial"].values, dtype=h5py.string_dtype(), compression="gzip"
        )
        if "dim" in label_meta.columns:
            model_grp.create_dataset("trial_dims", data=label_meta["dim"].values, dtype="i8", compression="gzip")
        else:
            model_grp.create_dataset("trial_dims", data=label_meta.index.values, dtype="i8", compression="gzip")
            
        # Main variants table (compressed)
        variants_grp = f.create_group("variants")
        variants_grp.create_dataset(
            "index_key", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True, compression="gzip"
        )
        variants_grp.create_dataset("rsid", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True, compression="gzip")
        variants_grp.create_dataset("chr", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True, compression="gzip")
        variants_grp.create_dataset("pos", (0,), maxshape=(None,), dtype="i8", chunks=True, compression="gzip")
        variants_grp.create_dataset("ref", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True, compression="gzip")
        variants_grp.create_dataset("alt", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True, compression="gzip")
        variants_grp.create_dataset("finished", (0,), maxshape=(None,), dtype="bool", chunks=True, compression="gzip")

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


def transfer_existing_predictions(source_h5_path, target_h5_path):
    """Transfer predictions from an existing HDF5 file to a new one"""
    if not os.path.exists(source_h5_path):
        print(f"Source HDF5 file not found: {source_h5_path}")
        return 0

    print(f"Transferring predictions from {source_h5_path} to {target_h5_path}...")

    transferred_count = 0

    with h5py.File(source_h5_path, "r") as src_f, h5py.File(target_h5_path, "r+") as tgt_f:
        # Get source and target data
        src_variants = src_f["variants"]
        src_results = src_f["results"]

        tgt_variants = tgt_f["variants"]
        tgt_results = tgt_f["results"]

        # Get score names from target file
        score_names = tgt_f.attrs.get("score_names", ["raw_diff"])

        # Build index mapping from source to target
        src_index_keys = src_variants["index_key"][:]
        src_finished = src_variants["finished"][:]

        tgt_index_keys = tgt_variants["index_key"][:]

        # Create lookup dictionary for target indices
        print("Building target index lookup...")
        tgt_index_lookup = {key.decode() if isinstance(key, bytes) else key: idx
                           for idx, key in enumerate(tgt_index_keys)}

        # Transfer finished predictions
        print("Transferring finished predictions...")

        # Find all finished indices at once
        finished_src_indices = np.where(src_finished)[0]
        print(f"Found {len(finished_src_indices)} finished predictions in source")

        if len(finished_src_indices) > 0:
            # Get keys for finished entries and map to target indices
            src_indices_list = []
            tgt_indices_list = []

            for src_idx in tqdm(finished_src_indices, desc="Mapping indices"):
                src_key = src_index_keys[src_idx]
                src_key_str = src_key.decode() if isinstance(src_key, bytes) else src_key

                # Check if this variant exists in target
                if src_key_str in tgt_index_lookup:
                    tgt_idx = tgt_index_lookup[src_key_str]
                    src_indices_list.append(src_idx)
                    tgt_indices_list.append(tgt_idx)

            # Convert to numpy arrays for vectorized operations
            src_indices = np.array(src_indices_list, dtype=np.int64)
            tgt_indices = np.array(tgt_indices_list, dtype=np.int64)
            transferred_count = len(src_indices)

            print(f"Matched {transferred_count} predictions to transfer")

            # Sort indices in increasing order (required by HDF5)
            sort_order = np.argsort(tgt_indices)
            src_indices = src_indices[sort_order]
            tgt_indices = tgt_indices[sort_order]

            # Copy predictions using vectorized operations
            for score_name in score_names:
                if score_name in src_results and score_name in tgt_results:
                    print(f"Copying {score_name}...")
                    tgt_results[score_name][tgt_indices, :] = src_results[score_name][src_indices, :]

            # Mark as finished in target using vectorized assignment
            tgt_variants["finished"][tgt_indices] = True

        print(f"Transferred {transferred_count} predictions")

    return transferred_count


@click.command()
@click.option("-f", "--enriched_sumstats", help="Path to enriched summary statistics CSV/CSV.GZ file")
@click.option("-e", "--experiment_name", help="Unique experiment name")
@click.option(
    "-fl", "--filelist", help="Path to file containing list of enriched summary statistics files (one per line)"
)
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 storage file")
@click.option("-l", "--label_meta", required=True, help="Path to label_meta.csv")
@click.option("--force", is_flag=True, help="Force recreate HDF5 file")
@click.option(
    "-s",
    "--score_names",
    multiple=True,
    default=["raw_diff"],
    help="Score names for results_grp (can be used multiple times)",
)
@click.option(
    "--load_existing",
    help="Path to existing HDF5 file to transfer predictions from",
)
def main(enriched_sumstats, experiment_name, filelist, hdf5_file, label_meta, force, score_names, load_existing):
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
            trial_names = f["model_meta/trial_dims"][:]
            # Get score names from existing file, or use default
            if "score_names" in f.attrs:
                score_names = f.attrs["score_names"]

    print(f"Existing variants in index: {len(existing_index)}")
    print(f"Score names: {score_names}")

    # Process each experiment separately to efficiently use existing_index
    new_variant_data = {"index_key": [], "rsid": [], "chr": [], "pos": [], "ref": [], "alt": [], "finished": []}
    experiment_data_dict = {}

    for exp_idx, (exp_name, enriched_df) in enumerate(enriched_df_dict.items()):
        print(
            f"Processing experiment {exp_idx + 1}/{len(enriched_df_dict)}: '{exp_name}' with {len(enriched_df)} variants..."
        )

        # Sort by index_key for efficient processing
        enriched_df = enriched_df.sort_values("index_key")

        experiment_data = []
        for _, variant in tqdm(enriched_df.iterrows(), desc=f"Processing {exp_name}"):
            index_key = variant["index_key"]

            additional_info = {k: variant[k] for k in variant.index if k not in REQUIRED_COLS}
            additional_info.update({"index_key": index_key})

            # Store experiment data regardless
            experiment_data.append(additional_info)

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

                for i in exp_df.columns:
                    exp_data_grp.create_dataset(i, data=exp_df[i].values, compression="gzip")

    # Transfer predictions from existing HDF5 file after adding all variants
    if load_existing:
        print("\nTransferring predictions from existing HDF5 file...")
        transferred_count = transfer_existing_predictions(load_existing, hdf5_file)
        print(f"Successfully transferred {transferred_count} predictions from {load_existing}")

    # Summary
    print("\n" + "=" * 60)
    print("INITIALIZATION SUMMARY")
    print("=" * 60)

    print(f"Total enriched variants: {total_variants}")
    print(f"New variants added to main table: {n_new_variants}")
    if load_existing:
        print(f"Predictions transferred from: {load_existing}")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Experiment data stored under: experiments/[experiment_name]")


if __name__ == "__main__":
    main()
