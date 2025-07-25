from pathlib import Path
import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_chunk_files(chunk_results_dir):
    """Collect all chunk result files (both final results and parts)"""
    chunk_dir = Path(chunk_results_dir)
    if not chunk_dir.exists():
        return []
    
    chunk_files = list(chunk_dir.glob("chunk_*_part_*.h5"))
    
    # Sort by chunk ID (and part number for part files)
    if chunk_files:
        chunk_files.sort(key=lambda x: (int(x.stem.split('_')[1]), int(x.stem.split('_')[3])))
    
    return chunk_files


def validate_chunk_file(chunk_file):
    """Validate chunk file integrity"""
    try:
        with h5py.File(chunk_file, 'r') as f:
            required_datasets = [
                "successful_indices",
                "successful_results",
                "error_indices",
                "error_info",
            ]
            # Check if it's a part file or final result file
            required_attrs = ['chunk_id', 'part_idx', "part_nsample"]

            for dataset in required_datasets:
                if dataset not in f:
                    return False, f"Missing dataset: {dataset}"

            for attr in required_attrs:
                if attr not in f.attrs:
                    return False, f"Missing attribute: {attr}"

            return True, "OK"

    except Exception as e:
        return False, f"Error reading file: {e}"


@click.command()
@click.option("-h5", "--hdf5_file", required=True, help="Path to main HDF5 file")
@click.option("-c", "--chunk_dir", help="Directory containing chunk results (default: <hdf5_dir>/chunk_results)")
@click.option("--cleanup", is_flag=True, help="Remove chunk files after successful merge")
@click.option("--force", is_flag=True, help="Force merge even if some chunks failed validation")
def main(hdf5_file, chunk_dir, cleanup, force):
    """Merge chunk results into main HDF5 file"""

    hdf5_path = Path(hdf5_file)
    if not hdf5_path.exists():
        print(f"Error: Main HDF5 file not found: {hdf5_file}")
        return

    # Default chunk directory
    if not chunk_dir:
        chunk_dir = hdf5_path.parent / "chunk_results"
    else:
        chunk_dir = Path(chunk_dir)

    print(f"Main HDF5 file: {hdf5_file}")
    print(f"Chunk results directory: {chunk_dir}")

    # Collect chunk files
    chunk_files = collect_chunk_files(chunk_dir)
    if not chunk_files:
        print("No chunk result files found!")
        return

    print(f"Found {len(chunk_files)} chunk files")

    # Validate chunk files
    print("Validating chunk files...")
    valid_chunks = []
    invalid_chunks = []

    for chunk_file in tqdm(chunk_files, desc="Validating"):
        is_valid, message = validate_chunk_file(chunk_file)
        if is_valid:
            valid_chunks.append(chunk_file)
        else:
            invalid_chunks.append((chunk_file, message))
            print(f"Invalid chunk {chunk_file.name}: {message}")

    print(f"Valid chunks: {len(valid_chunks)}")
    print(f"Invalid chunks: {len(invalid_chunks)}")

    if invalid_chunks and not force:
        print("Some chunks failed validation. Use --force to proceed anyway.")
        return

    if not valid_chunks:
        print("No valid chunks to merge!")
        return

    # Collect all results
    print("Collecting results from valid chunks...")
    all_indices = []
    all_results = []
    total_successful = 0
    chunk_info = []

    for chunk_file in tqdm(valid_chunks, desc="Loading chunks"):
        with h5py.File(chunk_file, 'r') as f:
            # Get successful results only
            successful_indices = f['successful_indices'][:]
            successful_results = f['successful_results'][:]

            all_indices.append(successful_indices)
            all_results.append(successful_results)
            total_successful += len(successful_indices)

            # Handle both part files and final result files
            chunk_info_dict = {
                "file": chunk_file.name,
                "chunk_id": f.attrs["chunk_id"],
                "part_idx": f.attrs["part_idx"],
                "successful_computations": len(successful_indices),
                "total_sample": f.attrs["part_nsample"],
                "completed_at": f.attrs.get("completed_at", f.attrs.get("saved_at", "unknown")),
            }

            chunk_info.append(chunk_info_dict)

    if not all_results:
        print("No successful results to merge!")
        return

    # Concatenate all results
    print("Merging results...")
    all_indices = np.concatenate(all_indices, dtype='i8')
    all_results = np.vstack(all_results)

    print(f"Total successful computations: {total_successful}")
    print(f"Result matrix shape: {all_results.shape}")

    # Update main HDF5 file
    print("Updating main HDF5 file...")
    with h5py.File(hdf5_file, 'r+') as f:
        variants_grp = f['variants']
        results_grp = f['results']

        # Update status and results in batch for much better performance
        print("Sorting indices for optimal HDF5 access...")
        # Sort indices for better access pattern (contiguous reads/writes are faster)
        sort_order = np.argsort(all_indices)
        sorted_indices = all_indices[sort_order]
        sorted_results = all_results[sort_order]

        print("Batch updating status...")
        # Create status array
        status_array = np.array(['computed'] * len(sorted_indices), dtype=h5py.string_dtype())

        print("Batch updating results...")
        # Use fancy indexing for batch updates (much faster than individual updates)
        variants_grp['status'][sorted_indices] = status_array
        results_grp['variant_effects'][sorted_indices, :] = sorted_results

        # Update metadata
        f.attrs['last_merge_at'] = pd.Timestamp.now().isoformat()
        f.attrs['total_merged_chunks'] = len(valid_chunks)
        f.attrs['total_successful_variants'] = total_successful

    # Save merge report
    merge_report = hdf5_path.parent / f"merge_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(chunk_info).to_csv(merge_report, index=False)
    print(f"Merge report saved: {merge_report}")

    # Cleanup chunk files if requested
    if cleanup:
        print("Cleaning up chunk files...")
        for chunk_file in valid_chunks:
            try:
                chunk_file.unlink()
                print(f"Removed: {chunk_file.name}")
            except Exception as e:
                print(f"Failed to remove {chunk_file.name}: {e}")

        # Remove empty directory
        try:
            if chunk_dir.exists() and not list(chunk_dir.iterdir()):
                chunk_dir.rmdir()
                print(f"Removed empty directory: {chunk_dir}")
        except Exception as e:
            print(f"Failed to remove directory {chunk_dir}: {e}")

    print("\n" + "="*60)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Merged {len(valid_chunks)} chunks")
    print(f"Updated {total_successful} variants")
    print(f"Main HDF5 file: {hdf5_file}")
    if invalid_chunks:
        print(f"Warning: {len(invalid_chunks)} chunks were invalid and skipped")


if __name__ == "__main__":
    main()
