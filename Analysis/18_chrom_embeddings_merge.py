#!/usr/bin/env python3
"""
Merge rank-sharded HDF5 embedding files into per-cell-type merged files.

Each rank worker writes merged_peaks_rank{N}_{celltype}.h5. This script
collects all rank shards for each cell type and merges them in original
BED row order.

Usage:
    python Analysis/18_chrom_embeddings_merge.py \
        --input_dir Results/chrom_emb \
        --output_dir Results/chrom_emb \
        --world_size 16 \
        --n_workers 8
"""

import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_celltypes(input_dir: Path, world_size: int) -> list[str]:
    """Return list of cell type safe-names found for rank 0."""
    prefix = "merged_peaks_rank0_"
    celltypes = []
    for p in sorted(input_dir.glob(f"{prefix}*.h5")):
        ct = p.name[len(prefix):-3]  # strip prefix and .h5
        celltypes.append(ct)
    return celltypes


def merge_celltype(args):
    input_dir, output_dir, celltype, world_size = args
    output_path = output_dir / f"merged_peaks_{celltype}.h5"
    if output_path.exists():
        logger.info(f"[{celltype}] already exists, skipping")
        return celltype, True

    # Collect (global_idx, emb, chrom, start, end, bin_start, bin_end) from all ranks
    records = []
    ref_attrs = None

    for rank in range(world_size):
        shard_path = input_dir / f"merged_peaks_rank{rank}_{celltype}.h5"
        if not shard_path.exists():
            logger.warning(f"[{celltype}] Missing rank {rank} shard: {shard_path}")
            continue

        with h5py.File(shard_path, 'r') as f:
            if ref_attrs is None:
                ref_attrs = dict(f.attrs)

            emb_group = f['embeddings']
            meta = f['metadata']

            # Sort keys numerically to match metadata iteration order
            sorted_keys = sorted(emb_group.keys(), key=lambda k: int(k))

            chrom_arr = meta['chrom'][:]
            start_arr = meta['start'][:]
            end_arr = meta['end'][:]
            bin_start_arr = meta['bin_start'][:]
            bin_end_arr = meta['bin_end'][:]

            for i, key in enumerate(sorted_keys):
                global_idx = int(key)
                emb = emb_group[key][:]
                records.append((
                    global_idx, emb,
                    chrom_arr[i], start_arr[i], end_arr[i],
                    bin_start_arr[i], bin_end_arr[i]
                ))

    if not records:
        logger.error(f"[{celltype}] No records found, skipping")
        return celltype, False

    # Sort by global index
    records.sort(key=lambda r: r[0])
    total = len(records)

    # Write merged file
    with h5py.File(output_path, 'w') as f:
        emb_group = f.create_group('embeddings')
        meta_group = f.create_group('metadata')

        chrom_list = []
        start_list = []
        end_list = []
        bin_start_list = []
        bin_end_list = []

        for global_idx, emb, chrom, start, end, bin_s, bin_e in records:
            emb_group.create_dataset(str(global_idx), data=emb)
            chrom_list.append(chrom)
            start_list.append(start)
            end_list.append(end)
            bin_start_list.append(bin_s)
            bin_end_list.append(bin_e)

        # Write metadata arrays
        meta_group.create_dataset('chrom', data=np.array(chrom_list))
        meta_group.create_dataset('start', data=np.array(start_list))
        meta_group.create_dataset('end', data=np.array(end_list))
        meta_group.create_dataset('bin_start', data=np.array(bin_start_list))
        meta_group.create_dataset('bin_end', data=np.array(bin_end_list))

        # Copy attrs, update n_ccres / rank / world_size
        for k, v in ref_attrs.items():
            f.attrs[k] = v
        f.attrs['n_ccres'] = total
        f.attrs['rank'] = 0
        f.attrs['world_size'] = 1

    logger.info(f"[{celltype}] Merged {total} records -> {output_path}")
    return celltype, True


def main():
    parser = argparse.ArgumentParser(description="Merge rank-sharded HDF5 embeddings")
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--world_size', type=int, default=16)
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    celltypes = discover_celltypes(args.input_dir, args.world_size)
    logger.info(f"Found {len(celltypes)} cell types")

    tasks = [(args.input_dir, args.output_dir, ct, args.world_size) for ct in celltypes]

    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            results = pool.map(merge_celltype, tasks)
    else:
        results = [merge_celltype(t) for t in tasks]

    success = sum(1 for _, ok in results if ok)
    logger.info(f"Done: {success}/{len(celltypes)} cell types merged successfully")


if __name__ == '__main__':
    main()
