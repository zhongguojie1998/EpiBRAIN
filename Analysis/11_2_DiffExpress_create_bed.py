#!/usr/bin/env python3
"""
Create BED file from DiffExpress data with exon annotations.
"""

import pandas as pd
import re


def parse_exon_coordinates(exon_string):
    """
    Parse exon string (chr:start-end;chr:start-end;...) into list of tuples.
    Returns list of (chr, start, end) tuples.
    """
    if pd.isna(exon_string) or exon_string == '':
        return []

    exons = []
    for exon in exon_string.split(';'):
        if not exon:
            continue
        match = re.match(r'(.+):(\d+)-(\d+)', exon)
        if match:
            chrom, start, end = match.groups()
            exons.append((chrom, int(start), int(end)))
    return exons


def get_exon_range(exons):
    """
    Get the min and max positions across all exons.
    GTF coordinates are 1-based closed [start, end] (both inclusive).
    Keep as 1-based for now.
    Returns (chr, min_pos, max_pos) in 1-based coordinates.
    """
    if not exons:
        return None, None, None

    # Assume all exons are from the same chromosome
    chrom = exons[0][0]
    all_positions = []
    for _, start, end in exons:
        # Keep 1-based coordinates
        all_positions.append(start)
        all_positions.append(end)

    return chrom, min(all_positions), max(all_positions)


def convert_exons_to_relative(exons, window_start, bin_size=32):
    """
    Convert exon coordinates to relative binned indices.
    GTF/BED coordinates are kept as 1-based closed [start, end] (both inclusive).
    For binning: relative_pos = coord - window_start gives 0 when at window start.
    For 1-based inclusive intervals, we need to add 1 to the end for proper binning.
    Returns list of relative bin index strings in Python 0-based indexing.
    """
    relative_exons = []
    for chrom, start, end in exons:
        # Calculate relative positions from window start (1-based coords)
        # If start == window_start, rel_start = 0 (first position in window)
        rel_start = start - window_start
        # For 1-based inclusive end, add 1 to convert to exclusive end for binning
        rel_end = end - window_start + 1

        # Convert to bins with Python 0-based indexing
        # bin_start uses floor division
        bin_start = rel_start // bin_size
        # bin_end uses ceiling division (exclusive end in Python indexing)
        bin_end = -(-rel_end // bin_size)  # ceiling division

        relative_exons.append(f"{bin_start}-{bin_end}")
    return relative_exons


def load_chromosome_lengths(fai_path):
    """
    Load chromosome lengths from .fai file.
    Returns dict mapping chromosome name to length.
    """
    chrom_lengths = {}
    with open(fai_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom_name = parts[0]
                chrom_length = int(parts[1])
                chrom_lengths[chrom_name] = chrom_length
    print(f"Loaded {len(chrom_lengths)} chromosome lengths")
    return chrom_lengths


def load_miniatlas_celltypes(config_path):
    """
    Load MiniAtlas celltypes from config file.
    Returns set of celltype names (without MiniAtlas- prefix).
    """
    print(f"Reading config file: {config_path}")
    df = pd.read_csv(config_path)

    # Get unique celltypes that start with MiniAtlas-
    miniatlas_celltypes = set()
    for celltype in df['celltype'].unique():
        if celltype.startswith('MiniAtlas-'):
            # Remove the MiniAtlas- prefix
            celltype_name = celltype.replace('MiniAtlas-', '')
            miniatlas_celltypes.add(celltype_name)

    print(f"Found {len(miniatlas_celltypes)} MiniAtlas celltypes")
    return sorted(miniatlas_celltypes)


def create_bed_file(input_csv, output_bed, config_path, fai_path):
    """
    Create BED file from DiffExpress data with exon annotations.
    Validates that coordinates are within chromosome boundaries.
    """
    # Load chromosome lengths
    chrom_lengths = load_chromosome_lengths(fai_path)

    # Load MiniAtlas celltypes
    miniatlas_celltypes = load_miniatlas_celltypes(config_path)
    miniatlas_celltypes_set = set(miniatlas_celltypes)

    # Read DiffExpress data
    print(f"\nReading DiffExpress data: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")

    # Open output file
    print(f"\nCreating BED file: {output_bed}")
    bed_lines = []
    skipped = 0
    processed = 0

    window_half_size = 524288 // 2  # 262144

    for idx, row in df.iterrows():
        gene = row['gene']
        celltype = row['celltype']
        strand = row['strand']
        exons_str = row['exons']

        # Check if this celltype exists in MiniAtlas
        if celltype not in miniatlas_celltypes_set:
            skipped += 1
            continue

        # Parse exons
        exons = parse_exon_coordinates(exons_str)
        if not exons:
            print(f"Warning: No exons found for gene {gene}, skipping")
            skipped += 1
            continue

        # Get exon range
        chrom, min_pos, max_pos = get_exon_range(exons)
        if chrom is None:
            skipped += 1
            continue

        # Calculate midpoint
        mid = (min_pos + max_pos) // 2

        # Calculate initial window
        window_start = mid - window_half_size
        window_end = mid + window_half_size

        # Validate and adjust coordinates based on chromosome boundaries
        if chrom not in chrom_lengths:
            print(f"Warning: Chromosome {chrom} not found in reference, skipping gene {gene}")
            skipped += 1
            continue

        chrom_length = chrom_lengths[chrom]

        # Adjust coordinates to be within chromosome boundaries
        # Note: Using 1-based coordinates where valid range is [1, chrom_length]
        if window_start < 1:
            # Region starts before chromosome start
            adjustment = 1 - window_start
            window_start = 1
            window_end = min(window_end + adjustment, chrom_length)
            print(f"Info: Adjusted window for {gene} on {chrom} - start was < 1")

        if window_end > chrom_length:
            # Region extends beyond chromosome end
            adjustment = window_end - chrom_length
            window_end = chrom_length
            window_start = max(window_start - adjustment, 1)
            print(f"Info: Adjusted window for {gene} on {chrom} - end was > {chrom_length}")

        # Final validation: ensure window is still valid
        if window_start >= window_end:
            print(f"Warning: Invalid window coordinates for gene {gene} on {chrom}, skipping")
            skipped += 1
            continue

        # Convert exons to relative coordinates
        relative_exons = convert_exons_to_relative(exons, window_start)
        relative_exons_str = ';'.join(relative_exons)

        # Determine RNA strand suffix (reversed: + -> RNAminus, - -> RNAplus)
        rna_suffix = "RNAminus" if strand == "+" else "RNAplus"

        # Get foreground celltype with MiniAtlas- prefix and strand info
        foreground_celltype = f"MiniAtlas-{celltype}_{rna_suffix}"

        # Get background celltypes (all other MiniAtlas celltypes) with strand info
        background_celltypes = [f"MiniAtlas-{ct}_{rna_suffix}" for ct in miniatlas_celltypes if ct != celltype]
        background_celltypes_str = ';'.join(background_celltypes)

        # Create BED line (tab-separated)
        bed_line = '\t'.join([
            chrom,                      # Column 1: chromosome
            str(window_start),          # Column 2: start
            str(window_end),            # Column 3: end
            strand,                     # Column 4: strand
            relative_exons_str,         # Column 5: relative exons
            gene,                       # Column 6: gene name
            foreground_celltype,        # Column 7: foreground celltype
            background_celltypes_str    # Column 8: background celltypes
        ])
        bed_lines.append(bed_line)
        processed += 1

    # Write to file
    with open(output_bed, 'w') as f:
        for line in bed_lines:
            f.write(line + '\n')

    print(f"\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total output lines: {len(bed_lines)}")


def main():
    # File paths
    config_path = "Data/data_config/basal_ganglia_miniatlas_drop_celltype_v1.csv"
    input_csv = "Data/source/DiffExpress/DiffExpress.tss.exons.top1000.csv"
    output_bed = "Data/source/DiffExpress/DiffExpress.tss.exons.top1000.bed"
    fai_path = "Data/source/hg38/hg38.fa.fai"

    # Create BED file
    create_bed_file(input_csv, output_bed, config_path, fai_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
