#!/usr/bin/env python3
"""
Script to link differential expression and differential peaks
1. Add TSS information to DiffExpress data
2. Find overlaps between TSS regions and differential peaks
"""

import pandas as pd
import numpy as np
import gzip
import os
from tqdm import tqdm

# Set working directory
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')

print("Step 1: Reading differential expression data...")
# Read the differential expression data
diff_expr = pd.read_csv('Data/source/DiffExpress/subclass_corrected_edgeR.dds')
print(f"Loaded {len(diff_expr)} differential expression entries")
print(f"Columns: {diff_expr.columns.tolist()}")

print("\nStep 2: Parsing GTF file to extract TSS information...")
# Parse GTF file to get TSS for each gene
gtf_file = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v48/gencode.v48.annotation.gtf.gz'

# Dictionary to store gene -> TSS mapping
# We'll store multiple transcripts per gene and take the most representative one
gene_tss_dict = {}

with gzip.open(gtf_file, 'rt') as f:
    for line in tqdm(f, desc="Parsing GTF"):
        # Skip comment lines
        if line.startswith('#'):
            continue

        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue

        chrom = fields[0]
        feature = fields[2]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attributes = fields[8]

        # Only process transcript entries
        if feature != 'transcript':
            continue

        # Parse attributes to get gene_name
        attr_dict = {}
        for attr in attributes.strip().rstrip(';').split(';'):
            attr = attr.strip()
            if ' ' in attr:
                key, value = attr.split(' ', 1)
                attr_dict[key] = value.strip('"')

        if 'gene_name' not in attr_dict:
            continue

        gene_name = attr_dict['gene_name']

        # Determine TSS based on strand
        if strand == '+':
            tss = start
        elif strand == '-':
            tss = end
        else:
            continue

        # Store TSS info (we'll keep first occurrence or handle multiple transcripts)
        if gene_name not in gene_tss_dict:
            gene_tss_dict[gene_name] = {
                'chrom': chrom,
                'tss': tss,
                'strand': strand,
                'start': start,
                'end': end
            }

print(f"Extracted TSS information for {len(gene_tss_dict)} genes")

print("\nStep 3: Adding TSS information to differential expression data...")
# Add TSS information to diff_expr
diff_expr['chrom'] = diff_expr['gene'].map(lambda x: gene_tss_dict.get(x, {}).get('chrom', None))
diff_expr['tss'] = diff_expr['gene'].map(lambda x: gene_tss_dict.get(x, {}).get('tss', None))
diff_expr['strand'] = diff_expr['gene'].map(lambda x: gene_tss_dict.get(x, {}).get('strand', None))

# Remove entries without TSS information
diff_expr_with_tss = diff_expr[diff_expr['tss'].notna()].copy()
print(f"Matched {len(diff_expr_with_tss)} entries with TSS information")
print(f"Missing TSS for {len(diff_expr) - len(diff_expr_with_tss)} entries")

# Save to file
os.makedirs('Data/source/DiffExpress', exist_ok=True)
diff_expr_with_tss.to_csv('Data/source/DiffExpress/DiffExpress.tss.csv', index=False)
print("Saved TSS-annotated data to Data/source/DiffExpress/DiffExpress.tss.csv")

print("\nStep 4: Reading differential peak data...")
# Read differential peak data
diff_peak = pd.read_csv('Data/source/DiffPeak/K27ac_hba_ccre_LR.csv')
print(f"Loaded {len(diff_peak)} differential peak entries")
print(f"Columns: {diff_peak.columns.tolist()}")

print("\nStep 5: Finding overlaps between TSS regions (±250kb) and differential peaks...")
# Define overlap window
window_size = 250000  # 250kb

# List to store overlapping pairs
overlaps = []

# Group diff_peak by chromosome for efficient lookup
diff_peak_by_chrom = {chrom: group for chrom, group in diff_peak.groupby('chrom') if 'chrom' in diff_peak.columns}

# Check if celltype column exists in both dataframes
tss_celltype_col = None
peak_celltype_col = None

for col in diff_expr_with_tss.columns:
    if 'celltype' in col.lower() or 'cell_type' in col.lower() or 'cell' in col.lower():
        tss_celltype_col = col
        break

for col in diff_peak.columns:
    if 'celltype' in col.lower() or 'cell_type' in col.lower() or 'cell' in col.lower():
        peak_celltype_col = col
        break

print(f"TSS celltype column: {tss_celltype_col}")
print(f"Peak celltype column: {peak_celltype_col}")

# Iterate through TSS entries
for tss_idx, tss_row in tqdm(diff_expr_with_tss.iterrows(), total=len(diff_expr_with_tss), desc="Finding overlaps"):
    tss_chrom = tss_row['chrom']
    tss_pos = tss_row['tss']
    tss_celltype = tss_row[tss_celltype_col] if tss_celltype_col else None

    # Define TSS window
    tss_start = tss_pos - window_size
    tss_end = tss_pos + window_size

    # Get peaks on the same chromosome
    if tss_chrom not in diff_peak_by_chrom:
        continue

    chrom_peaks = diff_peak_by_chrom[tss_chrom]

    # Find overlapping peaks
    for peak_idx, peak_row in chrom_peaks.iterrows():
        # Assume peak has start and end columns
        peak_start_col = None
        peak_end_col = None

        for col in ['start', 'Start', 'peak_start', 'chromStart']:
            if col in peak_row.index:
                peak_start_col = col
                break

        for col in ['end', 'End', 'peak_end', 'chromEnd']:
            if col in peak_row.index:
                peak_end_col = col
                break

        if peak_start_col is None or peak_end_col is None:
            continue

        peak_start = peak_row[peak_start_col]
        peak_end = peak_row[peak_end_col]
        peak_celltype = peak_row[peak_celltype_col] if peak_celltype_col else None

        # Check celltype match
        if tss_celltype_col and peak_celltype_col:
            if tss_celltype != peak_celltype:
                continue

        # Check overlap: peak overlaps with TSS window
        if peak_end >= tss_start and peak_start <= tss_end:
            # Store both rows with a combined record
            overlap_record = {}

            # Add TSS info with prefix
            for col in tss_row.index:
                overlap_record[f'tss_{col}'] = tss_row[col]

            # Add peak info with prefix
            for col in peak_row.index:
                overlap_record[f'peak_{col}'] = peak_row[col]

            # Add distance from TSS to peak center
            peak_center = (peak_start + peak_end) / 2
            overlap_record['distance_to_tss'] = abs(peak_center - tss_pos)

            overlaps.append(overlap_record)

print(f"\nFound {len(overlaps)} overlaps between TSS regions and differential peaks")

# Convert to DataFrame and save
if overlaps:
    overlap_df = pd.DataFrame(overlaps)
    os.makedirs('Data/source/DiffPeak', exist_ok=True)
    overlap_df.to_csv('Data/source/DiffPeak/DiffPeak.overlap.DiffTss.csv', index=False)
    print("Saved overlap data to Data/source/DiffPeak/DiffPeak.overlap.DiffTss.csv")
    print(f"\nOverlap DataFrame shape: {overlap_df.shape}")
    print(f"Columns: {overlap_df.columns.tolist()}")
else:
    print("No overlaps found!")

print("\nDone!")
