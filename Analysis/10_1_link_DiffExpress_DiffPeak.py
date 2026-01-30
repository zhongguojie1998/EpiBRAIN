#!/usr/bin/env python3
"""
Script to link differential expression and differential peaks
1. Add TSS information to DiffExpress data
2. Find overlaps between TSS regions and differential peaks
"""

import pandas as pd
import gzip
import os
from tqdm import tqdm
import pyranges as pr

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

# Parse feature name to extract chrom, start, end
print("\nStep 4.5: Parsing peak coordinates from 'feature name' column...")
def parse_feature_name(feature):
    """Parse feature name like 'chr10:47245631-47246130' into chrom, start, end"""
    try:
        chrom_part, coord_part = feature.split(':')
        start, end = coord_part.split('-')
        return chrom_part, int(start), int(end)
    except:
        return None, None, None

parsed = diff_peak['feature name'].apply(parse_feature_name)
diff_peak['chrom'] = [x[0] for x in parsed]
diff_peak['start'] = [x[1] for x in parsed]
diff_peak['end'] = [x[2] for x in parsed]

# Remove entries with failed parsing
diff_peak = diff_peak[diff_peak['chrom'].notna()].copy()
print(f"Successfully parsed {len(diff_peak)} peak entries")

print("\nStep 5: Finding overlaps between TSS regions (±250kb) and differential peaks using PyRanges...")
# Define overlap window
window_size = 250000  # 250kb

# Explicitly use known celltype columns
tss_celltype_col = 'celltype'
peak_celltype_col = 'subclass_corrected'

# Verify columns exist
if tss_celltype_col not in diff_expr_with_tss.columns:
    raise ValueError(f"Column '{tss_celltype_col}' not found in diff_expr_with_tss. Available columns: {diff_expr_with_tss.columns.tolist()}")
if peak_celltype_col not in diff_peak.columns:
    raise ValueError(f"Column '{peak_celltype_col}' not found in diff_peak. Available columns: {diff_peak.columns.tolist()}")

print(f"TSS celltype column: {tss_celltype_col}")
print(f"Peak celltype column: {peak_celltype_col}")

# Get unique celltypes
tss_celltypes = set(diff_expr_with_tss[tss_celltype_col].dropna().unique())
peak_celltypes = set(diff_peak[peak_celltype_col].dropna().unique())
common_celltypes = tss_celltypes & peak_celltypes

print(f"Found {len(tss_celltypes)} celltypes in TSS data")
print(f"Found {len(peak_celltypes)} celltypes in peak data")
print(f"Found {len(common_celltypes)} common celltypes to process")

if len(common_celltypes) == 0:
    print("Warning: No common celltypes found!")
    overlaps = []
else:
    # Process each celltype separately
    all_overlaps = []

    for celltype in tqdm(common_celltypes, desc="Processing celltypes"):
        # Filter by celltype
        tss_ct = diff_expr_with_tss[diff_expr_with_tss[tss_celltype_col] == celltype].copy()
        peak_ct = diff_peak[diff_peak[peak_celltype_col] == celltype].copy()

        if len(tss_ct) == 0 or len(peak_ct) == 0:
            continue

        # Prepare TSS regions with windows
        tss_ct['Start'] = (tss_ct['tss'] - window_size).clip(lower=0)
        tss_ct['End'] = tss_ct['tss'] + window_size
        tss_ct['Chromosome'] = tss_ct['chrom']
        tss_ct['tss_id'] = range(len(tss_ct))

        # Prepare peak regions
        peak_ct['Chromosome'] = peak_ct['chrom']
        peak_ct['Start'] = peak_ct['start']
        peak_ct['End'] = peak_ct['end']
        peak_ct['peak_id'] = range(len(peak_ct))

        # Convert to PyRanges objects
        tss_pr = pr.PyRanges(tss_ct[['Chromosome', 'Start', 'End', 'tss_id']])
        peak_pr = pr.PyRanges(peak_ct[['Chromosome', 'Start', 'End', 'peak_id']])

        # Find overlaps
        overlaps_pr = tss_pr.join(peak_pr, suffix='_peak')

        if len(overlaps_pr) > 0:
            overlaps_df = overlaps_pr.df

            # Rename PyRanges Start/End columns from peak
            overlaps_df = overlaps_df.rename(columns={
                'Start_peak': 'start_peak',
                'End_peak': 'end_peak'
            })

            # Merge with original data to get all columns
            overlaps_df = overlaps_df.merge(tss_ct.drop(columns=['Chromosome', 'Start', 'End']),
                                           on='tss_id', how='left')
            overlaps_df = overlaps_df.merge(peak_ct.drop(columns=['Chromosome', 'Start', 'End']),
                                           on='peak_id', how='left')

            # Add explicit celltype column for clarity
            overlaps_df['celltype'] = celltype

            # Add TSS region columns (TSS ± 511bp)
            overlaps_df['start_tss'] = overlaps_df['tss'] - 511
            overlaps_df['end_tss'] = overlaps_df['tss'] + 511

            # Calculate distance from TSS to peak center
            peak_center = (overlaps_df['start'] + overlaps_df['end']) / 2
            overlaps_df['distance_to_tss'] = abs(peak_center - overlaps_df['tss'])

            all_overlaps.append(overlaps_df)

    # Combine all celltype results
    if all_overlaps:
        print(f"\nCombining results from {len(all_overlaps)} celltypes...")
        overlaps_df = pd.concat(all_overlaps, ignore_index=True)

        # Rename columns with prefixes for clarity
        print("Formatting output...")
        tss_cols = [col for col in diff_expr_with_tss.columns if col not in ['Chromosome', 'Start', 'End', 'tss_id']]
        peak_cols = [col for col in diff_peak.columns if col not in ['Chromosome', 'Start', 'End', 'peak_id']]

        rename_dict = {}
        for col in tss_cols:
            if col in overlaps_df.columns:
                rename_dict[col] = f'tss_{col}'
        for col in peak_cols:
            if col in overlaps_df.columns and f'tss_{col}' not in rename_dict.values():
                rename_dict[col] = f'peak_{col}'

        overlaps_df = overlaps_df.rename(columns=rename_dict)

        # Remove PyRanges-specific columns
        overlaps_df = overlaps_df.drop(columns=['tss_id', 'peak_id'], errors='ignore')

        # Convert to list of dicts for compatibility
        overlaps = overlaps_df.to_dict('records')

        print(f"\nFound {len(overlaps)} overlaps between TSS regions and differential peaks (within matching celltypes)")
    else:
        overlaps = []
        print("No overlaps found!")

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
    
# Convert to bed files
overlap_df_bed = overlap_df[['Chromosome', 'start_tss', 'end_tss', 'tss_strand', 'tss_celltype', 'tss_logFC']]
overlap_df_bed = overlap_df_bed.rename(columns={
    'Chromosome': 'chrom',
    'start_tss': 'start',
    'end_tss': 'end',
    'tss_strand': 'strand',
    'tss_celltype': 'track'
})
overlap_df_bed = overlap_df_bed.drop_duplicates()
overlap_df_bed['start'] = overlap_df_bed['start'].astype(int)
overlap_df_bed['end'] = overlap_df_bed['end'].astype(int)
# change celltype to the track of H3K27Ac
overlap_df_bed['track'] = 'MiniAtlas-' + overlap_df_bed['track'].str.replace(' ', '_') + '_K27Ac'
# add negative tracks
overlap_df_bed['track_neg'] = 'all'
# filter by tss_logFC
overlap_df_bed_filter = overlap_df_bed[overlap_df_bed['tss_logFC'] > 2].copy()
overlap_df_bed_other = overlap_df_bed[overlap_df_bed['tss_logFC'] <= 2].copy()
print(f"\nFiltered overlap bed entries with tss_logFC <= 2: {len(overlap_df_bed_other)} entries")
# drop tss_logFC
overlap_df_bed_filter = overlap_df_bed_filter.drop(columns=['tss_logFC'])
print(f"\nFiltered overlap bed entries with tss_logFC > 2: {len(overlap_df_bed_filter)} entries")
overlap_df_bed_filter.to_csv('Data/source/DiffPeak/DiffPeak.overlap.DiffTss.filter.bed', sep='\t', index=False, header=False)
# drop tss_logFC
overlap_df_bed_other = overlap_df_bed_other.drop(columns=['tss_logFC'])
overlap_df_bed_other.to_csv('Data/source/DiffPeak/DiffPeak.overlap.DiffTss.other.bed', sep='\t', index=False, header=False)
# drop tss_logFC
overlap_df_bed = overlap_df_bed.drop(columns=['tss_logFC'])
overlap_df_bed.to_csv('Data/source/DiffPeak/DiffPeak.overlap.DiffTss.bed', sep='\t', index=False, header=False)
print("\nDone!")
