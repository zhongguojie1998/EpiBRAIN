#!/usr/bin/env python3
"""
Add exon annotations to DiffExpress TSS data from GENCODE GTF file.
"""

import gzip
import pandas as pd
from collections import defaultdict
import re


def parse_gtf_attributes(attr_string):
    """Parse GTF attribute string into a dictionary."""
    attrs = {}
    # Match key "value" patterns
    for match in re.finditer(r'(\w+)\s+"([^"]+)"', attr_string):
        attrs[match.group(1)] = match.group(2)
    return attrs


def extract_exons_from_gtf(gtf_path):
    """
    Extract exon information from GTF file.
    Returns a dictionary mapping gene names/IDs to lists of exon coordinates.
    """
    print(f"Reading GTF file: {gtf_path}")
    gene_exons = defaultdict(list)

    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]

            # Only process exon entries
            if feature_type != 'exon':
                continue

            chrom = fields[0]
            start = fields[3]
            end = fields[4]
            strand = fields[6]
            attributes = parse_gtf_attributes(fields[8])

            # Get gene identifiers
            gene_id = attributes.get('gene_id', '').split('.')[0]  # Remove version
            gene_name = attributes.get('gene_name', '')

            # Format exon as chrom:start-end
            exon_coord = f"{chrom}:{start}-{end}"

            # Store by both gene_id and gene_name
            if gene_id:
                gene_exons[gene_id].append(exon_coord)
            if gene_name:
                gene_exons[gene_name].append(exon_coord)

    print(f"Extracted exons for {len(gene_exons)} genes")
    return gene_exons


def add_exon_annotations(input_csv, output_csv, gtf_path):
    """
    Add exon annotations to DiffExpress data.
    """
    # Extract exons from GTF
    gene_exons = extract_exons_from_gtf(gtf_path)

    # Read DiffExpress data
    print(f"\nReading DiffExpress data: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} genes in DiffExpress data")

    # Add exons column
    print("\nMapping exons to genes...")
    exons_list = []
    genes_with_exons = 0
    genes_without_exons = []

    for gene in df['gene']:
        if gene in gene_exons:
            # Remove duplicates and join with semicolon
            unique_exons = list(dict.fromkeys(gene_exons[gene]))  # Preserves order
            exons_str = ';'.join(unique_exons)
            exons_list.append(exons_str)
            genes_with_exons += 1
        else:
            exons_list.append('')
            genes_without_exons.append(gene)

    df['exons'] = exons_list

    # Write output
    print(f"\nWriting output to: {output_csv}")
    df.to_csv(output_csv, index=False)

    # Report statistics
    print(f"\nSummary:")
    print(f"  Total genes: {len(df)}")
    print(f"  Genes with exons: {genes_with_exons}")
    print(f"  Genes without exons: {len(genes_without_exons)}")

    if genes_without_exons and len(genes_without_exons) <= 10:
        print(f"  Genes without exons: {', '.join(genes_without_exons)}")
    elif genes_without_exons:
        print(f"  First 10 genes without exons: {', '.join(genes_without_exons[:10])}")


def calculate_exon_span(exons_str):
    """
    Calculate the span from minimum start to maximum end of exons.
    Returns None if exons string is empty or invalid.
    """
    if pd.isna(exons_str) or exons_str == '':
        return None

    try:
        positions = []
        for exon in exons_str.split(';'):
            if not exon:
                continue
            # Parse chr:start-end
            match = re.match(r'(.+):(\d+)-(\d+)', exon)
            if match:
                start = int(match.group(2))
                end = int(match.group(3))
                positions.extend([start, end])

        if not positions:
            return None

        return max(positions) - min(positions)
    except:
        return None


def select_top_genes_by_celltype(input_csv, output_csv, top_n=1000, fdr_threshold=0.05, max_exon_span=524288):
    """
    Select top N genes with largest positive fold change for each celltype.
    Only includes genes with significant FDR and exon span <= max_exon_span.
    """
    print(f"\nReading annotated data: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Total rows: {len(df)}")
    print(f"Celltypes: {df['celltype'].unique()}")

    # Calculate exon spans
    print(f"\nCalculating exon spans...")
    df['exon_span'] = df['exons'].apply(calculate_exon_span)

    # Group by celltype and select top N by positive fold change
    print(f"\nSelecting top {top_n} genes by positive logFC for each celltype (FDR < {fdr_threshold}, exon span <= {max_exon_span})...")

    top_genes_list = []
    total_filtered_by_span = 0

    for celltype in df['celltype'].unique():
        celltype_df = df[df['celltype'] == celltype].copy()

        # Filter by significant FDR
        significant_df = celltype_df[celltype_df['FDR'] < fdr_threshold]

        # Filter by exon span
        before_span_filter = len(significant_df)
        significant_df = significant_df[
            (significant_df['exon_span'].notna()) &
            (significant_df['exon_span'] <= max_exon_span)
        ]
        filtered_by_span = before_span_filter - len(significant_df)
        total_filtered_by_span += filtered_by_span

        # Sort by logFC in descending order (largest positive fold change)
        # Keep taking genes until we have top_n valid genes
        celltype_top = significant_df.nlargest(top_n, 'logFC')

        if len(celltype_top) > 0:
            max_logfc = celltype_top['logFC'].max()
            min_logfc = celltype_top['logFC'].min()
            print(f"  {celltype}: selected {len(celltype_top)} genes (max logFC: {max_logfc:.2f}, min logFC: {min_logfc:.2f}, filtered by span: {filtered_by_span})")
        else:
            print(f"  {celltype}: no valid genes found (filtered by span: {filtered_by_span})")

        top_genes_list.append(celltype_top)

    print(f"\nTotal genes filtered by exon span: {total_filtered_by_span}")

    # Combine all celltypes
    result_df = pd.concat(top_genes_list, ignore_index=True)

    # Remove exon_span column before writing (it was only used for filtering)
    if 'exon_span' in result_df.columns:
        result_df = result_df.drop(columns=['exon_span'])

    # Write output
    print(f"\nWriting output to: {output_csv}")
    result_df.to_csv(output_csv, index=False)
    print(f"Total genes in output: {len(result_df)}")

    if len(result_df) > 0:
        print(f"Overall max logFC: {result_df['logFC'].max():.2f}")
        print(f"Overall min logFC: {result_df['logFC'].min():.2f}")


def main():
    # File paths
    gtf_path = "/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v48/gencode.v48.annotation.gtf.gz"
    input_csv = "Data/source/DiffExpress/DiffExpress.tss.csv"
    output_csv = "Data/source/DiffExpress/DiffExpress.tss.exons.csv"
    output_top1000_csv = "Data/source/DiffExpress/DiffExpress.tss.exons.top1000.csv"

    # Add exon annotations
    add_exon_annotations(input_csv, output_csv, gtf_path)

    # Select top 1000 genes by fold change for each celltype
    select_top_genes_by_celltype(output_csv, output_top1000_csv, top_n=1000)

    print("\nDone!")


if __name__ == "__main__":
    main()
