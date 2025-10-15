#!/usr/bin/env python3
"""
Simple function to extract transcript information from GTF file based on genomic regions.
Can be imported and used programmatically.
"""

import pandas as pd
import gzip
import re
from typing import List, Tuple

def parse_region(region: str) -> Tuple[str, int, int]:
    """Parse genomic region string like 'chr1:1000-2000' into components."""
    if ':' not in region or '-' not in region:
        raise ValueError(f"Invalid region format: {region}. Expected format: chr:start-end")
    
    chr_part, pos_part = region.split(':', 1)
    start_str, end_str = pos_part.split('-', 1)
    
    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        raise ValueError(f"Invalid coordinates in region: {region}")
    
    if start > end:
        raise ValueError(f"Start position ({start}) must be <= end position ({end})")
    
    return chr_part, start, end

def extract_transcript_id(attributes: str) -> str:
    """Extract transcript_id from GTF attributes column."""
    match = re.search(r'transcript_id "([^"]+)"', attributes)
    return match.group(1) if match else ""

def extract_gene_name(attributes: str) -> str:
    """Extract gene_name from GTF attributes column."""
    match = re.search(r'gene_name "([^"]+)"', attributes)
    return match.group(1) if match else ""

def extract_gene_id(attributes: str) -> str:
    """Extract gene_id from GTF attributes column."""
    match = re.search(r'gene_id "([^"]+)"', attributes)
    return match.group(1) if match else ""

def extract_gene_type(attributes: str) -> str:
    """Extract gene_type (or gene_biotype) from GTF attributes column."""
    # Try gene_type first (newer GENCODE versions)
    match = re.search(r'gene_type "([^"]+)"', attributes)
    if match:
        return match.group(1)
    # Fall back to gene_biotype (older GENCODE versions)
    match = re.search(r'gene_biotype "([^"]+)"', attributes)
    return match.group(1) if match else ""

def overlaps(region_start: int, region_end: int, transcript_start: int, transcript_end: int) -> bool:
    """Check if two genomic intervals overlap."""
    return not (region_end < transcript_start or region_start > transcript_end)

def get_transcripts_in_region(region: str, gtf_file: str = "Data/source/gencode.v48.annotation.gtf.gz", window_size=32, n_window=None,
                              filter_to_full_transcript=False, filter_to_longest=False, return_exon_bins_only=False,
                              filter_protein_coding=False) -> pd.DataFrame:
    """
    Extract all transcripts that overlap with given genomic region.

    Args:
        region: Genomic region in format 'chr:start-end'
        gtf_file: Path to GTF file
        window_size: Size of bins in base pairs (default: 32)
        n_window: Number of windows to adjust region (optional)
        filter_to_full_transcript: Only include transcripts fully contained within region
        filter_to_longest: Keep only longest transcript per gene
        return_exon_bins_only: If True, aggregate by gene_id and return exon bin indices for each gene
                               (combines all exons from all transcripts/isoforms of the same gene)
        filter_protein_coding: If True, only include protein_coding genes (default: False)

    Returns:
        When return_exon_bins_only=False (default):
            DataFrame with columns: transcriptID, geneName, geneType, chr, start, end, start_bin_idx, end_bin_idx
        When return_exon_bins_only=True:
            DataFrame aggregated by gene with columns: geneID, geneName, geneType, chr, start, end,
            start_bin_idx, end_bin_idx, exon_bins, num_exon_bins
            The 'exon_bins' column contains a list of bin indices corresponding to exons from
            ALL transcripts/isoforms of that gene (union of all exons).
        Bin indices are calculated by dividing the region into window_size-bp bins starting from 0
    """
    
    # Parse input region
    chr_name, region_start, region_end = parse_region(region)

    transcripts = []
    exons_dict = {}  # Will store exons for each gene/transcript if return_exon_bins_only=True
    gene_info = {}  # Will store gene-level info when aggregating by gene_id

    # if n_window is not None, adjust region coordinates
    if n_window is not None:
        target_window = (region_end - region_start) // window_size
        trim = (n_window - target_window) // 2
        region_start = max(0, region_start + trim * window_size)
        region_end = region_end - trim * window_size

    # Process GTF file
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            transcript_chr = parts[0]
            attributes = parts[8]

            # Skip if chromosome doesn't match
            if transcript_chr != chr_name:
                continue

            # Process transcript entries
            if feature_type == 'transcript':
                transcript_start = int(parts[3])
                transcript_end = int(parts[4])

                # Check if transcript overlaps with the region
                if overlaps(region_start, region_end, transcript_start, transcript_end):
                    transcript_id = extract_transcript_id(attributes)
                    gene_name = extract_gene_name(attributes)
                    gene_id = extract_gene_id(attributes)
                    gene_type = extract_gene_type(attributes)

                    # Apply protein_coding filter if requested
                    if filter_protein_coding and gene_type != "protein_coding":
                        continue

                    if transcript_id:
                        if filter_to_full_transcript:
                            # only include transcripts fully contained within the region
                            if transcript_start < region_start or transcript_end > region_end:
                                continue
                        # Calculate bin indices (window_size-bp bins starting from 0)
                        # Transcript coordinates relative to region start
                        transcript_rel_start = max(0, transcript_start - region_start)
                        transcript_rel_end = min(region_end - region_start, transcript_end - region_start)

                        # Calculate bin indices with bounds checking
                        max_bin_idx = (region_end - region_start) // window_size - 1
                        start_bin_idx = transcript_rel_start // window_size
                        end_bin_idx = min(transcript_rel_end // window_size, max_bin_idx)

                        if return_exon_bins_only:
                            # When aggregating by gene, store gene-level info
                            if gene_id not in gene_info:
                                gene_info[gene_id] = {
                                    'geneID': gene_id,
                                    'geneName': gene_name,
                                    'geneType': gene_type,
                                    'chr': transcript_chr,
                                    'start': transcript_start,
                                    'end': transcript_end,
                                    'start_bin_idx': start_bin_idx,
                                    'end_bin_idx': end_bin_idx
                                }
                            else:
                                # Update gene range to include this transcript
                                gene_info[gene_id]['start'] = min(gene_info[gene_id]['start'], transcript_start)
                                gene_info[gene_id]['end'] = max(gene_info[gene_id]['end'], transcript_end)
                                gene_info[gene_id]['start_bin_idx'] = min(gene_info[gene_id]['start_bin_idx'], start_bin_idx)
                                gene_info[gene_id]['end_bin_idx'] = max(gene_info[gene_id]['end_bin_idx'], end_bin_idx)

                            # Initialize exon list for this transcript
                            exons_dict[transcript_id] = {'gene_id': gene_id}
                        else:
                            # Original behavior: store transcript-level info
                            transcripts.append({
                                'transcriptID': transcript_id,
                                'geneName': gene_name,
                                'geneType': gene_type,
                                'chr': transcript_chr,
                                'start': transcript_start,
                                'end': transcript_end,
                                'start_bin_idx': start_bin_idx,
                                'end_bin_idx': end_bin_idx,
                                'length': transcript_end - transcript_start,
                                'length_bin': end_bin_idx - start_bin_idx
                            })

            # Process exon entries if requested
            elif return_exon_bins_only and feature_type == 'exon':
                exon_start = int(parts[3])
                exon_end = int(parts[4])
                transcript_id = extract_transcript_id(attributes)

                # Only process exons for transcripts we're tracking
                if transcript_id in exons_dict:
                    # Check if exon overlaps with the region
                    if overlaps(region_start, region_end, exon_start, exon_end):
                        # Calculate exon coordinates relative to region
                        exon_rel_start = max(0, exon_start - region_start)
                        exon_rel_end = min(region_end - region_start, exon_end - region_start)

                        # Calculate bin indices for this exon
                        exon_start_bin = exon_rel_start // window_size
                        # Use min to ensure we don't exceed the maximum bin index
                        # For a region_length, max bin index is (region_length // window_size) - 1
                        max_bin_idx = (region_end - region_start) // window_size - 1
                        exon_end_bin = min(exon_rel_end // window_size, max_bin_idx)

                        # Store exon bins - add to list if not exists
                        if 'exons' not in exons_dict[transcript_id]:
                            exons_dict[transcript_id]['exons'] = []

                        exons_dict[transcript_id]['exons'].append({
                            'start': exon_start,
                            'end': exon_end,
                            'start_bin': exon_start_bin,
                            'end_bin': exon_end_bin
                        })

    # Convert to DataFrame and sort
    if return_exon_bins_only:
        # When return_exon_bins_only=True, aggregate by gene_id
        # Calculate max valid bin index for the region
        max_bin_idx = (region_end - region_start) // window_size - 1

        # Aggregate all exons for each gene across all transcripts
        gene_exon_bins = {}
        for transcript_id, transcript_data in exons_dict.items():
            gene_id = transcript_data['gene_id']
            if gene_id not in gene_exon_bins:
                gene_exon_bins[gene_id] = set()

            # Add all exon bins from this transcript to the gene's exon set
            if 'exons' in transcript_data:
                for exon in transcript_data['exons']:
                    # Add all bins from start to end (inclusive)
                    for bin_idx in range(exon['start_bin'], exon['end_bin'] + 1):
                        # Only add valid bin indices
                        if 0 <= bin_idx <= max_bin_idx:
                            gene_exon_bins[gene_id].add(bin_idx)

        # Convert gene_info to DataFrame
        genes = []
        for gene_id, info in gene_info.items():
            gene_record = info.copy()
            # Add exon bins for this gene
            gene_record['exon_bins'] = sorted(list(gene_exon_bins.get(gene_id, set())))
            gene_record['num_exon_bins'] = len(gene_record['exon_bins'])
            gene_record['length'] = info['end'] - info['start']
            gene_record['length_bin'] = info['end_bin_idx'] - info['start_bin_idx']
            genes.append(gene_record)

        df = pd.DataFrame(genes)
        if not df.empty:
            df = df.sort_values(['chr', 'start', 'geneID']).reset_index(drop=True)
    else:
        # Original behavior: return transcript-level data
        df = pd.DataFrame(transcripts)
        if not df.empty:
            df = df.sort_values(['chr', 'start', 'transcriptID']).reset_index(drop=True)

    if filter_to_longest and not df.empty:
        # Keep only the longest transcript per gene
        df = df.loc[df.groupby('geneName')['length_bin'].idxmax()].reset_index(drop=True)

    return df

# Example usage
if __name__ == "__main__":
    # Example region
    example_region = "chr1:11000-15000"

    print("Example usage:")
    print(f"Region: {example_region}")

    # Basic usage
    result = get_transcripts_in_region(example_region)
    print(f"\nFound {len(result)} transcripts:")
    print(result.head(10))

    if len(result) > 10:
        print(f"... (showing first 10 out of {len(result)} total transcripts)")

    # Usage with exon bins (aggregated by gene)
    print("\n\nExample with exon bins only (aggregated by gene):")
    result_exons = get_transcripts_in_region(example_region, return_exon_bins_only=True)
    if not result_exons.empty:
        print(f"Found {len(result_exons)} genes with exon information:")
        print(result_exons[['geneID', 'geneName', 'geneType', 'start_bin_idx', 'end_bin_idx', 'num_exon_bins']].head(5))
        print("\nExample exon bins for first gene:")
        if len(result_exons) > 0:
            print(f"  Gene: {result_exons.iloc[0]['geneName']} ({result_exons.iloc[0]['geneID']})")
            print(f"  Gene type: {result_exons.iloc[0]['geneType']}")
            print(f"  Full range: bins {result_exons.iloc[0]['start_bin_idx']} to {result_exons.iloc[0]['end_bin_idx']}")
            print(f"  Exon bins: {result_exons.iloc[0]['exon_bins'][:20]}..." if len(result_exons.iloc[0]['exon_bins']) > 20 else f"  Exon bins: {result_exons.iloc[0]['exon_bins']}")
            print(f"  Number of exon bins: {result_exons.iloc[0]['num_exon_bins']}")
            print(f"  Note: Exon bins include all exons from all transcripts/isoforms of this gene")

    # Usage with protein_coding filter
    print("\n\nExample with protein_coding genes only:")
    result_protein = get_transcripts_in_region(example_region, return_exon_bins_only=True, filter_protein_coding=True)
    if not result_protein.empty:
        print(f"Found {len(result_protein)} protein-coding genes:")
        print(result_protein[['geneID', 'geneName', 'geneType', 'num_exon_bins']].head(5))
    else:
        print("No protein-coding genes found in this region")