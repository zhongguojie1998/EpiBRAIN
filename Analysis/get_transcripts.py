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

def overlaps(region_start: int, region_end: int, transcript_start: int, transcript_end: int) -> bool:
    """Check if two genomic intervals overlap."""
    return not (region_end < transcript_start or region_start > transcript_end)

def get_transcripts_in_region(region: str, gtf_file: str = "Data/source/gencode.v48.annotation.gtf.gz", window_size=32, n_window=None, filter_to_full_transcript=False, filter_to_longest=False) -> pd.DataFrame:
    """
    Extract all transcripts that overlap with given genomic region.
    
    Args:
        region: Genomic region in format 'chr:start-end'
        gtf_file: Path to GTF file
    
    Returns:
        DataFrame with columns: transcriptID, geneName, chr, start, end, start_bin_idx, end_bin_idx
        Bin indices are calculated by dividing the region into 32-bp bins starting from 0
    """
    
    # Parse input region
    chr_name, region_start, region_end = parse_region(region)
    
    transcripts = []

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
            
            # Only process transcript entries
            if parts[2] != 'transcript':
                continue
            
            transcript_chr = parts[0]
            transcript_start = int(parts[3])
            transcript_end = int(parts[4])
            attributes = parts[8]
            
            # Skip if chromosome doesn't match
            if transcript_chr != chr_name:
                continue
            
            # Check if transcript overlaps with the region
            if overlaps(region_start, region_end, transcript_start, transcript_end):
                transcript_id = extract_transcript_id(attributes)
                gene_name = extract_gene_name(attributes)
                if transcript_id:
                    if filter_to_full_transcript:
                        # only include transcripts fully contained within the region
                        if transcript_start < region_start or transcript_end > region_end:
                            continue
                    # Calculate bin indices (32-bp bins starting from 0)
                    # Transcript coordinates relative to region start
                    transcript_rel_start = max(0, transcript_start - region_start)
                    transcript_rel_end = min(region_end - region_start, transcript_end - region_start)
                    
                    # Calculate bin indices
                    start_bin_idx = transcript_rel_start // 32
                    end_bin_idx = transcript_rel_end // 32
                    
                    transcripts.append({
                        'transcriptID': transcript_id,
                        'geneName': gene_name,
                        'chr': transcript_chr,
                        'start': transcript_start,
                        'end': transcript_end,
                        'start_bin_idx': start_bin_idx,
                        'end_bin_idx': end_bin_idx,
                        'length': transcript_end - transcript_start,
                        'length_bin': end_bin_idx - start_bin_idx
                    })
    
    # Convert to DataFrame and sort
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
    
    result = get_transcripts_in_region(example_region)
    print(f"\nFound {len(result)} transcripts:")
    print(result.head(10))
    
    if len(result) > 10:
        print(f"... (showing first 10 out of {len(result)} total transcripts)")