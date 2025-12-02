#!/usr/bin/env python
"""
03_4_variant_effect_by_gene.py

Calculate gene-level variant effects from VCF file predictions.
For each variant, aggregates predictions over gene exons and computes
the difference between alt and ref alleles (alt - ref) at the gene level.

Workflow:
1. Load variant effect predictions (from 03_variant_effect.py output)
2. Load gene annotations from GTF
3. For each gene overlapping the variant region:
   - Extract exons for that gene
   - Use RNAplus track for plus-strand genes, RNAminus for minus-strand
   - Aggregate predictions over exon regions for both ref and alt
   - Calculate gene-level variant effect (alt - ref)
"""

import os
import sys
import warnings

import click
import h5py
import numpy as np
import pandas as pd
import pyranges as pr
from intervaltree import IntervalTree
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Import pygene for GTF parsing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pygene


def create_genes_pyranges(gtf):
    """
    Create a PyRanges object from GTF genes for efficient overlap queries.

    Args:
        gtf: pygene.GTF object

    Returns:
        PyRanges object with gene spans
    """
    gene_data = []
    for gene_id, gene in gtf.genes.items():
        gene_start, gene_end = gene.span()
        gene_data.append({
            'Chromosome': gene.chrom,
            'Start': gene_start - 1,  # Convert to 0-based
            'End': gene_end,
            'gene_id': gene_id,
            'Strand': gene.strand
        })

    genes_df = pd.DataFrame(gene_data)
    return pr.PyRanges(genes_df)


def get_gene_exon_intervals(gene):
    """
    Get merged exon intervals for a gene.

    Args:
        gene: Gene object

    Returns:
        List of (start, end) tuples representing merged exon regions
    """
    # Collect all exons from all transcripts
    gene_intervals = IntervalTree()
    for tx in gene.transcripts.values():
        for exon in tx.exons:
            gene_intervals[exon.start - 1 : exon.end] = True

    # Merge overlapping intervals
    gene_intervals.merge_overlaps()

    # Convert to list of tuples
    exon_intervals = [(interval.begin, interval.end) for interval in sorted(gene_intervals)]
    return exon_intervals


def aggregate_exons_from_prediction(prediction, exon_intervals, context_start, context_end, pool_width):
    """
    Aggregate predictions over exon regions.

    Args:
        prediction: Prediction array (n_bins, n_trials)
        exon_intervals: List of (start, end) tuples for exons (genomic coordinates)
        context_start: Start position of prediction context in genomic coordinates
        context_end: End position of prediction context in genomic coordinates
        pool_width: Width of each prediction bin in bp

    Returns:
        Tuple of (aggregated_values, n_bins) where:
            - aggregated_values: Aggregated prediction values for each trial
            - n_bins: Total number of bins used in aggregation
    """
    aggregated = []
    total_bins = 0

    for start, end in exon_intervals:
        # Check if exon overlaps with prediction context
        if end <= context_start or start >= context_end:
            continue

        # Clip to context boundaries
        exon_start_in_context = max(0, start - context_start)
        exon_end_in_context = min(context_end - context_start, end - context_start)

        # Convert to bin coordinates
        bin_start = int(np.round(exon_start_in_context / pool_width))
        bin_end = int(np.round(exon_end_in_context / pool_width))

        # Extract exon region from prediction
        if bin_end > bin_start:
            n_bins_in_exon = bin_end - bin_start
            total_bins += n_bins_in_exon
            exon_pred = prediction[bin_start:bin_end]
            aggregated.append(exon_pred)

    if len(aggregated) > 0:
        # Concatenate all exon predictions and sum
        all_exon_preds = np.concatenate(aggregated, axis=0)
        return all_exon_preds.sum(axis=0), total_bins
    else:
        return None, 0


def process_variant_by_gene(variant_h5_path, gtf, genes_pr, label_meta, pool_width=32):
    """
    Process a single variant and calculate gene-level effects.

    Args:
        variant_h5_path: Path to variant effect h5 file
        gtf: GTF object with gene annotations
        genes_pr: PyRanges object with gene spans for efficient overlap queries
        label_meta: DataFrame with label metadata (must have 'modality' and 'trial' columns)
        pool_width: Width of prediction bins in bp (default: 32)

    Returns:
        DataFrame with gene-level variant effects
    """
    # Load variant data
    with h5py.File(variant_h5_path, 'r') as f:
        pred_ref = f['data']['pred_wt'][:]  # (n_bins, n_trials)
        pred_alt = f['data']['pred_alt'][:]  # (n_bins, n_trials)
        context_start = f.attrs['context_start']
        context_end = f.attrs['context_end']
        chr_name = str(variant_h5_path).split('/')[-1].split('_')[0]  # Extract chr from filename
        pos = f.attrs['pos']
        ref = f.attrs['ref']
        alt = f.attrs['alt']

    # Get RNAminus and RNAplus track indices (lists for all cell types)
    rna_minus_tracks = []  # List of (idx, trial_name) tuples
    rna_plus_tracks = []   # List of (idx, trial_name) tuples

    for _, row in label_meta.iterrows():
        if row['modality'] == 'RNAplus': # 10x RNAplus and RNAminus are reversed
            rna_minus_tracks.append((row['dim'], row['trial']))
        elif row['modality'] == 'RNAminus':
            rna_plus_tracks.append((row['dim'], row['trial']))
        else:
            # append to both for non-RNA modalities
            rna_minus_tracks.append((row['dim'], row['trial']))
            rna_plus_tracks.append((row['dim'], row['trial']))

    if len(rna_minus_tracks) == 0 or len(rna_plus_tracks) == 0:
        print(f"Warning: Could not find RNAminus or RNAplus tracks in label metadata")
        return pd.DataFrame()

    # print(f"  Found {len(rna_minus_tracks)} RNAminus tracks and {len(rna_plus_tracks)} RNAplus tracks")

    # Find genes overlapping this region using PyRanges
    variant_pr = pr.PyRanges(pd.DataFrame({
        'Chromosome': [chr_name],
        'Start': [context_start],
        'End': [context_end]
    }))

    # Get overlapping genes
    overlapping_pr = genes_pr.overlap(variant_pr)

    if len(overlapping_pr) == 0:
        print(f"  Found 0 genes overlapping variant region")
        return pd.DataFrame()

    overlapping_gene_ids = overlapping_pr.df['gene_id'].tolist()
    overlapping_genes = [(gene_id, gtf.genes[gene_id]) for gene_id in overlapping_gene_ids]

    print(f"  Found {len(overlapping_genes)} genes overlapping variant region")

    # Calculate gene-level effects
    results = []

    for gene_id, gene in overlapping_genes:
        # Get exon intervals
        exon_intervals = get_gene_exon_intervals(gene)

        if len(exon_intervals) == 0:
            continue

        # Get gene span and calculate TSS
        gene_start, gene_end = gene.span()
        if gene.strand == '+':
            tss = gene_start
            tracks_to_use = rna_plus_tracks
        elif gene.strand == '-':
            tss = gene_end
            tracks_to_use = rna_minus_tracks
        else:
            continue  # Skip genes without strand information

        # Calculate distance from variant to TSS (signed: positive if downstream, negative if upstream)
        distance_to_tss = pos - tss

        # Get all track indices and names
        track_indices = [idx for idx, _ in tracks_to_use]
        track_names = [name for _, name in tracks_to_use]

        # Aggregate predictions over exons for all tracks at once
        ref_agg, gene_length_bins = aggregate_exons_from_prediction(
            pred_ref[:, track_indices], exon_intervals, context_start, context_end, pool_width
        )
        alt_agg, _ = aggregate_exons_from_prediction(
            pred_alt[:, track_indices], exon_intervals, context_start, context_end, pool_width
        )

        if ref_agg is not None and alt_agg is not None:
            # Calculate variant effects for all tracks at once
            variant_effects = alt_agg - ref_agg

            # Create DataFrame chunk for this gene with all tracks
            gene_results = pd.DataFrame({
                'chr': chr_name,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'gene_id': gene_id,
                'gene_name': gene.name if hasattr(gene, 'name') else gene_id,
                'gene_strand': gene.strand,
                'gene_length_bins': gene_length_bins,
                'gene_tss': tss,
                'distance_to_tss': distance_to_tss,
                'track': track_names,
                'ref_value': ref_agg,
                'alt_value': alt_agg,
                'variant_effect': variant_effects,
                'n_exons': len(exon_intervals)
            })
            results.append(gene_results)

    if len(results) > 0:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


# Global variables shared across worker threads
_worker_gtf = None
_worker_genes_pr = None


def process_single_variant(variant_info, variant_dir, label_meta, pool_width):
    """
    Process a single variant from the VCF file.
    Uses global _worker_gtf and _worker_genes_pr initialized in main process.

    Args:
        variant_info: Tuple of (chr_name, pos, ref, alt)
        variant_dir: Directory containing variant effect h5 files
        label_meta: DataFrame with label metadata
        pool_width: Width of prediction bins in bp

    Returns:
        DataFrame with gene-level variant effects for this variant, or None if file not found
    """
    global _worker_gtf, _worker_genes_pr

    chr_name, pos, ref, alt = variant_info

    variant_name = f"{chr_name}_{ref}{pos}{alt}"
    variant_h5 = f"{variant_dir}/{variant_name}.h5"

    if not os.path.exists(variant_h5):
        print(f"Warning: Variant file not found: {variant_h5}")
        return None

    # Process this variant using global worker variables
    variant_results = process_variant_by_gene(variant_h5, _worker_gtf, _worker_genes_pr, label_meta, pool_width)

    if len(variant_results) > 0:
        return variant_results
    return None


@click.command()
@click.option("--vcf", "-f", required=True, type=str, help="Path to the VCF file")
@click.option("--exp_name", "-e", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint name")
@click.option("--res_base", required=True, type=str, default="./Res", help="Results base directory")
@click.option("--log_base", required=True, type=str, default="./logs", help="Logs base directory")
@click.option("--genes_gtf", type=str, default="Data/source/gencode.v48.annotation.gtf.gz", help="Path to genes GTF file")
@click.option("--pool_width", type=int, default=32, help="Prediction bin width in bp")
@click.option("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 uses all cores)")
@click.option("--output", "-o", type=str, default=None, help="Output file path (default: <res_base>/<exp_name>/analysis_<chk>/var_eff/variant_effects_by_gene.tsv)")
def main(vcf, exp_name, chk, res_base, log_base, genes_gtf, pool_width, n_jobs, output):
    """Calculate gene-level variant effects from VCF predictions."""

    RES_BASE = os.path.abspath(res_base)
    LOG_BASE = os.path.abspath(log_base)

    # Set default output path
    if output is None:
        output = f"{RES_BASE}/{exp_name}/analysis_{chk}/var_eff/variant_effects_by_gene.tsv"

    # Initialize GTF in main process (will be shared with worker threads)
    print(f"Loading gene annotations from {genes_gtf}...")
    global _worker_gtf, _worker_genes_pr
    _worker_gtf = pygene.GTF(genes_gtf)
    print("Creating PyRanges object for efficient gene overlap queries...")
    _worker_genes_pr = create_genes_pyranges(_worker_gtf)

    # Load label metadata
    label_meta_path = f"{LOG_BASE}/{exp_name}/regression_label_meta.csv"
    print(f"Loading label metadata from {label_meta_path}...")
    label_meta = pd.read_csv(label_meta_path, index_col=None)

    # Read VCF file
    print(f"Reading VCF file: {vcf}")
    vcf_df = pd.read_csv(vcf, sep="\t", comment='#', header=None)

    # Process each variant
    variant_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/var_eff/raw_data/"

    # Prepare variant information list
    variant_infos = [
        (vcf_df.iloc[i, 0], vcf_df.iloc[i, 1], vcf_df.iloc[i, 3], vcf_df.iloc[i, 4])
        for i in range(len(vcf_df))
    ]

    print(f"\nProcessing {len(variant_infos)} variants in parallel with {n_jobs} jobs...")
    # Process variants in parallel using threading backend (shares memory)
    all_results = Parallel(
        n_jobs=n_jobs,
        verbose=10,
        require='sharedmem'
    )(
        delayed(process_single_variant)(variant_info, variant_dir, label_meta, pool_width)
        for variant_info in variant_infos
    )

    # Filter out None results
    all_results = [result for result in all_results if result is not None]

    # Combine all results
    if len(all_results) > 0:
        final_results = pd.concat(all_results, ignore_index=True)

        # Save results
        print(f"\nSaving results to {output}...")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        final_results.to_csv(output, sep='\t', index=False)

        print(f"Processed {len(final_results)} gene-variant pairs")
        print(f"Unique genes: {final_results['gene_id'].nunique()}")
        print(f"Unique gene names: {final_results['gene_name'].nunique()}")
        print(f"Unique variants: {len(final_results[['chr', 'pos', 'ref', 'alt']].drop_duplicates())}")

        # Summary statistics
        print("\nVariant effect summary:")
        print(final_results['variant_effect'].describe())
    else:
        print("No results generated!")

    print("\nDone!")


if __name__ == "__main__":
    main()
