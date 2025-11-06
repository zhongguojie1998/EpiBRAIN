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
import warnings

import click
import h5py
import numpy as np
import pandas as pd
import gzip
from intervaltree import IntervalTree
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Exon:
    """Simple exon class."""
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Transcript:
    """Simple transcript class."""
    def __init__(self, transcript_id):
        self.transcript_id = transcript_id
        self.exons = []


class Gene:
    """Simple gene class."""
    def __init__(self, gene_id, chrom, strand):
        self.gene_id = gene_id
        self.chrom = chrom
        self.strand = strand
        self.transcripts = {}
        self._start = float('inf')
        self._end = 0

    def add_transcript(self, transcript_id):
        if transcript_id not in self.transcripts:
            self.transcripts[transcript_id] = Transcript(transcript_id)
        return self.transcripts[transcript_id]

    def update_span(self, start, end):
        self._start = min(self._start, start)
        self._end = max(self._end, end)

    def span(self):
        return (self._start, self._end)


class GTF:
    """Simple GTF parser."""
    def __init__(self, gtf_file):
        self.genes = {}
        self._parse_gtf(gtf_file)

    def _parse_gtf(self, gtf_file):
        """Parse GTF file and extract gene/transcript/exon information."""
        print(f"Parsing GTF file: {gtf_file}")

        # Handle gzipped files
        if gtf_file.endswith('.gz'):
            f = gzip.open(gtf_file, 'rt')
        else:
            f = open(gtf_file, 'r')

        for line in f:
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

            # Parse attributes
            attr_dict = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                # Handle both "key value" and "key=value" formats
                if ' "' in attr:
                    key, value = attr.split(' "', 1)
                    value = value.rstrip('"')
                elif '=' in attr:
                    key, value = attr.split('=', 1)
                    value = value.strip('"')
                else:
                    continue
                attr_dict[key] = value

            gene_id = attr_dict.get('gene_id', attr_dict.get('gene', None))
            if not gene_id:
                continue

            # Create gene if needed
            if gene_id not in self.genes:
                self.genes[gene_id] = Gene(gene_id, chrom, strand)

            gene = self.genes[gene_id]
            gene.update_span(start, end)

            # Handle transcripts and exons
            if feature in ['transcript', 'exon']:
                transcript_id = attr_dict.get('transcript_id', attr_dict.get('transcript', None))
                if transcript_id:
                    transcript = gene.add_transcript(transcript_id)

                    if feature == 'exon':
                        exon = Exon(start, end)
                        transcript.exons.append(exon)

        f.close()
        print(f"Parsed {len(self.genes)} genes")


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
        Aggregated prediction values for each trial
    """
    aggregated = []

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
            exon_pred = prediction[bin_start:bin_end]
            aggregated.append(exon_pred)

    if len(aggregated) > 0:
        # Concatenate all exon predictions and sum
        all_exon_preds = np.concatenate(aggregated, axis=0)
        return all_exon_preds.sum(axis=0)
    else:
        return None


def process_variant_by_gene(variant_h5_path, gtf, label_meta, pool_width=32):
    """
    Process a single variant and calculate gene-level effects.

    Args:
        variant_h5_path: Path to variant effect h5 file
        gtf: GTF object with gene annotations
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
        if row['modality'] == 'RNAminus':
            rna_minus_tracks.append((row['dim'], row['trial']))
        elif row['modality'] == 'RNAplus':
            rna_plus_tracks.append((row['dim'], row['trial']))

    if len(rna_minus_tracks) == 0 or len(rna_plus_tracks) == 0:
        print(f"Warning: Could not find RNAminus or RNAplus tracks in label metadata")
        return pd.DataFrame()

    print(f"  Found {len(rna_minus_tracks)} RNAminus tracks and {len(rna_plus_tracks)} RNAplus tracks")

    # Find genes overlapping this region
    overlapping_genes = []
    for gene in gtf.genes.values():
        if gene.chrom != chr_name:
            continue

        gene_start, gene_end = gene.span()
        # Check if gene overlaps with prediction context
        if not (gene_end <= context_start or gene_start >= context_end):
            overlapping_genes.append(gene)

    print(f"  Found {len(overlapping_genes)} genes overlapping variant region")

    # Calculate gene-level effects
    results = []

    for gene in overlapping_genes:
        # Get exon intervals
        exon_intervals = get_gene_exon_intervals(gene)

        if len(exon_intervals) == 0:
            continue

        # Choose RNA tracks based on strand
        if gene.strand == '+':
            tracks_to_use = rna_plus_tracks
        elif gene.strand == '-':
            tracks_to_use = rna_minus_tracks
        else:
            continue  # Skip genes without strand information

        # Process each track for this gene
        for track_idx, track_name in tracks_to_use:
            # Aggregate predictions over exons for ref and alt
            ref_agg = aggregate_exons_from_prediction(
                pred_ref[:, [track_idx]], exon_intervals, context_start, context_end, pool_width
            )
            alt_agg = aggregate_exons_from_prediction(
                pred_alt[:, [track_idx]], exon_intervals, context_start, context_end, pool_width
            )

            if ref_agg is not None and alt_agg is not None:
                # Calculate variant effect (alt - ref)
                variant_effect = alt_agg[0] - ref_agg[0]

                results.append({
                    'chr': chr_name,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'gene_id': gene.gene_id,
                    'gene_strand': gene.strand,
                    'track': track_name,
                    'ref_value': ref_agg[0],
                    'alt_value': alt_agg[0],
                    'variant_effect': variant_effect,
                    'n_exons': len(exon_intervals)
                })

    return pd.DataFrame(results)


@click.command()
@click.option("--vcf", "-f", required=True, type=str, help="Path to the VCF file")
@click.option("--exp_name", "-e", required=True, type=str, help="Experiment name")
@click.option("--chk", required=True, type=str, help="Checkpoint name")
@click.option("--res_base", required=True, type=str, default="./Res", help="Results base directory")
@click.option("--log_base", required=True, type=str, default="./logs", help="Logs base directory")
@click.option("--genes_gtf", type=str, default="Data/source/gencode.v48.annotation.gtf.gz", help="Path to genes GTF file")
@click.option("--pool_width", type=int, default=32, help="Prediction bin width in bp")
@click.option("--output", "-o", type=str, default=None, help="Output file path (default: <res_base>/<exp_name>/analysis_<chk>/variant_effects_by_gene.tsv)")
def main(vcf, exp_name, chk, res_base, log_base, genes_gtf, pool_width, output):
    """Calculate gene-level variant effects from VCF predictions."""

    RES_BASE = os.path.abspath(res_base)
    LOG_BASE = os.path.abspath(log_base)

    # Set default output path
    if output is None:
        output = f"{RES_BASE}/{exp_name}/analysis_{chk}/variant_effects_by_gene.tsv"

    print(f"Loading gene annotations from {genes_gtf}...")
    gtf = GTF(genes_gtf)

    # Load label metadata
    label_meta_path = f"{LOG_BASE}/{exp_name}/regression_label_meta.csv"
    print(f"Loading label metadata from {label_meta_path}...")
    label_meta = pd.read_csv(label_meta_path, index_col=None)

    # Read VCF file
    print(f"Reading VCF file: {vcf}")
    vcf_df = pd.read_csv(vcf, sep="\t", comment='#', header=None)

    # Process each variant
    all_results = []
    variant_dir = f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/var_eff"

    print(f"\nProcessing {len(vcf_df)} variants...")
    for i in tqdm(range(len(vcf_df)), desc="Processing variants"):
        chr_name = vcf_df.iloc[i, 0]
        pos = vcf_df.iloc[i, 1]
        ref = vcf_df.iloc[i, 3]
        alt = vcf_df.iloc[i, 4]

        variant_name = f"{chr_name}_{ref}{pos}{alt}"
        variant_h5 = f"{variant_dir}/{variant_name}.h5"

        if not os.path.exists(variant_h5):
            print(f"Warning: Variant file not found: {variant_h5}")
            continue

        # Process this variant
        variant_results = process_variant_by_gene(variant_h5, gtf, label_meta, pool_width)

        if len(variant_results) > 0:
            all_results.append(variant_results)

    # Combine all results
    if len(all_results) > 0:
        final_results = pd.concat(all_results, ignore_index=True)

        # Save results
        print(f"\nSaving results to {output}...")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        final_results.to_csv(output, sep='\t', index=False)

        print(f"Processed {len(final_results)} gene-variant pairs")
        print(f"Unique genes: {final_results['gene_id'].nunique()}")
        print(f"Unique variants: {len(final_results[['chr', 'pos', 'ref', 'alt']].drop_duplicates())}")

        # Summary statistics
        print("\nVariant effect summary:")
        print(final_results['variant_effect'].describe())
    else:
        print("No results generated!")

    print("\nDone!")


if __name__ == "__main__":
    main()
