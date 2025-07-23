import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

refseq_map = {
    "NC_000001.11": "chr1",
    "NC_000002.12": "chr2",
    "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",
    "NC_000005.10": "chr5",
    "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",
    "NC_000008.11": "chr8",
    "NC_000009.12": "chr9",
    "NC_000010.11": "chr10",
    "NC_000011.10": "chr11",
    "NC_000012.12": "chr12",
    "NC_000013.11": "chr13",
    "NC_000014.9": "chr14",
    "NC_000015.10": "chr15",
    "NC_000016.10": "chr16",
    "NC_000017.11": "chr17",
    "NC_000018.10": "chr18",
    "NC_000019.10": "chr19",
    "NC_000020.11": "chr20",
    "NC_000021.9": "chr21",
    "NC_000022.11": "chr22",
    "NC_000023.11": "chrX",
    "NC_000024.10": "chrY",
}

def load_sum_stat(sum_stat_path):
    """Load GWAS summary statistics with pandas"""
    df = pd.read_csv(sum_stat_path, sep="\t", compression="gzip" if sum_stat_path.endswith(".gz") else None)

    # Validate required columns
    required_cols = ["SNP", "A1", "A2", "Z", "N"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean data and convert types
    df = df.dropna(subset=required_cols)
    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df.dropna(subset=["Z", "N"])

    # Rename columns for consistency
    df = df.rename(columns={"SNP": "rsid", "A1": "alt", "A2": "ref", "Z": "z_score", "N": "n_sample"})

    return df


def query_vcf_with_rsidx(rsids, vcf_path, rsidx_path):
    """Use rsidx command line to efficiently query VCF for rsIDs"""
    if not rsids:
        return {}

    # Check if rsidx file exists
    if not os.path.exists(rsidx_path):
        print(f"Error: rsidx file not found: {rsidx_path}")
        print("Please generate the rsidx file first using:")
        print(f"  rsidx index {vcf_path} {rsidx_path}")
        exit(1)

    # Write rsids to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        for rsid in rsids:
            temp_file.write(f"{rsid}\n")
        temp_file_path = temp_file.name

    print(f"Querying {len(rsids)} rsIDs using rsidx...")

    try:
        # Use rsidx command line to query with file
        cmd = ["rsidx", "search", vcf_path, rsidx_path, "--file", temp_file_path]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"Error: rsidx query failed: {result.stderr}")
            print(f"Temporary file with rsIDs: {temp_file_path}")
            print("Checking for problematic rsIDs...")

            # Read and show first few lines of temp file for debugging
            with open(temp_file_path, 'r') as f:
                lines = f.readlines()[:10]
                print(f"First 10 rsIDs in temp file:")
                for i, line in enumerate(lines):
                    print(f"  {i+1}: {line.strip()}")

            print(f"Temp file preserved at: {temp_file_path}")
            return {}

        # Parse results
        vcf_variants = {}
        print("Parsing...")
        for line in tqdm(result.stdout.strip().split("\n")):
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 5:
                chrom, pos, rsid, ref, alt_list = parts[:5]
                # Handle multiple ALT alleles
                alts = alt_list.split(",")
                vcf_variants[rsid] = {"chr": chrom, "pos": int(pos), "ref": ref, "alts": alts}

        print(f"rsidx query complete: found {len(vcf_variants)} variants")
        return vcf_variants

    finally:
        # Clean up temporary file (only if no error occurred)
        if result.returncode == 0:
            os.unlink(temp_file_path)
        # If there was an error, temp file is preserved for debugging


def validate_and_match_alleles(sum_stat_variant, vcf_variant, allow_ref_alt_reverse):
    """Validate exact allele match"""
    if not vcf_variant:
        return False, None, "VCF_NOT_FOUND"

    if vcf_variant["chr"] not in refseq_map:
        return False, None, "NOT_STD_CHROM"

    sum_ref = sum_stat_variant["ref"].upper()
    sum_alt = sum_stat_variant["alt"].upper()
    vcf_ref = vcf_variant["ref"].upper()
    vcf_alts = [alt.upper() for alt in vcf_variant["alts"]]

    # Only accept exact matches
    if sum_ref == vcf_ref and sum_alt in vcf_alts:
        return True, sum_alt, "MATCH"
    if allow_ref_alt_reverse:
        if sum_alt == vcf_ref and sum_ref in vcf_alts:
            return True, sum_ref, "MATCH_REVERSE"

    return False, None, f"MISMATCH_sumstat_{sum_ref}>{sum_alt}_vcf_{vcf_ref}>{','.join(vcf_alts)}"


def create_variant_index_key(chrom, pos, ref, alt):
    """Create unique index key for variant"""
    return f"{chrom}:{pos}:{ref}:{alt}"


@click.command()
@click.option("-s", "--sum_stat_file", required=True, help="Path to GWAS summary statistics .gz file")
@click.option("-v", "--vcf_file", required=True, help="Path to VCF file")
@click.option("-o", "--output_file", required=True, help="Output enriched summary statistics file")
@click.option(
    "-r",
    "--rsidx_file",
    help="Path to rsidx index file (.rsidx) (default to the same vcf naming with .rsidx in the end)",
)
@click.option("--output_dir", help="Output directory for logs (defaults to output file directory)")
@click.option("-e", "--experiment_name", help="Experiment name for log files (defaults to output file stem)")
@click.option("-a", "--allow_ref_alt_reverse", is_flag=True, help="Whether to allow ref alt be reversed when fetching and checking from standard vcf file")
def main(sum_stat_file, vcf_file, output_file, rsidx_file, output_dir, experiment_name, allow_ref_alt_reverse):
    """Enrich GWAS summary statistics with genomic coordinates"""

    output_path = Path(output_file)

    # Setup output directory and experiment name
    if output_dir is None:
        output_dir = output_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        experiment_name = output_path.stem

    if rsidx_file is None:
        rsidx_file = vcf_file + ".rsidx"

    # processing log
    log_file = output_dir / f"log_{experiment_name}.gz"

    print(f"Input: {sum_stat_file}")
    print(f"VCF: {vcf_file}")
    print(f"rsidx: {rsidx_file}")
    print(f"Output: {output_file}")

    # Load summary statistics
    print("Loading summary statistics...")
    sum_stat_df = load_sum_stat(sum_stat_file)
    print(f"Loaded {len(sum_stat_df)} variants from summary statistics")

    # Extract unique rsIDs for VCF query
    unique_rsids = sum_stat_df["rsid"].unique().tolist()
    # TODO: other kinds of input, including pure chr:pos ; ss index, now we just remove them
    legal_rsids = [i for i in unique_rsids if i.startswith("rs")]
    print(f"Unique rsIDs: {len(legal_rsids)}")

    # Query VCF with rsidx
    vcf_results = query_vcf_with_rsidx(legal_rsids, vcf_file, rsidx_file)

    # Process all variants (including duplicates)
    print("Processing variants...")
    enriched_data = []
    logs = []

    for i, variant in tqdm(sum_stat_df.iterrows()):
        rsid = variant["rsid"]
        vcf_variant = vcf_results.get(rsid)

        is_valid, matched_alt, validation_reason = validate_and_match_alleles(variant, vcf_variant, allow_ref_alt_reverse)

        if is_valid:
            # Create enriched record
            index_key = create_variant_index_key(
                refseq_map[vcf_variant["chr"]], vcf_variant["pos"], vcf_variant["ref"], matched_alt
            )

            enriched_record = {
                "rsid": rsid,
                "chr": refseq_map[vcf_variant["chr"]],
                "pos": vcf_variant["pos"],
                "ref": vcf_variant["ref"],
                "alt": matched_alt,
                "index_key": index_key,
                "reverse_map": "REVERSE" in validation_reason,
                "z_score": variant["z_score"],
                "n_sample": variant["n_sample"],
            }
            enriched_data.append(enriched_record)

        logs.append({"idx": i, "rsid": rsid, "log": validation_reason})

    # Save enriched data
    if enriched_data:
        enriched_df = pd.DataFrame(enriched_data)
        enriched_df.to_csv(
            output_file, index=False, sep="\t", compression="gzip" if output_file.endswith(".gz") else None
        )
        print(f"Enriched summary statistics saved: {output_file}")
        print(f"Columns: {', '.join(enriched_df.columns)}")
    else:
        print("No valid variants to save!")

    # Save logs
    if logs:
        pd.DataFrame(logs).to_csv(log_file, index=False, sep="\t", compression="gzip")
        print(f"All logs saved: {log_file}")

    # Summary
    total_input = len(sum_stat_df)
    success_rate = len(enriched_data) / total_input * 100 if total_input > 0 else 0

    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Input variants: {total_input}")
    print(f"Successfully enriched: {len(enriched_data)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
