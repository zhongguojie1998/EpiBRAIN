#!/bin/bash

# Script to filter sumstats file to keep only SNPs
# Usage: ./filter_snps.sh input.tsv.gz output.tsv.gz

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file.tsv.gz> <output_file.tsv.gz>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "Filtering SNPs from $INPUT_FILE to $OUTPUT_FILE"

# Process line by line, keeping header and filtering for SNPs
# SNPs are defined as variants where REF and ALT are both single nucleotides (A, T, C, G)
zcat "$INPUT_FILE" | awk '
BEGIN { FS=OFS="\t" }
NR==1 {
    print  # Print header
    next
}
{
    # Assumes REF and ALT are in typical columns (adjust column numbers if needed)
    # Common formats have REF in column 4 or 5, ALT in column 5 or 6
    # This checks all fields to find single nucleotides
    is_snp = 0
    for (i=1; i<=NF; i++) {
        if ($i ~ /^[ATCG]$/) {
            ref_col = i
            # Check if next column is also single nucleotide
            if ($(i+1) ~ /^[ATCG]$/) {
                is_snp = 1
                break
            }
        }
    }

    # Alternative approach: assume standard VCF-like format
    # Check if there are columns that look like REF and ALT (single bases)
    # Count fields that are single nucleotides
    single_nuc_count = 0
    for (i=1; i<=NF; i++) {
        if ($i ~ /^[ATCG]$/) {
            single_nuc_count++
        }
    }

    # If we have at least 2 single nucleotide fields, likely a SNP
    if (single_nuc_count >= 2) {
        print
    }
}
' | gzip > "$OUTPUT_FILE"

echo "Done! Output written to $OUTPUT_FILE"

# Print some statistics
TOTAL_LINES=$(zcat "$INPUT_FILE" | wc -l)
OUTPUT_LINES=$(zcat "$OUTPUT_FILE" | wc -l)
echo "Total input lines: $TOTAL_LINES"
echo "SNP lines (including header): $OUTPUT_LINES"
echo "Filtered SNPs: $((OUTPUT_LINES - 1))"
