#!/bin/bash
#
# Example usage of pyGenomeTracks visualization script
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_SCRIPT="${SCRIPT_DIR}/00_visualize_data_pygenometrack.py"

echo "pyGenomeTracks Visualization Examples"
echo "======================================"
echo ""

# Example 1: Plot a specific region
echo "Example 1: Plot a specific genomic region"
echo "  Region: chr1:10000000-10100000"
python3 ${VIZ_SCRIPT} \
    --region chr1:10000000-10100000 \
    --output-prefix region_chr1_10Mb \
    --width 40 \
    --dpi 300

echo ""
echo "Example 2: Auto-detect and plot 3 regions with highest annotation scores"
python3 ${VIZ_SCRIPT} \
    --auto-regions 3 \
    --output-prefix high_score \
    --width 40 \
    --dpi 300

echo ""
echo "Example 3: Plot specific tracks only"
python3 ${VIZ_SCRIPT} \
    --region chr1:10000000-10100000 \
    --tracks all BasalGanglia-Astrocyte_ATAC MiniAtlas-PVALB_ATAC \
    --output-prefix selected_tracks \
    --width 40 \
    --dpi 300

echo ""
echo "Example 4: Plot all tracks for chromosome 22"
python3 ${VIZ_SCRIPT} \
    --region chr22:20000000-20100000 \
    --chrom chr22 \
    --output-prefix chr22_example \
    --width 40 \
    --dpi 300

echo ""
echo "Example 5: Just create bedGraph files and config (no plotting)"
python3 ${VIZ_SCRIPT} \
    --output-prefix test

echo ""
echo "======================================"
echo "Examples complete! Check the visualizations/ directory for output files."
