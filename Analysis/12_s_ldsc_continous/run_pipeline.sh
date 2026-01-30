#!/bin/bash
#
# Main pipeline script for running S-LDSC with continuous annotations
#
# This pipeline implements the workflow from:
# - https://kevinlkx.github.io/analysis_pipelines/sldsc_pipeline.html
# - https://github.com/bulik/ldsc/wiki/Partitioned-Heritability-from-Continuous-Annotations
#
# Steps:
# 1. Create annotation files from numpy matrix (one per chromosome per track)
# 2. Compute LD scores for each annotation
# 3. Run S-LDSC regression with baseline + continuous annotation
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "S-LDSC Pipeline for Continuous Annotations"
echo "========================================================================"
echo ""
echo "Working directory: ${SCRIPT_DIR}"
echo "Date: $(date)"
echo ""

# Step 1: Create annotation files
echo "========================================================================"
echo "Step 1: Creating annotation files from numpy matrix"
echo "========================================================================"
python3 ${SCRIPT_DIR}/01_create_annotation_files.py

if [ $? -ne 0 ]; then
    echo "Error in Step 1: Annotation file creation failed"
    exit 1
fi

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Compute LD scores
echo "========================================================================"
echo "Step 2: Computing LD scores for each annotation"
echo "========================================================================"
bash ${SCRIPT_DIR}/02_compute_ld_scores.sh

if [ $? -ne 0 ]; then
    echo "Error in Step 2: LD score computation failed"
    exit 1
fi

echo ""
echo "Step 2 complete!"
echo ""

# Step 3: Run S-LDSC regression
echo "========================================================================"
echo "Step 3: Running S-LDSC regression"
echo "========================================================================"
bash ${SCRIPT_DIR}/03_run_sldsc_regression.sh

if [ $? -ne 0 ]; then
    echo "Error in Step 3: S-LDSC regression failed"
    exit 1
fi

echo ""
echo "Step 3 complete!"
echo ""

# Final summary
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Output directories:"
echo "  - Annotations: ${SCRIPT_DIR}/archive/annotations/"
echo "  - Results: ${SCRIPT_DIR}/archive/results/"
echo ""
echo "NOTE: This script outputs to archive/ for backward compatibility."
echo "      For new analyses, use run_pipeline_ultra_parallel.sh which outputs"
echo "      to organized *_by_trait/ directories."
echo ""
echo "Key files:"
echo "  - Individual results: ${SCRIPT_DIR}/archive/results/*.results"
echo "  - Summary table: ${SCRIPT_DIR}/archive/results/summary_results.txt"
echo ""
echo "To view summary results:"
echo "  cat ${SCRIPT_DIR}/archive/results/summary_results.txt"
echo ""
echo "========================================================================"
