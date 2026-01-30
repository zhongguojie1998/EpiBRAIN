#!/bin/bash
#
# Run TF-MoDISco on differential expression gradients
# This script processes all cell types in the modisco directory
#

# Define base paths
BASE_DIR="/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/modisco"

# MoDISco parameters (based on borzoi defaults)
NUM_SEQLETS=20000
WINDOW_SIZE=131072
TRIM_SIZE=24
INITIAL_FLANK=8
SEQLET_CORE_SIZE=18
SEQLET_FLANK_SIZE=8

# Slurm resource requirements
CPUS=4
MEMORY="200G"

echo "=================================================="
echo "Running TF-MoDISco for differential expression"
echo "=================================================="
echo "Base directory: ${BASE_DIR}"
echo "MoDISco parameters:"
echo "  Max seqlets: ${NUM_SEQLETS}"
echo "  Window size: ${WINDOW_SIZE}"
echo "  Trim size: ${TRIM_SIZE}"
echo "  Initial flank: ${INITIAL_FLANK}"
echo "  Seqlet core size: ${SEQLET_CORE_SIZE}"
echo "  Seqlet flank size: ${SEQLET_FLANK_SIZE}"
echo ""

# Find all cell types (based on .onehot.npz files)
for onehot_file in ${BASE_DIR}/*.onehot.npz; do
    # Extract cell type name from filename
    celltype=$(basename "${onehot_file}" .onehot.npz)

    echo "Processing cell type: ${celltype}"

    # Define file paths
    gradient_file="${BASE_DIR}/${celltype}.gradient.npz"
    output_file="${BASE_DIR}/${celltype}_modisco_results.h5"

    # Check if output already exists
    if [ -f "${output_file}" ]; then
        echo "  ✓ Results already exist: ${output_file}"
        echo "  Skipping..."
        continue
    fi

    # Check if gradient file exists
    if [ ! -f "${gradient_file}" ]; then
        echo "  ✗ Gradient file not found: ${gradient_file}"
        echo "  Skipping..."
        continue
    fi

    echo "  → Submitting MoDISco job for ${celltype}"

    # Submit MoDISco job via slurm
    slurmsub \
        -c ${CPUS} \
        -m ${MEMORY} \
        modisco motifs \
            -s "${onehot_file}" \
            -a "${gradient_file}" \
            -n ${NUM_SEQLETS} \
            -w ${WINDOW_SIZE} \
            -t ${TRIM_SIZE} \
            -g ${INITIAL_FLANK} \
            -z ${SEQLET_CORE_SIZE} \
            -f ${SEQLET_FLANK_SIZE} \
            -o "${output_file}"

    echo "  ✓ Job submitted"
    echo ""
done

echo "=================================================="
echo "All cell types processed!"
echo "=================================================="
