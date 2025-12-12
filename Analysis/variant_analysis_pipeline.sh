#!/bin/bash
# Comprehensive variant analysis pipeline
# This script runs a complete analysis pipeline for a variant including:
# 1. Quick inference
# 2. Saturation mutagenesis
# 3. Visualization with pygenometrack
# 4. Mutagenesis visualization
# 5. Motif interpretation
# 6. Motif interpretation plotting

set -e  # Exit on error

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --variant VARIANT           Variant in format chr:pos:ref:alt (e.g., chr11:113400106:G:T)"
    echo "  --track TRACK               Cell type/track name (e.g., BasalGanglia-STR-D1-MSN_RNAminus)"
    echo ""
    echo "Optional arguments:"
    echo "  --variant-name NAME         Variant name/RS ID (default: chr_pos_ref_alt)"
    echo "  --gene GENE                 Gene symbol (default: auto-detect from GTF)"
    echo "  --disease DISEASE           Disease name (e.g., Schizophrenia)"
    echo "  --gene-region REGION        Gene region in format chr:start-end (default: variant ± 262144bp)"
    echo "  --sat-region REGION         Saturation mutagenesis region chr:start-end (default: ±20bp around variant)"
    echo "  --trial-pos TRIAL_POS       Trial position for motif interpretation (default: derived from --track)"
    echo "  --exp-name NAME             Experiment name (default: full_finetune_original_loss_celltype_head_dim8_linear)"
    echo "  --checkpoint CHK            Checkpoint number (default: 20)"
    echo "  --output-base DIR           Base output directory (default: Analysis/figures)"
    echo "  --gtf-file PATH             GTF annotation file (default: Data/source/gencode.v48.annotation.gtf.gz)"
    echo "  --num-gpus NUM              Number of GPUs to use for saturation mutagenesis (default: 4)"
    echo "  --skip-steps STEPS          Comma-separated list of steps to skip (1-6)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Pipeline Steps:"
    echo "  Step 1: Quick inference - Predict effects of the variant"
    echo "  Step 2: Saturation mutagenesis - Test all possible SNVs in region (±20bp)"
    echo "  Step 3: Genome track visualization - Visualize predictions across gene region"
    echo "  Step 4: Mutagenesis visualization - Visualize saturation mutagenesis results"
    echo "  Step 5: Motif interpretation - Identify regulatory motifs affected by variant"
    echo "  Step 6: Motif plotting - Generate full and zoomed motif interpretation plots"
    echo ""
    echo "Example:"
    echo "  # Minimal usage (auto-detect gene, derive trial-pos from track):"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus"
    echo ""
    echo "  # Full specification:"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --variant-name rs1800497 --gene DRD2 --disease Schizophrenia"
    echo ""
    echo "  # Use 8 GPUs for saturation mutagenesis:"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --num-gpus 8"
    echo ""
    echo "Note: Track 'BasalGanglia-STR-D1-MSN_RNAminus' will auto-derive trial-pos as 'STR-D1-MSN_RNAminus'"
    exit 1
}

# Parse command line arguments
SKIP_STEPS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --variant-name)
            VARIANT_NAME="$2"
            shift 2
            ;;
        --gene)
            GENE="$2"
            shift 2
            ;;
        --track)
            TRACK="$2"
            shift 2
            ;;
        --disease)
            DISEASE="$2"
            shift 2
            ;;
        --gene-region)
            GENE_REGION="$2"
            shift 2
            ;;
        --sat-region)
            SAT_REGION="$2"
            shift 2
            ;;
        --trial-pos)
            TRIAL_POS="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --gtf-file)
            GTF_FILE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-steps)
            SKIP_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$VARIANT" ] || [ -z "$TRACK" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Set defaults
EXP_NAME=${EXP_NAME:-"full_finetune_original_loss_celltype_head_dim8_linear"}
CHECKPOINT=${CHECKPOINT:-20}
OUTPUT_BASE=${OUTPUT_BASE:-"Analysis/figures"}
GTF_FILE=${GTF_FILE:-"Data/source/gencode.v48.annotation.gtf.gz"}
NUM_GPUS=${NUM_GPUS:-4}

# Parse variant to extract chromosome and position
IFS=':' read -r CHR POS REF ALT <<< "$VARIANT"

# Auto-generate variant name if not provided
if [ -z "$VARIANT_NAME" ]; then
    VARIANT_NAME="${CHR}_${POS}_${REF}_${ALT}"
    echo "Variant name not provided, using: $VARIANT_NAME"
fi

# Auto-detect gene from GTF if not provided
if [ -z "$GENE" ]; then
    echo "Gene not provided, searching for closest gene in GTF file..."
    if [ ! -f "$GTF_FILE" ]; then
        echo "Error: GTF file not found at $GTF_FILE"
        exit 1
    fi

    # Extract genes from GTF and find the closest one
    # Use zcat if file is gzipped, otherwise cat
    if [[ "$GTF_FILE" == *.gz ]]; then
        CAT_CMD="zcat"
    else
        CAT_CMD="cat"
    fi

    # Find closest gene (looking for genes on the same chromosome)
    # GTF format: chr source feature start end score strand frame attributes
    GENE=$($CAT_CMD "$GTF_FILE" | \
        awk -v chr="$CHR" -v pos="$POS" '
        BEGIN { min_dist = 999999999; closest_gene = "unknown" }
        $1 == chr && $3 == "gene" {
            # Extract gene_name from attributes
            match($0, /gene_name "([^"]+)"/, arr)
            gene_name = arr[1]

            start = $4
            end = $5

            # Calculate distance (0 if inside gene, otherwise distance to nearest boundary)
            if (pos >= start && pos <= end) {
                dist = 0
            } else if (pos < start) {
                dist = start - pos
            } else {
                dist = pos - end
            }

            if (dist < min_dist) {
                min_dist = dist
                closest_gene = gene_name
            }
        }
        END { print closest_gene }
    ')

    if [ "$GENE" = "unknown" ] || [ -z "$GENE" ]; then
        echo "Error: Could not find a gene near variant $VARIANT in GTF file"
        exit 1
    fi

    echo "Closest gene found: $GENE"
fi

# Auto-generate trial_pos from track if not provided
# Transform: BasalGanglia-STR-D1-MSN_RNAminus -> STR-D1-MSN_RNAminus
# (Remove region prefix before first dash)
if [ -z "$TRIAL_POS" ]; then
    TRIAL_POS=$(echo "$TRACK" | sed 's/^[^-]*-//')
    echo "Trial position not provided, derived from track: $TRIAL_POS"
fi

# Generate output suffix based on disease
if [ -n "$DISEASE" ]; then
    OUTPUT_SUFFIX="${VARIANT_NAME}_${GENE}_${DISEASE}"
else
    OUTPUT_SUFFIX="${VARIANT_NAME}_${GENE}"
fi

# Calculate saturation mutagenesis region if not provided (±20bp around variant)
if [ -z "$SAT_REGION" ]; then
    SAT_START=$((POS - 20))
    SAT_END=$((POS + 20))
    SAT_REGION="${CHR}:${SAT_START}-${SAT_END}"
fi

# Calculate gene region if not provided (centered on gene center ± 262144bp)
if [ -z "$GENE_REGION" ]; then
    echo "Extracting gene coordinates from GTF..."

    # Use zcat if file is gzipped, otherwise cat
    if [[ "$GTF_FILE" == *.gz ]]; then
        CAT_CMD="zcat"
    else
        CAT_CMD="cat"
    fi

    # Extract gene start and end from GTF file
    GENE_COORDS=$($CAT_CMD "$GTF_FILE" | \
        awk -v gene="$GENE" -v chr="$CHR" '
        $1 == chr && $3 == "gene" {
            match($0, /gene_name "([^"]+)"/, arr)
            if (arr[1] == gene) {
                print $4, $5
                exit
            }
        }
    ')

    if [ -z "$GENE_COORDS" ]; then
        echo "Warning: Could not find gene $GENE in GTF, using variant position ± 262144bp"
        GENE_REGION_START=$((POS - 262144))
        GENE_REGION_END=$((POS + 262144))
    else
        read GENE_GTF_START GENE_GTF_END <<< "$GENE_COORDS"
        GENE_CENTER=$(( (GENE_GTF_START + GENE_GTF_END) / 2 ))
        echo "Gene $GENE center: $GENE_CENTER (from GTF: $GENE_GTF_START-$GENE_GTF_END)"

        GENE_REGION_START=$((GENE_CENTER - 262144))
        GENE_REGION_END=$((GENE_CENTER + 262144))
    fi

    # Ensure start is not negative
    if [ $GENE_REGION_START -lt 0 ]; then
        GENE_REGION_START=0
    fi
    GENE_REGION="${CHR}:${GENE_REGION_START}-${GENE_REGION_END}"
fi

# Create main output directory for this analysis
ANALYSIS_DIR="${OUTPUT_BASE}/${OUTPUT_SUFFIX}"
mkdir -p "${ANALYSIS_DIR}"

# Helper function to check if step should be skipped
should_skip() {
    local step=$1
    if [[ ",$SKIP_STEPS," == *",$step,"* ]]; then
        return 0
    fi
    return 1
}

echo "========================================="
echo "Variant Analysis Pipeline"
echo "========================================="
echo "Variant:       $VARIANT"
echo "Variant Name:  $VARIANT_NAME"
echo "Gene:          $GENE"
echo "Gene Region:   $GENE_REGION"
echo "Sat Region:    $SAT_REGION"
echo "Track:         $TRACK"
echo "Trial Pos:     $TRIAL_POS"
echo "Disease:       ${DISEASE:-N/A}"
echo "Experiment:    $EXP_NAME"
echo "Checkpoint:    $CHECKPOINT"
echo "Num GPUs:      $NUM_GPUS (for saturation mutagenesis)"
echo "Output Dir:    $ANALYSIS_DIR"
echo "========================================="
echo ""

# Step 1: Quick inference for the variant
if ! should_skip 1; then
    echo "[Step 1/6] Running quick inference for variant..."
    INFERENCE_DIR="${ANALYSIS_DIR}/inference"
    python Analysis/01_0_quick_inference_bigwig.py \
        --variant "$VARIANT" \
        --exp_name "$EXP_NAME" \
        --chk "$CHECKPOINT" \
        --output "$INFERENCE_DIR"
    echo "✓ Step 1 complete. Output: $INFERENCE_DIR"
    echo ""
else
    echo "[Step 1/6] Skipped."
    echo ""
fi

# Step 2: Saturation mutagenesis
if ! should_skip 2; then
    echo "[Step 2/6] Running saturation mutagenesis (using $NUM_GPUS GPUs)..."
    SAT_OUTPUT_DIR="${ANALYSIS_DIR}/sat_mutagenesis"

    python Analysis/01_0_quick_inference_bigwig.py \
        --sat_mutagenesis "$SAT_REGION" \
        --exp_name "$EXP_NAME" \
        --chk "$CHECKPOINT" \
        --output "$SAT_OUTPUT_DIR" \
        --gene "$GENE" \
        --num-gpus "$NUM_GPUS"
    echo "✓ Step 2 complete. Output: $SAT_OUTPUT_DIR"
    echo ""
else
    echo "[Step 2/6] Skipped."
    echo ""
fi

# Step 3: Visualization with pygenometrack
if ! should_skip 3; then
    echo "[Step 3/6] Running visualization with pygenometrack..."
    INFERENCE_DIR="${ANALYSIS_DIR}/inference"
    VIZ_OUTPUT="${ANALYSIS_DIR}/visualization.pdf"
    # Extract cell type pattern from full track name
    # Example: BasalGanglia-STR-D1-MSN_RNAminus -> BasalGanglia-STR-D1-MSN*
    # Keep region and cell type, replace assay type with wildcard
    CELL_TYPE_PATTERN=$(echo "$TRACK" | sed 's/_.*/*/')
    python Analysis/00_visualize_data_pygenometrack.py \
        --inference-dir "$INFERENCE_DIR" \
        --region "$GENE_REGION" \
        --output "$VIZ_OUTPUT" \
        --tracks "$CELL_TYPE_PATTERN" \
        --highlight "${CHR}:${POS}"

    # Append variant information to the existing line in visualization_vlines.bed
    VLINES_BED="${ANALYSIS_DIR}/visualization_vlines.bed"
    if [ -f "$VLINES_BED" ]; then
        echo "  Appending variant information to visualization_vlines.bed..."
        # Append variant name to the existing line
        sed -i "s/$/ ${VARIANT_NAME}_${REF}>${ALT}/" "$VLINES_BED"
    fi

    echo "✓ Step 3 complete. Output: $VIZ_OUTPUT"
    echo ""
else
    echo "[Step 3/6] Skipped."
    echo ""
fi

# Step 4: Visualize mutagenesis
if ! should_skip 4; then
    echo "[Step 4/6] Running mutagenesis visualization..."
    SAT_OUTPUT_DIR="${ANALYSIS_DIR}/sat_mutagenesis"
    python Analysis/03_6_visualize_mutagenesis.py \
        --input "$SAT_OUTPUT_DIR" \
        --track "$TRACK"
    echo "✓ Step 4 complete."
    echo ""
else
    echo "[Step 4/6] Skipped."
    echo ""
fi

# Step 5: Motif interpretation
if ! should_skip 5; then
    echo "[Step 5/6] Running motif gene diff interpretation..."
    python Analysis/02_motif_gene_diff_interpretation.py \
        --gene_name "$GENE" \
        --trial_pos "$TRIAL_POS" \
        -e "$EXP_NAME" \
        --chk "$CHECKPOINT" \
        -b random \
        --log_base ./logs \
        --chk_base ./Chk \
        --res_base ./Res \
        --processor gpu \
        --num_processes 1 \
        --num_threads 1 \
        --use_head regression
    echo "✓ Step 5 complete."
    echo ""
else
    echo "[Step 5/6] Skipped."
    echo ""
fi

# Step 6: Motif interpretation plotting
if ! should_skip 6; then
    echo "[Step 6/6] Running motif interpretation plotting..."
    # Parse gene region
    IFS=':' read -r GENE_CHR GENE_COORDS <<< "$GENE_REGION"
    IFS='-' read -r GENE_START GENE_END <<< "$GENE_COORDS"

    # Extract trial pos components (remove _RNAplus/RNAminus suffix)
    TRIAL_POS_SHORT=$(echo "$TRIAL_POS" | sed 's/_RNA.*//')
    TRIAL_STRAND=$(echo "$TRIAL_POS" | grep -o "RNA.*")

    # Create name base
    # Format: chr_start_end_gene_celltype_strand
    # Example: chr11_113137962_113662250_DRD2_STR-D1-MSN_minus
    NAME_BASE="${GENE_CHR}_${GENE_START}_${GENE_END}_${GENE}_${TRIAL_POS_SHORT}"
    if [ "$TRIAL_STRAND" = "RNAplus" ]; then
        NAME_BASE="${NAME_BASE}_plus"
    else
        NAME_BASE="${NAME_BASE}_minus"
    fi

    echo "  Using name base: $NAME_BASE"

    DATA_DIR="./Res/${EXP_NAME}/analysis_${CHECKPOINT}/raw_data/interp_diff"

    # Plot 1: Full region plot (without zoom)
    PLOT_OUTPUT_FULL="${ANALYSIS_DIR}/motif_interpretation_full.pdf"
    echo "  Creating full region plot..."
    python Analysis/02_motif_interpretation_plot.py \
        --data_dir "$DATA_DIR" \
        --name_base "$NAME_BASE" \
        --baseline random \
        --output "$PLOT_OUTPUT_FULL"

    # Plot 2: Zoomed region around variant (±100bp)
    ZOOM_START=$((POS - 100))
    ZOOM_END=$((POS + 100))
    PLOT_OUTPUT_ZOOM="${ANALYSIS_DIR}/motif_interpretation_zoom.pdf"
    echo "  Creating zoomed region plot (variant ±100bp)..."
    python Analysis/02_motif_interpretation_plot.py \
        --data_dir "$DATA_DIR" \
        --name_base "$NAME_BASE" \
        --baseline random \
        --output "$PLOT_OUTPUT_ZOOM" \
        --start "$ZOOM_START" \
        --end "$ZOOM_END" \
        --show_sequence

    echo "✓ Step 6 complete."
    echo "  Full plot:   $PLOT_OUTPUT_FULL"
    echo "  Zoomed plot: $PLOT_OUTPUT_ZOOM"
    echo ""
else
    echo "[Step 6/6] Skipped."
    echo ""
fi

echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo "All outputs are in: $ANALYSIS_DIR"
echo ""
echo "Output structure:"
echo "  $ANALYSIS_DIR/"
echo "  ├── inference/                        (Step 1: variant inference bigwig files)"
echo "  ├── sat_mutagenesis/                  (Step 2: saturation mutagenesis results)"
echo "  ├── visualization.pdf                 (Step 3: genome track visualization)"
echo "  ├── motif_interpretation_full.pdf     (Step 6: full region motif plot)"
echo "  └── motif_interpretation_zoom.pdf     (Step 6: zoomed motif plot)"
echo ""
echo "Note: Step 4 outputs are saved within sat_mutagenesis/ directory"
echo "      Step 5 outputs are saved in ./Res/${EXP_NAME}/analysis_${CHECKPOINT}/"
echo "========================================="
