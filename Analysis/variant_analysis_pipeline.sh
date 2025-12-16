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
    echo "  --midpoint POSITION         Central position for analysis in format chr:pos"
    echo "                              - Step 1: Switches to region inference mode (full gene region centered at midpoint)"
    echo "                              - Step 2: Inference context centered at midpoint (mutations still at variant ±20bp)"
    echo "                              - Steps 3,5,6: Inference/visualization centered at midpoint"
    echo "                              - Step 6 zoom: Always shows variant location (±50bp), not midpoint"
    echo "                              - Default: variant position for step 1-2; gene center from GTF for steps 3,5,6"
    echo "  --gene-region REGION        Gene region in format chr:start-end (default: midpoint/gene-center ± 262144bp)"
    echo "  --sat-region REGION         Saturation mutagenesis region chr:start-end (default: variant ±20bp)"
    echo "  --trial-pos TRIAL_POS       Trial position for motif interpretation (default: derived from --track)"
    echo "  --exp-name NAME             Experiment name (default: full_finetune_original_loss_celltype_head_dim8_linear)"
    echo "  --checkpoint CHK            Checkpoint number (default: 20)"
    echo "  --output-base DIR           Base output directory (default: Analysis/figures)"
    echo "  --gtf-file PATH             GTF annotation file (default: Data/source/gencode.v48.annotation.gtf.gz)"
    echo "  --num-gpus NUM              Number of GPUs to use for saturation mutagenesis (default: 4)"
    echo "  --method METHOD                  Attribution method for Step 5 (DeepLift, gradient_input, gradient_input_smooth; default: DeepLift)"
    echo "  --tomtom-db PATH                 Path to MEME motif database for TOMTOM (required for Step 7)"
    echo "  --tomtom-region REGION           Region for TOMTOM analysis (default: variant ± 10bp)"
    echo "  --tomtom-background-region REGION Region for calculating background nucleotide frequencies (default: variant ± 50bp)"
    echo "  --track-height HEIGHT            BigWig track height for Step 3 visualization (default: 2)"
    echo "  --gtf-height HEIGHT              GTF annotation track height for Step 3 visualization (default: 5)"
    echo "  --spacer-height HEIGHT           Spacer height between tracks for Step 3 visualization (default: 0.2)"
    echo "  --label-color COLOR              Color for label tracks in hex format (default: #1f77b4, blue)"
    echo "  --ref-color COLOR                Color for reference tracks in hex format (default: #d62728, red)"
    echo "  --alt-color COLOR                Color for alternative tracks in hex format (default: #2ca02c, green)"
    echo "  --diff-color COLOR               Color for diff tracks in hex format (default: #9467bd, purple)"
    echo "  --ref-alpha ALPHA                Transparency for reference tracks, 0.0-1.0 (default: 1.0, opaque)"
    echo "  --alt-alpha ALPHA                Transparency for alternative tracks, 0.0-1.0 (default: 1.0, opaque)"
    echo "  --alt-first                      Display alt track first, then overlay ref (default: ref first, then alt)"
    echo "  --sat-mutagenesis-viz-plot-width WIDTH   Width of saturation mutagenesis plot in inches (default: 12.0)"
    echo "  --sat-mutagenesis-viz-plot-height HEIGHT Height of saturation mutagenesis plot in inches (default: 4.0)"
    echo "  --motif-viz-plot-width WIDTH             Width of motif interpretation plots in inches (default: 8.0)"
    echo "  --motif-viz-plot-height HEIGHT           Height per subplot in motif plots in inches (default: 1.5)"
    echo "  --skip-steps STEPS               Comma-separated list of steps to skip (1-7)"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Pipeline Steps:"
    echo "  Step 1: Quick inference - Predict effects across region"
    echo "          (with --midpoint: full gene region; without: small window around variant)"
    echo "  Step 2: Saturation mutagenesis - Test all possible SNVs at variant ±20bp"
    echo "          (inference context can be centered elsewhere with --midpoint)"
    echo "  Step 3: Genome track visualization - Visualize predictions across gene region"
    echo "  Step 4: Mutagenesis visualization - Visualize saturation mutagenesis results (log2 fold change)"
    echo "  Step 5: Motif interpretation - Identify regulatory motifs (uses --method)"
    echo "  Step 6: Motif plotting - Generate full region plot and zoomed plot at variant (±100bp)"
    echo "  Step 7: TOMTOM analysis - Match motifs to known TF binding sites (optional, requires --tomtom-db)"
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
    echo "  # For large genes, specify a midpoint to center the analysis:"
    echo "  $0 --variant chr12:2236129:G:A --track MiniAtlas-L34IT_RNAminus \\"
    echo "     --variant-name rs1006737 --gene CACNA1C --disease Schizophrenia \\"
    echo "     --midpoint chr12:2300000"
    echo ""
    echo "  # Use gradient-based attribution instead of DeepLift:"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --method gradient_input"
    echo ""
    echo "  # Customize visualization track heights:"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --track-height 3 --gtf-height 7 --spacer-height 0.3"
    echo ""
    echo "  # Swap ref/alt colors (alt=red, ref=green):"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --ref-color '#2ca02c' --alt-color '#d62728'"
    echo ""
    echo "  # Make ref/alt tracks semi-transparent for better overlay visibility:"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --ref-alpha 0.5 --alt-alpha 0.5"
    echo ""
    echo "  # Display alt track first, then overlay ref (default is ref first):"
    echo "  $0 --variant chr11:113400106:G:T --track BasalGanglia-STR-D1-MSN_RNAminus \\"
    echo "     --alt-first"
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
        --midpoint)
            MIDPOINT="$2"
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
        --method)
            METHOD="$2"
            shift 2
            ;;
        --tomtom-db)
            TOMTOM_DB="$2"
            shift 2
            ;;
        --tomtom-region)
            TOMTOM_REGION="$2"
            shift 2
            ;;
        --tomtom-background-region)
            TOMTOM_BACKGROUND_REGION="$2"
            shift 2
            ;;
        --track-height)
            TRACK_HEIGHT="$2"
            shift 2
            ;;
        --gtf-height)
            GTF_HEIGHT="$2"
            shift 2
            ;;
        --spacer-height)
            SPACER_HEIGHT="$2"
            shift 2
            ;;
        --label-color)
            LABEL_COLOR="$2"
            shift 2
            ;;
        --ref-color)
            REF_COLOR="$2"
            shift 2
            ;;
        --alt-color)
            ALT_COLOR="$2"
            shift 2
            ;;
        --diff-color)
            DIFF_COLOR="$2"
            shift 2
            ;;
        --ref-alpha)
            REF_ALPHA="$2"
            shift 2
            ;;
        --alt-alpha)
            ALT_ALPHA="$2"
            shift 2
            ;;
        --alt-first)
            ALT_FIRST=true
            shift
            ;;
        --sat-mutagenesis-viz-plot-width)
            SAT_MUTAGENESIS_VIZ_PLOT_WIDTH="$2"
            shift 2
            ;;
        --sat-mutagenesis-viz-plot-height)
            SAT_MUTAGENESIS_VIZ_PLOT_HEIGHT="$2"
            shift 2
            ;;
        --motif-viz-plot-width)
            MOTIF_VIZ_PLOT_WIDTH="$2"
            shift 2
            ;;
        --motif-viz-plot-height)
            MOTIF_VIZ_PLOT_HEIGHT="$2"
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
METHOD=${METHOD:-"DeepLift"}
TRACK_HEIGHT=${TRACK_HEIGHT:-2}
GTF_HEIGHT=${GTF_HEIGHT:-5}
SPACER_HEIGHT=${SPACER_HEIGHT:-0.2}
LABEL_COLOR=${LABEL_COLOR:-"#1f77b4"}
REF_COLOR=${REF_COLOR:-"#d62728"}
ALT_COLOR=${ALT_COLOR:-"#2ca02c"}
DIFF_COLOR=${DIFF_COLOR:-"#9467bd"}
REF_ALPHA=${REF_ALPHA:-1.0}
ALT_ALPHA=${ALT_ALPHA:-1.0}
SAT_MUTAGENESIS_VIZ_PLOT_WIDTH=${SAT_MUTAGENESIS_VIZ_PLOT_WIDTH:-12.0}
SAT_MUTAGENESIS_VIZ_PLOT_HEIGHT=${SAT_MUTAGENESIS_VIZ_PLOT_HEIGHT:-4.0}
MOTIF_VIZ_PLOT_WIDTH=${MOTIF_VIZ_PLOT_WIDTH:-8.0}
MOTIF_VIZ_PLOT_HEIGHT=${MOTIF_VIZ_PLOT_HEIGHT:-1.5}

# Validate method
case "$METHOD" in
    DeepLift|gradient_input|gradient_input_smooth)
        # Valid method
        ;;
    *)
        echo "Error: Invalid method '$METHOD'"
        echo "Valid methods: DeepLift, gradient_input, gradient_input_smooth"
        exit 1
        ;;
esac

# Parse variant to extract chromosome and position
IFS=':' read -r CHR POS REF ALT <<< "$VARIANT"

# Parse midpoint if provided
if [ -n "$MIDPOINT" ]; then
    IFS=':' read -r MIDPOINT_CHR MIDPOINT_POS <<< "$MIDPOINT"
    echo "Midpoint specified: $MIDPOINT (will be used as center for analysis)"
    # Validate that midpoint chromosome matches variant chromosome
    if [ "$MIDPOINT_CHR" != "$CHR" ]; then
        echo "Warning: Midpoint chromosome ($MIDPOINT_CHR) differs from variant chromosome ($CHR)"
    fi
fi

# Auto-generate variant name if not provided
if [ -z "$VARIANT_NAME" ]; then
    VARIANT_NAME="${CHR}_${POS}_${REF}_${ALT}"
    echo "Variant name not provided, using: $VARIANT_NAME"
fi

# Set default TOMTOM region if not provided (variant ± 10bp)
if [ -z "$TOMTOM_REGION" ]; then
    TOMTOM_REGION_START=$((POS - 10))
    TOMTOM_REGION_END=$((POS + 10))
    TOMTOM_REGION="${CHR}:${TOMTOM_REGION_START}-${TOMTOM_REGION_END}"
fi

# Set default TOMTOM background region if not provided (variant ± 50bp)
if [ -z "$TOMTOM_BACKGROUND_REGION" ]; then
    TOMTOM_BACKGROUND_REGION_START=$((POS - 50))
    TOMTOM_BACKGROUND_REGION_END=$((POS + 50))
    TOMTOM_BACKGROUND_REGION="${CHR}:${TOMTOM_BACKGROUND_REGION_START}-${TOMTOM_BACKGROUND_REGION_END}"
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
# Note: This is the mutation region, NOT the inference region
if [ -z "$SAT_REGION" ]; then
    SAT_START=$((POS - 20))
    SAT_END=$((POS + 20))
    SAT_REGION="${CHR}:${SAT_START}-${SAT_END}"
fi

# Calculate gene region if not provided (centered on midpoint or gene center ± 262144bp)
if [ -z "$GENE_REGION" ]; then
    # If midpoint is specified, use it directly
    if [ -n "$MIDPOINT" ]; then
        echo "Using midpoint as center for gene region: $MIDPOINT_POS"
        REGION_CENTER=$MIDPOINT_POS
    else
        # Extract gene coordinates from GTF to find gene center
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
            REGION_CENTER=$POS
        else
            read GENE_GTF_START GENE_GTF_END <<< "$GENE_COORDS"
            GENE_CENTER=$(( (GENE_GTF_START + GENE_GTF_END) / 2 ))
            echo "Gene $GENE center: $GENE_CENTER (from GTF: $GENE_GTF_START-$GENE_GTF_END)"
            REGION_CENTER=$GENE_CENTER
        fi
    fi

    GENE_REGION_START=$((REGION_CENTER - 262144))
    GENE_REGION_END=$((REGION_CENTER + 262144))

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

# Helper function to calculate intersection of two regions
intersect_regions() {
    local region1=$1  # e.g., chr12:1000-5000
    local region2=$2  # e.g., chr12:2000-6000

    # Parse region1
    IFS=':' read -r chr1 coords1 <<< "$region1"
    IFS='-' read -r start1 end1 <<< "$coords1"

    # Parse region2
    IFS=':' read -r chr2 coords2 <<< "$region2"
    IFS='-' read -r start2 end2 <<< "$coords2"

    # Check chromosome match
    if [ "$chr1" != "$chr2" ]; then
        echo ""
        return 1
    fi

    # Calculate intersection
    local max_start=$((start1 > start2 ? start1 : start2))
    local min_end=$((end1 < end2 ? end1 : end2))

    # Check if regions overlap
    if [ $max_start -ge $min_end ]; then
        echo ""
        return 1
    fi

    echo "${chr1}:${max_start}-${min_end}"
    return 0
}

echo "========================================="
echo "Variant Analysis Pipeline"
echo "========================================="
echo "Variant:            $VARIANT"
echo "Variant Name:       $VARIANT_NAME"
echo "Gene:               $GENE"
echo "Midpoint:           ${MIDPOINT:-N/A (using defaults)}"
echo "Gene Region:        $GENE_REGION"
echo "Sat Region:         $SAT_REGION"
echo "Track:              $TRACK"
echo "Trial Pos:          $TRIAL_POS"
echo "Disease:            ${DISEASE:-N/A}"
echo "Experiment:         $EXP_NAME"
echo "Checkpoint:         $CHECKPOINT"
echo "Num GPUs:           $NUM_GPUS (for saturation mutagenesis)"
echo "Method:             $METHOD (for motif interpretation)"
echo "TOMTOM Region:      $TOMTOM_REGION"
echo "TOMTOM Background:  $TOMTOM_BACKGROUND_REGION"
echo "Output Dir:         $ANALYSIS_DIR"
echo "========================================="
echo ""

# Step 1: Quick inference for the variant
if ! should_skip 1; then
    INFERENCE_DIR="${ANALYSIS_DIR}/inference"

    # If midpoint is specified, run BOTH region and variant inference
    # Region inference: gives us full genomic context for ref predictions
    # Variant inference: gives us alt/diff predictions for the specific variant
    if [ -n "$MIDPOINT" ]; then
        echo "[Step 1/7] Running region inference across full gene region (centered at midpoint)..."
        echo "  Inference region: $GENE_REGION"
        python Analysis/01_0_quick_inference_bigwig.py \
            --region "$GENE_REGION" \
            --exp_name "$EXP_NAME" \
            --chk "$CHECKPOINT" \
            --output "$INFERENCE_DIR"

        echo "  Running variant inference for alt/diff predictions..."
        python Analysis/01_0_quick_inference_bigwig.py \
            --variant "$VARIANT" \
            --exp_name "$EXP_NAME" \
            --chk "$CHECKPOINT" \
            --output "$INFERENCE_DIR"

        # Track that we have data for the full gene region
        ACTUAL_INFERENCE_REGION="$GENE_REGION"
    else
        echo "[Step 1/7] Running quick inference for variant..."
        python Analysis/01_0_quick_inference_bigwig.py \
            --variant "$VARIANT" \
            --exp_name "$EXP_NAME" \
            --chk "$CHECKPOINT" \
            --output "$INFERENCE_DIR"

        # Variant inference generates data for a limited region around the variant
        # Typically uses context_length (262144 bp window = ±131072 bp)
        VARIANT_INFERENCE_START=$((POS - 131072))
        VARIANT_INFERENCE_END=$((POS + 131072))
        if [ $VARIANT_INFERENCE_START -lt 0 ]; then
            VARIANT_INFERENCE_START=0
        fi
        ACTUAL_INFERENCE_REGION="${CHR}:${VARIANT_INFERENCE_START}-${VARIANT_INFERENCE_END}"
        echo "  Inference data available for: $ACTUAL_INFERENCE_REGION"
    fi
    echo "✓ Step 1 complete. Output: $INFERENCE_DIR"
    echo ""
else
    echo "[Step 1/7] Skipped."
    echo ""
    # If Step 1 was skipped, we need to infer what region has data
    if [ -n "$MIDPOINT" ]; then
        ACTUAL_INFERENCE_REGION="$GENE_REGION"
    else
        VARIANT_INFERENCE_START=$((POS - 131072))
        VARIANT_INFERENCE_END=$((POS + 131072))
        if [ $VARIANT_INFERENCE_START -lt 0 ]; then
            VARIANT_INFERENCE_START=0
        fi
        ACTUAL_INFERENCE_REGION="${CHR}:${VARIANT_INFERENCE_START}-${VARIANT_INFERENCE_END}"
    fi
fi

# Step 2: Saturation mutagenesis
if ! should_skip 2; then
    echo "[Step 2/7] Running saturation mutagenesis (using $NUM_GPUS GPUs)..."
    SAT_OUTPUT_DIR="${ANALYSIS_DIR}/sat_mutagenesis"

    # Build command with optional region-center parameter
    CMD="python Analysis/01_0_quick_inference_bigwig.py \
        --sat_mutagenesis \"$SAT_REGION\" \
        --exp_name \"$EXP_NAME\" \
        --chk \"$CHECKPOINT\" \
        --output \"$SAT_OUTPUT_DIR\" \
        --gene \"$GENE\" \
        --num-gpus \"$NUM_GPUS\""

    # Add region-center if midpoint is specified
    if [ -n "$MIDPOINT" ]; then
        CMD="$CMD --region-center $MIDPOINT_POS"
        echo "  Mutation region: $SAT_REGION"
        echo "  Inference centered at: $MIDPOINT_POS"
    fi

    eval $CMD
    echo "✓ Step 2 complete. Output: $SAT_OUTPUT_DIR"
    echo ""
else
    echo "[Step 2/7] Skipped."
    echo ""
fi

# Step 3: Visualization with pygenometrack
if ! should_skip 3; then
    echo "[Step 3/7] Running visualization with pygenometrack..."
    INFERENCE_DIR="${ANALYSIS_DIR}/inference"
    VIZ_OUTPUT="${ANALYSIS_DIR}/visualization.pdf"

    # Calculate visualization region as intersection of:
    # 1. Gene-centered region (gene midpoint ± 262144bp)
    # 2. Variant-centered region (variant position ± 262144bp)

    # Variant-centered region (where we have inference data)
    VARIANT_VIZ_START=$((POS - 262144))
    VARIANT_VIZ_END=$((POS + 262144))
    if [ $VARIANT_VIZ_START -lt 0 ]; then
        VARIANT_VIZ_START=0
    fi
    VARIANT_CENTERED_REGION="${CHR}:${VARIANT_VIZ_START}-${VARIANT_VIZ_END}"

    # Intersect gene-centered region with variant-centered region
    VIZ_REGION=$(intersect_regions "$GENE_REGION" "$VARIANT_CENTERED_REGION")

    if [ -z "$VIZ_REGION" ]; then
        echo "  Warning: No overlap between gene region and variant region"
        echo "    Gene region:    $GENE_REGION"
        echo "    Variant region: $VARIANT_CENTERED_REGION"
        echo "  Using variant region for visualization"
        VIZ_REGION="$VARIANT_CENTERED_REGION"
    else
        echo "  Visualization region: $VIZ_REGION"
        echo "    Gene-centered:    $GENE_REGION"
        echo "    Variant-centered: $VARIANT_CENTERED_REGION"
        echo "    Intersection:     $VIZ_REGION"
    fi

    # Extract cell type pattern from full track name
    # Example: BasalGanglia-STR-D1-MSN_RNAminus -> BasalGanglia-STR-D1-MSN*
    # Keep region and cell type, replace assay type with wildcard
    CELL_TYPE_PATTERN=$(echo "$TRACK" | sed 's/_.*/*/')

    # Build visualization command
    VIZ_CMD="python Analysis/00_visualize_data_pygenometrack.py \
        --inference-dir \"$INFERENCE_DIR\" \
        --region \"$VIZ_REGION\" \
        --output \"$VIZ_OUTPUT\" \
        --tracks \"$CELL_TYPE_PATTERN\" \
        --highlight \"${CHR}:${POS}\" \
        --height \"$TRACK_HEIGHT\" \
        --gtf-height \"$GTF_HEIGHT\" \
        --spacer-height \"$SPACER_HEIGHT\" \
        --label-color \"$LABEL_COLOR\" \
        --ref-color \"$REF_COLOR\" \
        --alt-color \"$ALT_COLOR\" \
        --diff-color \"$DIFF_COLOR\" \
        --ref-alpha \"$REF_ALPHA\" \
        --alt-alpha \"$ALT_ALPHA\""

    # Add --alt-first flag if set
    if [ "$ALT_FIRST" = true ]; then
        VIZ_CMD="$VIZ_CMD --alt-first"
    fi

    eval $VIZ_CMD

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
    echo "[Step 3/7] Skipped."
    echo ""
fi

# Step 4: Visualize mutagenesis
if ! should_skip 4; then
    echo "[Step 4/7] Running mutagenesis visualization..."
    SAT_OUTPUT_DIR="${ANALYSIS_DIR}/sat_mutagenesis"
    SAT_FIGSIZE="${SAT_MUTAGENESIS_VIZ_PLOT_WIDTH},${SAT_MUTAGENESIS_VIZ_PLOT_HEIGHT}"
    python Analysis/03_6_visualize_mutagenesis.py \
        --input "$SAT_OUTPUT_DIR" \
        --track "$TRACK" \
        --log2fc \
        --figsize "$SAT_FIGSIZE"
    echo "✓ Step 4 complete."
    echo ""
else
    echo "[Step 4/7] Skipped."
    echo ""
fi

# Step 5: Motif interpretation
if ! should_skip 5; then
    echo "[Step 5/7] Running motif gene diff interpretation..."
    echo "  Attribution method: $METHOD"
    # Build command with optional region_center parameter
    CMD="python Analysis/02_motif_gene_diff_interpretation.py \
        --gene_name \"$GENE\" \
        --trial_pos \"$TRIAL_POS\" \
        -e \"$EXP_NAME\" \
        --chk \"$CHECKPOINT\" \
        -b random \
        --method \"$METHOD\" \
        --log_base ./logs \
        --chk_base ./Chk \
        --res_base ./Res \
        --processor gpu \
        --num_processes 1 \
        --num_threads 1 \
        --use_head regression"

    # Add region_center if midpoint is specified
    if [ -n "$MIDPOINT" ]; then
        CMD="$CMD --region_center $MIDPOINT_POS"
        echo "  Using custom region center: $MIDPOINT_POS"
    fi

    eval $CMD
    echo "✓ Step 5 complete."
    echo ""
else
    echo "[Step 5/7] Skipped."
    echo ""
fi

# Step 6: Motif interpretation plotting
if ! should_skip 6; then
    echo "[Step 6/7] Running motif interpretation plotting..."
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
        --output "$PLOT_OUTPUT_FULL" \
        --motif-viz-plot-width "$MOTIF_VIZ_PLOT_WIDTH" \
        --motif-viz-plot-height "$MOTIF_VIZ_PLOT_HEIGHT"

    # Plot 2: Zoomed region around variant (±50bp)
    # Always zoom to variant location, regardless of midpoint setting
    ZOOM_START=$((POS - 50))
    ZOOM_END=$((POS + 50))
    PLOT_OUTPUT_ZOOM="${ANALYSIS_DIR}/motif_interpretation_zoom.pdf"
    echo "  Creating zoomed region plot (variant ±50bp: ${CHR}:${ZOOM_START}-${ZOOM_END})..."
    python Analysis/02_motif_interpretation_plot.py \
        --data_dir "$DATA_DIR" \
        --name_base "$NAME_BASE" \
        --baseline random \
        --output "$PLOT_OUTPUT_ZOOM" \
        --start "$ZOOM_START" \
        --end "$ZOOM_END" \
        --show_sequence \
        --motif-viz-plot-width "$MOTIF_VIZ_PLOT_WIDTH" \
        --motif-viz-plot-height "$MOTIF_VIZ_PLOT_HEIGHT"

    echo "✓ Step 6 complete."
    echo "  Full plot:   $PLOT_OUTPUT_FULL"
    echo "  Zoomed plot: $PLOT_OUTPUT_ZOOM"
    echo ""
else
    echo "[Step 6/7] Skipped."
    echo ""
fi

# Step 7: TOMTOM analysis (optional, requires --tomtom-db)
if ! should_skip 7; then
    if [ -z "$TOMTOM_DB" ]; then
        echo "[Step 7/7] TOMTOM analysis - Skipped (no --tomtom-db specified)"
        echo ""
    else
        echo "[Step 7/7] Running TOMTOM analysis..."
        echo "  Region: $TOMTOM_REGION"
        echo "  Background: $TOMTOM_BACKGROUND_REGION"
        echo "  Database: $TOMTOM_DB"

        # Check if TOMTOM database exists
        if [ ! -f "$TOMTOM_DB" ]; then
            echo "  Error: TOMTOM database not found: $TOMTOM_DB"
            echo "  Skipping TOMTOM analysis"
            echo ""
        else
            # Use NAME_BASE and DATA_DIR from Step 6 context
            # Parse gene region to construct NAME_BASE if Step 6 was skipped
            if [ -z "$NAME_BASE" ]; then
                IFS=':' read -r GENE_CHR GENE_COORDS <<< "$GENE_REGION"
                IFS='-' read -r GENE_START GENE_END <<< "$GENE_COORDS"

                TRIAL_POS_SHORT=$(echo "$TRIAL_POS" | sed 's/_RNA.*//')
                TRIAL_STRAND=$(echo "$TRIAL_POS" | grep -o "RNA.*")

                NAME_BASE="${GENE_CHR}_${GENE_START}_${GENE_END}_${GENE}_${TRIAL_POS_SHORT}"
                if [ "$TRIAL_STRAND" = "RNAplus" ]; then
                    NAME_BASE="${NAME_BASE}_plus"
                else
                    NAME_BASE="${NAME_BASE}_minus"
                fi
            fi

            if [ -z "$DATA_DIR" ]; then
                DATA_DIR="./Res/${EXP_NAME}/analysis_${CHECKPOINT}/raw_data/interp_diff"
            fi

            TOMTOM_OUTPUT_DIR="${ANALYSIS_DIR}/tomtom"

            python Analysis/02_motif_region_tomtom.py \
                --data_dir "$DATA_DIR" \
                --name_base "$NAME_BASE" \
                --baseline random \
                --region "$TOMTOM_REGION" \
                --background-region "$TOMTOM_BACKGROUND_REGION" \
                --output_dir "$TOMTOM_OUTPUT_DIR" \
                --meme_db "$TOMTOM_DB"

            echo "✓ Step 7 complete."
            echo "  Results: $TOMTOM_OUTPUT_DIR/tomtom.html"
            echo ""
        fi
    fi
else
    echo "[Step 7/7] Skipped."
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
echo "  ├── motif_interpretation_zoom.pdf     (Step 6: zoomed motif plot)"
echo "  └── tomtom/                           (Step 7: TOMTOM motif matching results)"
echo ""
echo "Note: Step 4 outputs are saved within sat_mutagenesis/ directory"
echo "      Step 5 outputs are saved in ./Res/${EXP_NAME}/analysis_${CHECKPOINT}/"
echo "========================================="
