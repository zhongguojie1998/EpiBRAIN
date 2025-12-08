#!/bin/bash
# ==============================================================================
# Complete Analysis Pipeline for BICAN Model
# ==============================================================================
# This script runs all analysis steps from 00 to 11 for a trained model
#
# Usage: bash pipeline.sh <model_name> <chk_number> [options]
#
# Arguments:
#   model_name: Name of the model (e.g., full_finetune_original_loss_celltype_head_dim8_linear)
#   chk_number: Checkpoint epoch number to use (e.g., 20)
#
# Options:
#   --skip-01: Skip correlation analysis (section 01)
#   --skip-02: Skip motif interpretation analysis (section 02)
#   --skip-03: Skip variant effect analysis (section 03)
#   --skip-09: Skip ABC analysis (section 09)
#   --skip-10: Skip differential peak analysis (section 10)
#   --skip-11: Skip differential expression analysis (section 11)
#   --only-<section>: Only run specific section (e.g., --only-01)
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# Parse Arguments
# ==============================================================================

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: bash pipeline.sh <model_name> <chk_number> [options]"
    exit 1
fi

MODEL_NAME=$1
CHK=$2
shift 2

# Parse optional arguments
SKIP_01=false
SKIP_02=false
SKIP_03=false
SKIP_09=false
SKIP_10=false
SKIP_11=false
ONLY_SECTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-01) SKIP_01=true; shift ;;
        --skip-02) SKIP_02=true; shift ;;
        --skip-03) SKIP_03=true; shift ;;
        --skip-09) SKIP_09=true; shift ;;
        --skip-10) SKIP_10=true; shift ;;
        --skip-11) SKIP_11=true; shift ;;
        --only-*)
            ONLY_SECTION="${1#--only-}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Setup Paths and Directories
# ==============================================================================

# Get the root directory (parent of Analysis/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="${ROOT_DIR}/Analysis"
LOG_BASE="${ROOT_DIR}/logs"
CHK_BASE="${ROOT_DIR}/Chk"
RES_BASE="${ROOT_DIR}/Res"
DATA_BASE="${ROOT_DIR}/Data"

# Model-specific directories
MODEL_LOG_DIR="${LOG_BASE}/${MODEL_NAME}"
MODEL_CHK_DIR="${CHK_BASE}/${MODEL_NAME}"
MODEL_RES_DIR="${RES_BASE}/${MODEL_NAME}"
MODEL_ANALYSIS_DIR="${MODEL_RES_DIR}/analysis_${CHK}"

echo "=================================================="
echo "Starting Analysis Pipeline"
echo "=================================================="
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHK}"
echo "Root directory: ${ROOT_DIR}"
echo "Analysis directory: ${ANALYSIS_DIR}"
echo "Results will be saved to: ${MODEL_ANALYSIS_DIR}"
echo "=================================================="

# Check if model checkpoint exists
if [ ! -f "${MODEL_CHK_DIR}/chk_epoch_${CHK}.pt" ]; then
    echo "Error: Checkpoint file not found: ${MODEL_CHK_DIR}/chk_epoch_${CHK}.pt"
    exit 1
fi

# Create output directories
mkdir -p "${MODEL_ANALYSIS_DIR}"

# ==============================================================================
# Helper Functions
# ==============================================================================

should_run_section() {
    local section=$1

    # If ONLY_SECTION is set, only run that section
    if [ -n "$ONLY_SECTION" ]; then
        [ "$section" == "$ONLY_SECTION" ] && return 0 || return 1
    fi

    # Otherwise check if section should be skipped
    local skip_var="SKIP_${section}"
    [ "${!skip_var}" == "true" ] && return 1 || return 0
}

print_section_header() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}

# ==============================================================================
# Section 00: Visualization and Data Exploration
# ==============================================================================

if should_run_section "00"; then
    print_section_header "Section 00: Data Visualization"

    # Example: Visualize BigWig tracks for SNCA gene region with pyGenomeTracks
    echo ""
    echo "Note: For other visualizations, use:"
    echo ""
    echo "# Visualize specific region with track patterns:"
    echo "python ${ANALYSIS_DIR}/00_visualize_data_pygenometrack.py \\"
    echo "  --inference-dir ${RES_BASE}/bigwig/<INFERENCE_DIR>/ \\"
    echo "  --region <CHR>:<START>-<END> \\"
    echo "  --output <OUTPUT>.pdf \\"
    echo "  --tracks \"<TRACK_PATTERN>\" \\"
    echo "  --highlight <CHR>:<POSITION>"
    echo ""
    echo "Section 00: Completed"
fi

# ==============================================================================
# Section 01: Correlation Analysis
# ==============================================================================

if should_run_section "01"; then
    print_section_header "Section 01: Correlation Analysis"

    # 01_1: Basic correlation test
    echo "Running basic correlation test..."
    python "${ANALYSIS_DIR}/01_1_test_correlation.py" \
        -e "${MODEL_NAME}" \
        --chk "${CHK}" \
        --res_base "${RES_BASE}" \
        --log_base "${LOG_BASE}"

    # 01_2: Across cell types correlation
    echo "Running cross-celltype correlation..."
    python "${ANALYSIS_DIR}/01_2_test_correlation_across_celltypes.py" \
        -e "${MODEL_NAME}" \
        --chk "${CHK}" \
        --res_base "${RES_BASE}" \
        --log_base "${LOG_BASE}"

    # 01_5: Gene-level correlation
    echo "Running gene-level correlation..."
    python "${ANALYSIS_DIR}/01_5_test_correlation_by_gene.py" \
        -e "${MODEL_NAME}" \
        --chk "${CHK}" \
        -s Test \
        --res_base "${RES_BASE}" \
        --log_base "${LOG_BASE}" \
        --data_base "${DATA_BASE}" \
        --genes_gtf "${DATA_BASE}/source/gencode.v48.annotation.gtf.gz"

    # 01_6: Generate plots
    echo "Generating correlation plots..."
    python "${ANALYSIS_DIR}/01_6_plots.py" \
        -e "${MODEL_NAME}" \
        --chk "${CHK}" \
        --res_base "${RES_BASE}" \
        --log_base "${LOG_BASE}"

    # 01_0: Quick inference to BigWig (example for SNCA variant)
    echo ""
    echo "Running quick inference for SNCA variant (example)..."
    python "${ANALYSIS_DIR}/01_0_quick_inference_bigwig.py" \
        --variant chr4:89753280:G:A \
        --exp_name "${MODEL_NAME}" \
        --chk "${CHK}" \
        --output "${RES_BASE}/bigwig/rs356182_SNCA_parkinsons"
    
    # visualzie
    echo "Visualizing SNCA gene region (example)..."
    python "${ANALYSIS_DIR}/00_visualize_data_pygenometrack.py" \
        --inference-dir "${RES_BASE}/bigwig/rs356182_SNCA_parkinsons/" \
        --region chr4:89442816-89967104 \
        --output "${MODEL_ANALYSIS_DIR}/SNCA_visualization.pdf" \
        --tracks "BasalGanglia-STR-D1*" \
        --highlight chr4:89753280

    echo ""
    echo "Note: For other variants or regions, use:"
    echo ""
    echo "# Quick inference for variant effect:"
    echo "python ${ANALYSIS_DIR}/01_0_quick_inference_bigwig.py \\"
    echo "  --variant <CHR>:<POS>:<REF>:<ALT> \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --output ${RES_BASE}/bigwig/<OUTPUT_DIR>"
    echo ""
    echo "Section 01: Completed"
fi

# ==============================================================================
# Section 02: Motif Interpretation
# ==============================================================================

if should_run_section "02"; then
    print_section_header "Section 02: Motif Interpretation (Attribution Analysis)"

    echo "Running gene-based interpretation for SNCA with gradient×input method..."
    python "${ANALYSIS_DIR}/02_motif_gene_diff_interpretation.py" \
        --gene_name SNCA \
        --trial_pos STR-D1-MSN_RNAplus \
        -e "${MODEL_NAME}" \
        --chk "${CHK}" \
        -b random \
        --log_base "${LOG_BASE}" \
        --chk_base "${CHK_BASE}" \
        --res_base "${RES_BASE}" \
        --processor gpu \
        --num_processes 1 \
        --num_threads 1 \
        --use_head regression

    echo ""
    echo "Plotting interpretation results for SNCA gene (example region)..."
    python "${ANALYSIS_DIR}/02_motif_interpretation_plot.py" \
        --data_dir "${MODEL_RES_DIR}/analysis_${CHK}/raw_data/interp_diff" \
        --name_base chr4_89507186_90031474_SNCA_STR-D1-MSN_plus \
        --baseline random \
        --output "${MODEL_ANALYSIS_DIR}/chr4_89507186_90031474_SNCA_STR-D1-MSN_plus.pdf" \
        --show_sequence \
        --start 89753200 \
        --end 89753360

    echo ""
    echo "Note: For other genes or attribution methods, use:"
    echo ""
    echo "# Gene-based interpretation (recommended - supports multiple attribution methods):"
    echo "python ${ANALYSIS_DIR}/02_motif_gene_diff_interpretation_DeepLift.py \\"
    echo "  --gene_name <GENE> \\"
    echo "  --trial_pos <CELLTYPE> \\"
    echo "  -e ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --method gradient_input  # or DeepLift, or gradient_input_smooth"
    echo ""
    echo "# BED-based interpretation (for custom regions):"
    echo "python ${ANALYSIS_DIR}/02_motif_gene_diff_interpretation.py \\"
    echo "  --gene_name <GENE> \\"
    echo "  --trial_pos <CELLTYPE> \\"
    echo "  -e ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  -b random \\"
    echo "  --log_base ${LOG_BASE} \\"
    echo "  --chk_base ${CHK_BASE} \\"
    echo "  --res_base ${RES_BASE} \\"
    echo "  --processor gpu \\"
    echo "  --num_processes 1 \\"
    echo "  --num_threads 1 \\"
    echo "  --use_head regression"
    echo ""
    echo "# Plot interpretation results:"
    echo "python ${ANALYSIS_DIR}/02_motif_interpretation_plot.py \\"
    echo "  --data_dir ${MODEL_RES_DIR}/analysis_${CHK}/raw_data/interp_diff \\"
    echo "  --name_base <NAME_BASE> \\"
    echo "  --baseline random \\"
    echo "  --output <OUTPUT_FILE>.pdf \\"
    echo "  --show_sequence \\"
    echo "  --start <START_BP> \\"
    echo "  --end <END_BP>"
    echo ""
    echo "Section 02: Completed"
fi

# ==============================================================================
# Section 03: Variant Effect Prediction
# ==============================================================================

if should_run_section "03"; then
    print_section_header "Section 03: Variant Effect Prediction"

    echo "Running variant effect screen for eQTL analysis..."
    bash Analysis/03_variant_effect_screen/script.sh \
        --vcf ${DATA_BASE}/source/eQTL/all.vcf \
        --output ${DATA_BASE}/source/eQTL/${MODEL_NAME}.chk${CHK}.h5 \
        --model ${MODEL_CHK_DIR}/chk_epoch_${CHK}_packaged.pkl \
        --config ${MODEL_LOG_DIR}/overall_setting.yaml \
        --label_meta ${MODEL_LOG_DIR}/regression_label_meta.csv \
        --experiment eQTL \
        --chunks 6 \
        --mode slurm \
        --untransform

    echo "Running gene-level variant effect analysis for Jang2025 SingleBrain..."
    python ${ANALYSIS_DIR}/03_4_variant_effect_by_gene.py \
        --vcf ${DATA_BASE}/source/Jang2025_SingleBrain/finemapped_variants.vcf \
        --exp_name ${MODEL_NAME} \
        --chk ${CHK} \
        --n_jobs 36 \
        --untransform

    echo "Section 03: Completed"
fi

# ==============================================================================
# Section 04-08: Additional Analyses
# ==============================================================================

if should_run_section "04"; then
    print_section_header "Section 04-08: Specialized Analyses"

    echo "Section 04: View prediction differences (interactive)"
    echo "Section 05: Transcripts performance analysis"
    echo "Section 06-07: eQTL analysis (requires GTEx data)"
    echo "Section 08: TraitGym analysis (requires trait data)"
    echo ""
    echo "These sections require specific data files and are typically run separately."
    echo "Please refer to individual script documentation for usage."
fi

# ==============================================================================
# Section 09: ABC (Activity-By-Contact) Analysis
# ==============================================================================

if should_run_section "09"; then
    print_section_header "Section 09: ABC Analysis"

    echo "Note: ABC analysis is a multi-step process."
    echo ""
    echo "Step 1: Prepare predictions as BigWig files"
    echo "python ${ANALYSIS_DIR}/09_0_pt_to_bigwig.py \\"
    echo "  -e ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 2: Prepare ABC input files"
    echo "python ${ANALYSIS_DIR}/09_1_ABC_prepare.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 3: Run ABC (submit SLURM job)"
    echo "sbatch ${ANALYSIS_DIR}/09_2_ABC_run.slurm ${MODEL_NAME} ${CHK}"
    echo ""
    echo "Step 4: Screen significant attributions"
    echo "python ${ANALYSIS_DIR}/09_3_ABC_screen_significant_attributions.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 5: Generate plots"
    echo "python ${ANALYSIS_DIR}/09_4_ABC_screen_significant_attributions_plot.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Section 09: Skipping (requires manual execution of multi-step workflow)"
fi

# ==============================================================================
# Section 10: Differential Peak Analysis
# ==============================================================================

if should_run_section "10"; then
    print_section_header "Section 10: Differential Peak Analysis"

    echo "Note: Differential peak analysis workflow"
    echo ""
    echo "Step 1: Link differential expression to differential peaks"
    echo "python ${ANALYSIS_DIR}/10_1_link_DiffExpress_DiffPeak.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 2: Run differential peak attributions (submit SLURM job)"
    echo "sbatch ${ANALYSIS_DIR}/10_2_Diff_peak_run.slurm ${MODEL_NAME} ${CHK}"
    echo ""
    echo "Step 3: Screen significant attributions"
    echo "python ${ANALYSIS_DIR}/10_3_screen_DiffPeak_attributions.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 4: Generate plots"
    echo "python ${ANALYSIS_DIR}/10_4_screen_DiffPeak_attributions_plot.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Section 10: Skipping (requires differential expression data)"
fi

# ==============================================================================
# Section 11: Differential Expression Analysis with TF-MoDISco
# ==============================================================================

if should_run_section "11"; then
    print_section_header "Section 11: Differential Expression & TF-MoDISco Analysis"

    echo "Note: Comprehensive differential expression analysis workflow"
    echo ""
    echo "Step 1: Identify differentially expressed genes"
    echo "python ${ANALYSIS_DIR}/11_1_DiffExpress.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE} \\"
    echo "  --log_base ${LOG_BASE}"
    echo ""
    echo "Step 2: Create BED files for differential genes"
    echo "python ${ANALYSIS_DIR}/11_2_DiffExpress_create_bed.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 3: Run gradient×input attributions (submit SLURM job)"
    echo "bash ${ANALYSIS_DIR}/11_3_DiffExpress_run_smooth_gradient.sh ${MODEL_NAME} ${CHK}"
    echo ""
    echo "Step 4: Run TF-MoDISco motif discovery"
    for ct in ACBGM AST CBGA L23IT L2IT L34IT L35IT L45IT L4IT L56IT L56NP L5IT L6B L6CT L6IT-1 L6IT-2 MGC OGC OPC PV-CHC PVALB URL VIP; do
        slurmsub -p cpu -c 24 -m 150G -t 48:00:00 \
            "python ${ANALYSIS_DIR}/11_4_DiffExpress_TFMoDisco_borzoi.py \
            ${MODEL_RES_DIR}/analysis_${CHK}/raw_data/interp_diff_gradient_input/ \
            --fasta ${DATA_BASE}/Ref/hg38/hg38.fa \
            --context_length 131072 \
            -c 131072 \
            -o tfm_out \
            --baseline grad_input \
            --n_jobs 1 \
            --modisco_n_cores 24 \
            --celltype $ct"
    done
    echo ""
    echo "Step 5: Run TOMTOM motif matching"
    echo "python ${ANALYSIS_DIR}/11_5_DiffExpress_TOMTOM.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 6: Map to cCREs"
    echo "python ${ANALYSIS_DIR}/11_6_DiffExpress_cCRE.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 7: Filter cCREs by cutoff"
    echo "python ${ANALYSIS_DIR}/11_7_filter_cCRE_by_cutoff.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Step 8: Generate final plots"
    echo "python ${ANALYSIS_DIR}/11_8_plots.py \\"
    echo "  --exp_name ${MODEL_NAME} \\"
    echo "  --chk ${CHK} \\"
    echo "  --res_base ${RES_BASE}"
    echo ""
    echo "Section 11: Skipping (requires differential expression data and multiple steps)"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_section_header "Pipeline Summary"

if [ -n "$ONLY_SECTION" ]; then
    echo "Ran only section: ${ONLY_SECTION}"
else
    echo "Completed sections:"
    $SKIP_01 || echo "  ✓ Section 01: Correlation Analysis"
    $SKIP_02 || echo "  ✓ Section 02: Motif Interpretation (info provided)"
    $SKIP_03 || echo "  ✓ Section 03: Variant Effect (info provided)"
    $SKIP_09 || echo "  ✓ Section 09: ABC Analysis (info provided)"
    $SKIP_10 || echo "  ✓ Section 10: Differential Peak (info provided)"
    $SKIP_11 || echo "  ✓ Section 11: Differential Expression (info provided)"
fi

echo ""
echo "Results saved to: ${MODEL_ANALYSIS_DIR}"
echo ""
echo "=================================================="
echo "Pipeline Completed"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review correlation results in Section 01"
echo "2. For interpretation analysis, prepare region BED files and run Section 02 commands"
echo "3. For variant analysis, prepare VCF files and run Section 03 commands"
echo "4. For specialized analyses (09-11), follow the multi-step workflows provided"
echo ""
echo "For detailed documentation on each analysis, refer to individual script help:"
echo "  python <script.py> --help"
echo ""
