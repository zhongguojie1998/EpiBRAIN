#!/bin/bash
#
#SBATCH --job-name=webportal_variant         # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu # email
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu                      # GPU partition (inference + attribution)
#SBATCH --nodes=1                            # nodes
#SBATCH --ntasks-per-node=1                  # 1 task per node
#SBATCH --cpus-per-task=8                    # cores per task
#SBATCH --gres=gpu:1                         # gpus
#SBATCH --mem=64gb
#SBATCH --time 24:00:00                      # max execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                   # output file name
#
# Web-portal wrapper around Analysis/variant_analysis_pipeline.sh.
#
# Produces bigWig deliverables for a single variant:
#   - predicted tracks (ref + alt)          <ANALYSIS_DIR>/inference/{pred,alt}/*.bw   (Step 1)
#   - diff tracks                           <ANALYSIS_DIR>/inference/diff/*.bw         (Step 1)
#   - attribution-score track (optional)    <ANALYSIS_DIR>/attribution_score.bw       (Step 5 -> bigWig)
#   - motif zoom plot (optional)            <ANALYSIS_DIR>/motif_interpretation_zoom.pdf (Step 6 zoom only)
#   - saturation mutagenesis (optional)     <ANALYSIS_DIR>/sat_mutagenesis/            (Step 2, behind --sat)
#
# Attribution (Step 5 + zoom plot + bigWig) runs ONLY when --track is given.
# Visualization PDF (Step 3), mutagenesis-viz PDF (Step 4), full motif plot
# (Step 6 full) and TOMTOM (Step 7) are always skipped -- deliverables are tracks.
#
# Usage:
#   sbatch WebPortal/run_variant_analysis.slurm \
#       --variant chr11:113400106:G:T \
#       [--track BasalGanglia-STR-D1-MSN_RNAminus] \
#       [--gene DRD2 | --region chr11:113400000-113401000] \
#       [--sat] [--no-labels] \
#       [--exp-name NAME] [--checkpoint N] [--output-base DIR] \
#       [--num-gpus N] [--disease NAME] [--fai PATH] [--baseline random]
#
#   --variant     chr:pos:ref:alt           (required)
#   --track       attribution track name    (optional; no attribution if omitted)
#   --gene        gene symbol               (optional; requires --track; overrides --region)
#   --region      chr:start-end             (optional; requires --track)
#   --sat         run saturation mutagenesis (Step 2)
#   --no-labels   skip inference/label/ bigwigs (observed coverage, identical
#                 for every variant -- serve them statically instead; ~13s)
#
set -euo pipefail

# Under sbatch the script is copied to the spool dir, so BASH_SOURCE is useless;
# SLURM_SUBMIT_DIR is the dir sbatch was launched from (the project root by
# convention). Fall back to BASH_SOURCE for direct `bash WebPortal/...` runs.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python
PIPELINE="Analysis/variant_analysis_pipeline.sh"
CONVERTER="WebPortal/attribution_to_bigwig.py"

# ----- defaults (kept in sync with variant_analysis_pipeline.sh) -----
VARIANT=""
TRACK=""
GENE=""
REGION=""
RUN_SAT=false
NO_LABELS=false
EXP_NAME="full_finetune_original_loss_celltype_head_dim8_linear_full_atlas"
CHECKPOINT=17
OUTPUT_BASE="Analysis/figures"
NUM_GPUS=1
DISEASE=""
BASELINE="random"
FAI="Data/source/hg38/hg38.fa.fai"

usage() {
    sed -n '17,40p' "${BASH_SOURCE[0]}"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)      VARIANT="$2";     shift 2 ;;
        --track)        TRACK="$2";       shift 2 ;;
        --gene)         GENE="$2";        shift 2 ;;
        --region)       REGION="$2";      shift 2 ;;
        --sat)          RUN_SAT=true;     shift ;;
        --no-labels)    NO_LABELS=true;   shift ;;
        --exp-name)     EXP_NAME="$2";    shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";  shift 2 ;;
        --output-base)  OUTPUT_BASE="$2"; shift 2 ;;
        --num-gpus)     NUM_GPUS="$2";    shift 2 ;;
        --disease)      DISEASE="$2";     shift 2 ;;
        --baseline)     BASELINE="$2";    shift 2 ;;
        --fai)          FAI="$2";         shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ----- validation -----
if [ -z "$VARIANT" ]; then
    echo "Error: --variant is required" >&2
    usage
fi
if { [ -n "$GENE" ] || [ -n "$REGION" ]; } && [ -z "$TRACK" ]; then
    echo "Error: --gene/--region only affect attribution; they require --track" >&2
    exit 1
fi
if [ -n "$GENE" ] && [ -n "$REGION" ]; then
    echo "Warning: both --gene and --region given; --gene overrides --region" >&2
    REGION=""
fi

# ----- environment (mirrors inference.slurm / 09_2_ABC_run.slurm) -----
date; hostname; pwd
export USE_TF=0   # keep transformers off its TF path; see variant_analysis_pipeline.sh
export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include

# ----- build skip-steps -----
# Always skip: 3 (viz PDF), 4 (mut-viz PDF), 6 (full motif plot), 7 (TOMTOM).
# We generate the Step-6 ZOOM plot ourselves below to avoid the full plot.
SKIP="3,4,6,7"
$RUN_SAT || SKIP="2,${SKIP}"               # Step 2 only with --sat
[ -z "$TRACK" ] && SKIP="${SKIP},5"        # Step 5 (attribution) only with --track

# ----- assemble pipeline command -----
# --track is mandatory in the pipeline even though Step 1/2 don't use it;
# pass a harmless placeholder when the caller omitted it (all track-using
# steps are skipped in that case).
PIPE_TRACK="${TRACK:-NA_PLACEHOLDER}"
PIPE_ARGS=(
    --variant "$VARIANT"
    --track "$PIPE_TRACK"
    --exp-name "$EXP_NAME"
    --checkpoint "$CHECKPOINT"
    --output-base "$OUTPUT_BASE"
    --num-gpus "$NUM_GPUS"
    --skip-steps "$SKIP"
)
$NO_LABELS        && PIPE_ARGS+=(--no-labels)
[ -n "$GENE" ]    && PIPE_ARGS+=(--gene "$GENE")
[ -n "$REGION" ]  && PIPE_ARGS+=(--region "$REGION")
[ -n "$DISEASE" ] && PIPE_ARGS+=(--disease "$DISEASE")

PIPE_LOG="$(mktemp "${TMPDIR:-/tmp}/webportal_pipeline.XXXXXX.log")"
trap 'rm -f "$PIPE_LOG"' EXIT

echo "=== Running pipeline (skip-steps: $SKIP) ==="
bash "$PIPELINE" "${PIPE_ARGS[@]}" 2>&1 | tee "$PIPE_LOG"

# Recover the analysis directory the pipeline chose (mode-agnostic).
ANALYSIS_DIR="$(grep -m1 '^Output Dir:' "$PIPE_LOG" | awk '{print $NF}')"
if [ -z "$ANALYSIS_DIR" ]; then
    echo "Error: could not parse 'Output Dir:' from pipeline output" >&2
    exit 1
fi
echo "=== Pipeline outputs in: $ANALYSIS_DIR ==="

# ----- attribution post-processing (only when --track was given) -----
if [ -n "$TRACK" ]; then
    # Step 5 prints the plot command containing --data_dir and --name_base.
    NAME_BASE="$(grep -oP '(?<=--name_base )[^ ]+' "$PIPE_LOG" | head -1)"
    DATA_DIR="$(grep -oP '(?<=--data_dir )[^ ]+' "$PIPE_LOG" | head -1)"
    [ -z "$DATA_DIR" ] && DATA_DIR="./Res/${EXP_NAME}/analysis_${CHECKPOINT}/raw_data/interp_diff"

    if [ -z "$NAME_BASE" ]; then
        echo "Error: could not determine NAME_BASE from Step 5 output" >&2
        exit 1
    fi
    echo "=== Attribution name_base: $NAME_BASE (data_dir: $DATA_DIR) ==="

    # (a) attribution scores -> bigWig deliverable
    ATTR_BW="${ANALYSIS_DIR}/attribution_score.bw"
    echo "=== Converting attribution scores to bigWig: $ATTR_BW ==="
    "$PYTHON" "$CONVERTER" \
        --data-dir "$DATA_DIR" \
        --name-base "$NAME_BASE" \
        --baseline "$BASELINE" \
        --output "$ATTR_BW" \
        --fai "$FAI"

    # (b) Step-6 zoom plot only (variant +/- 50 bp)
    IFS=':' read -r CHR POS _REF _ALT <<< "$VARIANT"
    ZOOM_START=$((POS - 50))
    ZOOM_END=$((POS + 50))
    ZOOM_PDF="${ANALYSIS_DIR}/motif_interpretation_zoom.pdf"
    echo "=== Generating motif zoom plot (${CHR}:${ZOOM_START}-${ZOOM_END}): $ZOOM_PDF ==="
    "$PYTHON" Analysis/02_motif_interpretation_plot.py \
        --data_dir "$DATA_DIR" \
        --name_base "$NAME_BASE" \
        --baseline "$BASELINE" \
        --output "$ZOOM_PDF" \
        --start "$ZOOM_START" \
        --end "$ZOOM_END" \
        --show_sequence
fi

echo "========================================="
echo "WebPortal variant analysis complete."
echo "  Predicted tracks: ${ANALYSIS_DIR}/inference/{pred,alt}/"
echo "  Diff tracks:      ${ANALYSIS_DIR}/inference/diff/"
[ -n "$TRACK" ] && echo "  Attribution track: ${ANALYSIS_DIR}/attribution_score.bw"
[ -n "$TRACK" ] && echo "  Motif zoom plot:   ${ANALYSIS_DIR}/motif_interpretation_zoom.pdf"
$RUN_SAT && echo "  Sat. mutagenesis:  ${ANALYSIS_DIR}/sat_mutagenesis/"
echo "========================================="
date
