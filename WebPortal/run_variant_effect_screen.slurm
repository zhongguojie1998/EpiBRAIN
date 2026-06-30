#!/bin/bash
#
#SBATCH --job-name=webportal_vep_screen      # name
#SBATCH --mail-user=gz2294@cumc.columbia.edu # email
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                            # nodes
#SBATCH --ntasks-per-node=1                  # 1 task per node
#SBATCH --cpus-per-task=4                    # cores (orchestrator + merge + extract)
#SBATCH --mem=32gb
#SBATCH --time 24:00:00                      # max execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                   # output file name
#
# Web-portal wrapper around Analysis/03_variant_effect_screen/script.sh.
#
# This is an ORCHESTRATOR job: script.sh submits one GPU sbatch job per chunk,
# polls until they finish, then merges results into HDF5. This driver itself
# needs no GPU -- it only launches/waits/merges and then extracts the table.
#
# Deliverable: a variant-by-track fold-change CSV (NO bigWig is produced).
#   - VCF INFO carries gene_ID/gene_name  -> gene_lfc   (exon-aggregated LFC)
#   - plain VCF                           -> local_raw_log_diff (+-992bp LFC)
# Detection is automatic from the VCF (override with --gene-lfc on|off).
#
# Usage:
#   sbatch WebPortal/run_variant_effect_screen.slurm \
#       --vcf path/to/variants.vcf[.gz] \
#       [--model-preset full_atlas|dim8_ckpt20] \
#       [--model PKL_OR_PT --config CFG --label_meta CSV] \
#       [--output H5] [--table CSV] [--experiment NAME] \
#       [--gtf GTF] [--chunks N] [--gene-lfc auto|on|off] [--force]
#
#   --vcf           input VCF (required)
#   --model-preset  full_atlas (default) or dim8_ckpt20
#   --gene-lfc      auto (default; detect gene_ID/gene_name in VCF) | on | off
#
set -euo pipefail

# Under sbatch the script is copied to the spool dir, so BASH_SOURCE is useless;
# SLURM_SUBMIT_DIR is the dir sbatch was launched from (the project root by
# convention). Fall back to BASH_SOURCE for direct `bash WebPortal/...` runs.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python
SCREEN="Analysis/03_variant_effect_screen/script.sh"
EXTRACTOR="WebPortal/screen_h5_to_table.py"

# ----- defaults -----
VCF=""
MODEL_PRESET="full_atlas"
MODEL=""
CONFIG=""
LABEL_META=""
H5=""
TABLE=""
EXPERIMENT=""
GTF="Data/source/gencode.v48.annotation.gtf.gz"
CHUNKS=1
GENE_LFC_MODE="auto"
FORCE=""

usage() { sed -n '14,33p' "${BASH_SOURCE[0]}"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --vcf)          VCF="$2";           shift 2 ;;
        --model-preset) MODEL_PRESET="$2";  shift 2 ;;
        --model)        MODEL="$2";         shift 2 ;;
        --config)       CONFIG="$2";        shift 2 ;;
        --label_meta)   LABEL_META="$2";    shift 2 ;;
        --output)       H5="$2";            shift 2 ;;
        --table)        TABLE="$2";         shift 2 ;;
        --experiment)   EXPERIMENT="$2";    shift 2 ;;
        --gtf)          GTF="$2";           shift 2 ;;
        --chunks)       CHUNKS="$2";        shift 2 ;;
        --gene-lfc)     GENE_LFC_MODE="$2"; shift 2 ;;
        --force)        FORCE="--force";    shift ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ----- validation -----
if [ -z "$VCF" ]; then echo "Error: --vcf is required" >&2; usage; fi
if [ ! -f "$VCF" ]; then echo "Error: VCF not found: $VCF" >&2; exit 1; fi

# ----- resolve model preset (explicit --model overrides) -----
if [ -z "$MODEL" ]; then
    case "$MODEL_PRESET" in
        full_atlas)
            MODEL="Chk/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/chk_epoch_17.pt"
            CONFIG="${CONFIG:-logs/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/overall_setting.yaml}"
            LABEL_META="${LABEL_META:-logs/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas/regression_label_meta.csv}"
            ;;
        dim8_ckpt20)
            MODEL="Chk/full_finetune_original_loss_celltype_head_dim8_linear/chk_epoch_20_packaged.pkl"
            LABEL_META="${LABEL_META:-logs/full_finetune_original_loss_celltype_head_dim8_linear/regression_label_meta.csv}"
            ;;
        *)
            echo "Error: unknown --model-preset '$MODEL_PRESET' (full_atlas|dim8_ckpt20)" >&2
            exit 1 ;;
    esac
fi
# A .pt model is packaged at runtime by script.sh and needs --config.
if [[ "$MODEL" == *.pt ]] && [ -z "$CONFIG" ]; then
    echo "Error: a .pt model requires --config" >&2; exit 1
fi
for f in "$MODEL" "$LABEL_META"; do
    [ -f "$f" ] || { echo "Error: required file not found: $f" >&2; exit 1; }
done

# ----- output paths -----
VCF_STEM="$(basename "$VCF")"; VCF_STEM="${VCF_STEM%.gz}"; VCF_STEM="${VCF_STEM%.vcf}"
EXPERIMENT="${EXPERIMENT:-$VCF_STEM}"
H5="${H5:-Analysis/figures/variant_effect_screen/${VCF_STEM}/${VCF_STEM}.h5}"
mkdir -p "$(dirname "$H5")"

# ----- decide gene_lfc vs local LFC -----
# gene_lfc requires gene_ID/gene_name in the VCF INFO column.
detect_gene_anno() {
    local reader="cat"
    [[ "$VCF" == *.gz ]] && reader="zcat"
    $reader "$VCF" | grep -v '^#' | head -n 1000 | grep -qE 'gene_ID=|gene_name='
}
case "$GENE_LFC_MODE" in
    on)   USE_GENE_LFC=true ;;
    off)  USE_GENE_LFC=false ;;
    auto) if detect_gene_anno; then USE_GENE_LFC=true; else USE_GENE_LFC=false; fi ;;
    *)    echo "Error: --gene-lfc must be auto|on|off" >&2; exit 1 ;;
esac
if $USE_GENE_LFC; then
    [ -f "$GTF" ] || { echo "Error: --gtf not found (required for gene_lfc): $GTF" >&2; exit 1; }
    SCORE="gene_lfc"
else
    SCORE="local_raw_log_diff"
fi

# ----- environment (mirrors inference.slurm; puts BICAN env python on PATH) -----
date; hostname; pwd
export CUDA_HOME=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
CONDA_PREFIX=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include:${CONDA_PREFIX}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
# script.sh hard-requires workingHOME (only used for VPN path translation in slurm mode).
export workingHOME="${workingHOME:-$HOME}"

echo "========================================="
echo "VCF:          $VCF"
echo "Experiment:   $EXPERIMENT"
echo "Model:        $MODEL"
[ -n "$CONFIG" ] && echo "Config:       $CONFIG"
echo "Label meta:   $LABEL_META"
echo "Score:        $SCORE  (gene_lfc=$USE_GENE_LFC)"
echo "HDF5:         $H5"
echo "Chunks:       $CHUNKS"
echo "========================================="

# ----- assemble and run the screen (slurm mode: submits chunk jobs, polls, merges) -----
SCREEN_ARGS=(
    --vcf "$VCF"
    --output "$H5"
    --model "$MODEL"
    --label_meta "$LABEL_META"
    --experiment "$EXPERIMENT"
    --chunks "$CHUNKS"
    --mode slurm
)
[ -n "$CONFIG" ] && SCREEN_ARGS+=(--config "$CONFIG")
[ -n "$FORCE" ]  && SCREEN_ARGS+=("$FORCE")
if $USE_GENE_LFC; then
    SCREEN_ARGS+=(--gene_lfc --gtf "$GTF")
fi

echo "=== Running variant effect screen ==="
bash "$SCREEN" "${SCREEN_ARGS[@]}"

# ----- extract the variant-by-track table -----
TABLE="${TABLE:-${H5%.h5}_${SCORE}_table.csv}"
echo "=== Extracting fold-change table ($SCORE) -> $TABLE ==="
"$PYTHON" "$EXTRACTOR" \
    --h5 "$H5" \
    --experiment "$EXPERIMENT" \
    --score "$SCORE" \
    --output "$TABLE"

echo "========================================="
echo "WebPortal variant-effect screen complete."
echo "  HDF5:  $H5"
echo "  Table: $TABLE  (rows=variants, cols=tracks, value=$SCORE)"
echo "========================================="
date
