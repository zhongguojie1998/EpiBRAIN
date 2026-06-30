#!/bin/bash
# Test for WebPortal/run_variant_analysis.slurm (script 1).
#
# The slurm wrapper's only NEW code is attribution_to_bigwig.py (predicted/diff
# bigwigs + zoom plot are produced by the unchanged upstream pipeline, which
# needs a GPU). This test therefore:
#   1. syntax-checks the slurm wrapper
#   2. runs attribution_to_bigwig.py on the REAL Step-5 attribution output of the
#      reference run rs6581593 / LRRK2 / BasalGanglia-Microglia (CPU only)
#   3. verifies the resulting bigWig (chrom, span, coverage) against metadata
#
# The full GPU pipeline is not submitted here; the sbatch command is printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python
OUT="${TMPDIR:-/tmp}/webportal_test1"; mkdir -p "$OUT"

DATA_DIR="Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/raw_data/interp_diff"
NAME_BASE="chr12_40020870_40545158_LRRK2_BasalGanglia-Microglia_minus"
ATTR_BW="$OUT/attribution_score.bw"

echo "### [1/3] syntax-check slurm wrapper"
bash -n WebPortal/run_variant_analysis.slurm
echo "OK"

echo "### [2/3] run attribution_to_bigwig.py on real LRRK2 Step-5 output"
"$PYTHON" WebPortal/attribution_to_bigwig.py \
    --data-dir "$DATA_DIR" \
    --name-base "$NAME_BASE" \
    --baseline random \
    --output "$ATTR_BW" \
    --fai Data/source/hg38/hg38.fa.fai

echo "### [3/3] verify bigWig against metadata"
"$PYTHON" - "$DATA_DIR" "$NAME_BASE" "$ATTR_BW" <<'PY'
import sys, numpy as np, pyBigWig
data_dir, name_base, bw_path = sys.argv[1:4]
m = np.load(f"{data_dir}/{name_base}_metadata.npy", allow_pickle=True).item()
imp = np.load(f"{data_dir}/{name_base}_random_importance.npy")
ws, nw, ctx, rs = m['window_size'], m['n_window'], m['context_length'], m['real_start']
trim = (ctx // ws - nw) // 2
total_bp = nw * ws
start = rs + trim * ws
span = 1 if imp.shape[0] == total_bp else ws
end = start + (imp.shape[0] * span)

bw = pyBigWig.open(bw_path)
hdr = bw.chroms()
assert m['chr_name'] in hdr, f"chrom {m['chr_name']} missing from header {hdr}"
n_int = bw.header()['nBasesCovered']
mean_real = np.nanmean(imp.astype('f8'))
got_mean = bw.stats(m['chr_name'], start, end, type="mean", exact=True)[0]
bw.close()

assert n_int == imp.shape[0] * span, f"covered {n_int} != {imp.shape[0]*span}"
assert abs(got_mean - mean_real) < 1e-3 * (abs(mean_real) + 1e-9), \
    f"mean mismatch: bw={got_mean} npy={mean_real}"
print(f"PASS: {m['chr_name']}:{start}-{end}  values={imp.shape[0]} span={span} "
      f"covered={n_int}bp mean={got_mean:.6g}")
PY

echo ""
echo "### full pipeline (GPU) -- submit manually:"
echo "sbatch WebPortal/run_variant_analysis.slurm \\"
echo "    --variant chr12:40021869:A:G --track BasalGanglia-Microglia_RNAminus \\"
echo "    --gene LRRK2 --exp-name full_finetune_original_loss_celltype_head_dim8_linear --checkpoint 20"
echo ""
echo "ALL TEST 1 CHECKS PASSED"
