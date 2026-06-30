#!/bin/bash
# Test for WebPortal/run_variant_effect_screen.slurm (script 2).
#
# The slurm wrapper's NEW code is screen_h5_to_table.py plus the gene_lfc
# auto-detection. The screen itself needs GPUs, so this test:
#   1. syntax-checks the slurm wrapper
#   2. builds a small gene-annotated VCF subset from the eQTL feather
#      (WebPortal/test_data/eqtl_subset.vcf) and checks gene_lfc auto-detection
#   3. runs screen_h5_to_table.py on a REAL merged screen HDF5
#      (Data/source/eQTL/full_finetune.dim8.chk20.h5) and verifies the table
#
# The full GPU screen is not submitted here; the sbatch command is printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python
OUT="${TMPDIR:-/tmp}/webportal_test2"; mkdir -p "$OUT"

FEATHER="Analysis/07_eQTL/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather"
VCF="WebPortal/test_data/eqtl_subset.vcf"
H5="Data/source/eQTL/full_finetune.dim8.chk20.h5"
TABLE="$OUT/eqtl_subset_local_raw_log_diff_table.csv"
mkdir -p WebPortal/test_data

echo "### [1/4] syntax-check slurm wrapper"
bash -n WebPortal/run_variant_effect_screen.slurm
echo "OK"

echo "### [2/4] build small gene-annotated VCF subset from feather -> $VCF"
"$PYTHON" - "$FEATHER" "$VCF" <<'PY'
import sys, pandas as pd
feather, vcf = sys.argv[1:3]
df = pd.read_feather(feather, columns=["variant_id", "gene_id"]).drop_duplicates("variant_id").head(8)
with open(vcf, "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write('##INFO=<ID=gene_ID,Number=1,Type=String,Description="Ensembl gene id">\n')
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for vid, gid in zip(df.variant_id, df.gene_id):
        chrom, pos, ref, alt = vid.split("_")
        f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\tgene_ID={gid}\n")
print(f"wrote {len(df)} variants")
PY

echo "### [3/4] gene_lfc auto-detection on the VCF"
if grep -v '^#' "$VCF" | head -n 1000 | grep -qE 'gene_ID=|gene_name='; then
    echo "PASS: gene annotation detected -> would deliver gene_lfc"
else
    echo "FAIL: gene annotation NOT detected"; exit 1
fi

echo "### [4/4] run screen_h5_to_table.py on real screen HDF5"
"$PYTHON" WebPortal/screen_h5_to_table.py \
    --h5 "$H5" \
    --experiment eQTL \
    --score local_raw_log_diff \
    --output "$TABLE"

"$PYTHON" - "$H5" "$TABLE" <<'PY'
import sys, h5py, pandas as pd
h5, table = sys.argv[1:3]
with h5py.File(h5, "r") as f:
    n_tracks = f["model_meta/trial_names"].shape[0]
t = pd.read_csv(table)
id_cols = ["index_key", "rsid", "chr", "pos", "ref", "alt"]
assert list(t.columns[:6]) == id_cols, f"bad id cols: {list(t.columns[:6])}"
assert t.shape[1] == 6 + n_tracks, f"cols {t.shape[1]} != 6+{n_tracks}"
assert t.shape[0] > 0, "no rows"
track_block = t.iloc[:, 6:]
assert track_block.notna().any().any(), "all track values NaN"
print(f"PASS: table {t.shape[0]} variants x {n_tracks} tracks; "
      f"e.g. {t['index_key'].iloc[0]} -> {track_block.iloc[0,0]:.4g}")
PY

echo ""
echo "### full screen (GPU) -- submit manually:"
echo "sbatch WebPortal/run_variant_effect_screen.slurm --vcf $VCF --model-preset full_atlas"
echo ""
echo "ALL TEST 2 CHECKS PASSED"
