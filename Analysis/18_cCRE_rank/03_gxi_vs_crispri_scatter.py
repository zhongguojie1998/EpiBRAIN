"""Scatter of gxi attribution vs. in-silico CRISPRi effect on matched
(cCRE, gene, cell_type) triples.

  x = crispri log2fc   (predicted RNA fold-change on silencing the cCRE)
  y = gxi  attr_mean   (mean gradient×input attribution over the cCRE window)

Overplotting at ~10^5 points → hexbin density (single-hue sequential, log count).
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)

# gxi y-axis mode: attr_mean (default) | attr_max | attr_min | attr_sum
#   | signed_peak = per-triple extreme of {attr_max, attr_min} by |value|, sign kept
GXI_COL = sys.argv[1] if len(sys.argv) > 1 else "attr_mean"
GXI_LABEL = {"attr_mean": "mean", "attr_max": "max", "attr_min": "min",
             "attr_sum": "sum", "signed_peak": "signed peak"}.get(GXI_COL, GXI_COL)

D = "Analysis/18_cCRE_rank/output/full_finetune_original_loss_celltype_head_dim8_linear_full_atlas_17"
FIG_DIR = "Analysis/18_cCRE_rank/figures"
os.makedirs(FIG_DIR, exist_ok=True)

KEYS = ["cCRE_id", "gene", "cell_type"]

gxi_cols = ["attr_max", "attr_min"] if GXI_COL == "signed_peak" else [GXI_COL]
gxi = pd.read_csv(f"{D}/gxi/gxi_all.csv", usecols=KEYS + gxi_cols)
cri = pd.read_csv(f"{D}/crispri/crispri_all.csv", usecols=KEYS + ["log2fc"])

if GXI_COL == "signed_peak":
    # extreme by absolute value, keeping the original sign
    gxi["signed_peak"] = np.where(
        gxi["attr_max"].abs() >= gxi["attr_min"].abs(), gxi["attr_max"], gxi["attr_min"])

m = gxi.merge(cri, on=KEYS, how="inner").dropna(subset=[GXI_COL, "log2fc"])
x = m["log2fc"].to_numpy()
y = m[GXI_COL].to_numpy()
n = len(m)
rho, _ = spearmanr(x, y)
r, _ = pearsonr(x, y)
print(f"matched triples: {n:,}  Spearman={rho:.3f}  Pearson={r:.3f}")

# ---- style: light surface, recessive axes, dark ink ----
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
})

MARK = "#2f6f8f"  # single primary hue for the point cloud

# robust limits (0.5–99.5 pct) so a few outliers don't flatten the cloud
xlo, xhi = np.percentile(x, 0.5), np.percentile(x, 99.5)
ylo, yhi = np.percentile(y, 0.5), np.percentile(y, 99.5)

fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=200)
ax.axhline(0, color=GRID, lw=1.0, zorder=1)
ax.axvline(0, color=GRID, lw=1.0, zorder=1)
# ~6x10^5 points overplot heavily → tiny, low-alpha, rasterized marks
ax.scatter(x, y, s=2, c=MARK, alpha=0.04, linewidths=0, rasterized=True, zorder=2)
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)

ax.set_xlabel("CRISPRi effect  (log2 fold-change, silenced / ref)")
ax.set_ylabel(f"Gradient×input attribution  ({GXI_LABEL} over cCRE)")
ax.set_title("Enhancer attribution vs. in-silico CRISPRi effect", fontsize=12, pad=10)
ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.text(0.03, 0.97,
        f"matched (cCRE×gene×cell type) = {n:,}\nSpearman ρ = {rho:.3f}\nPearson r = {r:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, lw=0.8, alpha=0.9))

fig.tight_layout()
out = f"{FIG_DIR}/gxi_{GXI_LABEL.replace(' ', '_')}_vs_crispri_scatter.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved: {out}")
