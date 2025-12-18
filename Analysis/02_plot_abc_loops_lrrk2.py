#!/usr/bin/env python3
"""
One-time script to plot ABC loops for LRRK2 using pyGenomeTracks
"""

import pandas as pd
import subprocess
from pathlib import Path

# Paths
ABC_FILE = "Data/source/ABC/merged_abc_results_LRRK2.tsv"
OUTPUT_DIR = Path("Analysis/figures/ABC_plots")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Output files
LINKS_FILE = OUTPUT_DIR / "lrrk2_abc_links.bedpe"
CONFIG_FILE = OUTPUT_DIR / "lrrk2_tracks.ini"
PLOT_FILE = OUTPUT_DIR / "lrrk2_abc_loops.pdf"

print("Loading ABC data...")
df = pd.read_csv(ABC_FILE, sep="\t")

print(f"Loaded {len(df)} enhancer-gene interactions for LRRK2")
print(f"Chromosome: {df['chr'].unique()}")
print(f"Target gene TSS: {df['TargetGeneTSS'].unique()}")

# Create links file in bedpe format (chr1 start1 end1 chr2 start2 end2 score)
# For each enhancer, create a link from enhancer to TSS
print("\nCreating links file...")
links_data = []

for _, row in df.iterrows():
    # Enhancer coordinates
    chr1 = row['chr']
    start1 = row['start']
    end1 = row['end']

    # TSS coordinates (create a small window around TSS)
    chr2 = row['chr']
    tss = row['TargetGeneTSS']
    start2 = tss - 500  # 500bp window around TSS
    end2 = tss + 500

    # ABC score
    score = row['ABC.Score']

    # Cell type for coloring
    cell_type = row['CellType']

    links_data.append({
        'chr1': chr1,
        'start1': start1,
        'end1': end1,
        'chr2': chr2,
        'start2': start2,
        'end2': end2,
        'score': score,
        'cell_type': cell_type
    })

links_df = pd.DataFrame(links_data)

# Save as bedpe format (without header)
links_df[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'score']].to_csv(
    LINKS_FILE, sep="\t", header=False, index=False
)

print(f"Saved {len(links_df)} links to {LINKS_FILE}")

# Determine plotting region - fixed range 40,020kb to 40,475kb
chr_name = df['chr'].iloc[0]
plot_start = 40_020_000  # 40,020kb
plot_end = 40_475_000    # 40,475kb

region = f"{chr_name}:{plot_start}-{plot_end}"

print(f"\nPlotting region: {region}")
print(f"Region size: {(plot_end - plot_start) / 1e6:.2f} Mb")

# Create configuration file for pyGenomeTracks
print("\nCreating configuration file...")
config_content = f"""[x-axis]

[spacer]
height = 0.5

[links]
file = {LINKS_FILE}
title = ABC Enhancer-Gene Links for LRRK2
height = 4
links_type = arcs
color = RdYlBu_r
line_width = 0.5
alpha = 0.8
min_value = 0
max_value = 1.0
compact_arcs_level = 1
orientation = inverted
"""

with open(CONFIG_FILE, 'w') as f:
    f.write(config_content)

print(f"Saved configuration to {CONFIG_FILE}")

# Run pyGenomeTracks
print(f"\nGenerating plot...")
cmd = [
    "pyGenomeTracks",
    "--tracks", str(CONFIG_FILE),
    "--region", region,
    "--outFileName", str(PLOT_FILE),
    "--width", "38",
    "--title", f"ABC Predictions for LRRK2 (n={len(df)} links)"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n✓ Successfully created plot: {PLOT_FILE}")
    print(f"\nSummary:")
    print(f"  - Total links: {len(df)}")
    print(f"  - Cell types: {df['CellType'].nunique()}")
    print(f"  - ABC score range: {df['ABC.Score'].min():.3f} - {df['ABC.Score'].max():.3f}")
    print(f"  - Distance range: {df['distance'].min()} - {df['distance'].max()} bp")
else:
    print(f"\n✗ Error running pyGenomeTracks:")
    print(result.stderr)
    print("\nStdout:")
    print(result.stdout)
