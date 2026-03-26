"""
TF-MoDISco pipeline for MiniAtlas peaks gradient x input attributions.

Subcommands:
  prepare      — Extract center regions from .pt tensors, build NPZ + BED per cell type
  modisco      — Generate bash script with modisco motifs commands per cell type
  postprocess  — Extract seqlets, CWMs, build MEME file, run TOMTOM
  aggregate    — Collect CWMs across all cell types, cluster, and visualize
"""

import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import h5py
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pt_filename(fname):
    """Parse a grad_input .pt filename into (chr, start, end, trial, celltype).

    Filename format: {chr}_{start}_{end}_{trial}_grad_input.pt
    Trial format: MiniAtlas-{CT}_K27Ac
    """
    stem = fname.replace("_grad_input.pt", "")
    # Find '_MiniAtlas-' to split coordinates from trial
    m = re.search(r"_MiniAtlas-", stem)
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname}")
    coord_part = stem[: m.start()]
    trial = stem[m.start() + 1 :]  # strip leading '_'

    # coord_part = chr_start_end
    parts = coord_part.split("_")
    chr_name = parts[0]
    start = int(parts[1])
    end = int(parts[2])

    # Extract cell type from trial
    ct = trial.replace("MiniAtlas-", "").replace("_K27Ac", "")
    return chr_name, start, end, trial, ct


def discover_celltypes(pt_dir):
    """Scan .pt files and group by cell type.

    Returns dict: celltype -> list of (chr, start, end, trial, filename)
    """
    ct_map = defaultdict(list)
    for fname in sorted(os.listdir(pt_dir)):
        if not fname.endswith("_grad_input.pt"):
            continue
        chr_name, start, end, trial, ct = parse_pt_filename(fname)
        ct_map[ct].append((chr_name, start, end, trial, fname))
    return dict(ct_map)


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------

def prepare_one_celltype(ct, peaks, pt_dir, out_dir, center_bp):
    """Prepare NPZ + BED for one cell type."""
    ct_dir = os.path.join(out_dir, ct)

    # Skip if both NPZ files already exist
    attr_path = os.path.join(ct_dir, "attributions.npz")
    seq_path = os.path.join(ct_dir, "sequences.npz")
    if os.path.exists(attr_path) and os.path.exists(seq_path):
        return ct, len(peaks), True

    os.makedirs(ct_dir, exist_ok=True)

    # Write peaks.bed
    bed_path = os.path.join(ct_dir, "peaks.bed")
    with open(bed_path, "w") as f:
        for chr_name, start, end, trial, _ in peaks:
            f.write(f"{chr_name}\t{start}\t{end}\t{chr_name}:{start}-{end}\t{trial}\n")

    context_len = 524288
    mid = context_len // 2
    half = center_bp // 2
    n_peaks = len(peaks)

    # Pre-allocate output arrays to avoid holding full tensors in memory
    # Each .pt is [1, 524288, 4] (~8MB) but we only keep [center_bp, 4] (~2KB)
    attribs_t = np.empty((n_peaks, 4, center_bp), dtype=np.float32)
    onehot_t = np.empty((n_peaks, 4, center_bp), dtype=np.float32)

    for i, (chr_name, start, end, trial, fname) in enumerate(peaks):
        pt_path = os.path.join(pt_dir, fname)
        tensor = torch.load(pt_path, map_location="cpu")  # [1, 524288, 4]
        # Extract center region and immediately convert to numpy
        a = tensor[0, mid - half : mid - half + center_bp, :].numpy()  # [center_bp, 4]
        del tensor

        # Reconstruct one-hot from this single peak
        best_nuc = np.argmax(np.abs(a), axis=-1)  # [center_bp]
        oh = np.zeros_like(a)
        oh[np.arange(center_bp), best_nuc] = 1.0

        # Store transposed [4, center_bp]
        attribs_t[i] = a.T
        onehot_t[i] = oh.T

    np.savez_compressed(os.path.join(ct_dir, "attributions.npz"), attribs_t)
    np.savez_compressed(os.path.join(ct_dir, "sequences.npz"), onehot_t)

    return ct, len(peaks), False


@click.command("prepare")
@click.option("--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--celltype", default=None, type=str,
              help="Comma-separated cell types to process (default: all)")
@click.option("--center_bp", default=500, type=int)
@click.option("--n_jobs", default=4, type=int)
def cmd_prepare(exp_name, chk, celltype, center_bp, n_jobs):
    """Extract center attribution regions into NPZ files per cell type."""
    pt_dir = f"Res/{exp_name}/analysis_{chk}/raw_data/interp_gradient_input"
    out_dir = f"Res/{exp_name}/analysis_{chk}/modisco_miniatlas"

    print(f"Scanning {pt_dir} ...")
    ct_map = discover_celltypes(pt_dir)
    print(f"Found {len(ct_map)} cell types, {sum(len(v) for v in ct_map.values())} total peaks")

    # Filter to requested cell types
    if celltype:
        requested = [c.strip() for c in celltype.split(",")]
        ct_map = {k: v for k, v in ct_map.items() if k in requested}
        missing = set(requested) - set(ct_map.keys())
        if missing:
            print(f"WARNING: cell types not found: {missing}")
    print(f"Processing {len(ct_map)} cell types")

    if n_jobs <= 1:
        for ct, peaks in sorted(ct_map.items()):
            ct_name, n, skipped = prepare_one_celltype(ct, peaks, pt_dir, out_dir, center_bp)
            print(f"  {ct_name}: {n} peaks{' (skipped)' if skipped else ''}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(prepare_one_celltype, ct, peaks, pt_dir, out_dir, center_bp): ct
                for ct, peaks in ct_map.items()
            }
            for future in as_completed(futures):
                ct_name, n, skipped = future.result()
                print(f"  {ct_name}: {n} peaks{' (skipped)' if skipped else ''}")

    print("Done.")


# ---------------------------------------------------------------------------
# Subcommand: modisco
# ---------------------------------------------------------------------------

@click.command("modisco")
@click.option("--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--celltype", default=None, type=str,
              help="Comma-separated cell types (default: all with NPZ)")
@click.option("--center_bp", default=500, type=int)
@click.option("--num_seqlets", default=20000, type=int)
@click.option("--output_script", default=None, type=str,
              help="Path to output bash script (default: auto)")
def cmd_modisco(exp_name, chk, celltype, center_bp, num_seqlets, output_script):
    """Generate bash script with modisco motifs commands."""
    base_dir = f"Res/{exp_name}/analysis_{chk}/modisco_miniatlas"

    # Find cell types with prepared NPZ files
    celltypes = []
    for d in sorted(os.listdir(base_dir)):
        ct_dir = os.path.join(base_dir, d)
        if os.path.isdir(ct_dir) and os.path.exists(os.path.join(ct_dir, "attributions.npz")):
            celltypes.append(d)

    if celltype:
        requested = [c.strip() for c in celltype.split(",")]
        celltypes = [c for c in celltypes if c in requested]

    if not celltypes:
        print("No cell types found with prepared NPZ files.")
        return

    if output_script is None:
        output_script = os.path.join(base_dir, "run_modisco.sh")

    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for ct in celltypes:
        ct_dir = os.path.join(base_dir, ct)
        lines.append(f"# {ct}")
        lines.append(
            f"cd {os.path.abspath(ct_dir)} && "
            f"modisco motifs "
            f"-s sequences.npz -a attributions.npz "
            f"-n {num_seqlets} -w {center_bp} "
            f"-o modisco_results.h5 -v"
        )
        lines.append("")

    with open(output_script, "w") as f:
        f.write("\n".join(lines))
    os.chmod(output_script, 0o755)
    print(f"Wrote {output_script} with {len(celltypes)} cell types")


# ---------------------------------------------------------------------------
# Subcommand: postprocess
# ---------------------------------------------------------------------------

def ic_clip_pwm(pwm, threshold=0.3):
    """Clip PWM columns with low information content."""
    # pwm: [L, 4]
    eps = 1e-10
    p = pwm / (pwm.sum(axis=1, keepdims=True) + eps)
    p = np.clip(p, eps, 1.0)
    ic = 2.0 + (p * np.log2(p)).sum(axis=1)  # [L]
    # Find contiguous region above threshold
    above = ic >= threshold
    if not above.any():
        return pwm, 0, len(pwm)
    first = np.argmax(above)
    last = len(above) - 1 - np.argmax(above[::-1])
    return pwm[first : last + 1], first, last + 1


def write_meme_header(f):
    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies\n")
    f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")


def write_meme_motif(f, name, pwm):
    """Write one motif in MEME format. pwm: [L, 4] (A, C, G, T)."""
    L = pwm.shape[0]
    f.write(f"MOTIF {name}\n")
    f.write(f"letter-probability matrix: alength= 4 w= {L}\n")
    for row in pwm:
        row_norm = row / (row.sum() + 1e-10)
        f.write(f" {row_norm[0]:.6f}  {row_norm[1]:.6f}  {row_norm[2]:.6f}  {row_norm[3]:.6f}\n")
    f.write("\n")


def postprocess_one_celltype(ct, base_dir, meme_db, center_bp):
    """Post-process modisco results for one cell type."""
    ct_dir = os.path.join(base_dir, ct)
    h5_path = os.path.join(ct_dir, "modisco_results.h5")
    bed_path = os.path.join(ct_dir, "peaks.bed")

    if not os.path.exists(h5_path):
        return ct, "skipped (no modisco_results.h5)"

    # Load peaks for coordinate mapping
    peaks = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            peaks.append((parts[0], int(parts[1]), int(parts[2])))

    cwm_dir = os.path.join(ct_dir, "cwms")
    os.makedirs(cwm_dir, exist_ok=True)

    seqlet_records = []
    cluster_records = []
    all_seqlet_logos = []
    motif_pwms = {}  # name -> clipped PWM

    with h5py.File(h5_path, "r") as h5:
        for direction in ["pos_patterns", "neg_patterns"]:
            if direction not in h5:
                continue
            dir_prefix = direction.split("_")[0]
            grp = h5[direction]
            pattern_names = sorted([k for k in grp.keys() if k.startswith("pattern_")])

            for pname in pattern_names:
                pat = grp[pname]

                # --- CWMs ---
                if "contrib_scores" in pat:
                    cwm = pat["contrib_scores"][:]  # [L, 4]
                    np.save(os.path.join(cwm_dir, f"{dir_prefix}_{pname}_cwm.npy"), cwm)
                if "sequence" in pat:
                    seq = pat["sequence"][:]
                    np.save(os.path.join(cwm_dir, f"{dir_prefix}_{pname}_sequence.npy"), seq)

                    # Build PWM for MEME from sequence (PFM)
                    clipped, _, _ = ic_clip_pwm(seq, threshold=0.3)
                    motif_name = f"{dir_prefix}_{pname}"
                    motif_pwms[motif_name] = clipped

                # --- Seqlets ---
                if "seqlets" not in pat:
                    continue
                seqlets = pat["seqlets"]
                starts = seqlets["start"][:]
                ends = seqlets["end"][:]
                example_idxs = seqlets["example_idx"][:]
                is_revcomp = seqlets["is_revcomp"][:] if "is_revcomp" in seqlets else np.zeros(len(starts), dtype=bool)

                # Seqlet contrib scores for logos
                if "contrib_scores" in seqlets:
                    logos = seqlets["contrib_scores"][:]  # [n_seqlets, L, 4]
                    all_seqlet_logos.append(logos)

                full_pattern_name = f"{dir_prefix}_{pname}"
                for i in range(len(starts)):
                    eidx = int(example_idxs[i])
                    s = int(starts[i])
                    e = int(ends[i])
                    rc = bool(is_revcomp[i])
                    strand = "-" if rc else "+"

                    # Map to genomic coords
                    peak_chr, peak_start, peak_end = peaks[eidx]
                    peak_center = (peak_start + peak_end) // 2
                    offset_start = peak_center - center_bp // 2
                    gen_start = offset_start + s
                    gen_end = offset_start + e

                    seqlet_id = f"{full_pattern_name}.{i}"
                    seqlet_records.append(
                        (peak_chr, gen_start, gen_end, seqlet_id, 0, strand)
                    )
                    cluster_records.append(
                        (seqlet_id, full_pattern_name, dir_prefix, eidx,
                         peak_chr, gen_start, gen_end, strand)
                    )

    # Write seqlets.bed
    with open(os.path.join(ct_dir, "seqlets.bed"), "w") as f:
        for rec in seqlet_records:
            f.write("\t".join(str(x) for x in rec) + "\n")

    # Write seqlet_logos.npy
    if all_seqlet_logos:
        np.save(os.path.join(ct_dir, "seqlet_logos.npy"),
                np.concatenate(all_seqlet_logos, axis=0))

    # Write cluster_report.tsv
    with open(os.path.join(ct_dir, "cluster_report.tsv"), "w") as f:
        f.write("seqlet_id\tpattern_name\tpattern_direction\texample_idx\tchr\tstart\tend\tstrand\n")
        for rec in cluster_records:
            f.write("\t".join(str(x) for x in rec) + "\n")

    # Write motifs.meme
    meme_path = os.path.join(ct_dir, "motifs.meme")
    if motif_pwms:
        with open(meme_path, "w") as f:
            write_meme_header(f)
            for name, pwm in motif_pwms.items():
                if pwm.shape[0] >= 3:  # skip very short motifs
                    write_meme_motif(f, name, pwm)

    # Run TOMTOM
    tomtom_dir = os.path.join(ct_dir, "tomtom")
    if motif_pwms and meme_db and os.path.exists(meme_db):
        import subprocess
        cmd = [
            "tomtom", "-dist", "pearson", "-evalue", "-thresh", "1.0",
            "-oc", tomtom_dir, meme_path, meme_db
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return ct, f"TOMTOM failed: {result.stderr[:200]}"

    n_seqlets = len(seqlet_records)
    n_motifs = len(motif_pwms)
    return ct, f"{n_seqlets} seqlets, {n_motifs} motifs"


@click.command("postprocess")
@click.option("--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--celltype", default=None, type=str,
              help="Comma-separated cell types (default: all with modisco_results.h5)")
@click.option("--meme_db", default="Data/source/meme-5.4.1/motif_databases/HOCOMOCO/H12CORE_meme_format.meme",
              type=str)
@click.option("--center_bp", default=500, type=int)
@click.option("--n_jobs", default=4, type=int)
def cmd_postprocess(exp_name, chk, celltype, meme_db, center_bp, n_jobs):
    """Extract seqlets, CWMs, build MEME file, run TOMTOM."""
    base_dir = f"Res/{exp_name}/analysis_{chk}/modisco_miniatlas"

    # Find cell types with modisco results
    celltypes = []
    for d in sorted(os.listdir(base_dir)):
        ct_dir = os.path.join(base_dir, d)
        if os.path.isdir(ct_dir) and os.path.exists(os.path.join(ct_dir, "modisco_results.h5")):
            celltypes.append(d)

    if celltype:
        requested = [c.strip() for c in celltype.split(",")]
        celltypes = [c for c in celltypes if c in requested]

    if not celltypes:
        print("No cell types found with modisco_results.h5")
        return

    print(f"Post-processing {len(celltypes)} cell types")

    meme_db = os.path.abspath(meme_db)

    if n_jobs <= 1:
        for ct in celltypes:
            ct_name, status = postprocess_one_celltype(ct, base_dir, meme_db, center_bp)
            print(f"  {ct_name}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(postprocess_one_celltype, ct, base_dir, meme_db, center_bp): ct
                for ct in celltypes
            }
            for future in as_completed(futures):
                ct_name, status = future.result()
                print(f"  {ct_name}: {status}")

    print("Done.")


# ---------------------------------------------------------------------------
# Subcommand: aggregate
# ---------------------------------------------------------------------------

@click.command("aggregate")
@click.option("--exp_name", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--method", default="both", type=click.Choice(["umap", "tsne", "both"]),
              help="Dimensionality reduction method")
@click.option("--max_seqlets_per_ct", default=2000, type=int,
              help="Max seqlets to sample per cell type (0 = all)")
@click.option("--min_seqlets_per_pattern", default=100, type=int,
              help="Skip patterns with fewer seqlets than this")
@click.option("--min_seqlet_attr", default=2.0, type=float,
              help="Min sum(abs(contrib_scores)) for a seqlet to keep (0 = no filter)")
@click.option("--min_pattern_attr", default=0.0, type=float,
              help="Min CWM sum(abs(contrib_scores)) for a pattern to keep (0 = no filter)")
@click.option("--n_neighbors", default=500, type=int,
              help="K for coarse gapped-kmer neighbor search")
@click.option("--min_overlap", default=0.7, type=float,
              help="Minimum overlap fraction for Jaccard sliding window")
@click.option("--perplexity", default=10.0, type=float,
              help="t-SNE perplexity for density adaptation")
@click.option("--skip_leiden", is_flag=True, default=False,
              help="Stop after UMAP/t-SNE, skip Leiden clustering and downstream")
@click.option("--n_leiden_runs", default=50, type=int,
              help="Number of Leiden seeds for clustering")
@click.option("--min_cluster_size", default=20, type=int,
              help="Minimum seqlets per cluster to keep")
@click.option("--merge_threshold", default=0.8, type=float,
              help="Pearson correlation threshold for merging similar clusters")
@click.option("--meme_db",
              default="Data/source/meme-5.4.1/motif_databases/HOCOMOCO/H12CORE_meme_format.meme",
              type=str, help="MEME database for TOMTOM comparison")
@click.option("--seed", default=42, type=int)
def cmd_aggregate(exp_name, chk, method, max_seqlets_per_ct,
                  min_seqlets_per_pattern, min_seqlet_attr,
                  min_pattern_attr, n_neighbors,
                  min_overlap, perplexity, skip_leiden, n_leiden_runs,
                  min_cluster_size, merge_threshold, meme_db, seed):
    """Aggregate seqlets across cell types using modisco's native similarity."""
    import subprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modiscolite.core import Seqlet
    from modiscolite.affinitymat import (cosine_similarity_from_seqlets,
                                         jaccard_from_seqlets)
    from modiscolite.tfmodisco import _density_adaptation
    from modiscolite.cluster import LeidenCluster

    base_dir = f"Res/{exp_name}/analysis_{chk}/modisco_miniatlas"
    out_dir = os.path.join(base_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # --- Step 1: Load seqlets as modisco Seqlet objects ---
    # Modisco processes pos_patterns (sign=1) and neg_patterns (sign=-1)
    # in separate pipelines. The `sign` parameter controls which attribution
    # positions matter for gapped k-mer extraction: sign=1 picks the most
    # positive attributions, sign=-1 picks the most negative. We must
    # preserve this separation in the coarse neighbor search.
    print("Loading seqlets from all cell types...")
    pos_seqlets = []     # seqlets from pos_patterns
    pos_celltypes = []
    pos_patterns = []
    neg_seqlets = []     # seqlets from neg_patterns
    neg_celltypes = []
    neg_patterns = []

    for ct in sorted(os.listdir(base_dir)):
        ct_dir = os.path.join(base_dir, ct)
        h5_path = os.path.join(ct_dir, "modisco_results.h5")
        if not os.path.isfile(h5_path):
            continue

        ct_pos_seqlets = []
        ct_pos_patterns = []
        ct_neg_seqlets = []
        ct_neg_patterns = []
        with h5py.File(h5_path, "r") as h5:
            for direction in ["pos_patterns", "neg_patterns"]:
                if direction not in h5:
                    continue
                dir_prefix = direction.split("_")[0]
                is_pos = dir_prefix == "pos"
                grp = h5[direction]
                for pname in sorted(k for k in grp.keys() if k.startswith("pattern_")):
                    pat = grp[pname]
                    if "seqlets" not in pat:
                        continue
                    sq = pat["seqlets"]
                    required = ["contrib_scores", "hypothetical_contribs",
                                "sequence", "start", "end", "example_idx",
                                "is_revcomp"]
                    if not all(k in sq for k in required):
                        continue

                    # Filter: skip patterns with too few seqlets
                    n_seqlets_in_pat = sq["contrib_scores"].shape[0]
                    if n_seqlets_in_pat < min_seqlets_per_pattern:
                        continue

                    # Filter: skip patterns with weak CWM attribution
                    if min_pattern_attr > 0 and "contrib_scores" in pat:
                        cwm_strength = np.abs(pat["contrib_scores"][:]).sum()
                        if cwm_strength < min_pattern_attr:
                            continue

                    contrib = sq["contrib_scores"][:]        # [n, L, 4]
                    hyp = sq["hypothetical_contribs"][:]     # [n, L, 4]
                    seq = sq["sequence"][:]                   # [n, L, 4]
                    starts = sq["start"][:]
                    ends = sq["end"][:]
                    example_idxs = sq["example_idx"][:]
                    is_revcomp = sq["is_revcomp"][:]

                    full_pname = f"{ct}_{dir_prefix}_{pname}"
                    for i in range(contrib.shape[0]):
                        # Filter: skip seqlets with weak attribution
                        if min_seqlet_attr > 0:
                            attr_mag = np.abs(contrib[i]).sum()
                            if attr_mag < min_seqlet_attr:
                                continue

                        s = Seqlet(example_idx=int(example_idxs[i]),
                                   start=int(starts[i]),
                                   end=int(ends[i]),
                                   is_revcomp=bool(is_revcomp[i]))
                        s.contrib_scores = contrib[i]
                        s.hypothetical_contribs = hyp[i]
                        s.sequence = seq[i]
                        if is_pos:
                            ct_pos_seqlets.append(s)
                            ct_pos_patterns.append(full_pname)
                        else:
                            ct_neg_seqlets.append(s)
                            ct_neg_patterns.append(full_pname)

        # Subsample pos and neg independently
        for seqlets_list, patterns_list, out_s, out_ct, out_p in [
            (ct_pos_seqlets, ct_pos_patterns, pos_seqlets, pos_celltypes, pos_patterns),
            (ct_neg_seqlets, ct_neg_patterns, neg_seqlets, neg_celltypes, neg_patterns),
        ]:
            if not seqlets_list:
                continue
            n_ct = len(seqlets_list)
            if max_seqlets_per_ct > 0 and n_ct > max_seqlets_per_ct:
                idx = rng.choice(n_ct, max_seqlets_per_ct, replace=False)
                idx.sort()
                seqlets_list = [seqlets_list[i] for i in idx]
                patterns_list = [patterns_list[i] for i in idx]
                n_ct = max_seqlets_per_ct
            out_s.extend(seqlets_list)
            out_ct.extend([ct] * n_ct)
            out_p.extend(patterns_list)

        n_pos = len(ct_pos_seqlets)
        n_neg = len(ct_neg_seqlets)
        if n_pos or n_neg:
            parts = []
            if n_pos:
                parts.append(f"{min(n_pos, max_seqlets_per_ct) if max_seqlets_per_ct > 0 else n_pos} pos")
            if n_neg:
                parts.append(f"{min(n_neg, max_seqlets_per_ct) if max_seqlets_per_ct > 0 else n_neg} neg")
            print(f"  {ct}: {', '.join(parts)}")

    # Combine pos + neg, keeping track of the boundary
    n_pos = len(pos_seqlets)
    n_neg = len(neg_seqlets)
    all_seqlets = pos_seqlets + neg_seqlets
    all_celltypes = pos_celltypes + neg_celltypes
    all_patterns = pos_patterns + neg_patterns
    N = len(all_seqlets)

    if N == 0:
        print("No seqlets found.")
        return

    celltypes = np.array(all_celltypes)
    patterns = np.array(all_patterns)
    print(f"Total: {N} seqlets ({n_pos} pos, {n_neg} neg) "
          f"from {len(set(celltypes))} cell types")

    # Save metadata (cell_classes assigned later, write after class mapping)
    _metadata_path = os.path.join(out_dir, "seqlet_metadata.tsv")

    # --- Step 2: Compute affinity via modisco's two-stage approach ---
    # Run coarse + fine similarity separately for pos (sign=1) and neg
    # (sign=-1) seqlets, matching how modisco processes them internally.
    # Then assemble a combined distance matrix.

    def _compute_affinity(seqlets, sign, label):
        """Run modisco's 3-stage affinity pipeline on one sign group.

        Returns (density_affmat_sparse, distance_matrix_dense).
        """
        n = len(seqlets)
        k = min(n_neighbors, n - 1)
        print(f"  [{label}] Stage 1: Coarse cosine similarity "
              f"({n} seqlets, k={k})...")
        _, seqlet_nbrs = cosine_similarity_from_seqlets(
            seqlets=seqlets, n_neighbors=k, sign=sign)

        print(f"  [{label}] Stage 2: Fine Jaccard "
              f"(min_overlap={min_overlap})...")
        fine_affmat = jaccard_from_seqlets(
            seqlets=seqlets, min_overlap=min_overlap,
            seqlet_neighbors=seqlet_nbrs)

        print(f"  [{label}] Stage 3: Density adaptation "
              f"(perplexity={perplexity})...")
        density_aff = _density_adaptation(
            fine_affmat, seqlet_nbrs, tsne_perplexity=perplexity)
        return density_aff, _affinity_to_distance(density_aff)

    # Build block-diagonal distance matrix: pos-pos and neg-neg blocks
    # use modisco similarity; cross-block (pos-neg) gets max distance.
    dist_mat = np.zeros((N, N), dtype=np.float64)

    def _affinity_to_distance(sparse_aff):
        """Convert sparse affinity → dense symmetric distance."""
        dense = sparse_aff.toarray()
        dense = (dense + dense.T) / 2
        d = np.zeros_like(dense)
        nz = dense > 0
        d[nz] = -np.log(dense[nz])
        d[~nz] = d[nz].max() if nz.any() else 1.0
        np.fill_diagonal(d, 0.0)
        return d

    pos_sparse_aff = None
    neg_sparse_aff = None

    if n_pos > 1:
        pos_sparse_aff, pos_dist = _compute_affinity(
            pos_seqlets, sign=1, label="pos")
        dist_mat[:n_pos, :n_pos] = pos_dist

    if n_neg > 1:
        neg_sparse_aff, neg_dist = _compute_affinity(
            neg_seqlets, sign=-1, label="neg")
        dist_mat[n_pos:, n_pos:] = neg_dist

    # Cross-block: pos↔neg get max distance (they were discovered with
    # opposite signs and should not be considered similar)
    max_dist = dist_mat.max() if dist_mat.max() > 0 else 1.0
    dist_mat[:n_pos, n_pos:] = max_dist
    dist_mat[n_pos:, :n_pos] = max_dist

    dist_mat = dist_mat.astype(np.float32)
    np.save(os.path.join(out_dir, "distance_matrix.npy"), dist_mat)

    # --- Cell class mapping ---
    CT_TO_CLASS = {}
    for ct in ("L23IT", "L2IT", "L34IT", "L35IT", "L4IT", "L56IT", "L56NP",
               "L5ET", "L5IT", "L6B", "L6CT", "L6IT-1", "L6IT-2", "URL"):
        CT_TO_CLASS[ct] = "Excitatory"
    for ct in ("ACBGM", "CBGA", "LAMP5", "LAMP5-LHX6", "MSN", "PAX6",
               "PV-CHC", "PVALB", "SNCG", "SST", "SST-CHODL", "VIP"):
        CT_TO_CLASS[ct] = "Inhibitory"
    for ct in ("AST", "ENDO", "FBL", "IMMUNE", "MGC", "OGC", "OPC"):
        CT_TO_CLASS[ct] = "Non-neuron"

    cell_classes = np.array([CT_TO_CLASS.get(ct, "Unknown") for ct in celltypes])
    unique_classes = sorted(set(cell_classes))
    CLASS_COLORS = {
        "Excitatory": "#E64B35",
        "Inhibitory": "#4DBBD5",
        "Non-neuron": "#00A087",
        "Unknown":    "#999999",
    }

    # --- Shared color maps ---
    unique_cts = sorted(set(celltypes))
    n_cts = len(unique_cts)
    if n_cts > 20:
        cmap1 = plt.cm.get_cmap("tab20", 20)
        cmap2 = plt.cm.get_cmap("tab20b", 20)
        ct_colors = {ct: (cmap1(i % 20) if i < 20 else cmap2(i % 20))
                     for i, ct in enumerate(unique_cts)}
    else:
        cmap = plt.cm.get_cmap("tab20", max(n_cts, 20))
        ct_colors = {ct: cmap(i) for i, ct in enumerate(unique_cts)}

    unique_pats = sorted(set(patterns))
    n_pats = len(unique_pats)
    pat_cmap = plt.cm.get_cmap("hsv", max(n_pats, 10))
    pat_colors = {p: pat_cmap(i / max(n_pats, 1)) for i, p in enumerate(unique_pats)}

    # Write metadata with class column
    with open(_metadata_path, "w") as f:
        f.write("celltype\tcell_class\tpattern\tsign\n")
        for i in range(N):
            sign_label = "pos" if i < n_pos else "neg"
            f.write(f"{celltypes[i]}\t{cell_classes[i]}\t"
                    f"{patterns[i]}\t{sign_label}\n")

    # --- Step 4: UMAP and/or t-SNE on precomputed distance ---
    methods = ["umap", "tsne"] if method == "both" else [method]

    for m in methods:
        print(f"Computing {m.upper()} embedding...")
        if m == "umap":
            import umap
            reducer = umap.UMAP(n_neighbors=min(30, N - 1), min_dist=0.3,
                                metric="precomputed", random_state=seed)
            embedding = reducer.fit_transform(dist_mat)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(30, N // 4),
                           metric="precomputed", random_state=seed,
                           init="random")
            embedding = reducer.fit_transform(dist_mat)

        np.save(os.path.join(out_dir, f"{m}_embedding.npy"), embedding)

        # Plot colored by cell type
        print(f"  Plotting {m} by cell type...")
        fig, ax = plt.subplots(figsize=(12, 10))
        for ct in unique_cts:
            mask = celltypes == ct
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[ct_colors[ct]], label=ct, s=3, alpha=0.5, rasterized=True)
        ax.legend(fontsize=6, ncol=max(1, (n_cts + 14) // 15), markerscale=4,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xlabel(f"{m.upper()} 1")
        ax.set_ylabel(f"{m.upper()} 2")
        ax.set_title(f"Seqlets across {n_cts} cell types ({N:,} seqlets)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_by_celltype.pdf"),
                    dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{m}_by_celltype.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plot colored by pattern
        print(f"  Plotting {m} by pattern...")
        fig, ax = plt.subplots(figsize=(12, 10))
        for pat in unique_pats:
            mask = patterns == pat
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[pat_colors[pat]], label=pat, s=3, alpha=0.5, rasterized=True)
        if n_pats <= 30:
            ax.legend(fontsize=5, ncol=max(1, (n_pats + 9) // 10), markerscale=4,
                      loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xlabel(f"{m.upper()} 1")
        ax.set_ylabel(f"{m.upper()} 2")
        ax.set_title(f"Seqlets colored by pattern ({n_pats} patterns)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_by_pattern.pdf"),
                    dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{m}_by_pattern.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plot colored by cell class
        print(f"  Plotting {m} by cell class...")
        fig, ax = plt.subplots(figsize=(10, 8))
        for cls in unique_classes:
            mask = cell_classes == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=CLASS_COLORS.get(cls, "#999999"), label=cls,
                       s=3, alpha=0.4, rasterized=True)
        ax.legend(fontsize=10, markerscale=6,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xlabel(f"{m.upper()} 1")
        ax.set_ylabel(f"{m.upper()} 2")
        ax.set_title(f"Seqlets by cell class ({N:,} seqlets)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_by_class.pdf"),
                    dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{m}_by_class.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    if skip_leiden:
        print(f"Outputs in {out_dir}/")
        print("Done (--skip_leiden).")
        return

    # --- Step 5: Leiden clustering on density-adapted affinity ---
    # Cluster pos and neg seqlets separately (matching modisco's pipeline),
    # then combine into a single label array.
    print("Leiden clustering...")
    cluster_names = np.empty(N, dtype=object)

    if n_pos > 1 and pos_sparse_aff is not None:
        pos_labels = LeidenCluster(pos_sparse_aff,
                                   n_seeds=n_leiden_runs)
        for i, lab in enumerate(pos_labels):
            cluster_names[i] = f"pos_cluster_{lab}"
    elif n_pos == 1:
        cluster_names[0] = "pos_cluster_0"

    if n_neg > 1 and neg_sparse_aff is not None:
        neg_labels = LeidenCluster(neg_sparse_aff,
                                   n_seeds=n_leiden_runs)
        for i, lab in enumerate(neg_labels):
            cluster_names[n_pos + i] = f"neg_cluster_{lab}"
    elif n_neg == 1:
        cluster_names[n_pos] = "neg_cluster_0"

    unique_clusters = sorted(set(cluster_names))
    print(f"  Found {len(unique_clusters)} clusters")

    # --- Step 6: Compute initial CWMs per cluster ---
    print("Computing CWMs per cluster...")

    def _compute_cluster_cwms(cluster_names, unique_clusters):
        """Compute CWM and PFM for each cluster."""
        cwms = {}   # cname -> [L, 4]
        pfms = {}   # cname -> [L, 4]
        for cname in unique_clusters:
            mask = cluster_names == cname
            n_in = mask.sum()
            if n_in < min_cluster_size:
                continue
            indices = np.where(mask)[0]
            contrib_sum = np.zeros_like(all_seqlets[indices[0]].contrib_scores)
            seq_sum = np.zeros_like(all_seqlets[indices[0]].sequence)
            for idx in indices:
                contrib_sum += all_seqlets[idx].contrib_scores
                seq_sum += all_seqlets[idx].sequence
            cwms[cname] = contrib_sum / n_in
            pfms[cname] = seq_sum / n_in
        return cwms, pfms

    cwms, pfms = _compute_cluster_cwms(cluster_names, unique_clusters)
    print(f"  {len(cwms)} clusters pass min_cluster_size={min_cluster_size}")

    # --- Step 6b: Merge similar clusters by CWM Pearson correlation ---
    if merge_threshold < 1.0 and len(cwms) > 1:
        print(f"Merging similar clusters (Pearson r >= {merge_threshold})...")
        from modiscolite.affinitymat import pearson_correlation

        def _cwm_pearson(cwm_a, cwm_b, min_ovlp):
            """Best Pearson correlation between two CWMs with sliding
            window alignment, considering both orientations."""
            # pearson_correlation expects [1, L, 4] arrays
            a = cwm_a[None, :, :]
            b = cwm_b[None, :, :]
            # Forward
            fwd = pearson_correlation(a, b, min_overlap=min_ovlp)
            best_fwd = fwd[0, :, 0].max()
            # Reverse complement
            b_rc = cwm_b[::-1, ::-1][None, :, :]
            rev = pearson_correlation(a, b_rc, min_overlap=min_ovlp)
            best_rev = rev[0, :, 0].max()
            return max(best_fwd, best_rev)

        # Compute pairwise similarity between cluster CWMs
        kept_names = sorted(cwms.keys())
        n_cl = len(kept_names)
        sim_mat = np.zeros((n_cl, n_cl))
        for i in range(n_cl):
            sim_mat[i, i] = 1.0
            for j in range(i + 1, n_cl):
                s = _cwm_pearson(cwms[kept_names[i]], cwms[kept_names[j]],
                                 min_ovlp=min_overlap)
                sim_mat[i, j] = s
                sim_mat[j, i] = s

        # Average-linkage hierarchical merge: only merge two groups
        # if the mean pairwise correlation between all their members
        # exceeds the threshold.  This prevents single-linkage chaining.
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        dist_condensed = squareform(1.0 - sim_mat, checks=False)
        Z = linkage(dist_condensed, method="average")
        labels = fcluster(Z, t=1.0 - merge_threshold,
                          criterion="distance")
        # labels: 1-indexed cluster IDs

        root_to_members = defaultdict(list)
        for i, lab in enumerate(labels):
            root_to_members[lab].append(kept_names[i])
        n_merges = n_cl - len(root_to_members)

        merge_map = {}
        for members in root_to_members.values():
            # Pick the member with most seqlets as canonical
            canonical = max(members,
                            key=lambda c: (cluster_names == c).sum())
            for m in members:
                merge_map[m] = canonical

        if n_merges > 0:
            # Relabel seqlets
            for i in range(N):
                old = cluster_names[i]
                if old in merge_map:
                    cluster_names[i] = merge_map[old]

            unique_clusters = sorted(set(cluster_names))
            # Recompute CWMs after merge
            cwms, pfms = _compute_cluster_cwms(cluster_names, unique_clusters)
            print(f"  Merged {n_merges} pairs -> {len(cwms)} clusters")
        else:
            print("  No clusters to merge")

    # --- Save CWMs and build MEME motifs ---
    cwm_dir = os.path.join(out_dir, "cwms")
    os.makedirs(cwm_dir, exist_ok=True)
    motif_pwms = {}

    cluster_report_rows = []
    for cname in sorted(cwms.keys()):
        cwm = cwms[cname]
        pfm = pfms[cname]
        mask = cluster_names == cname
        n_in_cluster = mask.sum()

        np.save(os.path.join(cwm_dir, f"{cname}_cwm.npy"), cwm)
        np.save(os.path.join(cwm_dir, f"{cname}_pfm.npy"), pfm)

        clipped, _, _ = ic_clip_pwm(pfm, threshold=0.3)
        if clipped.shape[0] >= 3:
            motif_pwms[cname] = clipped

        indices = np.where(mask)[0]
        ct_counts = defaultdict(int)
        cls_counts = defaultdict(int)
        for idx in indices:
            ct_counts[celltypes[idx]] += 1
            cls_counts[cell_classes[idx]] += 1
        ct_summary = ", ".join(f"{ct}:{n}" for ct, n in
                               sorted(ct_counts.items(), key=lambda x: -x[1]))
        cls_summary = ", ".join(f"{cls}:{n}" for cls, n in
                                sorted(cls_counts.items(), key=lambda x: -x[1]))
        cluster_report_rows.append((cname, n_in_cluster, cls_summary, ct_summary))

    print(f"  {len(motif_pwms)} clusters with MEME motifs")

    # --- Write cluster report ---
    report_path = os.path.join(out_dir, "cluster_report.tsv")
    with open(report_path, "w") as f:
        f.write("cluster\tn_seqlets\tclass_composition\tcelltype_composition\n")
        for row in cluster_report_rows:
            f.write("\t".join(str(x) for x in row) + "\n")

    # Write per-seqlet cluster assignments
    assign_path = os.path.join(out_dir, "seqlet_clusters.tsv")
    with open(assign_path, "w") as f:
        f.write("seqlet_idx\tcelltype\tcell_class\tpattern\tsign\tcluster\n")
        for i in range(N):
            sign_label = "pos" if i < n_pos else "neg"
            f.write(f"{i}\t{celltypes[i]}\t{cell_classes[i]}\t"
                    f"{patterns[i]}\t{sign_label}\t{cluster_names[i]}\n")

    # --- Step 7: Write MEME and run TOMTOM ---
    meme_path = os.path.join(out_dir, "motifs.meme")
    if motif_pwms:
        print("Writing MEME file...")
        with open(meme_path, "w") as f:
            write_meme_header(f)
            for name, pwm in sorted(motif_pwms.items()):
                write_meme_motif(f, name, pwm)

        meme_db_abs = os.path.abspath(meme_db)
        tomtom_dir = os.path.join(out_dir, "tomtom")
        if os.path.exists(meme_db_abs):
            print("Running TOMTOM...")
            cmd = [
                "tomtom", "-dist", "pearson", "-evalue", "-thresh", "1.0",
                "-oc", tomtom_dir, meme_path, meme_db_abs
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  TOMTOM failed: {result.stderr[:300]}")
            else:
                print(f"  TOMTOM results in {tomtom_dir}/")
        else:
            print(f"  MEME database not found: {meme_db_abs}, skipping TOMTOM")

    # --- Step 8: Plot by cluster ---
    n_clusters = len(unique_clusters)
    cl_cmap = plt.cm.get_cmap("tab20", max(n_clusters, 20))
    cl_colors = {c: cl_cmap(i % 20) for i, c in enumerate(unique_clusters)}

    for m in methods:
        emb_path = os.path.join(out_dir, f"{m}_embedding.npy")
        if not os.path.exists(emb_path):
            continue
        embedding = np.load(emb_path)

        print(f"  Plotting {m} by cluster...")
        fig, ax = plt.subplots(figsize=(12, 10))
        for cname in unique_clusters:
            mask = cluster_names == cname
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[cl_colors[cname]], label=cname,
                       s=3, alpha=0.5, rasterized=True)
        if n_clusters <= 30:
            ax.legend(fontsize=5, ncol=max(1, (n_clusters + 9) // 10),
                      markerscale=4,
                      loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xlabel(f"{m.upper()} 1")
        ax.set_ylabel(f"{m.upper()} 2")
        ax.set_title(f"Seqlets by cluster ({n_clusters} clusters, {N:,} seqlets)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_by_cluster.pdf"),
                    dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{m}_by_cluster.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Outputs in {out_dir}/")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """TF-MoDISco pipeline for MiniAtlas peaks gradient x input."""
    pass


cli.add_command(cmd_prepare, "prepare")
cli.add_command(cmd_modisco, "modisco")
cli.add_command(cmd_postprocess, "postprocess")
cli.add_command(cmd_aggregate, "aggregate")


if __name__ == "__main__":
    cli()
