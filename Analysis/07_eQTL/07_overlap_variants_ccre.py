#!/usr/bin/env python
"""Build union cCRE BEDs per organ (Cortex / Basal_ganglia) and flag variants
from ag_like.brain.vcf that overlap any cCRE.

Outputs (under output.ag_like/):
  - ccre_union_Cortex.bed
  - ccre_union_Basal_ganglia.bed
  - variants_in_ccre_Cortex.tsv         (single column: variant_id)
  - variants_in_ccre_Basal_ganglia.tsv
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

import pandas as pd

PWD = f'{os.environ["workingHOME"]}/BICAN'
VCF = os.path.join(PWD, 'Data/source/eQTL/alphagenome/ag_like.brain.vcf')

CORTEX_PEAK_DIR = os.path.join(PWD, 'Data/source/MiniAtlas_ATAC_peak')
BG_SUBCLASS_TXT = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/BICAN/ATAC-seq/Subclass.peaks/Subclass.peaks.txt'

OUT_DIR = os.path.join(PWD, 'Analysis/07_eQTL/output.ag_like')


def _run(cmd: list[str]) -> None:
    print('[run]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_cortex_union(out_bed: str) -> None:
    beds = [p for p in sorted(glob.glob(os.path.join(CORTEX_PEAK_DIR, '*.bed')))
            if os.path.basename(p) != 'test.bed']
    print(f'[Cortex] merging {len(beds)} subclass BEDs')
    with tempfile.NamedTemporaryFile('w', suffix='.bed', delete=False) as tmp:
        for bed in beds:
            with open(bed) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    tmp.write(f'{parts[0]}\t{parts[1]}\t{parts[2]}\n')
        tmp_path = tmp.name
    try:
        sorted_bed = out_bed + '.sorted'
        _run(['sort', '-k1,1', '-k2,2n', tmp_path, '-o', sorted_bed])
        with open(out_bed, 'w') as out:
            subprocess.run(['bedtools', 'merge', '-i', sorted_bed], check=True, stdout=out)
        os.remove(sorted_bed)
    finally:
        os.remove(tmp_path)
    n = sum(1 for _ in open(out_bed))
    print(f'[Cortex] union BED: {out_bed} ({n} intervals)')


def build_basal_ganglia_union(out_bed: str) -> None:
    print(f'[Basal_ganglia] parsing {BG_SUBCLASS_TXT}')
    peaks = pd.read_csv(BG_SUBCLASS_TXT, sep='\t', usecols=['Peaks'])['Peaks']
    rows = []
    for peak in peaks:
        chrom, rng = peak.split(':')
        start, end = rng.split('-')
        rows.append((chrom, int(start), int(end)))
    df = pd.DataFrame(rows, columns=['chr', 'start', 'end'])
    df = df.sort_values(['chr', 'start', 'end']).reset_index(drop=True)
    raw = out_bed + '.raw'
    df.to_csv(raw, sep='\t', header=False, index=False)
    try:
        with open(out_bed, 'w') as out:
            subprocess.run(['bedtools', 'merge', '-i', raw], check=True, stdout=out)
    finally:
        os.remove(raw)
    n = sum(1 for _ in open(out_bed))
    print(f'[Basal_ganglia] union BED: {out_bed} ({n} intervals)')


def vcf_to_bed(vcf_path: str, out_bed: str) -> int:
    n = 0
    with open(vcf_path) as f, open(out_bed, 'w') as out:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            chrom, pos, vid = parts[0], int(parts[1]), parts[2]
            out.write(f'{chrom}\t{pos - 1}\t{pos}\t{vid}\n')
            n += 1
    return n


def intersect_variants(variant_bed: str, ccre_bed: str, out_tsv: str) -> None:
    cmd = ['bedtools', 'intersect', '-u', '-a', variant_bed, '-b', ccre_bed]
    print('[run]', ' '.join(cmd), '>', out_tsv, flush=True)
    with open(out_tsv, 'w') as out:
        out.write('variant_id\n')
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        hits = 0
        for line in res.stdout.splitlines():
            vid = line.split('\t')[3]
            out.write(vid + '\n')
            hits += 1
    print(f'  -> {hits} variants in cCRE')


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    variant_bed = os.path.join(OUT_DIR, '_variants.bed')
    n = vcf_to_bed(VCF, variant_bed)
    print(f'Variants BED: {variant_bed} ({n} rows)')

    cortex_bed = os.path.join(OUT_DIR, 'ccre_union_Cortex.bed')
    bg_bed = os.path.join(OUT_DIR, 'ccre_union_Basal_ganglia.bed')
    build_cortex_union(cortex_bed)
    build_basal_ganglia_union(bg_bed)

    intersect_variants(variant_bed, cortex_bed,
                       os.path.join(OUT_DIR, 'variants_in_ccre_Cortex.tsv'))
    intersect_variants(variant_bed, bg_bed,
                       os.path.join(OUT_DIR, 'variants_in_ccre_Basal_ganglia.tsv'))

    os.remove(variant_bed)


if __name__ == '__main__':
    sys.exit(main())
