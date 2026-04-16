# %%
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

GTEX_DIR = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/GTEx/v10/SuSiE/'
TSS_FILE = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v39/gencode.v39.hg38.all.sga'
OUT_DIR = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/source/eQTL/alphagenome'
os.makedirs(OUT_DIR, exist_ok=True)

NEG_PIP_CUTOFF = 0.01
POS_PIP_CUTOFF = 0.9
N_BINS = 10
RANDOM_SEED = 42

# %%
# load GENCODE V39: compute median TSS per gene across all transcripts
tss_raw = pd.read_csv(TSS_FILE, sep='\t', skipinitialspace=True, header=None)
tss_raw.columns = ['chr', 'type', 'pos', 'strand', 'tag_counts', 'info']
tss_raw['transcript'] = tss_raw['info'].apply(lambda x: x.split('..')[0])
tss_raw['gene'] = tss_raw['info'].apply(lambda x: x.split('..')[1])
tss_raw['gene_ID'] = tss_raw['info'].apply(lambda x: x.split('..')[2])

# median TSS position per gene (use first gene_ID encountered per gene name)
tss_median = (
    tss_raw.groupby('gene')
    .agg(median_tss=('pos', 'median'), gene_ID=('gene_ID', 'first'))
    .reset_index()
)
tss_median_by_name = tss_median.set_index('gene')
# also index by gene_ID for fallback lookup
tss_median_by_id = (
    tss_raw.groupby('gene_ID')
    .agg(median_tss=('pos', 'median'), gene=('gene', 'first'))
    .reset_index()
    .set_index('gene_ID')
)

# %%
def get_median_tss_distance(variant_pos, gene_name):
    if gene_name in tss_median_by_name.index:
        row = tss_median_by_name.loc[gene_name]
        return abs(variant_pos - row['median_tss']), row.name, row['gene_ID']
    if gene_name in tss_median_by_id.index:
        row = tss_median_by_id.loc[gene_name]
        return abs(variant_pos - row['median_tss']), row['gene'], row.name
    return np.nan, np.nan, np.nan

# %%
gtex_files = os.listdir(GTEX_DIR)
tissues = [f.replace('.v10.eQTLs.SuSiE_summary.parquet', '')
           for f in gtex_files if f.endswith('.v10.eQTLs.SuSiE_summary.parquet')]

def process_tissue(tissue):
    g = pd.read_parquet(f'{GTEX_DIR}/{tissue}.v10.eQTLs.SuSiE_summary.parquet')
    g['chr'] = g['variant_id'].apply(lambda x: x.split('_')[0])
    g['pos'] = g['variant_id'].apply(lambda x: int(x.split('_')[1]))
    g['ref'] = g['variant_id'].apply(lambda x: x.split('_')[2])
    g['alt'] = g['variant_id'].apply(lambda x: x.split('_')[3])
    # filter to SNPs only
    g = g[(g['ref'].str.len() == 1) & (g['alt'].str.len() == 1)]
    # filter to positive (pip >= 0.9) or negative (pip < 0.01)
    g = g[(g['pip'] >= POS_PIP_CUTOFF) | (g['pip'] < NEG_PIP_CUTOFF)].copy()
    if g.empty:
        return None
    # compute distance to median TSS of the target gene
    results = g.apply(
        lambda row: get_median_tss_distance(row['pos'], row['gene_name']), axis=1
    )
    g['distance'] = results.apply(lambda x: x[0])
    g['gene_name'] = results.apply(lambda x: x[1])
    g['gene_ID'] = results.apply(lambda x: x[2])
    g = g.dropna(subset=['distance'])
    g['label'] = np.where(g['pip'] >= POS_PIP_CUTOFF, 'positive', 'negative')
    g['tissue'] = tissue
    pre_pos = (g['label'] == 'positive').sum()
    pre_neg = (g['label'] == 'negative').sum()
    # distance balancing: log10(distance), 10 equal-width bins,
    # within each bin downsample negatives per gene to match positive count
    g['log_distance'] = np.log10(g['distance'].clip(lower=1))
    log_min = g['log_distance'].min()
    log_max = g['log_distance'].max()
    bin_edges = np.linspace(log_min, log_max, N_BINS + 1)
    g['dist_bin'] = pd.cut(g['log_distance'], bins=bin_edges, include_lowest=True, labels=False)
    balanced_parts = []
    for bin_idx in range(N_BINS):
        bin_data = g[g['dist_bin'] == bin_idx]
        pos = bin_data[bin_data['label'] == 'positive']
        neg = bin_data[bin_data['label'] == 'negative']
        if pos.empty:
            continue
        pos_counts = pos.groupby('gene_name').size()
        neg_sampled = []
        sampled_idx = set()
        for gene, n_pos in pos_counts.items():
            neg_gene = neg[neg['gene_name'] == gene]
            if neg_gene.empty:
                continue
            if len(neg_gene) > n_pos:
                neg_gene = neg_gene.sample(n=n_pos, random_state=RANDOM_SEED)
            neg_sampled.append(neg_gene)
            sampled_idx.update(neg_gene.index)
        # if total sampled negatives < total positives, fill from remaining negatives in other genes
        n_neg_sampled = sum(len(s) for s in neg_sampled)
        n_shortfall = len(pos) - n_neg_sampled
        if n_shortfall > 0:
            remaining = neg[~neg.index.isin(sampled_idx)]
            if not remaining.empty:
                extra = remaining.sample(n=min(n_shortfall, len(remaining)), random_state=RANDOM_SEED)
                neg_sampled.append(extra)
        balanced_parts.append(pos)
        if neg_sampled:
            balanced_parts.append(pd.concat(neg_sampled, axis=0))
    if not balanced_parts:
        return None
    g = pd.concat(balanced_parts, axis=0, ignore_index=True)
    pos_n = (g['label'] == 'positive').sum()
    neg_n = (g['label'] == 'negative').sum()
    print(f"  {tissue}: before=({pre_pos} pos, {pre_neg} neg) -> after=({pos_n} pos, {neg_n} neg)")
    return g

results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_tissue)(tissue) for tissue in sorted(tissues)
)
all_rows = [r for r in results if r is not None]

# %%
balanced = pd.concat(all_rows, axis=0, ignore_index=True)
print(f"After balancing: {(balanced['label']=='positive').sum()} positive, {(balanced['label']=='negative').sum()} negative")
print(f"Total: {len(balanced)} variant/gene/tissue tuples")

# %%
def make_vcf(df: pd.DataFrame) -> pd.DataFrame:
    vcf = df[['chr', 'pos', 'variant_id', 'ref', 'alt']].copy()
    vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
    vcf['QUAL'] = '.'
    vcf['FILTER'] = '.'
    vcf['INFO'] = df.apply(
        lambda row: f"label={row['label']};gene_name={row['gene_name']};gene_ID={row['gene_ID']};distance={row['distance']:.0f}",
        axis=1
    ).values
    return vcf

# %%
# write all-tissue balanced VCF and CSV
make_vcf(balanced).to_csv(f'{OUT_DIR}/ag_like.all.vcf', sep='\t', index=False, header=True)
balanced.to_csv(f'{OUT_DIR}/ag_like.all.info.csv', index=False)

# write brain-tissue balanced VCF and CSV
brain = balanced[balanced['tissue'].str.lower().str.startswith('brain')].copy()
make_vcf(brain).to_csv(f'{OUT_DIR}/ag_like.brain.vcf', sep='\t', index=False, header=True)
brain.to_csv(f'{OUT_DIR}/ag_like.brain.info.csv', index=False)
print(f"Brain: {(brain['label']=='positive').sum()} positive, {(brain['label']=='negative').sum()} negative")
for t, grp in brain.groupby('tissue'):
    pos_n = (grp['label'] == 'positive').sum()
    neg_n = (grp['label'] == 'negative').sum()
    print(f"  {t}: positive={pos_n}, negative={neg_n}")
# %%
