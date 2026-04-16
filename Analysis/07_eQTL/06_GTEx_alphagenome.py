# %%
import pandas as pd
import numpy as np
import os

FEATHER_FILE = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Analysis/07_eQTL_new/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather'
TSS_FILE = '/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v39/gencode.v39.hg38.all.sga'
OUT_DIR = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/source/eQTL/alphagenome'
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# load GENCODE V39: median TSS per gene (matches 06_GTEx_ag_like.get_median_tss_distance)
tss_raw = pd.read_csv(TSS_FILE, sep='\t', skipinitialspace=True, header=None)
tss_raw.columns = ['chr', 'type', 'pos', 'strand', 'tag_counts', 'info']
tss_raw['transcript'] = tss_raw['info'].apply(lambda x: x.split('..')[0])
tss_raw['gene'] = tss_raw['info'].apply(lambda x: x.split('..')[1])
tss_raw['gene_ID'] = tss_raw['info'].apply(lambda x: x.split('..')[2])

tss_median_by_name = (
    tss_raw.groupby('gene')
    .agg(median_tss=('pos', 'median'), gene_ID=('gene_ID', 'first'))
)
tss_median_by_id = (
    tss_raw.groupby('gene_ID')
    .agg(median_tss=('pos', 'median'), gene=('gene', 'first'))
)
# base ENSG (strip version) for robust AlphaGenome feather matching
tss_median_by_id_base = tss_median_by_id.copy()
tss_median_by_id_base.index = tss_median_by_id_base.index.str.split('.').str[0]
tss_median_by_id_base = tss_median_by_id_base[~tss_median_by_id_base.index.duplicated(keep='first')]

def get_median_tss_distance(variant_pos, gene_name):
    if gene_name in tss_median_by_name.index:
        row = tss_median_by_name.loc[gene_name]
        return abs(variant_pos - row['median_tss']), row.name, row['gene_ID']
    if gene_name in tss_median_by_id.index:
        row = tss_median_by_id.loc[gene_name]
        return abs(variant_pos - row['median_tss']), row['gene'], row.name
    base = gene_name.split('.')[0] if isinstance(gene_name, str) else gene_name
    if base in tss_median_by_id_base.index:
        row = tss_median_by_id_base.loc[base]
        return abs(variant_pos - row['median_tss']), row['gene'], gene_name
    return np.nan, np.nan, np.nan

# %%
# load AlphaGenome eQTL catalogue predictions
df = pd.read_feather(FEATHER_FILE)
print(f"Loaded feather: {df.shape}")

# keep only labeled rows (positive/negative)
df = df[df['target'].notna()].copy()
print(f"Labeled rows: {len(df)}")

# parse variant_id: chrN_POS_REF_ALT
parts = df['variant_id'].str.split('_', expand=True)
df['chr'] = parts[0]
df['pos'] = parts[1].astype(int)
df['ref'] = parts[2]
df['alt'] = parts[3]

# filter to SNPs only
df = df[(df['ref'].str.len() == 1) & (df['alt'].str.len() == 1)].copy()
print(f"SNPs: {len(df)}")

# binary label
df['label'] = np.where(df['target'] >= 0.5, 'positive', 'negative')

# dedupe on (variant, gene, tissue)
df = df.drop_duplicates(subset=['variant_id', 'gene_id', 'tissue']).reset_index(drop=True)

# distance to median TSS of the target gene (per variant+gene, merged back by tissue)
print("Computing median TSS distance...")
dist_keys = df[['gene_id', 'pos']].drop_duplicates()
results = dist_keys.apply(
    lambda row: get_median_tss_distance(row['pos'], row['gene_id']), axis=1
)
dist_keys['distance'] = results.apply(lambda x: x[0])
dist_keys['gene_name'] = results.apply(lambda x: x[1])
dist_keys['gene_ID'] = results.apply(lambda x: x[2])
df = df.merge(dist_keys, on=['gene_id', 'pos'], how='left')
df = df.dropna(subset=['distance']).reset_index(drop=True)

pos_n = (df['label'] == 'positive').sum()
neg_n = (df['label'] == 'negative').sum()
print(f"Total: {pos_n} positive, {neg_n} negative, {len(df)} variant/gene/tissue tuples")

# %%
def make_vcf(d: pd.DataFrame) -> pd.DataFrame:
    vcf = d[['chr', 'pos', 'variant_id', 'ref', 'alt']].copy()
    vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
    vcf['QUAL'] = '.'
    vcf['FILTER'] = '.'
    vcf['INFO'] = d.apply(
        lambda row: f"label={row['label']};gene_name={row['gene_name']};gene_ID={row['gene_ID']};tissue={row['tissue']};distance={row['distance']:.0f}",
        axis=1
    ).values
    return vcf

# %%
# write all-tissue VCF + info CSV
make_vcf(df).to_csv(f'{OUT_DIR}/ag.all.vcf', sep='\t', index=False, header=True)
df.to_csv(f'{OUT_DIR}/ag.all.info.csv', index=False)

# write brain-tissue VCF + info CSV
brain = df[df['tissue'].str.lower().str.startswith('brain')].copy()
make_vcf(brain).to_csv(f'{OUT_DIR}/ag.brain.vcf', sep='\t', index=False, header=True)
brain.to_csv(f'{OUT_DIR}/ag.brain.info.csv', index=False)
print(f"Brain: {(brain['label']=='positive').sum()} positive, {(brain['label']=='negative').sum()} negative")
for t, grp in brain.groupby('tissue'):
    pos_n = (grp['label'] == 'positive').sum()
    neg_n = (grp['label'] == 'negative').sum()
    print(f"  {t}: positive={pos_n}, negative={neg_n}")
# %%
