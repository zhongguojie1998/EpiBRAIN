# 
import pandas as pd
import numpy as np
import os
# 
all_tissue = pd.DataFrame()
# get names of all tissues
getx_files = os.listdir('/gpfs/commons/groups/ren_lab/guojiezhong/Data/GTEx/v10/SuSiE/')
tissues = [f.split('/')[-1].replace('.v10.eQTLs.SuSiE_summary.parquet', '') for f in getx_files if f.endswith('.v10.eQTLs.SuSiE_summary.parquet')]
for tissue in tissues:
    print(tissue)
    gtex = pd.read_parquet(f'/gpfs/commons/groups/ren_lab/guojiezhong/Data/GTEx/v10/SuSiE/{tissue}.v10.eQTLs.SuSiE_summary.parquet')
    gtex['chr'] = gtex['variant_id'].apply(lambda x: x.split('_')[0])
    gtex['pos'] = gtex['variant_id'].apply(lambda x: int(x.split('_')[1]))
    gtex['ref'] = gtex['variant_id'].apply(lambda x: x.split('_')[2])
    gtex['alt'] = gtex['variant_id'].apply(lambda x: x.split('_')[3])
    gtex['build'] = gtex['variant_id'].apply(lambda x: x.split('_')[4])
    # filter to only contain snp
    gtex = gtex[(gtex['ref'].str.len() == 1) & (gtex['alt'].str.len() == 1)]
    #  read tss file
    tss = pd.read_csv('/gpfs/commons/groups/ren_lab/guojiezhong/Data/GENCODE/v48/gencode.v48.hg38.sga', sep='\t', skipinitialspace=True, header=None)
    tss.columns = ['chr', 'type', 'pos', 'strand', 'tag_counts', 'info']
    tss['transcript'] = tss['info'].apply(lambda x: x.split('..')[0])
    tss['gene'] = tss['info'].apply(lambda x: x.split('..')[1])
    #  get closest distance to tss
    def get_closest_tss(row):
        chr = row['chr']
        pos = row['pos']
        tss_chr = tss[tss['chr'] == chr]
        if tss_chr.empty:
            return np.nan, np.nan, np.nan, np.nan
        tss_chr['distance'] = abs(tss_chr['pos'] - pos)
        closest_tss = tss_chr.loc[tss_chr['distance'].idxmin()]
        return closest_tss['transcript'], closest_tss['gene'], closest_tss['distance'], closest_tss['strand']
    gtex[['closest_transcript', 'closest_gene', 'closest_distance', 'closest_strand']] = gtex.apply(get_closest_tss, axis=1, result_type='expand')
    #  filter to PIP ≥ 0.9
    positive = gtex[gtex['pip'] >= 0.9].copy()
    negative = gtex[gtex['pip'] < 0.0001].copy()
    if not positive.empty:
        positive.loc[positive.index, 'label'] = 'positive'
    if not negative.empty:
        negative.loc[negative.index, 'label'] = 'negative'
    all = pd.concat([positive, negative], axis=0, ignore_index=True)
    all['group'] = '>35k'
    for dist_cutoff, name in zip([35000, 12000, 3000], ['12k-35k', '3k-12k', '<3k']):
        all.loc[all.index[all['closest_distance'] <= dist_cutoff], 'group'] = name
    #  transform to vcf file
    os.makedirs(f'../Data/source/eQTL/{tissue}', exist_ok=True)
    for group in all['group'].unique().tolist() + ['all']:
        if group == 'all':
            all_group = all.copy()
        else:
            all_group = all[all['group'] == group].copy()
        vcf = all_group[['chr', 'pos', 'variant_id', 'ref', 'alt']].copy()
        vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
        vcf['QUAL'] = '.'
        vcf['FILTER'] = '.'
        vcf['INFO'] = all_group['label'].copy()
        vcf.to_csv(f'../Data/source/eQTL/{tissue}/{group.replace('>', '_').replace('<', '_').replace('-', '_')}.vcf', sep='\t', index=False, header=True)
    all.to_csv(f'../Data/source/eQTL/{tissue}/info.csv', index=False)
    all['tissue'] = tissue
    all_tissue = pd.concat([all_tissue, all], axis=0, ignore_index=True)
for group in all_tissue['group'].unique().tolist() + ['all']:
        if group == 'all':
            all_group = all_tissue.copy()
        else:
            all_group = all_tissue[all_tissue['group'] == group].copy()
        vcf = all_group[['chr', 'pos', 'variant_id', 'ref', 'alt']].copy()
        vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT']
        vcf['QUAL'] = '.'
        vcf['FILTER'] = '.'
        vcf['INFO'] = all_group['label'].copy()
        vcf.to_csv(f'../Data/source/eQTL/{group.replace('>', '_').replace('<', '_').replace('-', '_')}.vcf', sep='\t', index=False, header=True)
all_tissue.to_csv(f'../Data/source/eQTL/info.csv', index=False)