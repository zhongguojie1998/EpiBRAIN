# %%
import h5py
import numpy as np
import pandas as pd
import sys
import os
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/Analysis/03_variant_effect_screen')
from get_full_var_info import query_vcf_with_rsidx
# %%
# read file
res = h5py.File('/nfs/user/Users/dl3738/BICAN/data/source/GWAS_Var/res_file.h5', mode='r')
# %%
tracks = res.attrs["trial_names"]
exp_index = res["experiments/schizophrenia/index_key"][:]
exp_zscore = res["experiments/schizophrenia/z_score"][:]
exp_reverse_map = res["experiments/schizophrenia/reverse_map"][:]
all_variant_index = res["variants/index_key"][:]
all_variant_to_pos = {val: idx for idx, val in enumerate(all_variant_index)}
positions = [all_variant_to_pos[val] for val in exp_index]
# %%
exp_var_score = res["results/variant_effects"][positions, :]
final_score = exp_var_score * (1 - 2 * exp_reverse_map).reshape(-1, 1)
variant_effects = pd.DataFrame(final_score, columns=tracks)
# %%
variants = pd.DataFrame({'CHR': res['variants/chr'],
                         'BP': res['variants/pos'],
                         'SNP': res['variants/rsid'], 
                         'A1': res['variants/ref'],
                         'A2': res['variants/alt'],})
# %%
# use the file "/share/vault/Users/gz2294/BICAN_Data/GWAStraits/hapmap3.snp.bed" to convert to hg19 position
rsid_hg19 = pd.read_csv('/share/vault/Users/gz2294/BICAN_Data/GWAStraits/hapmap3.snp.bed',
                        sep='\t', header=None, names=['CHR', 'start', 'end', 'SNP'])
# %%
rsid_hg19.index = rsid_hg19['SNP'].values
variants['SNP'] = variants['SNP'].astype(str)
variants['A1'] = variants['A1'].astype(str)
variants['A2'] = variants['A2'].astype(str)
variants.index = variants['SNP'].values
# remove the chr prefix and convert to int
variants['CHR'] = variants['CHR'].astype(str).str.replace('chr', '').astype(int)
# %%
# get the chrom positions in hg19
variants_hg19 = query_vcf_with_rsidx(variants['SNP'].values,
                                     '../Data/Ref/hg19/dbSNP/GCF_000001405.40.gz',
                                     '../Data/Ref/hg19/dbSNP/GCF_000001405.40.rsidx',
                                     n_processes=10)
# save to pickle file
import pickle
with open('sldp.variants.hg19.pkl', 'wb') as f:
    pickle.dump(variants_hg19, f)
# %%
