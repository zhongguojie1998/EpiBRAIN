# %%
from datasets import load_dataset
import os
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')

dataset1 = load_dataset("songlab/TraitGym", "complex_traits", split="test")
dataset2 = load_dataset("songlab/TraitGym", "mendelian_traits", split="test")

# %% first save to csv files
dataset1.to_csv("Data/source/TraitGym/complex_traits_test.csv", index=False)
dataset2.to_csv("Data/source/TraitGym/mendelian_traits_test.csv", index=False)
# %% next read them in and do some preprocessing
import pandas as pd
import numpy as np
dataset1 = pd.read_csv("Data/source/TraitGym/complex_traits_test.csv")
dataset2 = pd.read_csv("Data/source/TraitGym/mendelian_traits_test.csv")
# %% 
dataset1['chrom'] = 'chr' + dataset1['chrom'].astype(str)
dataset2['chrom'] = 'chr' + dataset2['chrom'].astype(str)
dataset1['ID'] = dataset1['chrom'].astype(str) + '_' + dataset1['pos'].astype(str) + '_' + dataset1['ref'] + '_' + dataset1['alt']
dataset2['ID'] = dataset2['chrom'].astype(str) + '_' + dataset2['pos'].astype(str) + '_' + dataset2['ref'] + '_' + dataset2['alt']
dataset1['ID'] = dataset1['ID'].astype(str)
dataset2['ID'] = dataset2['ID'].astype(str)
# %% reorder columns
dataset1 = dataset1.iloc[:, [0, 1, 12, 2, 3] + list(range(4, 12))]
dataset2 = dataset2.iloc[:, [0, 1, 9, 2, 3] + list(range(4, 9))]
# %% rename columns 0:5
dataset1.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT'] + list(dataset1.columns[5:])
dataset2.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT'] + list(dataset2.columns[5:])
# %% save again
dataset1.to_csv("Data/source/TraitGym/complex_traits_test.vcf", index=False, sep='\t')
dataset2.to_csv("Data/source/TraitGym/mendelian_traits_test.vcf", index=False, sep='\t')
dataset1.shape, dataset2.shape
# %% drop duplicates
dataset1 = dataset1.drop_duplicates(subset=['ID'])
dataset2 = dataset2.drop_duplicates(subset=['ID'])
dataset1.shape, dataset2.shape
# %% save again
dataset1.to_csv("Data/source/TraitGym/complex_traits_test.vcf", index=False, sep='\t')
dataset2.to_csv("Data/source/TraitGym/mendelian_traits_test.vcf", index=False, sep='\t')
