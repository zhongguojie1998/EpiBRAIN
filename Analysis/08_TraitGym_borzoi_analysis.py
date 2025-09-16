# %%
import h5py
import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from functools import partial
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PWD}/../')
os.chdir(f'{PWD}/../')
# %% load borzoi results
borzoi_h5 = h5py.File(f'Data/source/TraitGym/complex_traits_test.h5', 'r')
log_square = borzoi_h5['results/log_square'][:]
borzoi_info_df = pd.DataFrame({'chr': borzoi_h5['variants/chr'][:],
                               'pos': borzoi_h5['variants/pos'][:],
                               'ref': borzoi_h5['variants/ref'][:],
                               'alt': borzoi_h5['variants/alt'][:]})
borzoi_info_df.index = borzoi_info_df['chr'].astype(str) + '_' + borzoi_info_df['pos'].astype(str) + '_' + borzoi_info_df['ref'].astype(str) + '_' + borzoi_info_df['alt'].astype(str) + '_b38'
# %% read in TraitGym files