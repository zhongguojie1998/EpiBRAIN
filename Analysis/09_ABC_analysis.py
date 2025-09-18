# %%
import pandas as pd
import numpy as np
import os
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')

# %%
abc = pd.read_csv("Data/source/H3K27ac_abc_filtcelltype_conns.txt", sep='\t')
# %% check uniq pairs of CellType:chr:start.y:end.y
abc['id'] = abc['CellType'] + ':' + abc['chr'] + ':' + abc['start.y'].astype(str) + ':' + abc['end.y'].astype(str)
print(len(abc['id'].unique()) * 15 / 20 / 3600) # GPU hours
# %% two types of chunks, one is TSS ± 500bp (1024 in total), the other is whole gene body
# output format ix chr_name, start, end, _, trial
trial_file = pd.read_csv('Data/basel_ganglia_complete_v2/raw_label_meta.csv')
# %%
abc_example = abc.iloc[:10,]
# %%
abc_bed = pd.DataFrame({
    'chr': abc_example['chr'],
    'start': abc_example['start.y']-511,
    'end': abc_example['end.y']+511,
    'strand': '.',
    'trial': trial_file['trial'].values[4]
})
# %%
abc_bed.to_csv('Data/source/ABC/abc_example.bed', sep='\t', header=False, index=False)
# %%
