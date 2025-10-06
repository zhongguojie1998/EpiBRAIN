# %%
import pandas as pd
import numpy as np
import os
PWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{PWD}/../')

# %%
abc = pd.read_csv("Data/source/ABC/H3K27ac_abc_filtcelltype_conns.txt", sep='\t')
# %% check uniq pairs of CellType:chr:start.y:end.y
abc['id'] = abc['CellType'] + ':' + abc['chr'] + ':' + abc['start.y'].astype(str) + ':' + abc['end.y'].astype(str)
print(len(abc['id'].unique()) * 15 / 20 / 3600) # GPU hours
# %% two types of chunks, one is TSS ± 500bp (1024 in total), the other is whole gene body
# output format ix chr_name, start, end, _, trial
trial_file = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/raw_label_meta.csv')
# %% do four runs, one for ATAC tracks, K27Ac tracks, one for RNAplus, one for RNAminus
# Pre-create a lookup dictionary for faster trial matching
trial_lookup = {}
for modal in ['ATAC', 'K27Ac', 'RNAplus', 'RNAminus']:
    trial_lookup[modal] = trial_file[trial_file['modality'] == modal].set_index('cell_type')['trial'].to_dict()

for modal in ['ATAC', 'K27Ac', 'RNAplus', 'RNAminus']:
    # Vectorized operations
    abc_modal = abc.copy()
    abc_modal['cell_type_key'] = 'MiniAtlas-' + abc_modal['CellType']
    abc_modal['trial'] = abc_modal['cell_type_key'].map(trial_lookup[modal])

    # Filter out rows with no matching trial
    missing_trials = abc_modal[abc_modal['trial'].isna()]
    if len(missing_trials) > 0:
        for ct in missing_trials['CellType'].unique():
            print(f'No trial found for {ct} {modal}')

    abc_modal = abc_modal.dropna(subset=['trial'])

    # Create BED format output
    abc_bed = pd.DataFrame({
        'chr': abc_modal['chr'],
        'start': abc_modal['start.y'] - 511,
        'end': abc_modal['end.y'] + 511,
        'strand': '.',
        'trial': abc_modal['trial']
    })

    abc_bed.to_csv(f'Data/source/ABC/abc_{modal.lower()}.bed', sep='\t', header=False, index=False)

# %%
