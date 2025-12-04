# %%
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from dotenv import load_dotenv
load_dotenv()
PWD = f'{os.environ["workingHOME"]}/BICAN'
import sys
os.chdir(PWD)
sys.path.append(PWD)

# %%
# read homer result
res_dir = 'Res/full_finetune_original_loss_celltype_head_dim8_linear/analysis_20/homer/'
homer_dirs = os.listdir(res_dir)
homer_dirs = [f for f in homer_dirs if f.endswith('_10')]

# %% do for each dir
homer_res = {}
for homer_dir in homer_dirs:
    print(f'Processing {homer_dir}...')
    motif_df = pd.read_csv(f'{res_dir}/{homer_dir}/knownResults.txt', sep='\t')
    # filter for significant motifs (q-value < 0.05)
    motif_df = motif_df[motif_df['q-value (Benjamini)'] < 0.05]
    # get cell type name
    cell_type = homer_dir.split('_')[0]
    print(f'  Found {len(motif_df)} significant motifs for cell type {cell_type}')
    homer_res[cell_type] = motif_df
    
    
# %% load jaspar motifs
from pyjaspar import jaspardb
jdb_obj = jaspardb(release='JASPAR2022')

# %%
motif_name = jdb_obj.fetch_motif_by_id(homer_res['AST']['Motif Name'].iloc[0].split('_')[0])

# %% read tomtom results
motif_df = pd.read_csv('tfm_out/CBGA/tomtom.txt', sep='\t')
# filter for q-value < 0.05
motif_df = motif_df[motif_df['q-value'] < 0.001]
# fetch motif details
motif_db = pd.read_csv('meme-5.4.1/motif_databases/CIS-BP_2.00/Homo_sapiens.txt', sep=' ', header=None, names=['info', 'motif_id', 'gene_name'])
motif_df['Gene Name'] = motif_db['gene_name'].groupby(motif_db['motif_id']).first().loc[motif_df['Target ID'].values].values
# if () exist in the motif_df['Gene Name'], remove it and extract the first bracket
extracted = motif_df['Gene Name'].str.extract(r'\((.*?)\)', expand=False)
motif_df['Gene Name'] = extracted.fillna(motif_df['Gene Name'])

# %% overlap the Gene Name to Differential Expression List
diff_express_list = pd.read_csv('Data/source/DiffExpress/subclass_corrected_edgeR.dds')
diff_express_list = diff_express_list[diff_express_list['celltype'] == 'CBGA']
motif_df['Gene Name'].isin(diff_express_list['gene']).sum(), motif_df['Gene Name'].__len__()

# %%
