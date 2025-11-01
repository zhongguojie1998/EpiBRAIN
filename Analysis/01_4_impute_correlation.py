# %%
import multiprocessing as mp
import os
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")

PWD = f'{os.getenv("workingHOME")}/BICAN'
sys.path.append(f'{PWD}/')
os.chdir(PWD)
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# %%
test_res = torch.load(f"Res/full_finetune_original_loss_celltype_head_dim8_linear/Test_preds_epoch_25.pt")
# %%
pred_track_info = pd.read_csv('logs/full_finetune_original_loss_celltype_head_dim8_linear/regression_label_meta.csv')
label_track_info = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/raw_label_meta.csv')
# %% compare miniatlas track 93 and label_track 3
miniatlas_non_neuron_names = ['MiniAtlas-AST', 'MiniAtlas-OGC', 'MiniAtlas-OPC', 'MiniAtlas-MGC']
basalganglia_non_neuron_names = ['BasalGanglia-Astrocyte', 'BasalGanglia-Oligodendrocyte', 
                                 'BasalGanglia-OPC', 'BasalGanglia-Microglia']

# %% plot the correlation
from scipy.stats import pearsonr
for miniatlas_name, basalganglia_name in zip(miniatlas_non_neuron_names, basalganglia_non_neuron_names):
    i = pred_track_info['dim'][(pred_track_info['cell_type'] == miniatlas_name) & (pred_track_info['modality'] == 'ATAC')].values[0] + 3
    # K9Me3 is ATAC +3
    j = label_track_info.index[(label_track_info['cell_type'] == basalganglia_name) & (label_track_info['modality'] == 'K9Me3')].values[0]
    pred = test_res['pred']['regression'][:, :, i].numpy().flatten()
    label = test_res['label']['regression'][:, :, j].numpy().flatten()
    r, p = pearsonr(pred, label)
    print(f"Correlation between {miniatlas_name} and {basalganglia_name}: r={r:.4f}, p={p:.4e}")
 # %% check STR-Hybrid-MSN track vs ground truth
celltype = 'BasalGanglia-STR-Hybrid-MSN'
label_track_info = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/raw_label_meta.csv')
label_res = torch.load(f"Res/full_finetune_original_loss_celltype_head_dim8/Test_preds_epoch_25.pt")
# %% get the dims correctly
i = pred_track_info['dim'][(pred_track_info['cell_type'] == celltype) & (pred_track_info['modality'] == 'ATAC')].values[0] + 3
j = label_track_info.index[(label_track_info['cell_type'] == celltype) & (label_track_info['modality'] == 'K9Me3')].values[0]
# reorder to make the indexes match
pred_idxs = np.argsort(test_res['index'])
label_idxs = np.argsort(label_res['index'])
pred = test_res['pred']['regression'][pred_idxs, :, i].numpy().flatten()
label = label_res['label']['regression'][label_idxs, :, j].numpy().flatten()
r, p = pearsonr(pred, label)
print(f"Correlation between imputed {celltype} and ground truth {celltype}: r={r:.4f}, p={p:.4e}")
# %%
