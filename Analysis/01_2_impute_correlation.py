# %%
import multiprocessing as mp
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PWD = '/gpfs/commons/groups/ren_lab/guojiezhong/BICAN'
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
test_res = torch.load(f"Res/basal_ganglia_miniatlas_drop_celltype_v1/Test_preds_epoch_150.pt")
# %%
pred_track_info = pd.read_csv('logs/basal_ganglia_miniatlas_drop_celltype_v1/regression_label_meta.csv')
label_track_info = pd.read_csv('Data/basal_ganglia_miniatlas_drop_celltype_v1/raw_label_meta.csv')
# %% compare miniatlas track 93 and label_track 3
miniatlas_k9me3_non_neuron_dims = [93, 189, 195, 183]
basalganglia_k9me3_non_neuron_dims = [3, 99, 51, 45]
miniatlas_non_neuron_names = ['MiniAtlas-AST', 'MiniAtlas-OGC', 'MiniAtlas-OPC', 'MiniAtlas-MGC']
basalganglia_non_neuron_names = ['BasalGanglia-Astrocyte', 'BasalGanglia-Oligodendrocyte', 
                                 'BasalGanglia-OPC', 'BasalGanglia-Microglia']

# %% plot the correlation
from scipy.stats import pearsonr
for i, j, miniatlas_name, basalganglia_name in zip(miniatlas_k9me3_non_neuron_dims, 
                                                    basalganglia_k9me3_non_neuron_dims,
                                                    miniatlas_non_neuron_names,
                                                    basalganglia_non_neuron_names):
    pred = test_res['pred']['regression'][:, :, i].numpy().flatten()
    label = test_res['label']['regression'][:, :, j].numpy().flatten()
    r, p = pearsonr(pred, label)
    print(f"Correlation between {miniatlas_name} and {basalganglia_name}: r={r:.4f}, p={p:.4e}")
# %% check other three modalities
for modality in ['ATAC', 'K27Ac', 'K27Me3']:
    for miniatlas_name, basalganglia_name in zip(miniatlas_non_neuron_names, basalganglia_non_neuron_names):
        pred = test_res['pred']['regression'][:, :, pred_track_info['dim'][(pred_track_info['cell_type'] == miniatlas_name) & (pred_track_info['modality'] == modality)].values[0]].numpy().flatten()
        label = test_res['label']['regression'][:, :, label_track_info.index[(label_track_info['cell_type'] == basalganglia_name) & (label_track_info['modality'] == modality)].values[0]].numpy().flatten()
        r, p = pearsonr(pred, label)
        print(f"Correlation between {miniatlas_name} and {basalganglia_name} {modality}: r={r:.4f}, p={p:.4e}")

# %%
