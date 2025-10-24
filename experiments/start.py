"""Basic functions for starting experiments."""
import os, sys
import yaml
import pickle
import io
import torch
import copy
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator

# get the root path
root = get_parent_path(lvl=1)
# define the model type and file
MODEL_TYPE = 'llama_60m'
FILE = '20251007_134341'
# define the low-rank configuration
rank_cfg = {
        'o_proj':    0.25,
        'q_proj':    0.25,
        'k_proj':    0.25,
        'v_proj':    0.25,
        'gate_proj': 0.35,
        'down_proj': 0.35,
        'up_proj':   0.35
    }
# define the number of layers to remove for partial low-rank evaluation
nr_remove = [3, 6, 9]

# set up paths and load config
path_folder = os.path.join(root, 'data', 'salad', MODEL_TYPE, FILE)
path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')
with open(path_cfg) as f:
    cfg = yaml.safe_load(f)
# set up the hyperparameters
seed = cfg['seed']
set_seed(seed)
max_length = cfg.get('max_length', 1024)
batch_size = cfg.get('eval_batch_size', 8)
# build the model
# TODO: might need to change the model structure for the memory efficient evaluation
model = get_model(path_cfg_model)
# load the original model weights X
load_model(model, os.path.join(path_folder, 'model.pth'))
# load the low-rank and sparse components L and S from all files starting with 'matrix'
LL = {}
SS = {}
files = os.listdir(path_folder)
rank_files = [f for f in files if f.startswith('matrix')]
for f in rank_files:
    LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
    for key in LL_part:
        LL[key] = LL_part[key]
        SS[key] = SS_part[key]

# get the layers trained with SALAAD
# they should be the same as keys in LL and SS
layers = [entry['name'] for entry in cfg['layers']]
# get the number of ranks of each layer for the later evaluation
rank_quantile_target = {entry['name']: entry['params']['rate_rank'] for entry in cfg['layers']}
energy_quantile = {entry['name']: entry['params']['energy'] for entry in cfg['layers']}


rank_quantile_energy = {}
rank_quantile_specify = {}
rank_diff = {}
rate_sparsity = {}
layer_dim = {}

for key in energy_quantile:
    _L = LL[key]
    _S = SS[key]
    layer_dim[key] = _L.shape

    rate_sparsity[key] = torch.sum(_S != 0).item() / _S.numel()

    _, s, _ = torch.linalg.svd(_L, full_matrices=False)
    energy = torch.cumsum(s, dim=0) / torch.sum(s)
    rank = torch.sum(energy < energy_quantile[key]).item() + 1
    rank_quantile_energy[key] = rank / len(s)

    if key.endswith('o_proj'):
        rank_quantile_specify[key] = min(rank_cfg['o_proj'], rank_quantile_energy[key])
    elif key.endswith('q_proj'):
        rank_quantile_specify[key] = min(rank_cfg['q_proj'], rank_quantile_energy[key])
    elif key.endswith('k_proj'):
        rank_quantile_specify[key] = min(rank_cfg['k_proj'], rank_quantile_energy[key])
    elif key.endswith('v_proj'):
        rank_quantile_specify[key] = min(rank_cfg['v_proj'], rank_quantile_energy[key])
    elif key.endswith('gate_proj'):
        rank_quantile_specify[key] = min(rank_cfg['gate_proj'], rank_quantile_energy[key])
    elif key.endswith('down_proj'): 
        rank_quantile_specify[key] = min(rank_cfg['down_proj'], rank_quantile_energy[key])
    elif key.endswith('up_proj'):
        rank_quantile_specify[key] = min(rank_cfg['up_proj'], rank_quantile_energy[key])

    rank_diff[key] = rank_quantile_energy[key] - rank_quantile_specify[key]

rank_quantile_partial_list = [
    copy.deepcopy(rank_quantile_specify) for _ in range(len(nr_remove))
]
# sort according to rank diff
sorted_layers = sorted(rank_diff.items(), key=lambda item: item[1], reverse=True)
for i in range(len(nr_remove)):
    _nr_remove = nr_remove[i]
    for j in range(_nr_remove):
        layer_name = sorted_layers[j][0]
        rank_quantile_partial_list[i][layer_name] = rank_quantile_energy[layer_name]

