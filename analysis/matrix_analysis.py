"""Basic functions for starting experiments."""
import os, sys
import yaml
import pickle
import io
import torch
import copy
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator
from salad.ialm import fit_torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

ROOT =  get_parent_path(lvl=1)
MODEL_TYPE = 'llama_60m'
FILE = '20251006_092303'
LAYER = 0
LAYER_TYPE = 'down_proj'  # 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj'

layer_name = f'layers.{LAYER}.self_attn.{LAYER_TYPE}' if LAYER_TYPE in ['o_proj', 'q_proj', 'k_proj', 'v_proj'] else f'layers.{LAYER}.mlp.{LAYER_TYPE}'

path_folder = os.path.join(ROOT, 'data', 'salad', MODEL_TYPE, FILE)
path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')

with open(path_cfg) as f:
    cfg = yaml.safe_load(f)

model = get_model(path_cfg_model)
# load the original model weights X
load_model(model, os.path.join(path_folder, 'model.pth'))

X = get_weight(model, 'model.'+layer_name).data.to(device)

LL = {}
SS = {}
files = os.listdir(path_folder)
rank_files = [f for f in files if f.startswith('matrix')]
for f in rank_files:
    LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
    for key in LL_part:
        LL[key] = LL_part[key]
        SS[key] = SS_part[key]

L = LL[layer_name].to(device)
S = SS[layer_name].to(device)

L_S = L + S

# A0, E0 = rpca.ealm.fit_torch(X, device=device, epsilon1=1e-3, epsilon2=1e-2)
A0, E0 = fit_torch(X, device=device, epsilon1=1e-2, epsilon2=1e-2)
A1, E1 = fit_torch(L_S, device=device, epsilon1=1e-2, epsilon2=1e-2)

X_loss1 = torch.linalg.norm(X - (A0 + E0), ord='fro')
X_loss2 = torch.linalg.norm(L_S - (A1 + E1), ord='fro')

L_loss1 = torch.linalg.norm(L - A0, ord='fro')
L_loss2 = torch.linalg.norm(L - A1, ord='fro')

S_loss1 = torch.linalg.norm(S - E0, ord='fro')
S_loss2 = torch.linalg.norm(S - E1, ord='fro')

# get the number of rank of L
rank_L = get_rank(L, energy_quantile=0.999)
rank_A0 = get_rank(A0, energy_quantile=0.999)
rank_A1 = get_rank(A1, energy_quantile=0.999)
# get the sparsity of S
sparsity_S = (S.abs() < 1.7e-2).sum().item() / S.numel()
sparsity_E0 = (E0.abs() < 1.7e-2).sum().item() / E0.numel()
sparsity_E1 = (E1.abs() < 1.7e-2).sum().item() / E1.numel()


print(f'For layer {layer_name}:')
print(f'Rank of L: {rank_L}, Rank of A0: {rank_A0}, Rank of A1: {rank_A1}')
print(f'Sparsity of S: {sparsity_S*100:.2f}%, Sparsity of E0: {sparsity_E0*100:.2f}%, Sparsity of E1: {sparsity_E1*100:.2f}%')
print(f'Loss on X decomposition: Method0 {X_loss1:.4f}, Method1 {X_loss2:.4f}')
print(f'Loss on L recovery: Method0 {L_loss1:.4f}, Method1 {L_loss2:.4f}')
print(f'Loss on S recovery: Method0 {S_loss1:.4f}, Method1 {S_loss2:.4f}')