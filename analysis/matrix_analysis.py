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

import rpca.ealm
import rpca.ialm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

ROOT =  get_parent_path(lvl=1)
MODEL_TYPE = 'llama_60m'
FILE = '20251006_092303'
LAYER = 0
LAYER_TYPE = 'o_proj'  # 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj'

layer_name = f'layers.{LAYER}.self_attn.{LAYER_TYPE}' if LAYER_TYPE in ['o_proj', 'q_proj', 'k_proj', 'v_proj'] else f'layers.{LAYER}.mlp.{LAYER_TYPE}'

path_folder = os.path.join(ROOT, 'data', 'salad', MODEL_TYPE, FILE)
path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')

with open(path_cfg) as f:
    cfg = yaml.safe_load(f)

model = get_model(path_cfg_model)
# load the original model weights X
load_model(model, os.path.join(path_folder, 'model.pth'))

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

X = L + S

# A0, E0 = rpca.ealm.fit_torch(X, device=device, epsilon1=1e-3, epsilon2=1e-2)
A, E = rpca.ialm.fit_torch(X, device=device, epsilon1=1e-4, epsilon2=1e-2)

X_loss = torch.linalg.norm(X - (L + S), ord='fro')
L_loss = torch.linalg.norm(L - A, ord='fro')
S_loss = torch.linalg.norm(S - E, ord='fro')

# get the number of rank of L
rank_L = torch.linalg.matrix_rank(L).item()
rank_A = torch.linalg.matrix_rank(A).item()
# get the sparsity of S
sparsity_S = (S.abs() < 1e-5).sum().item() / S.numel()
sparsity_E = (E.abs() < 1e-5).sum().item() / E.numel()  

print(f'X loss (Frobenius norm): {X_loss.item():.6f}')
print(f'L loss (Frobenius norm): {L_loss.item():.6f}')
print(f'S loss (Frobenius norm): {S_loss.item():.6f}')

print(f'Rank of L: {rank_L}, Rank of A: {rank_A}')
print(f'Sparsity of S: {sparsity_S*100:.2f}%, Sparsity of E: {sparsity_E*100:.2f}%')