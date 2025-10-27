"""Basic functions for starting experiments."""
import os, sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import rpca.ialm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

LAYER = 0
LAYER_TYPE = 'down_proj'  # 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj'

layer_name = f'layers.{LAYER}.self_attn.{LAYER_TYPE}' if LAYER_TYPE in ['o_proj', 'q_proj', 'k_proj', 'v_proj'] else f'layers.{LAYER}.mlp.{LAYER_TYPE}'

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

X = get_weight(model, 'model.'+layer_name).data.to(device)

A, E = rpca.ialm.fit_torch(X, device=device, epsilon1=1e-2, epsilon2=1e-2)

X_loss = torch.linalg.norm(X - (A + E), ord='fro')

# get the number of rank of L
rank_A = get_rank(A, energy_quantile=0.999)
# get the sparsity of S
sparsity_E = (E.abs() < 1.7e-2).sum().item() / E.numel()


print(f'For layer {layer_name}:')
print(f'Rank of A: {rank_A}')
print(f'Sparsity of S: {sparsity_E*100:.2f}%')
print(f'Loss on X decomposition: Method0 {X_loss:.4f}')