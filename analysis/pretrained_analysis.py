"""Basic functions for starting experiments."""
import os, sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

import rpca.ialm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

hf_login_once()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

LAYER = 11
LAYER_TYPE = 'down_proj'  # 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj'

layer_name = f'layers.{LAYER}.self_attn.{LAYER_TYPE}' if LAYER_TYPE in ['o_proj', 'q_proj', 'k_proj', 'v_proj'] else f'layers.{LAYER}.mlp.{LAYER_TYPE}'

model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B"

model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )

# 只把目标权重搬到 GPU（若可用）进行 RPCA
rpca_device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = get_weight(model, 'model.' + layer_name).detach().to(torch.float32).to(rpca_device)

m, n = X.shape
# 用你实现的 torch 版 IALM；dtype 用 FP32，避免半精度 SVD 问题
A, E = rpca.ialm.fit_torch(X, 
                lambda_ = 1.0 / np.sqrt(max(m, n)),
                device=rpca_device, 
                dtype=torch.float32, 
                epsilon1=1e-2, 
                epsilon2=1e-2)

X_loss = torch.linalg.norm(X - (A + E), ord='fro')

rank_A = get_rank(A, energy_quantile=0.999)
sparsity_E = (E.abs() < 1.7e-2).sum().item() / E.numel()

print(f'For layer {layer_name}:')
print(f'Rank of A: {rank_A}/{min(m, n)}')
print(f'Sparsity of S: {sparsity_E*100:.2f}%')
print(f'Loss on X decomposition: Method0 {X_loss:.4f}')