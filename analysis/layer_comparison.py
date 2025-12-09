import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.ialm import fit_torch

root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rate = 100000000.0

def get_path(path_part: dict) -> str: 
    return os.path.join(root, 'data', path_part['folder'], path_part['model_type'], path_part['file'])

def get_layer_name(nr_layer: int, type_layer: str) -> str:
    if type_layer in ['q', 'k', 'v', 'o']:
        return f'layers.{nr_layer}.self_attn.{type_layer}_proj'
    elif type_layer in ['up', 'down', 'gate']:
        return f'layers.{nr_layer}.mlp.{type_layer}_proj'
    elif type_layer == 'embed':
        return 'embed_tokens'
    elif type_layer == 'lm_head':
        return 'lm_head'
    else:
        raise ValueError(f'Unknown layer type: {type_layer}')

def main(path_part1: dict,
         path_part2: dict,
         layer_name: str) -> None:
    
    path1 = get_path(path_part1)
    path2 = get_path(path_part2)

    if 'embed' in layer_name:
        if 'incl_embedding' in path_part1['folder']:
            L, S = get_layer_weight(path1, layer_name)
            X = get_layer_weight(path2, layer_name, target='X')

        elif 'incl_embedding' in path_part2['folder']:
            X = get_layer_weight(path1, layer_name, target='X')
            L, S = get_layer_weight(path2, layer_name)


        diff = torch.norm(X - (L + S), p='fro').item()
        total_elements = X.numel()
        avg = diff / np.sqrt(total_elements)
        print(f'Difference: {diff} | Average: {avg}')

        row, col = L.shape
        L_hat, S_hat = fit_torch(X, lambda_ = 1.0 / np.sqrt(max(row, col)),
                           device=device, 
                           dtype=torch.float32, 
                           epsilon1=1e-2, 
                           epsilon2=1e-2)
            
        diff = torch.norm(X.to(device) - L_hat.to(device) - S_hat.to(device), p='fro').item()
        avg = diff / np.sqrt(total_elements)
        print(f'{diff} | {avg}')

        diff = torch.norm(S.to(device) - S_hat.to(device), p='fro').item()
        avg = diff / np.sqrt(total_elements)
        print(f'{diff} | {avg}')    

        max_value = torch.max(S.abs()).item()
        eps = max_value / rate 

        nr_total = row * col

        nr_nonzero = torch.sum(S.abs() > eps).item()
        rate_density1 = nr_nonzero / nr_total

        max_value = torch.max(S_hat.abs()).item()
        eps = max_value / rate
        nr_nonzero = torch.sum(S_hat.abs() > eps).item()
        rate_density2 = nr_nonzero / nr_total
        print(f'Rate density incl. embedding: {rate_density1:.6f}')
        print(f'Rate density excl. embedding: {rate_density2:.6f}')

        _, s1, _ = torch.linalg.svd(L, full_matrices=False)     # including embedding
        _, s2, _ = torch.linalg.svd(L_hat, full_matrices=False) # excluding embedding

        energy1 = torch.cumsum(s1, dim=0) / torch.sum(s1)
        rank1 = torch.sum(energy1 < 0.999).item() + 1
        rank_quantile1 = rank1 / len(s1)
        energy2 = torch.cumsum(s2, dim=0) / torch.sum(s2)
        rank2 = torch.sum(energy2 < 0.999).item() + 1
        rank_quantile2 = rank2 / len(s2)
        
        print(f'Rank quantile incl. embedding: {rank_quantile1:.6f}')
        print(f'Rank quantile excl. embedding: {rank_quantile2:.6f}')

        # plot the distribution of the abs. values of all entries in S1 and S2
        plt.figure(figsize=(8,6))
        plt.hist(S_hat.cpu().numpy().flatten(), bins=300, alpha=0.5, label='excl. embedding')
        plt.hist(S.cpu().numpy().flatten(), bins=300, alpha=0.5, label='incl. embedding')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title(f'Distribution of Sparse Component Values for Layer {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # plot singular values
        plt.figure(figsize=(8,6))
        plt.semilogy(s1.detach().cpu().numpy(), label='incl. embedding')
        plt.semilogy(s2.detach().cpu().numpy(), label='excl. embedding')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title(f'Singular Values of Low-Rank Component for Layer {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif 'lm_head' in layer_name:
        if 'incl_embedding' in path1:
            X1 = get_layer_weight(path2, layer_name, target='X') # excluding embedding
            X2 = get_layer_weight(path1, layer_name, target='X') # including embedding
        elif 'incl_embedding' in path2:
            X1 = get_layer_weight(path1, layer_name, target='X') # excluding embedding
            X2 = get_layer_weight(path2, layer_name, target='X') # including embedding

        L1, S1 = fit_torch(X1, lambda_ = 1.0 / np.sqrt(max(X1.shape)),
                           device=device, 
                           dtype=torch.float32, 
                           epsilon1=1e-2, 
                           epsilon2=1e-2)
        
        L2, S2 = fit_torch(X2, lambda_ = 1.0 / np.sqrt(max(X2.shape)),
                           device=device, 
                           dtype=torch.float32, 
                           epsilon1=1e-2, 
                           epsilon2=1e-2)

        max_value = torch.max(S1.abs()).item()
        eps = max_value / rate 

        row, col = L1.shape
        nr_total = row * col

        nr_nonzero = torch.sum(S1.abs() > eps).item()
        rate_density1 = nr_nonzero / nr_total

        max_value = torch.max(S2.abs()).item()
        eps = max_value / rate
        nr_nonzero = torch.sum(S2.abs() > eps).item()
        rate_density2 = nr_nonzero / nr_total
        print(f'Rate density excl. embedding: {rate_density1:.6f}')
        print(f'Rate density incl. embedding: {rate_density2:.6f}')

        _, s1, _ = torch.linalg.svd(L1, full_matrices=False)
        _, s2, _ = torch.linalg.svd(L2, full_matrices=False)

        # calculate 99.9% energy rank
        energy1 = torch.cumsum(s1, dim=0) / torch.sum(s1)
        rank1 = torch.sum(energy1 < 0.999).item() + 1
        rank_quantile1 = rank1 / len(s1)
        energy2 = torch.cumsum(s2, dim=0) / torch.sum(s2)
        rank2 = torch.sum(energy2 < 0.999).item() + 1
        rank_quantile2 = rank2 / len(s2)

        print(f'Rank quantile excl. embedding: {rank_quantile1:.6f}')
        print(f'Rank quantile incl. embedding: {rank_quantile2:.6f}')

        # plot singular values
        plt.figure(figsize=(8,6))
        plt.semilogy(s1.cpu().numpy(), label='excl. embedding')
        plt.semilogy(s2.cpu().numpy(), label='incl. embedding')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title(f'Singular Values of Sparse Component for Layer {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # plot the distribution of the abs. values of all entries in S1 and S2
        plt.figure(figsize=(8,6))
        plt.hist(S1.cpu().numpy().flatten(), bins=300, alpha=0.5, label='excl. embedding')
        plt.hist(S2.cpu().numpy().flatten(), bins=300, alpha=0.5, label='incl. embedding')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title(f'Distribution of Sparse Component Values for Layer {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        if 'incl_embedding' in path2:
            L1, S1 = get_layer_weight(path1, layer_name)
            L2, S2 = get_layer_weight(path2, layer_name)
        elif 'incl_embedding' in path1:
            L1, S1 = get_layer_weight(path2, layer_name)
            L2, S2 = get_layer_weight(path1, layer_name)

        _, s1, _ = torch.linalg.svd(L1, full_matrices=False)
        _, s2, _ = torch.linalg.svd(L2, full_matrices=False)

        # plot singular values
        plt.figure(figsize=(8,6))
        plt.semilogy(s1.cpu().numpy(), label='excl. embedding')
        plt.semilogy(s2.cpu().numpy(), label='incl. embedding')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title(f'Singular Values of Sparse Component for Layer {layer_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

        
    print('here')

if __name__ == '__main__':
    MODEL_TYPES = [
        'llama_60m',
        'llama_130m',
        'llama_350m',
        'llama_1b',
    ]

    FOLDERS = [
        'baseline',
        'incl_embedding',
    ]

    FILES = [
        '20251204_135646',
        '20251204_152747',
    ]

    nr_layer = 2
    type_layer = 'embed'  # embed, q, v, q, k, up, down, gate

    layer_name = get_layer_name(nr_layer, type_layer)
    path_part1 = determine_path_part(MODEL_TYPES, FOLDERS, FILES[0])
    path_part2 = determine_path_part(MODEL_TYPES, FOLDERS, FILES[1])

    main(path_part1=path_part1,
         path_part2=path_part2,
         layer_name=layer_name)