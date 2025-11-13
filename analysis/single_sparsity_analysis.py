"""Script for analyzing sparsity patterns of single models.
"""
import os, sys
import pickle
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str) -> None:
    """
    Main function for single model sparsity analysis.
    Args:
        MODEL_TYPE: Type of the model (e.g., 'llama_60m').
        FOLDER: Folder containing the model and config files.
    """
    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file)
    
    # read the matrices
    LL = {}
    SS = {}
    files = os.listdir(path_folder)
    rank_files = [f for f in files if f.startswith('matrix')]
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            LL[key] = LL_part[key]
            SS[key] = SS_part[key]

    with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
        layer_info = pickle.load(f)
    diff = 0
    for layer_name in LL:
        L = LL[layer_name]
        S = SS[layer_name]
        # check the sparsity of S
        eps = torch.max(S.abs()).item() / 50.0
        s1 = torch.sum(S.abs() > eps).item()
        s2 = layer_info[layer_name]['nonzero'][-1]
        # print the largest and smallest absolute values in S
        # print(f'Layer: {layer_name}, max abs in S: {torch.max(S.abs()).item()}')
        print(f'Layer: {layer_name}, nonzero: {s1}/{s2}')

        diff += abs(s1 - s2)
        # plot the distribution of the absolute values of the nonzero elements in S
        # and log scale the x-axis
        # S_nonzero = S[S != 0].abs().cpu().numpy()
        # plt.figure()
        # plt.hist(S_nonzero, bins=50, density=True)
        # plt.xscale('log')
        # plt.title(f'Layer: {layer_name}, Non-zero elements in S')
        # plt.xlabel('Absolute value')
        # plt.ylabel('Density')
        # # plt.savefig(os.path.join(path_folder, f'sparsity_{layer_name.replace(".", "_")}.png'))
        # # plt.close()
        # plt.show()

        # check the rank of L
        # _, s, _ = torch.linalg.svd(L, full_matrices=False)
        # energy = torch.cumsum(s, dim=0) / torch.sum(s)
        # r1 = torch.sum(energy < 0.999).item() + 1
        # r2 = layer_info[layer_name]['rank'][-1]
        # print(f'Layer: {layer_name}, rank: {r1}/{r2}')
        # print(s[r2:r1])
    print(f'Total difference in non-zero counts: {diff/1e6:.2f} million')

if __name__ == '__main__':
    MODEL_TYPE = 'llama_130m'
    FOLDER = 'ablation'  
    # file = '20251029_162352'   # 60m
    file = '20251103_085721'   # 130m
    main(MODEL_TYPE=MODEL_TYPE,
         FOLDER=FOLDER,
         file=file)