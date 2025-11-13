"""Script for analyzing sparsity patterns of single models.
"""
import os, sys

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

    for layer_name in LL:
        L = LL[layer_name]
        S = SS[layer_name]
        # check the sparsity of S
        s1 = torch.count_nonzero(S).item()
        s2 = layer_info[layer_name]['nonzero'][-1]
        # print(f'Layer: {layer_name}, nonzero: {s1}/{s2}')

        # check the rank of L
        _, s, _ = torch.linalg.svd(L, full_matrices=False)
        energy = torch.cumsum(s, dim=0) / torch.sum(s)
        r1 = torch.sum(energy < 0.999).item() + 1
        r2 = layer_info[layer_name]['rank'][-1]
        print(f'Layer: {layer_name}, rank: {r1}/{r2}')
        print(s[r2:r1])


if __name__ == '__main__':
    MODEL_TYPE = 'llama_130m'
    FOLDER = 'ablation'
    # file = '20251029_162352'
    file = '20251103_085721'
    main(MODEL_TYPE=MODEL_TYPE,
         FOLDER=FOLDER,
         file=file)