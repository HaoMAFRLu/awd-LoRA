import pickle
import os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import *

root = get_parent_path(lvl=1)

def get_row(model_type: str, file: str) -> dict:
    """
    Get a row of statistics for the model from the saved file.
    Args:
        model_type: Type of the model (e.g., 'CNN', 'GPT').
        file: Name of the file containing the statistics.
    Returns:
        A dictionary with statistics for the model.
    """
    path = os.path.join(root, 'data', 'lowspa_ddp', model_type, file)
    with open(os.path.join(path, 'eval_results.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return [file,
            stats['baseline']['loss'], 
            stats['baseline_lowrank']['loss'],
            stats['lowspa']['loss'], 
            stats['lowspa_without_sparsity']['loss'],
            stats['lowspa_lowrank_without_sparsity']['loss'], 
            stats['lowspa_lowrank']['loss'], 
            stats['lowspa_lowrank_lowrank']['loss'],
            stats['lowspa_lowrank_sparsity']['loss'], 
            stats['lowspa_lowrank_lowrank_sparsity']['loss']]

def main(model_type: str, files: list) -> None:
    headers = [f"model", f"baseline", f"90% baseline", 
               f"X", f"X-S", f"90% X-S", 
               f"L", f"90% L", f"L+S", f"90% L+S"]
    rows = []
    for file in files:
        rows.append(get_row(model_type, file))
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    model_type = 'GPT'
    files = ['20250730_215710',
             '20250731_084208'
             ]
    main(model_type=model_type,
         files=files)