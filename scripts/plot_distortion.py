"""Plot rate-distortion curves from saved results.
"""
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

def read_distortion(model_type: str,
                    folder: str,
                    file: str) -> dict:
    path_folder = os.path.join(root, 'data', folder, model_type, file)
    _files = os.listdir(path_folder)
    distortion_files = [f for f in _files if f.startswith('distortion_')]   
    exps = []
    for file in distortion_files:
        with open(os.path.join(path_folder, file), 'rb') as f:
            data = pickle.load(f)
        exp = {
            'ppl': data['ppl'],
            'gamma': data['gamma'],
            'params': int(np.floor(data['nr_params']/1e6)),
            'flops': data['flops_cost'].flops_per_token_linear/1e6,  # in MFLOPS
            'memory': data['memory_cost'].total_bytes/1e9,  # in GB
        }
        exps.append(exp)
    return exps

def get_curve(exp: list,
              x_args: str,
              y_args: str) -> tuple:
    
    x_vals = [e[x_args] for e in exp]
    y_vals = [e[y_args] for e in exp]



def plot_figures(results: list,
                 x_args: str,
                 y_args: str,
                 path_fig: str,
                 base_name: str,
                 is_save: bool=True,
                 is_tikz: bool=False) -> None:
    

    # first get the curve for each experiment based on the given x and y args
    # then plot them together
    # write the curve as a function

    plt.figure(figsize=(6, 4))
    
    for i, exps in enumerate(results):
        x_vals = [exp[x_args] for exp in exps]
        y_vals = [-exp[y_args] for exp in exps]
        
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        
        plt.plot(x_vals, y_vals, marker='o', label=f'Exp {i+1}')    
    
    plt.legend()
    plt.xlabel(x_args)
    plt.ylabel(y_args)
    plt.show()

def main(path_parts: list) -> None:
    results = []
    for path_part in path_parts:
        model_type = path_part['model_type']
        folder = path_part['folder']
        file = path_part['file']

        results.append(read_distortion(model_type=model_type,
                                        folder=folder,
                                        file=file))
    
    path_fig = os.path.join(root, 'data', 'figures', 'rate_distortion')
    mkdir(path_fig)

    plot_figures(results=results,
                 x_args='params',
                 y_args='ppl',
                 path_fig=path_fig,
                 base_name=model_type,
                 is_save=True,
                 is_tikz=False)

    print('here')

if __name__ == "__main__":
    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'vanilla_bf16',  
        'baseline_fp32',
    ]

    files = [
        # '20251209_104454',  # for quick test
        '20251209_204846',
        '20251204_135646',
    ]

    path_parts = []
    for file in files:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)
        path_parts.append(path_part) 

    main(path_parts=path_parts)   