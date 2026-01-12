"""Plot rate-distortion curves from saved results.
"""
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)
color_mapping = {
    'llama_9m': 'b',
    'llama_60m': 'g',
    'llama_130m': 'r',
    'llama_350m': 'c',
    'llama_1b': 'm',
}

def read_distortion(model_type: str,
                    folder: str,
                    file: str) -> dict:
    path_folder = os.path.join(root, 'data', folder, model_type, file)
    _files = os.listdir(path_folder)
    distortion_files = [f for f in _files if f.startswith('comp_kappa')]   
    exps = []
    for file in distortion_files:
        with open(os.path.join(path_folder, file), 'rb') as f:
            data = pickle.load(f)

        gamma_list = []
        ppl_list = []
        nr_params = None
        for _data in data:
            gamma_list.append(_data['gamma'])
            ppl_list.append(_data['ppl'])
            if nr_params is None:
                nr_params = int(np.floor(_data['nr_params']/1e6))
            else:
                assert nr_params == int(np.floor(_data['nr_params']/1e6))

        exp = {
            'model_type': model_type,
            'ppl': ppl_list,
            'gamma': gamma_list,
            'params': int(np.floor(_data['nr_params']/1e6)),
        }

        exps.append(exp)
    return exps

def plot_figures(results: list,
                 path_fig: str,
                 is_save: bool=True,
                 is_tikz: bool=False) -> None:
    
    # first get the curve for each experiment based on the given x and y args
    # then plot them together
    # write the curve as a function

    plt.figure(figsize=(6, 4))

    for exps in results:
        for i, exp in enumerate(exps):
            model_type = exp['model_type']
            nr_params = exp['params']
            x_vals = exp['gamma']  # in million
            y_vals = [xx for xx in exp['ppl']]
            
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            y_vals = np.array(y_vals)[sorted_indices]
            # plot log x
            plt.plot(x_vals, y_vals, marker='o', label=f'{model_type}, {nr_params}', color=color_mapping.get(exp['model_type'], 'k'))  
    
    plt.legend()
    if is_save:
        path_save = os.path.join(path_fig, f'comp_kappa.png')
        plt.savefig(path_save, dpi=300)
        print(f'Figure saved to {path_save}')

    if is_tikz:
        path_save_tikz = os.path.join(path_fig, f'comp_kappa.tex')
        tikzplotlib.save(path_save_tikz)
        print(f'TikZ figure saved to {path_save_tikz}')

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
    
    path_fig = os.path.join(root, 'data', 'figures', 'comp_kappa')
    mkdir(path_fig)

    plot_figures(results=results,
                 path_fig=path_fig,
                 is_save=True,
                 is_tikz=True)

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
        # '20251204_135646',  # 60m, baseline fp32
        # '20251204_152747',  # 60m, head fp32
        # '20251227_222811',  # 60m, head bf16
        # '20251209_204846',  # 60m, vanilla bf16
        '20251202_164626',  # 130m, baseline fp32
        # '20251203_144749',  # 130m, head fp32
        # '20251227_222332',  # 130m, head bf16
        # '20251209_232356',  # 130m, vanilla bf16
        '20251203_102315',  # 350m, baseline fp32
        # '20251204_134313',  # 350m, head fp32
        # '20260102_233510',  # 350m, head bf16
        # '20251209_233045',  # 350m, vanilla bf16
        # '20251130_125959',  # 1b, baseline fp32
        # '20251213_234650',  # 1b, vanilla bf16
    ]

    path_parts = []
    for file in files:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)
        path_parts.append(path_part) 

    main(path_parts=path_parts)   