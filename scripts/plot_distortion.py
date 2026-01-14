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

        opt_ppl = np.inf
        opt_gamma = None
        opt_flops = None
        opt_memory = None
        opt_nr_params = None

        for _data in data:
            if _data['ppl'] < opt_ppl:
                opt_ppl = _data['ppl']
                opt_gamma = _data['gamma']
                opt_flops = _data['flops_cost']
                opt_memory = _data['memory_cost']
                opt_nr_params = _data['nr_params']

        exp = {
            'ppl': opt_ppl,
            'gamma': opt_gamma,
            'params': int(np.floor(opt_nr_params/1e6)),
            'flops': opt_flops.flops_per_token_linear/1e6,  # in MFLOPS
            'memory': opt_memory.total_bytes/1e9,  # in GB
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
        y_vals = [-np.log(exp[y_args]) for exp in exps]
        
        sorted_indices = np.argsort(x_vals)
        x_vals = np.array(x_vals)[sorted_indices]
        y_vals = np.array(y_vals)[sorted_indices]
        # plot log x
        plt.semilogx(x_vals, y_vals, marker='o', label=f'Exp {i+1}')
        # plt.plot(x_vals, y_vals, marker='o', label=f'Exp {i+1}')    
    
    if is_save:
        path_save = os.path.join(path_fig, f'{base_name}_{x_args}_{y_args}.png')
        plt.savefig(path_save, dpi=300)
        print(f'Figure saved to {path_save}')

    if is_tikz:
        path_save_tikz = os.path.join(path_fig, f'{base_name}_{x_args}_{y_args}.tex')
        tikzplotlib.save(path_save_tikz)
        print(f'TikZ figure saved to {path_save_tikz}')

    plt.legend()
    plt.xlabel(x_args)
    plt.ylabel(y_args)
    plt.show()

def main(path_parts: list,
         x_args: str='params') -> None:
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
                 x_args=x_args,
                 y_args='ppl',
                 path_fig=path_fig,
                 base_name=model_type,
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
        '20251204_135646',  # 60m, baseline fp32
        # '20251204_152747',  # 60m, head fp32
        # '20251227_222811',  # 60m, head bf16
        '20251209_204846',  # 60m, vanilla bf16
        '20251202_164626',  # 130m, baseline fp32
        # '20251203_144749',  # 130m, head fp32
        # '20251227_222332',  # 130m, head bf16
        '20251209_232356',  # 130m, vanilla bf16
        '20251203_102315',  # 350m, baseline fp32
        # '20251204_134313',  # 350m, head fp32
        # '20260102_233510',  # 350m, head bf16
        '20251209_233045',  # 350m, vanilla bf16
        '20251130_125959',  # 1b, baseline fp32
        '20251213_234650',  # 1b, vanilla bf16
    ]

    path_parts = []
    for file in files:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)
        path_parts.append(path_part) 

    main(path_parts=path_parts,
         x_args='params',)   