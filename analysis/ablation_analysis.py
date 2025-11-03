# Pull the files from cluster
import os, sys
import pickle
import torch
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.utils_analysis import *

root = get_parent_path(lvl=1)

key_word_map = {
    'X': 'X',
    'X-S': 'X_without_S',
    'LoR(X-S)': 'lowrank_X_without_S',
    'L': 'L',
    'LoR(L)': 'lowrank_L',
    'L+S': 'L_with_S',
    'par L+S': 'par_L_with_S',
    'LoR(L)+S': 'lowrank_L_with_S',
    'spe LoR(L)+S': 'lowrank_L_with_S_specify',
    'par LoR(L)+S': 'par_lowrank_L_with_S',
    'par LoR(L)+S_0': 'par_lowrank_L_with_S_0',
    'par LoR(L)+S_1': 'par_lowrank_L_with_S_1',
    'par LoR(L)+S_2': 'par_lowrank_L_with_S_2',
    'par LoR(L)+S_3': 'par_lowrank_L_with_S_3',
}

target_keys = ['X', 'L+S', 
               'par LoR(L)+S_0',
               'par LoR(L)+S_1',
               'par LoR(L)+S_2']

layer_names = ['self_attn.q_proj', 'self_attn.k_proj', 
               'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']

def get_layer_stats(data: dict,
                    layer_keys: list,
                    metric: str) -> list:
    """
    Get layer-wise statistics for a specific metric.
    Args:
        data: Dictionary containing layer data.
        layer_keys: List of layer keys to extract.
        metric: Metric to extract (e.g., 'loss', 'rank').
    Returns:
        A list of metric values for the specified layers.
    """
    stats = []
    for key in layer_keys:
        if metric == 'loss':
            stats.append(data[key][metric][-1])
        elif metric == 'rank':
            stats.append(data[key][metric][-1]/data[key]['total_rank'][-1])
        elif metric == 'nonzero':
            stats.append(data[key][metric][-1]/data[key]['total_elements'][-1])
    return stats

def get_results(model_type: str,
                folder:str,
                file: str,
                layer_keys: list,
                nr_layers: int) -> dict:
    """
    """
    path = os.path.join(root, 'data', folder, model_type, file)
    with open(os.path.join(path, MODEL_TYPE+'.yaml'), 'rb') as f:
        cfg = yaml.safe_load(f)
    
    rho = cfg['layers'][0]['params']['rho_dict']['rho']
    alpha = cfg['layers'][0]['params']['alpha_dict']['rate_decay']
    beta = cfg['layers'][0]['params']['beta_dict']['rate_decay']


    with open(os.path.join(path, 'eval_results.pkl'), 'rb') as f:
        stats = pickle.load(f)
    eval = stats['eval_test_results']

    with open(os.path.join(path, 'layer_info.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    df = build_item(
        exp_id=file, 
        rho=rho, 
        alpha=alpha, 
        beta=beta,
        data=data, 
        layer_names=layer_keys,
    )

    result = {
        'exp_id': file,
        'rho': float(rho),
        'alpha': float(alpha),
        'beta': float(beta),
    }

    for i in range(len(target_keys)):
        key_name = target_keys[i]
        map_key = key_word_map[key_name]    
        result[f'ppl_{key_name}'] = eval[map_key]['ppl']
        
    return df, result

def get_layer_keys(MODEL_TYPE: str,
                   folder: str,
                   file: str) -> list:
    path = os.path.join(root, 'data', folder, MODEL_TYPE, file)

    # load yaml file
    with open(os.path.join(path, MODEL_TYPE+'.yaml'), 'rb') as f:
        cfg = yaml.safe_load(f)
    layers = cfg['layers']
    keys = [entry['name'] for entry in layers]
    return keys

def _plot_heatmap(data: list, 
                  path: str,
                  values: str) -> None:

    df = pd.DataFrame(data)
    df_pivot = df.pivot_table(index=['alpha'], columns='beta', values=values)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f'Heatmap for {values}')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.savefig(os.path.join(path, f'{values}.png'))
    plt.close()

def plot_violin(data: dict,
                path: str) -> None:
    pass

def plot_heatmap(data: dict,
                 path: str) -> None:
    """Plot heatmap of perplexity results.
    """
    target_values = []
    for key in target_keys:
        target_values.append(f'ppl_{key}')

    for values in target_values:
        _plot_heatmap(data, path, values=values)

def main(MODEL_TYPE: str,
         files: list) -> None:
    path_fig = os.path.join(root, 'data', 'figures')
    mkdir(path_fig)

    folder, file = files[0]
    layer_keys = get_layer_keys(MODEL_TYPE, folder, file)
    nr_layers = len(layer_keys) // 7

    # results = {}
    dfs0 = []
    dfs1 = []
    
    df_list = []
    for (FOLDER, file) in files:
        df, result = get_results(MODEL_TYPE, FOLDER, file, layer_keys, nr_layers)
        # results.update(result)
        if result['rho'] == 1e-6:
            dfs0.append(result)
        elif result['rho'] == 1e-7:
            dfs1.append(result)
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    plot_violin_grid(
        df_all,
        scope_type='block',
        save_prefix=os.path.join(path_fig, "violin_grids_all"),
    )
    # plot_heatmap(dfs0, path=os.path.join(path_fig, str(dfs0[0]['rho'])))
    # plot_heatmap(dfs1, path=os.path.join(path_fig, str(dfs1[0]['rho'])))

    # plot_violin(dfs0)

if __name__ == "__main__":
    MODEL_TYPE = 'llama_60m'
    FOLDERS = ['ablation']
    files = []

    for FOLDER in FOLDERS:
        path = os.path.join(root, 'data', FOLDER, MODEL_TYPE)
        _files = os.listdir(path)
        for file in _files:
            files.append((FOLDER, file))

    main(MODEL_TYPE=MODEL_TYPE, files=files)