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

short_name_map = {
    'X': 'X',
    'L_with_S': 'L_S',
    'par_lowrank_L_with_S_0': 'pL_S0',
    'par_lowrank_L_with_S_1': 'pL_S1',
    'par_lowrank_L_with_S_2': 'pL_S2',
}

target_keys = ['X', 'L+S', 
               'par LoR(L)+S_0',
               'par LoR(L)+S_1',
               'par LoR(L)+S_2']

layer_names = ['self_attn.q_proj', 'self_attn.k_proj', 
               'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']

def get_results(model_type: str,
                folder:str,
                file: str,
                layer_keys: list) -> dict:
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
        eval_dict=eval,
        key_word_map=key_word_map,
        target_keys=target_keys
    )
        
    return df

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

def main(MODEL_TYPE: str,
         files: list) -> None:
    path_fig = os.path.join(root, 'data', 'figures', MODEL_TYPE)
    mkdir(path_fig)

    folder, file = files[0]
    layer_keys = get_layer_keys(MODEL_TYPE, folder, file)

    df_list = []
    for (FOLDER, file) in files:
        df = get_results(MODEL_TYPE, FOLDER, file, layer_keys)
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    
    for scope in ['all', 'layer', 'subcomp', 'block']:
        plot_violin_grid(
            df_all,
            scope_type=scope,
            save_prefix=os.path.join(path_fig, "violin_grids_all"),
            path=path_fig,
        )



    plot_ppl_grid(
        df_all,
        eval_order=[key_word_map[k] for k in target_keys],
        yscale='log',
        y_range=(15, 10000),
        path=path_fig,
        short_name_map=short_name_map,
    )

    # plot_heatmap(dfs0, path=os.path.join(path_fig, str(dfs0[0]['rho'])))
    # plot_heatmap(dfs1, path=os.path.join(path_fig, str(dfs1[0]['rho'])))


if __name__ == "__main__":
    MODEL_TYPE = 'llama_130m'
    FOLDERS = ['ablation']
    files = []

    for FOLDER in FOLDERS:
        path = os.path.join(root, 'data', FOLDER, MODEL_TYPE)
        _files = os.listdir(path)
        for file in _files:
            files.append((FOLDER, file))

    main(MODEL_TYPE=MODEL_TYPE, files=files)