"""Basic functions for starting experiments."""
import os, sys
import yaml
import torch
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.uia import UIA
from salad.operators import *


# get the root path
root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_all_models(MODEL_TYPES: list=['llama_60m'],
                   FOLDERS: list=['baseline'],
                   FILE: str=None,
                   nr_params: float=646.5,
                   gamma: float=0.5) -> None:
    """
    """
    # set up paths and load config
    path_folder = None
    for _MODEL_TYPE in MODEL_TYPES:
        for _FOLDER in FOLDERS:
            _path_folder = os.path.join(root, 'data', _FOLDER, _MODEL_TYPE, FILE)
            if os.path.exists(_path_folder):
                path_folder = _path_folder
                FOLDER = _FOLDER
                MODEL_TYPE = _MODEL_TYPE

    if path_folder is None:
        raise ValueError(f"Paht does not exist!")
    
    path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
    path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)
    # set up the hyperparameters
    seed = cfg['seed']
    set_seed(seed)
    max_length = cfg.get('max_length', 1024)
    batch_size = cfg.get('eval_batch_size', 8)

    """
    Load the original model: X
    """
    model_original = get_model(path_cfg_model)
    load_model(model_original, os.path.join(path_folder, 'model.pth'))
    
    """
    Load the spars and low-rank model: L+S
    """
    # Step 1: load the trained sparse matrices L, and low-rank matrices S
    # for the specified layers
    LL = {}
    SS = {}
    files = os.listdir(path_folder)
    rank_files = [f for f in files if f.startswith('matrix')]
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            LL[key] = LL_part[key]
            SS[key] = SS_part[key]

    
    
    # Step 2: get the specified layers
    layers = [entry['name'] for entry in cfg['layers']]

    # Step 3: get the linear layers in the model for cross evaluation
    # model_layers = get_model_layer_names(model_original)

    # Step 3: load the original model X again
    model_LS = get_model(path_cfg_model)
    load_model(model_LS, os.path.join(path_folder, 'model.pth'))
    model_LS.to(device)

    uia = UIA(LL, SS, model_LS, 
              layer_info=layer_info, 
              rate=100000000.0,
              rank=rank)
    # Step 4: replace the specified layers in the original model with L+S
    # Step 4.1: replace the specified layers with low-rank matirces
    opt_replace(model_LS, layers, LL, device)
    # Step 4.2: add sparse matrices
    opt_lowrank(model_LS, layers, rank_quantile)
    _SS = re_sparse(SS, rate_density)
    opt_add(model_LS, layers, _SS) 
    
    print('finished')

if __name__ == '__main__':
    MODEL_TYPES = [
                  'llama_60m',
                  'llama_130m',
                  'llama_350m',
                  'llama_1b'
                ]
    FOLDERS = [
              'incl_embedding',
              'baseline'
            ]
    # FILE = '20251204_152747'
    FILE = '20251204_135646'
    gamma = 0.25
    nr_params = 646.5
    load_all_models(MODEL_TYPES,
                    FOLDERS,
                    FILE,
                    nr_params,
                    gamma)