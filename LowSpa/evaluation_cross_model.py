import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import pickle
from tqdm import tqdm
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import *
from lowspa_ddp.model_register import get_model, get_dataloader
from lowspa_ddp.cross_evaluation import CrossEvaluator

root = get_parent_path(lvl=1)

def find_yaml(folder: str) -> list:
    folder = Path(folder)
    files = []
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.yaml', '.yml'):
            files.append(p)
    return files

def load_model(pth: dict, cfg: dict) -> torch.nn.Module:
    """
    Load the model from the given path dictionary.
    
    Args:
        pth_dict (dict): Dictionary containing paths to model files.
        model_type (str): Type of the model to load.
    
    Returns:
        nn.Module: The loaded model.
    """
    model = get_model(cfg)
    
    ckpt = torch.load(pth, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    clean_sd = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        clean_sd[k] = v

    model.load_state_dict(clean_sd, strict=True)
    return model

def get_lowspa_layers(pth: str) -> tuple:
    """Load data from the files"""
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        with open(pth, 'rb') as f:
            obj = pickle.load(f) 
    finally:
        torch.storage._load_from_bytes = orig
    return obj['LL'], obj['SS']


def main(model_type: str,
         file_baseline: str=None,
         file_lowspa: str=None) -> None:
    
    if file_baseline is not None:
        path_baseline = os.path.join(root, 'data', 'baseline', model_type, file_baseline)
        cfg_baseline = read_cfg(find_yaml(path_baseline)[0])
        model_baseline = load_model(os.path.join(path_baseline, 'model.pth'), cfg_baseline['model'],)

    if file_lowspa is not None:
        path_lowspa = os.path.join(root, 'data', 'lowspa_ddp', model_type, file_lowspa)
        cfg_lowspa = read_cfg(find_yaml(path_lowspa)[0])
        model_lowspa = load_model(os.path.join(path_lowspa, 'model.pth'), cfg_lowspa['model'])
        LL, SS = get_lowspa_layers(os.path.join(path_lowspa, 'results.pkl'))

    cfg_dataloader = cfg_lowspa['dataloader']
    cfg_dataloader['split'] = 'val'
    cfg_dataloader['batch_size'] = 100
    data_loader = get_dataloader(cfg_baseline['model']['name'],
                                 cfg_dataloader)

    # with '.weight' suffix
    layers = [entry['name'] for entry in cfg_lowspa['layers']]
    evaluator = CrossEvaluator(model_type=model_type,
                               baseline=model_baseline,
                               lowspa_model=model_lowspa,
                               data_loader=data_loader,
                               layers=layers,
                               LL=LL,
                               SS=SS,
                               energy_quantile=0.9)
    
    # evaluator.test_opts()
    evaluator.eval_baseline()
    evaluator.eval_lowspa()

    with open(os.path.join(path_lowspa, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(evaluator.eval_results, f)

if __name__ == "__main__":
    model_type = 'GPT'
    file_baseline = '20250730_221825'
    file_lowspa = '20250730_215710'
    main(model_type=model_type, 
         file_baseline=file_baseline, 
         file_lowspa=file_lowspa)