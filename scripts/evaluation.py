"""Evaluate the models trained with Salad on the validation set.
"""
import os, sys
import yaml
import pickle
import io
import torch
import copy
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator

root = get_parent_path(lvl=1)

# def load_model(model: torch.nn.Module, pth: str) -> torch.nn.Module:
#     """
#     Load the model from the given path.
    
#     Args:
#         model (torch.nn.Module): The model to load.
#         pth (str): Path to the model checkpoint.
    
#     Returns:
#         torch.nn.Module: The loaded model.
#     """
#     ckpt = torch.load(pth, map_location="cpu")
#     state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
#     clean_sd = {}
#     for k, v in state_dict.items():
#         while k.startswith("module."):
#             k = k[len("module."):]
#         clean_sd[k] = v

#     model.load_state_dict(clean_sd, strict=True)

def load_model(model, pth):
    # names = get_linear_layers_name(model)
    # p = get_weight(model, names[0]).clone()

    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # DDP?
    is_ddp = isinstance(model, DDP)
    prefix = "module."

    has_module_prefix = all(k.startswith(prefix) for k in sd.keys())

    if has_module_prefix and not is_ddp:
        # from DDP ckpt -> model
        sd = {k[len(prefix):]: v for k, v in sd.items()}
    elif (not has_module_prefix) and is_ddp:
        # from ckpt -> DDP model
        sd = {prefix + k: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_model] missing:", missing)
        print("[load_model] unexpected:", unexpected)

    # pp = get_weight(model, names[0]).clone()

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

def get_eval_data(split: str, 
                  seed_for_shuffle: int = 42, 
                  tokenizer=None, 
                  max_length=1024,
                  batch_size: int = 32):
    _data = get_data(seed_for_shuffle, split=split)
    _data_mapped = _data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    _data_mapped.batch = lambda batch_size: batch_fn(_data_mapped, batch_size)
    return _data_mapped

def get_ex_layers(layers: list, model, LL: dict, SS: dict, nr_remove: int) -> list:
    ex_layers = []
    loss = {}
    _list = []
    for layer in layers:
        L = LL[layer]
        S = SS[layer]

        X = model.get_submodule('model.'+layer).weight.data
        loss[layer] = torch.norm(X - L - S, p='fro').item() / X.numel()  # average per element
        _list.append(torch.norm(X - L - S, p='fro').item())

    sorted_layers = sorted(loss.items(), key=lambda item: item[1], reverse=True)
    for i in range(nr_remove):
        ex_layers.append(sorted_layers[i][0])

    return ex_layers

def main(cfg_version: str,
         path_folder: str,
         nr_remove: int,
         rank_cfg: dict) -> None:
    # load the config
    path_cfg = os.path.join(path_folder, cfg_version+'.yaml')
    path_cfg_model = os.path.join(path_folder, cfg_version+'_model.json')

    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']
    max_length = cfg['max_length']
    batch_size = cfg['batch_size']
    set_seed(seed)

    # get the model and load the checkpoint
    model = get_model(path_cfg_model)

    load_model(model, os.path.join(path_folder, 'model.pth'))
    LL, SS = get_lowspa_layers(os.path.join(path_folder, 'matrix.pkl'))

    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    pad_idx = tokenizer.pad_token_id
    # get the data loader
    val_loader = get_eval_data('validation', seed_for_shuffle=cfg['seed_for_shuffle'],
                             tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    train_loader = get_eval_data('train', seed_for_shuffle=cfg['seed_for_shuffle'],
                              tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    
    layers = [entry['name'] for entry in cfg['layers']]
    
    rank_quantile_target = {entry['name']: entry['params']['rate_rank'] for entry in cfg['layers']}

    energy_quantile = {entry['name']: entry['params']['energy'] for entry in cfg['layers']}
    
    rank_quantile_energy = {}
    rank_quantile_specify = {}
    rank_diff = {}
    rate_sparsity = {}
    layer_dim = {}

    for key in energy_quantile:
        _L = LL[key]
        _S = SS[key]
        layer_dim[key] = _L.shape

        rate_sparsity[key] = torch.sum(_S != 0).item() / _S.numel()

        _, s, _ = torch.linalg.svd(_L, full_matrices=False)
        energy = torch.cumsum(s, dim=0) / torch.sum(s)
        rank = torch.sum(energy < energy_quantile[key]).item() + 1
        rank_quantile_energy[key] = rank / len(s)

        if key.endswith('o_proj'):
            rank_quantile_specify[key] = rank_cfg['o_proj']
        elif key.endswith('q_proj'):
            rank_quantile_specify[key] = rank_cfg['q_proj']
        elif key.endswith('k_proj'):
            rank_quantile_specify[key] = rank_cfg['k_proj']
        elif key.endswith('v_proj'):
            rank_quantile_specify[key] = rank_cfg['v_proj']
        elif key.endswith('gate_proj'):
            rank_quantile_specify[key] = rank_cfg['gate_proj']
        elif key.endswith('down_proj'): 
            rank_quantile_specify[key] = rank_cfg['down_proj']
        elif key.endswith('up_proj'):
            rank_quantile_specify[key] = rank_cfg['up_proj']

        rank_diff[key] = rank_quantile_energy[key] - rank_quantile_specify[key]

    rank_quantile_partial = copy.deepcopy(rank_quantile_specify)
    # sort according to rank diff
    sorted_layers = sorted(rank_diff.items(), key=lambda item: item[1], reverse=True)
    for i in range(nr_remove):
        layer_name = sorted_layers[i][0]
        rank_quantile_partial[layer_name] = rank_quantile_energy[layer_name]

    ex_layers = get_ex_layers(layers, model, LL, SS, nr_remove)

    evaluator = CrossEvaluator(model_type=cfg_version,
                               model=model,
                               train_loader=train_loader,
                               test_loader=val_loader,
                               layers=layers,
                               ex_layers=ex_layers,
                               pad_idx=pad_idx,
                               LL=LL,
                               SS=SS,
                               rank_quantile_target=rank_quantile_target,
                               rank_quantile_energy=rank_quantile_energy,
                               rank_quantile_specify=rank_quantile_specify,
                               rank_quantile_partial=rank_quantile_partial,
                               rate_sparsity=rate_sparsity,
                               layer_dim=layer_dim,
                               batch_size=10)
    
    evaluator.collect_results()
    data = {
        'eval_train_results': evaluator.eval_train_results,
        'eval_test_results': evaluator.eval_test_results
    }
    with open(os.path.join(path_folder, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # cfg_version = 'llama_9m'
    # file = '20250903_165259'

    # cfg_version = 'llama_60m'
    # file = '20251006_143955'
    # file = '20251005_130200'

    cfg_version = 'llama_130m'
    file = '20251006_223931'

    rank_cfg = {
        'o_proj': 0.30,
        'q_proj': 0.30,
        'k_proj': 0.30,
        'v_proj': 0.30,
        'gate_proj': 0.35,
        'down_proj': 0.35,
        'up_proj': 0.35
    }
    path_folder = os.path.join(root, 'data', 'salad', cfg_version, file)
    main(cfg_version, path_folder, nr_remove=15, rank_cfg=rank_cfg)
    