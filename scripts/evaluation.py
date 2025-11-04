"""Evaluate the models trained with Salad on the validation set.
"""
import os, sys
import yaml
import pickle
import torch
import copy
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator

root = get_parent_path(lvl=1)

def main(cfg_version: str,
         path_folder: str,
         nr_remove: list,
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
    # list all files in the folder
    # and load dictionary LL and SS from all files starting with 'matrix_'
    # at last, combine them into one dictionary
    LL = {}
    SS = {}
    files = os.listdir(path_folder)
    rank_files = [f for f in files if f.startswith('matrix')]
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            LL[key] = LL_part[key]
            SS[key] = SS_part[key]

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

        rate_sparsity[key] = torch.sum(_S != 0).item() / _S.numel()  # actually it is density

        _, s, _ = torch.linalg.svd(_L, full_matrices=False)
        energy = torch.cumsum(s, dim=0) / torch.sum(s)
        rank = torch.sum(energy < energy_quantile[key]).item() + 1
        rank_quantile_energy[key] = rank / len(s)

        if key.endswith('o_proj'):
            rank_quantile_specify[key] = min(rank_cfg['o_proj'], rank_quantile_energy[key])
        elif key.endswith('q_proj'):
            rank_quantile_specify[key] = min(rank_cfg['q_proj'], rank_quantile_energy[key])
        elif key.endswith('k_proj'):
            rank_quantile_specify[key] = min(rank_cfg['k_proj'], rank_quantile_energy[key])
        elif key.endswith('v_proj'):
            rank_quantile_specify[key] = min(rank_cfg['v_proj'], rank_quantile_energy[key])
        elif key.endswith('gate_proj'):
            rank_quantile_specify[key] = min(rank_cfg['gate_proj'], rank_quantile_energy[key])
        elif key.endswith('down_proj'): 
            rank_quantile_specify[key] = min(rank_cfg['down_proj'], rank_quantile_energy[key])
        elif key.endswith('up_proj'):
            rank_quantile_specify[key] = min(rank_cfg['up_proj'], rank_quantile_energy[key])

        rank_diff[key] = rank_quantile_energy[key] - rank_quantile_specify[key]

    rank_quantile_partial_list = [
        copy.deepcopy(rank_quantile_specify) for _ in range(len(nr_remove))
    ]
    # sort according to rank diff
    sorted_layers = sorted(rank_diff.items(), key=lambda item: item[1], reverse=True)
    for i in range(len(nr_remove)):
        _nr_remove = nr_remove[i]
        for j in range(_nr_remove):
            layer_name = sorted_layers[j][0]
            rank_quantile_partial_list[i][layer_name] = rank_quantile_energy[layer_name]

    # for i in range(nr_remove):
    #     layer_name = sorted_layers[i][0]
    #     rank_quantile_partial[layer_name] = rank_quantile_energy[layer_name]

    ex_layers = get_ex_layers(layers, model, LL, SS, nr_remove[0])

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
                               rank_quantile_partial_list=rank_quantile_partial_list,
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
    # file = '20251009_205606'

    # cfg_version = 'llama_60m'
    # file = '20251028_001027'

    # cfg_version = 'llama_130m'
    # file = '20251006_223931'

    # cfg_version = 'llama_350m'
    # file = '20251012_120507'

    cfg_version = 'llama_1b'
    file = '20251016_233939'

    # setting for 350m
    # rank_cfg = {
    #     'o_proj': 0.25,
    #     'q_proj': 0.25,
    #     'k_proj': 0.25,
    #     'v_proj': 0.25,
    #     'gate_proj': 0.35,
    #     'down_proj': 0.35,
    #     'up_proj': 0.35
    # }

    # setting for 1b
    rank_cfg = {
        'o_proj': 0.25,
        'q_proj': 0.25,
        'k_proj': 0.25,
        'v_proj': 0.25,
        'gate_proj': 0.35,
        'down_proj': 0.35,
        'up_proj': 0.35
    }

    path_folder = os.path.join(root, 'data', 'salad', cfg_version, file)
    main(cfg_version, path_folder, nr_remove=[3, 6, 9], rank_cfg=rank_cfg)
    # main(cfg_version, path_folder, nr_remove=[35, 40, 45], rank_cfg=rank_cfg)  # this setting is only for 130m model
    