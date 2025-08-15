"""Evaluate the models trained with Salad on the validation set.
"""
import os, sys
import yaml
import pickle
import io
import torch
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

def main(cfg_version: str,
         path_folder: str) -> None:
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

    # exclude_layers = ['transformer.h.0.mlp.c_fc.weight',
    #                  'transformer.h.0.mlp.c_proj.weight']
    
    exclude_layers = []
    layers = [layer for layer in layers if layer not in exclude_layers]

    evaluator = CrossEvaluator(model_type=cfg_version,
                               model=model,
                               train_loader=train_loader,
                               test_loader=val_loader,
                               layers=layers,
                               pad_idx=pad_idx,
                               LL=LL,
                               SS=SS,
                               rank_quantile=0.25,
                               batch_size=10)
    
    evaluator.collect_results()
    data = {
        'eval_train_results': evaluator.eval_train_results,
        'eval_test_results': evaluator.eval_test_results
    }
    with open(os.path.join(path_folder, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    cfg_version = 'llama_60m'
    file = '20250814_150324'
    path_folder = os.path.join(root, 'data', 'salad', cfg_version, file)
    main(cfg_version, path_folder)
    