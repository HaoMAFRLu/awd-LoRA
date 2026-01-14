"""This script is used to analyze rate-distortion trade-offs
"""
import sys, os
import pickle
import numpy as np
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.uia import UIA
from salad.static_rpca import StaticRPCA
from salad.slr_block import *

root = get_parent_path(lvl=1)

def ddp_setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str,
         rank: int=0,
         precision: float=torch.bfloat16) -> None:
    
    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file)
    path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
    path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')
    
    with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
        layer_info = pickle.load(f)
    
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)

    layers = [entry['name'] for entry in cfg['layers']]

    seed = cfg['seed']
    max_length = cfg['max_length']
    batch_size = cfg['batch_size']
    set_seed(seed)

    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    print(f'[rank {rank}] Current device: {torch.cuda.current_device()}')

    # load the model
    model = get_model(path_cfg_model)
    model.to(precision)
    load_model(model, os.path.join(path_folder, 'model.pth'))
    model.to(device)

    LL = {}
    SS = {}
    files = os.listdir(path_folder)
    if FOLDER == 'vanilla':
        rank_files = [f for f in files if f.startswith('rpca_X')]
    else:
        rank_files = [f for f in files if f.startswith('matrix')]
        
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            if 'lm_head' in key:
                LL[key] = LL_part[key].to(device).t()
                SS[key] = SS_part[key].to(device).t()
            else:
                LL[key] = LL_part[key].to(device)
                SS[key] = SS_part[key].to(device)
    print(f'[rank {rank}] Low-rank and sparse components loaded.')

    uia = UIA(LL, SS, model, 
              layer_info=layer_info, 
              rate=100000000.0,
              rank=rank)
   
    params = 8.8  # the parameter budget
    gamma = 0.5  # the allocation ratio

    rank_quantile, rate_density, return_state = uia.allocate(params_tgt=params, gamma=gamma)

    if return_state == 0:  # 0 if the allocation succeeded
        rank_quantile, rate_density = uia.post_allocate(rank_quantile, rate_density, params_tgt=params) 

        abs_list = []

        nr_total_params = sum(p.numel() for p in model.parameters())
        print(f'Total model parameters: {nr_total_params/1e6:.2f}M')

        for layer_name in rank_quantile.keys():
            print(f'[rank {rank}] Processing layer: {layer_name}')

            r_ratio = rank_quantile[layer_name]
                
            L = LL[layer_name]
            S = SS[layer_name]

            m, n = L.shape

            r = max(1, int(min(m, n) * r_ratio))

            U, _s, Vh = torch.linalg.svd(L, full_matrices=False)
            A = U[:, :r] @ torch.diag(torch.sqrt(_s[:r]))
            B = torch.diag(torch.sqrt(_s[:r])) @ Vh[:r, :]

            abs_list.append(ABSFactor(name=layer_name, A=A, B=B, S=S))

        replaced = replace_linears(model, abs_list, strict=True)

        # target_names = ['model.'+x.name for x in abs_list]  # 你传入的层名
        # check_replaced_modules(model, target_names)

        nr_total_params = sum(p.numel() for p in model.parameters())
        print(f'Total model parameters: {nr_total_params/1e6:.2f}M')
    

if __name__ == "__main__":
    hf_login_once()
    rank, world_size = ddp_setup()

    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'vanilla',
        'baseline', 
        'incl_embedding'
    ]

    files = [
        '20251209_104454',
    ]

    if rank == 0:
        # print all file names
        print('All files to process:')
        for f in files:
            print(f)

    files = sorted(files)

    my_files = files[rank::world_size]
    if isinstance(my_files, str):
        my_files = [my_files]

    if not my_files:
        print(f"[rank {rank}] No file assigned. Exit.")
        sys.exit(0)

    nr = 0
    for file in my_files:
        nr += 1
        print(f'[rank {rank}] Processing folder: {file}')

        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)

        MODEL_TYPE = path_part['model_type']
        FOLDER = path_part['folder']

        path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file)

        main(MODEL_TYPE=MODEL_TYPE,
             FOLDER=FOLDER,
             file=file,
             rank=rank,
             precision=torch.bfloat16)

        print(f'[rank {rank}] Finished folder: {file}')
        print(f'[rank {rank}] ----------{nr}/{len(my_files)}----------')