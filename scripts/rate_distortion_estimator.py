"""This script is used to analyze rate-distortion trade-offs
"""
import sys, os
import pickle
import numpy as np
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.uia import UIA
from salad.slr_block import *
from salad.memory_estimator import *
from salad.flops_estimator import *
from salad.cross_evaluator import CrossEvaluator

root = get_parent_path(lvl=1)

def ddp_setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def destory():
    dist.destroy_process_group()

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str,
         rank: int=0,
         params_tgt: list=[1.0],
         precision: float=torch.bfloat16) -> None:
    
    gamma_list = [round(x, 2) for x in np.arange(0.20, 0.90, 0.1)]

    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file)
    path_cfg = os.path.join(path_folder, MODEL_TYPE+'.yaml')
    path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')
    
    with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
        layer_info = pickle.load(f)
    
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)

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
    if 'vanilla' in FOLDER:
        rank_files = [f for f in files if f.startswith('rpca_X')]
    else:
        rank_files = [f for f in files if f.startswith('matrix')]
    
    layers = []
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            if 'embed' in key or 'lm_head' in key:
                continue
            else:
                LL[key] = LL_part[key]
                SS[key] = SS_part[key]
                layers.append(key)

    # nr_total_params = sum(p.numel() for p in model.parameters())
    
    print(f'[rank {rank}] Preparing data loaders...')
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    pad_idx = tokenizer.pad_token_id
    # get the data loader
    val_loader = get_eval_data('validation', seed_for_shuffle=cfg['seed_for_shuffle'],
                             tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    train_loader = get_eval_data('train', seed_for_shuffle=cfg['seed_for_shuffle'],
                              tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    print(f'[rank {rank}] Data loaders ready.')

    uia = UIA(LL, SS, model, 
              layer_info=layer_info, 
              rate=100000000.0,
              rank=rank)
   
    print(f'[rank {rank}] Setting up evaluator...')
    evaluator = CrossEvaluator(model_type=MODEL_TYPE,
                               model=model,
                               train_loader=train_loader,
                               test_loader=val_loader,
                               layers=layers,
                               pad_idx=pad_idx,
                               LL=LL,
                               SS=SS,
                               layer_dim=uia.dim,
                               batch_size=10)
    print(f'[rank {rank}] Evaluator ready.')

    for params in params_tgt:
        
        print(f'[rank {rank}] Processing target params: {params}M')

        opt_ppl = 10000
        opt_rank_quantile = None
        opt_rate_density = None
        opt_gamma = None

        for gamma in gamma_list:
            print(f'[rank {rank}]   Trying gamma: {gamma}')
            gamma = np.clip(gamma, 0, 1)

            _rank_quantile, _rate_density = uia.allocate(params_tgt=params, gamma=gamma)
            rank_quantile, rate_density = uia.post_allocate(_rank_quantile, _rate_density, params_tgt=params) 

            results = evaluator._eval_par_lowrank_lowrank_sparsity(val_loader, rank_quantile, rate_density) 
            ppl = results['ppl']

            if ppl < opt_ppl:
                opt_ppl = ppl
                opt_rank_quantile = rank_quantile
                opt_rate_density = rate_density
                opt_gamma = gamma

        print(f'[rank {rank}] Finished target params: {params}M')
        nr_params = uia.check_params(opt_rank_quantile, opt_rate_density)
        sepc = {}
        for layer_name in opt_rank_quantile.keys():
            sepc['model.'+layer_name] = {
                "rank_ratio": float(opt_rank_quantile[layer_name]),
                "density": float(opt_rate_density[layer_name])
            }
        flops_cost = estimate_per_token_flops(model, sepc)
        memory_cost = estimate_inference_memory_cost(model, sepc)

        data = {
            'gamma': opt_gamma,
            'ppl': opt_ppl,
            'nr_params': nr_params,
            'flops_cost': flops_cost,
            'memory_cost': memory_cost,
        }

        file_name = f'distortion_{params}'
        with open(os.path.join(path_folder, file_name+'.pkl'), 'wb') as f:
            pickle.dump(data, f)

            
if __name__ == "__main__":
    hf_login_once()
    rank, world_size = ddp_setup()

    params_tgts = {
        'llama_9m':   [8.5, 8.2],
        'llama_60m':  [49.5, 46.5, 43.5, 40.5, 37.5],
        'llama_130m': [147.5, 137.5, 127.5, 117.5, 107.5, 97.5],
        'llama_350m': [253.5, 233.5, 213.5, 193.5, 173.5, 153.5],
        'llama_1b':   [646.5, 609.5],
    }

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
        'incl_embedding',
        'vanilla_bf16',  
        'baseline_fp32',
    ]

    files = [
        # '20251209_104454',  # for quick test
        # '20251209_204846',
        # '20251204_135646',
        '20251202_164626',
        '20251209_232356',
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
             params_tgt=params_tgts[MODEL_TYPE],
             precision=torch.bfloat16)

        print(f'[rank {rank}] Finished folder: {file}')
        print(f'[rank {rank}] ----------{nr}/{len(my_files)}----------')
    
    destory()