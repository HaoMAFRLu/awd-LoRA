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
from salad.simple_timer import SimpleTimer

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
    
    gamma_list = [round(x, 2) for x in np.arange(0.20, 0.90, 0.05)]

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
                LL[key] = LL_part[key].to(device)
                SS[key] = SS_part[key].to(device)
                layers.append(key)

    # nr_total_params = sum(p.numel() for p in model.parameters())
    
    print(f'[rank {rank}] Preparing data loaders...')
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    pad_idx = tokenizer.pad_token_id
    # get the data loader
    val_loader = get_eval_data(cfg, 'validation',
                             tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    train_loader = get_eval_data(cfg, 'train',
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

    timers = {
        'time': SimpleTimer('time'),
    }

    for params in params_tgt:
        
        print(f'[rank {rank}] Processing target params: {params}M')

        results = []

        if params < uia.nr_params_total / 1e6: # the target params should be smaller than the capacity

            for gamma in gamma_list:
                print(f'[rank {rank}]   Trying gamma: {gamma}')
                gamma = np.clip(gamma, 0, 1)

                rank_quantile, rate_density, return_state = uia.allocate(params_tgt=params, gamma=gamma)        
                rank_quantile, rate_density = uia.post_allocate(rank_quantile, rate_density, params_tgt=params) 

                if return_state == 0:  # only evaluate the model if allocation is successful
                    with timers['time']:
                        outputs = evaluator._eval_par_lowrank_lowrank_sparsity(val_loader, rank_quantile, rate_density) 
                    tt = timers['time'].total/60
                    print(f'[rank {rank}]   Time taken: {tt:.1f} mins')
                    timers['time'].reset()

                    nr_params = uia.check_params(rank_quantile, rate_density)

                    sepc = {}
                    for layer_name in rank_quantile.keys():
                        sepc['model.'+layer_name] = {
                            "rank_ratio": float(rank_quantile[layer_name]),
                            "density": float(rate_density[layer_name])
                        }
                    flops_cost = estimate_per_token_flops(model, sepc)
                    memory_cost = estimate_inference_memory_cost(model, sepc)
                    
                    data = {
                        'gamma': gamma,
                        'rank_quantile': rank_quantile,
                        'rate_density': rate_density,
                        'ppl': outputs['ppl'],
                        'nr_params': nr_params,
                        'flops_cost': flops_cost,
                        'memory_cost': memory_cost,
                    }

                    results.append(data)

            if len(results) > 0:
                file_name = f'comp_kappa_{params}'
                with open(os.path.join(path_folder, file_name+'.pkl'), 'wb') as f:
                    pickle.dump(results, f)

            
if __name__ == "__main__":
    hf_login_once()
    rank, world_size = ddp_setup()

    # params_tgts = {
    #     'llama_9m':   [10.5, 8.5, 8.2],
    #     'llama_60m':  [65.5, 60.5, 55.5, 50.5, 45.5, 40.5, 35.5],
    #     'llama_130m': [150.5, 140.5, 130.5, 120.5, 110.5, 100.5, 90.5, 80.5],
    #     'llama_350m': [400.5, 360.5, 320.5, 280.5, 240.5, 200.5, 160.5, 120.5],
    #     'llama_1b':   [1500.5, 1300.5, 1100.5, 900.5, 700.5, 600.5, 500.5, 400.5],
    # }

    params_tgts = {
        'llama_9m':   [12.5, 9.5, 8.5, 6.5, 5.5],
        'llama_60m':  [64.5, 60.5, 56.5, 52.5, 48.5, 44.5, 40.5, 36.5],
        'llama_130m': [150.5, 140.5, 130.5, 120.5, 110.5, 100.5, 90.5, 80.5],
        'llama_350m': [400.5, 360.5, 320.5, 280.5, 250.5, 230.5, 210.5, 190.5, 170.5, 150.5],
        'llama_1b':   [1500.5, 1300.5, 1100.5, 900.5, 780.5, 730.5, 680.5, 630.5, 580.5, 530.5],
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
