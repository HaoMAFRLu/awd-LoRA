"""Evaluate the models trained with Salad on the validation set.
"""
import os, sys
import yaml
import pickle
import torch
import copy
from transformers import AutoTokenizer
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model, get_data
from salad.cross_evaluator import CrossEvaluator
from salad.uia import UIA
from salad.simple_timer import SimpleTimer

root = get_parent_path(lvl=1)

def ddp_setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def main(cfg_version: str,
         path_folder: str,
         params_tgt: list,
         gamma_list: list,
         rank: int=0) -> None:
    # load the config
    path_cfg = os.path.join(path_folder, cfg_version+'.yaml')
    path_cfg_model = os.path.join(path_folder, cfg_version+'_model.json')

    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']
    max_length = cfg['max_length']
    batch_size = cfg['batch_size']
    set_seed(seed)

    # print current device
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    print(f'[rank {rank}] Current device: {torch.cuda.current_device()}')

    # get the model and load the checkpoint
    print(f'[rank {rank}] Loading model...')
    model = get_model(path_cfg_model)
    load_model(model, os.path.join(path_folder, 'model.pth'))
    model.to(device)
    print(f'[rank {rank}] Model loaded.')
    # list all files in the folder
    # and load dictionary LL and SS from all files starting with 'matrix_'
    # at last, combine them into one dictionary
    print(f'[rank {rank}] Loading low-rank and sparse components...')
    LL = {}
    SS = {}
    files = os.listdir(path_folder)
    rank_files = [f for f in files if f.startswith('matrix')]
    for f in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
        for key in LL_part:
            LL[key] = LL_part[key].to(device)
            SS[key] = SS_part[key].to(device)
    print(f'[rank {rank}] Low-rank and sparse components loaded.')

    with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
        layer_info = pickle.load(f)
    
    # get the tokenizer
    print(f'[rank {rank}] Preparing data loaders...')
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    pad_idx = tokenizer.pad_token_id
    # get the data loader
    val_loader = get_eval_data('validation', seed_for_shuffle=cfg['seed_for_shuffle'],
                             tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    train_loader = get_eval_data('train', seed_for_shuffle=cfg['seed_for_shuffle'],
                              tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    print(f'[rank {rank}] Data loaders ready.')


    print(f'[rank {rank}] Read layers...')
    layers = [entry['name'] for entry in cfg['layers']]
    print(f'[rank {rank}] Layers read.')

    print(f'[rank {rank}] Setting up UIA...')
    uia = UIA(LL, SS, model, 
              layer_info=layer_info, 
              rate=100000000.0,
              rank=rank)
    print(f'[rank {rank}] UIA ready.')

    print(f'[rank {rank}] Setting up evaluator...')
    evaluator = CrossEvaluator(model_type=cfg_version,
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
        'energy': SimpleTimer('energy'),
        'original': SimpleTimer('original')
    }

    print(f'[rank {rank}] Collecting results for full-rank + sparsity model...')
    with timers['energy']:
        evaluator.collect_single_results(uia.rank_quantile_energy,
                                        uia.rate_density)  # evaluate the full-rank + sparsity model and store the results
    print(f'[rank {rank}] Energy-based evaluation time: {timers["energy"].total/60:.1f} mins.')

    print(f'[rank {rank}] Collecting results for original model...')
    with timers['original']:
        evaluator.collect_model_results()  # evaluate the original model and store the results
    print(f'[rank {rank}] Original model evaluation time: {timers["original"].total/60:.1f} mins.')    


    for gamma in gamma_list:
        rank_quantile_list = []
        rate_density_list = []

        for params in params_tgt:
            rank_quantile, rate_density = uia.allocate(params_tgt=params, gamma=gamma)
            rank_quantile_list.append(rank_quantile)
            rate_density_list.append(rate_density)

        for i in range(len(rank_quantile_list)):
            nr_params = uia.check_params(rank_quantile_list[i], rate_density_list[i])
            print(f'Number of parameters: {nr_params/1e6:.2f} Million')
        
        evaluator.collect_results(rank_quantile_list, rate_density_list)

        data = {
            'eval_train_results': evaluator.eval_train_results,
            'eval_test_results': evaluator.eval_test_results
        }
        # with open(os.path.join(path_folder, 'eval_results.pkl'), 'wb') as f:
        #     pickle.dump(data, f)

        with open(os.path.join(path_folder, 'eval_results_'+str(gamma)+'.pkl'), 'wb') as f:
            pickle.dump(data, f)

    dist.destroy_process_group()
    
if __name__ == "__main__":
    params_tgt = {
        'llama_9m':   [7.5, 6.5, 5.5],
        'llama_60m':  [49.5, 46.5, 44.5],
        'llama_130m': [99.5, 97.5, 94.5],
        'llama_350m': [194.5, 185.5],
        'llama_1b':   [669.5, 646.5, 609.5],
    }
    
    MODEL_TYPE = 'llama_350m'
    FOLDERS = ['ablation']
    # gamma_list = [1.0]
    gamma_list = np.arange(0.4, 1.05, 0.05).tolist()

    print('Setting up DDP...')
    rank, world_size = ddp_setup()
    print(f'DDP setup done. World size: {world_size}.')
    # rank = 0

    files = []
    for FOLDER in FOLDERS:
        _path = os.path.join(root, 'data', FOLDER, MODEL_TYPE)
        files.extend(os.listdir(_path))

    # _files = []
    # for file in files:
    #     if file.startswith('20251120'):
    #         _files.append(file)
    
    _files = [
        # '20251110_094016',
        # '20251104_131243',
        # '20251107_105320',
        # '20251127_103821',
        # '20251127_130254',
        # '20251016_233939',
        '20251128_134521',
        '20251128_134339',
        '20251128_134206',
    ]

    if rank == 0:
        # print all file names
        print('All files to process:')
        for f in _files:
            print(f)

    _files = sorted(_files)

    my_files = _files[rank::world_size]
    # my_files = files[6]

    if isinstance(my_files, str):
        my_files = [my_files]

    # if rank == 0:
    #     # my_files = ['20251029_162352']
    #     my_files = ['20251104_130755']    
    #     # my_files = ['20251009_205606']
    # else:
    #     my_files = []   

    if not my_files:
        print(f"[rank {rank}] No file assigned. Exit.")
        sys.exit(0)

    nr = 0
    for file in my_files:
        nr += 1
        print(f'[rank {rank}] Processing folder: {file}')
    
        for FOLDER in FOLDERS:
            path = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file)
            if os.path.exists(path):
                path_folder = path
                break

        main(MODEL_TYPE,
             path_folder,
             params_tgt[MODEL_TYPE],
             gamma_list=gamma_list,
             rank=rank)

        print(f'[rank {rank}] Finished folder: {file}')
        print(f'[rank {rank}] ----------{nr}/{len(my_files)}----------')

    if rank == 0:
        print('All ranks done.')