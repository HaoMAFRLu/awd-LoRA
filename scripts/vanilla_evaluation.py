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

root = get_parent_path(lvl=1)

def ddp_setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def main(path_part: dict,
         rank: int=0) -> None:
    # load the config
    model_type = path_part['model_type']
    folder = path_part['folder']
    file = path_part['file']

    path_folder = os.path.join(root, 'data', folder, model_type, file)
    path_cfg = os.path.join(path_folder, model_type+'.yaml')
    path_cfg_model = os.path.join(path_folder, model_type+'_model.json')

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
    model.to(torch.bfloat16)
    print(f'[rank {rank}] Model loaded.')

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

    print(f'[rank {rank}] Setting up evaluator...')
    evaluator = CrossEvaluator(model_type=model_type,
                               model=model,
                               train_loader=train_loader,
                               test_loader=val_loader,
                               pad_idx=pad_idx,
                               batch_size=10)
    print(f'[rank {rank}] Evaluator ready.')
    
    evaluator.collect_model_results()  # evaluate the original model and store the results
    data = {
            'eval_train_results': evaluator.model_results_train,
            'eval_test_results': evaluator.model_results_test
        }

    with open(os.path.join(path_folder, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    MODEL_TYPES = [
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = ['vanilla']

    print('Setting up DDP...')
    rank, world_size = ddp_setup()
    print(f'DDP setup done. World size: {world_size}.')
    # rank = 0
    
    files = ['20251209_232356']
    
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

        main(path_part, rank=rank)

        print(f'[rank {rank}] Finished folder: {file}')
        print(f'[rank {rank}] ----------{nr}/{len(my_files)}----------')

    if rank == 0:
        print('All ranks done.')
    
    dist.destroy_process_group()