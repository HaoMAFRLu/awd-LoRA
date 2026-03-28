"""This script is used to train a model using the SALAD framework.
"""
import os, sys

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

import yaml
from datetime import datetime
import shutil
import transformers
import argparse
import torch.distributed as dist
import socket 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.trainer_salad import SALADTrainer
from salad.register import get_model, get_data

transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

root = get_parent_path(lvl=1)

def _init_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rho', type=float, default=None, help='Rho')
    parser.add_argument('--alpha_rate', type=float, default=None, help='Alpha Rate')
    parser.add_argument('--beta_rate', type=float, default=None, help='Beta Rate')
    parser.add_argument('--dalpha', type=float, default=None, help='Delta Alpha')
    parser.add_argument('--dbeta', type=float, default=None, help='Delta Beta')

    return parser.parse_args()

def main(cfg_version: str, 
         path_cfg: str,
         path_cfg_model: str,
         folder: str,
         rho: float,
         alpha_rate: float,
         beta_rate: float,
         dalpha: float,
         dbeta: float,
         exclude_layers: list=None) -> None:
    
    rank, world_size = _init_distributed()
    
    # fk the error 429!!!!!!
    # fk the error 502!!!!!!
    hf_login_once()

    print(f'[Rank {rank}] initializing...')
    print(f'[Rank {rank}]: Total world size: {world_size}')
    print(f"[Rank {rank}]: {dist.get_rank()} | [HOST]: {socket.gethostname()}")
    
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # load the config
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)
    
    target_layers = [entry['name'] for entry in cfg['layers']]

    if rho is not None and alpha_rate is not None and beta_rate is not None:
        for layer in cfg['layers']:
            if 'embed' in layer['name'] or 'lm_head' in layer['name']:
                layer['params']['alpha_dict']['rate_decay'] = alpha_rate
                layer['params']['beta_dict']['rate_decay'] = beta_rate 
                layer['params']['alpha_dict']['drate'] = dalpha
                layer['params']['beta_dict']['drate'] = dbeta
            else:
                layer['params']['rho_dict']['rho'] = rho
                layer['params']['alpha_dict']['rate_decay'] = alpha_rate
                layer['params']['beta_dict']['rate_decay'] = beta_rate 
                layer['params']['alpha_dict']['drate'] = dalpha
                layer['params']['beta_dict']['drate'] = dbeta

    if exclude_layers is not None:
        cfg['layers'] = [
            layer for layer in cfg['layers']
            if not any(ex in layer['name'] for ex in exclude_layers)
        ]

    seed = cfg['seed']
    set_seed(seed)

    if rank == 0:
        # create a unique folder name based on current datetime
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_folder = os.path.join(root, 'data', folder, cfg_version, folder_name)
        mkdir(path_folder)
        shutil.copytree(os.path.join(root, 'salad'), 
                        os.path.join(path_folder, 'salad'), 
                        dirs_exist_ok=True, 
                        copy_function=shutil.copy2) 
    
        # shutil.copy(path_cfg, path_folder)
        output_path = os.path.join(path_folder, cfg_version+'.yaml')
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        shutil.copy(path_cfg_model, path_folder)
    else:
        folder_name = None
    
    # broadcast the path folder to all ranks
    path_folder_list = [path_folder if rank == 0 else None]
    dist.broadcast_object_list(path_folder_list, src=0)
    path_folder = path_folder_list[0]

    # get the data loader
    model = get_model(path_cfg_model)

    # time.sleep(2.0 * rank)  # 3s per rank is a good starting point
    data = get_data(cfg["seed_for_shuffle"])
    # dist.barrier()

    ddp_trainer = SALADTrainer(model, data, cfg, 
                               rank=rank, 
                               world_size=world_size,
                               folder_name=folder_name)
    ddp_trainer.train(path_folder=path_folder)
    
if __name__ == "__main__":
    args = parse_args()

    cfg_version = 'llama_130m'
    folder = 'review_wall_clock'
    path_cfg = os.path.join(root, 'scripts', 'configs', cfg_version+'.yaml')
    path_cfg_model = os.path.join(root, 'scripts', 'configs', cfg_version+'_model.json')

    # exclude_layers = ['q_proj', 'k_proj']
    exclude_layers = None

    main(cfg_version, path_cfg, path_cfg_model, folder,
         args.rho, args.alpha_rate, args.beta_rate, args.dalpha, args.dbeta,
         exclude_layers=exclude_layers)