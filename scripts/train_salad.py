"""This script is used to train a model using the SALAD framework.
"""
import os, sys
import yaml
from datetime import datetime
import shutil
import transformers
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.trainer_salad import SALADTrainer
from salad.register import get_model, get_data

transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

root = get_parent_path(lvl=1)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha_rate', type=float, default=None, help='Alpha Rate')
    parser.add_argument('--beta_rate', type=float, default=None, help='Beta Rate')

    return parser.parse_args()

def main(cfg_version: str, 
         path_cfg: str,
         path_cfg_model: str,
         alpha_rate: float,
         beta_rate: float) -> None:
    
    # load the config
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)
    
    if alpha_rate is not None and beta_rate is not None:
        for layer in cfg['layers']:
            layer['params']['alpha_dict']['rate_decay'] = alpha_rate
            layer['params']['beta_dict']['rate_decay'] = beta_rate 

    seed = cfg['seed']
    set_seed(seed)

    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_folder = os.path.join(root, 'data', 'salad', cfg_version, folder_name)
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
    
    # get the data loader
    model = get_model(path_cfg_model)
    data = get_data(cfg['seed_for_shuffle'])

    ddp_trainer = SALADTrainer(model, data, cfg)
    ddp_trainer.train(path_folder=path_folder)
    
if __name__ == "__main__":
    args = parse_args()

    cfg_version = 'llama_60m'
    path_cfg = os.path.join(root, 'scripts', 'configs', cfg_version+'.yaml')
    path_cfg_model = os.path.join(root, 'scripts', 'configs', cfg_version+'_model.json')


    main(cfg_version, path_cfg, path_cfg_model, args.alpha_rate, args.beta_rate)