"""This script is used to train a model using the SALAD framework.
"""
import os, sys
import yaml
from datetime import datetime
import shutil
import transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.trainer_salad import SALADTrainer
from salad.register import get_model, get_data

transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

root = get_parent_path(lvl=1)

def main(cfg_version: str, 
         path_cfg: str,
         path_cfg_model) -> None:
    # load the config
    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)
    
    seed = cfg['seed']
    set_seed(seed)

    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_folder = os.path.join(root, 'data', 'salad', cfg_version, folder_name)
    mkdir(path_folder)
    shutil.copytree(os.path.join(root, 'salad'), 
                    os.path.join(path_folder, 'salad'), 
                    dirs_exist_ok=True, 
                    copy_function=shutil.copy2) 
    shutil.copy(path_cfg, path_folder)
    shutil.copy(path_cfg_model, path_folder)
    
    # get the data loader
    model = get_model(path_cfg_model)
    data = get_data(cfg['seed_for_shuffle'])

    ddp_trainer = SALADTrainer(model, data, cfg)
    ddp_trainer.train(path_folder=path_folder)
    
if __name__ == "__main__":
    cfg_version = 'llama_350m'
    path_cfg = os.path.join(root, 'scripts', 'configs', cfg_version+'.yaml')
    path_cfg_model = os.path.join(root, 'scripts', 'configs', cfg_version+'_model.json')
    main(cfg_version, path_cfg, path_cfg_model)