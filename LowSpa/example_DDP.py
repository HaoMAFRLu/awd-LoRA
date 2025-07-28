import os, sys
import yaml
from datetime import datetime
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import *
from lowspa_ddp.lowspa_trainer import LowSpaTrainer
from lowspa_ddp.model_register import get_model_and_dataloader
from lowspa_ddp.utils import *

root = get_parent_path(lvl=1)

def main(model_type: str,
         path_config: str) -> None:
    # load the config
    with open(path_config) as f:
        cfg = yaml.safe_load(f)
    
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_folder = os.path.join(root, 'data', 'lowspa_ddp', model_type, folder_name)
    mkdir(path_folder)
    shutil.copy(path_config, path_folder)
    
    params = cfg.get('training')
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    num_workers = params['num_workers']
    # get the data loader
    model, train_loader, test_loader = get_model_and_dataloader(model_type, batch_size, num_workers)
    ddp_trainer = LowSpaTrainer(model, train_loader, cfg)
    ddp_trainer.train(num_epochs=num_epochs, path_folder=path_folder)
    
if __name__ == "__main__":
    model_type = 'GPT'
    config_version = 'gpt_test'

    path_config = os.path.join(root, 'lowspa_ddp', 'configs', config_version+'.yaml')
    main(model_type, path_config)