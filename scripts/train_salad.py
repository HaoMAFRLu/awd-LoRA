"""This script is used to train a model using the SALAD framework.
"""
import os, sys

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

import json
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


def _write_effective_model_config(
    source_path: str,
    destination_path: str,
    training_config: dict,
) -> None:
    """Write the exact architecture config used for this training run."""
    with open(source_path, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    training_mode = training_config.get("training_mode")
    if training_mode == "consensus":
        consensus_config = training_config.get("consensus_salaad")
        if not isinstance(consensus_config, dict):
            raise ValueError(
                "training_mode='consensus' requires a consensus_salaad section"
            )
        components = consensus_config.get("components")
        if components is not None:
            model_consensus = model_config.get("consensus_salaad")
            if not isinstance(model_consensus, dict):
                raise ValueError(
                    "the model config requires a consensus_salaad section "
                    "when training config selects consensus components"
                )
            model_consensus["components"] = components
    elif training_mode == "loop":
        loop_config = training_config.get("loop")
        if not isinstance(loop_config, dict):
            raise ValueError("training_mode='loop' requires a loop section")

        counts = {
            "num_entry_blocks": loop_config.get("num_entry_blocks", 1),
            "num_blocks_per_loop": loop_config.get("num_blocks_per_loop"),
            "num_exit_blocks": loop_config.get("num_exit_blocks", 1),
            "num_loops": loop_config.get("num_loops"),
        }
        for name, value in counts.items():
            minimum = 1 if name in {"num_blocks_per_loop", "num_loops"} else 0
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(
                    f"loop.{name} must be an integer >= {minimum}, got {value!r}"
                )

        # A plain loop reuses the exact same physical blocks on every pass.
        # It therefore must not inherit loop-specific ConsensusLinear weights
        # from a consensus model config.
        model_config.pop("consensus_salaad", None)
        model_config["loop"] = counts
        model_config["num_hidden_layers"] = (
            counts["num_entry_blocks"]
            + counts["num_blocks_per_loop"]
            + counts["num_exit_blocks"]
        )
        model_config["use_cache"] = False

    with open(destination_path, "w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=4)
        file.write("\n")

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
    parser.add_argument('--cfg_version', type=str, default='llama_consensus_60m', help='Config version in configs/')
    parser.add_argument('--folder', type=str, default=None, help='Override output folder under data/')

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
    folder = folder or cfg.get('output_folder', 'review_wall_clock')
    if cfg.get("model_config"):
        path_cfg_model = os.path.join(root, 'configs', cfg["model_config"])

    if rho is not None and alpha_rate is not None and beta_rate is not None:
        if cfg.get('training_mode', 'salad') != 'salad':
            raise ValueError(
                "rho/alpha/beta command-line overrides require training_mode='salad'"
            )
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

    if exclude_layers is not None and cfg.get('training_mode', 'salad') == 'salad':
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
        _write_effective_model_config(
            path_cfg_model,
            os.path.join(path_folder, os.path.basename(path_cfg_model)),
            cfg,
        )
    else:
        folder_name = None
    
    # broadcast the path folder to all ranks
    path_folder_list = [path_folder if rank == 0 else None]
    dist.broadcast_object_list(path_folder_list, src=0)
    path_folder = path_folder_list[0]

    # Every rank loads the effective config saved with this run. This keeps
    # model construction and future checkpoint evaluation consistent when a
    # training config overrides the consensus components.
    effective_model_config = os.path.join(
        path_folder, os.path.basename(path_cfg_model)
    )
    model = get_model(effective_model_config)

    # time.sleep(2.0 * rank)  # 3s per rank is a good starting point
    data = get_data(cfg)
    # dist.barrier()

    ddp_trainer = SALADTrainer(model, data, cfg, 
                               rank=rank, 
                               world_size=world_size,
                               folder_name=folder_name)
    ddp_trainer.train(path_folder=path_folder)
    
if __name__ == "__main__":
    args = parse_args()

    cfg_version = args.cfg_version
    folder = args.folder
    path_cfg = os.path.join(root, 'configs', cfg_version+'.yaml')
    path_cfg_model = os.path.join(root, 'configs', cfg_version+'_model.json')

    # exclude_layers = ['q_proj', 'k_proj']
    exclude_layers = None

    main(cfg_version, path_cfg, path_cfg_model, folder,
         args.rho, args.alpha_rate, args.beta_rate, args.dalpha, args.dbeta,
         exclude_layers=exclude_layers)
