"""This script is used to show that the RPCA algorithm can recover low-rank and sparse components 
from the sum of low-rank matrices L and sparse matrices S.
"""
import sys, os
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model
from salad.static_rpca import StaticRPCA

hf_login_once()
ROOT = get_parent_path(lvl=1)

def init_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world

def destroy():
    dist.destroy_process_group()

def main(MODEL_TYPE: str, 
         FOLDER: str,
         file: str,
         rank: int=0) -> None:

    path_folder = os.path.join(ROOT, 'data', FOLDER, MODEL_TYPE, file)
    path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')

    model = get_model(path_cfg_model)
    # load the original model weights X
    load_model(model, os.path.join(path_folder, 'model.pth'))

    static_rpca = StaticRPCA(model, path_folder, rank)
    static_rpca.recover_X()
    # static_rpca.recover_LS()

if __name__ == "__main__":
    rank, world_size = init_distributed()   

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
        'head_fp32',
        'head_bf16',
        'vanilla_bf16',
        'baseline_fp16',
    ]

    files = [
        '20251209_204846',
        '20251209_232356',
        '20251209_233045',
        '20251213_234650',
        # '20251209_104454',   # for quick test
    ]

    files = sorted(files)
    myfiles = files[rank::world_size]
    
    if isinstance(myfiles, str):
        myfiles = [myfiles]

    for file in myfiles:
        print(f'[Rank {rank}]: Processing folder: {file}')

        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)

        MODEL_TYPE = path_part['model_type']
        FOLDER = path_part['folder']
        
        main(MODEL_TYPE, 
             FOLDER,
             file,
             rank)
        
        print(f'[Rank {rank}]: Finished folder: {file}')
    
    destroy()