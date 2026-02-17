"""Resave the model as HuggingFace format.
"""
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(MODEL_TYEP: str, 
         FOLDER: str, 
         file: str,
         precision: str=torch.bfloat16) -> None:
    
    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYEP, file)

    path_cfg = os.path.join(path_folder, MODEL_TYEP+'.yaml')
    path_cfg_model = os.path.join(path_folder, MODEL_TYEP+'_model.json')

    with open(path_cfg) as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']
    max_length = cfg['max_length']
    batch_size = cfg['batch_size']
    set_seed(seed)
    
    model = get_model(path_cfg_model)
    model.to(precision)
    
    load_model(model, os.path.join(path_folder, 'model.pth'))
    model.to(device)

    path_folder_resave = os.path.join(path_folder, 'model_resave')
    mkdir(path_folder_resave)

    # save the model in HuggingFace format
    model.save_pretrained(path_folder_resave, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    tokenizer.save_pretrained(path_folder_resave)

if __name__== '__main__':

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
        'head',
        'baseline_fp32',
        'head_fp32',
        'head_bf16',
        'vanilla_bf16',
    ]

    FILES = [
        '20251209_104454'
    ]

    precisin = torch.bfloat16

    for file in FILES:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)
        MODEL_TYEP = path_part['model_type']
        FOLDER = path_part['folder']
        main(MODEL_TYEP, 
             FOLDER, 
             file,
             precision=precisin)