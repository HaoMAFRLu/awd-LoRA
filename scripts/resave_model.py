"""Resave the model as HuggingFace format.
"""
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.uia import UIA
from salad.operators import *

root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(MODEL_TYEP: str, 
         FOLDER: str, 
         file: str,
         gamma: float=0.2,
         params: float=100000000.0,
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

    path_folder_resave_folder = os.path.join(path_folder_resave, 'vanilla')
    mkdir(path_folder_resave_folder)

    model.save_pretrained(path_folder_resave_folder, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    tokenizer.save_pretrained(path_folder_resave_folder)

    if 'vanilla' not in FOLDER:
        # if it's not a vanilla model, save two variants
        LL = {}
        SS = {}
        files = os.listdir(path_folder)
        rank_files = [f for f in files if f.startswith('matrix')]
        for f in rank_files:
            LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
            for key in LL_part:
                if 'lm_head' in key:
                    LL[key] = LL_part[key].to(device).t()
                    SS[key] = SS_part[key].to(device).t()
                else:
                    LL[key] = LL_part[key].to(device)
                    SS[key] = SS_part[key].to(device)

        with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
            layer_info = pickle.load(f)

        uia = UIA(LL, SS, model, 
                layer_info=layer_info, 
                rate=100000000.0,
                rank=0)
        
        layers = [entry['name'] for entry in cfg['layers']]
        gamma = np.clip(gamma, 0, 1)

        _rank_quantile, _rate_density, return_flag = uia.allocate(params_tgt=params, gamma=gamma)
        rank_quantile, rate_density = uia.post_allocate(_rank_quantile, _rate_density, params_tgt=params) 

        # double check the allocation
        nr_params = uia.check_params(rank_quantile, rate_density)
        print('-' * 50)
        print(f'Number of parameters: {nr_params/1e6:.2f} Million')
        print(f'Target parameters: {params:.2f} Million')
        print(f'State of return flag: {return_flag}')
        print(f'States:\n'
            f'0: success\n'
            f'1: total params less than target\n'
            f'2: no enought params to reduce in both L and S\n'
            f'3: no enought params to reduce in L\n'
            f'4: no enought params to reduce in S')
        print('-' * 50)
        
        XX = opt_slr(LL, SS, rank_quantile, rate_density, layers, device)
        opt_replace(model, layers, XX, device)  # replace partial layers with low-rank matrices L

        path_folder_resave_folder = os.path.join(path_folder_resave, 'surrogate')
        mkdir(path_folder_resave_folder)

        # save the model in HuggingFace format
        model.save_pretrained(path_folder_resave_folder, safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
        tokenizer.save_pretrained(path_folder_resave_folder)

if __name__== '__main__':
    params_tgt = {
        'llama_9m':   6.5,
        'llama_60m':  44.5,
        'llama_130m': 97.5,
        'llama_350m': 194.5,
        'llama_1b':   646.5,
    }
    
    gamma_list = {
        'llama_9m':   0.5,
        'llama_60m':  0.7,
        'llama_130m': 0.6,
        'llama_350m': 0.6,
        'llama_1b':   0.8,
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
        'head',
        'baseline_fp32',
        'head_fp32',
        'head_bf16',
        'vanilla_bf16',
    ]

    FILES = [
        # '20251229_134048'
        '20251130_125959',
        # '20251213_234650', # vanilla bf16 1b
    ]

    precisin = torch.bfloat16

    for file in FILES:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)
        MODEL_TYPE = path_part['model_type']
        FOLDER = path_part['folder']
        main(MODEL_TYPE, 
             FOLDER, 
             file,
             gamma=gamma_list[MODEL_TYPE],
             params=params_tgt[MODEL_TYPE],
             precision=precisin)