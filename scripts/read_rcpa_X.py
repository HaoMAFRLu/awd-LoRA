import sys, os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)
layers_mapping = {
    'k': 'self_attn.k_proj',
    'q': 'self_attn.q_proj',
    'v': 'self_attn.v_proj',
    'o': 'self_attn.o_proj',
    'up': 'mlp.up_proj',
    'down': 'mlp.down_proj',
    'gate': 'mlp.gate_proj',
}
block_list = ['k', 'q', 'v', 'o', 'up', 'down', 'gate']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(MODEL_TYPE: str,
            FOLDER: str,
            file: str,
            nr_layers: list) -> None:
    
        path_folder = os.path.join(root, 'data',  FOLDER, MODEL_TYPE, file)
            
        with open(os.path.join(path_folder, 'rpca_X_rank_3.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        svs = {}
        SS = {}
        LL = {}

        for nr_layer in nr_layers:
            for block in block_list:
                layer_name = f'layers.{nr_layer}.{layers_mapping[block]}'
                sv = data['svs'][layer_name]
                S = data['SS'][layer_name]
                L = data['LL'][layer_name]
                svs[layer_name] = sv.to('cpu')
                SS[layer_name] = S.to('cpu')
                LL[layer_name] = L.to('cpu')
        
        new_data = {'svs': svs, 
                    'SS': SS,
                    'LL': LL}

        with open(os.path.join(path_folder, f'rpca_X_small.pkl'), 'wb') as f:
            pickle.dump(new_data, f)

if __name__ == "__main__":
    nr_layers = [0, 12, 23]
    MODEL_TYPE = 'llama_1b'
    FOLDER = 'vanilla_bf16'
    file = '20251213_234650'

    main(
        MODEL_TYPE=MODEL_TYPE,
        FOLDER=FOLDER,
        file=file,
        nr_layers=nr_layers,
    )
