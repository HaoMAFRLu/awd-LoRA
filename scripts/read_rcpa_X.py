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
        
        # svs = {}
        # SS = {}
        # LL = {}

        svs = data['svs']
        SS = data['SS']
        LL = data['LL']

        rank_ratio_list = []
        sparsity_level_list = []

        for key in svs:
            sv = svs[key].to('cpu')
            S = SS[key].to('cpu')
            L = LL[key].to('cpu')

            # calculate 0.999 energy coverage
            singular_values = sv.cpu().numpy()
            squared_singular_values = singular_values ** 2
            total_energy = squared_singular_values.sum()
            cumulative_energy = 0.0
            rank_999 = 0
            for i, value in enumerate(squared_singular_values):
                cumulative_energy += value
                if cumulative_energy / total_energy >= 0.999:
                    rank_999 = i + 1
                    break
            rank_ratio = rank_999 / len(singular_values)
            rank_ratio_list.append(rank_ratio)

            # calculate sparsity level
            S_cpu = S.cpu().numpy()
            S_max = np.max(np.abs(S_cpu))
            epsilon = 1e-8 * S_max
            total_elements = S_cpu.size
            nonzero_elements = np.sum(S_cpu > epsilon)
            sparsity_level = 1 - (nonzero_elements / total_elements)
            sparsity_level_list.append(sparsity_level)
        
        rank_ratio_list = np.array(rank_ratio_list)
        sparsity_level_list = np.array(sparsity_level_list)

        # mean values
        mean_rank_ratio = np.mean(rank_ratio_list)
        mean_sparsity_level = np.mean(sparsity_level_list)
        # std values
        std_rank_ratio = np.std(rank_ratio_list)
        std_sparsity_level = np.std(sparsity_level_list)

        print(rank_ratio_list)
        print(sparsity_level_list)
        
        print('================ Summary ================')
        print(f'Sparsity Level: {mean_sparsity_level:.3f} ± {std_sparsity_level:.3f}')
        print(f'Rank Ratio: {mean_rank_ratio:.3f} ± {std_rank_ratio:.3f}')
        print('=========================================')


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
