"""Analyze the single model
"""
import sys, os
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

layer_type = ['self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj',
              'self_attn.v_proj', 'mlp.down_proj', 'mlp.up_proj', 'mlp.gate_proj']

def get_layer_info(path_folder: str) -> dict:
    with open(os.path.join(path_folder, 'layer_info.pkl'), 'rb') as f:
        layer_info = pickle.load(f)
    return layer_info

def plot_figs(layer_info: dict,
              nr_layers: int,
              layer_type: list,
              path_fig: str) -> None:
    """Plot the rank and sparsity convergence for each layer and each type.
    There are nr_layers * layer_type sub-plots, with the x-axis being the layer index,
    and y-axis being the layer type. For each sub-plot, the the x-axis is the step index,
    and the left y-axis is the rank, right y-axis is the sparsity.
    """
    # first create a figure with sub-plots, then plot each layer and each type
    fig, axes = plt.subplots(len(layer_type), nr_layers, figsize=(4 * nr_layers, 3 * len(layer_type)), squeeze=False)

    for j, lt in enumerate(layer_type):        # 纵轴: layer_type
        for i in range(nr_layers):             # 横轴: layer index
            layer_name = f'layers.{i}.{lt}'
            if layer_name not in layer_info:
                continue

            ax1 = axes[j, i]
            ax2 = ax1.twinx()

            _layer_info = layer_info[layer_name]

            ranks = np.array(_layer_info['rank']) / _layer_info['total_rank'][0]
            density = np.array(_layer_info['nonzero']) / _layer_info['total_elements'][0]

            ax1.plot(ranks, 'g-', label='Rank')
            ax2.plot(density, 'b--', label='Sparsity')

            ax1.set_xlabel('Step Index')
            ax1.set_ylabel('Rank', color='g')
            ax2.set_ylabel('Sparsity', color='b')
            ax1.set_title(f'Layer: {layer_name}')

            ax1.tick_params(axis='y', labelcolor='g')
            ax2.tick_params(axis='y', labelcolor='b')

            ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(path_fig, 'rank_sparsity_convergence.pdf'))
    plt.close()

def main(MODEL_TYPE: str,
         FOLDER: str,
         FILE: str) -> None:
    
    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, FILE)
    layer_info = get_layer_info(path_folder)

    layer_names = [key for key in layer_info.keys() if 'layers' in key]
    nr_layers = len(layer_names) // 7

    path_fig = os.path.join(root, 'data', 'figures', FOLDER, MODEL_TYPE, FILE)
    mkdir(path_fig)

    plot_figs(layer_info, nr_layers, layer_type, path_fig)



if __name__ == "__main__":
    MODEL_TYPES = [
        # 'llama_60m',
        # 'llama_130m',
        'llama_350m',
        # 'llama_1b',
    ]
    FOLDERS = [
        'ablation',
        # 'salad',
    ]

    # MODEL_TYPE = 'llama_350m'
    # FOLDER = 'ablation'

    for MODEL_TYPE in MODEL_TYPES:
        for FOLDER in FOLDERS:
            path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE)
            files = os.listdir(path_folder)

            files = ['20251126_101939']
            for FILE in files:
                main(MODEL_TYPE=MODEL_TYPE, FOLDER=FOLDER, FILE=FILE)