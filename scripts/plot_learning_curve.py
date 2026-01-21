"""This script is used to plot the learning curve
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

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

def get_info(path):
    path_file = os.path.join(path, 'layer_info.pkl')
    with open(path_file, 'rb') as f:
        info = pickle.load(f)
    return info

def plot_loss_diff(loss: list,
                   diff: list,
                   path_folder_fig: str,
                   stride: int=10) -> None:
    epochs = list(range(1, len(loss)+1))
    epochs_ds = epochs[::stride]
    loss_ds = loss[::stride]
    diff_ds = diff[::stride]
    # plot in independent figures
    # do the downsampling but keep the length the same
    plt.figure()
    plt.plot(epochs_ds, loss_ds, label='Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    file_name = 'loss'
    plt.savefig(os.path.join(path_folder_fig, f'{file_name}.png'))
    tikzplotlib.save(os.path.join(path_folder_fig, f'{file_name}.tex'))


    plt.figure()
    plt.plot(epochs_ds, diff_ds, label='Diff', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Diff')
    file_name = 'diff'
    plt.savefig(os.path.join(path_folder_fig, f'{file_name}.png'))
    tikzplotlib.save(os.path.join(path_folder_fig, f'{file_name}.tex'))

def plot_loss_layer(diff: list,
                    rank: list,
                    sparsity: list,
                    path_folder_fig: str,
                    nr_layer: int,
                    layer_type: str,
                    stride: int) -> None:
    epochs = list(range(1, len(diff)+1))
    epochs_ds = epochs[::stride]
    diff_ds = diff[::stride]
    rank_ds = rank[::stride]
    sparsity_ds = sparsity[::stride]

    plt.figure()
    plt.plot(epochs_ds, diff_ds, label='Layer Diff', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Layer Diff')
    file_name = f'layer_{nr_layer}_{layer_type}_diff'
    plt.savefig(os.path.join(path_folder_fig, f'{file_name}.png'))
    tikzplotlib.save(os.path.join(path_folder_fig, f'{file_name}.tex'))
    
    plt.figure()
    plt.plot(epochs_ds, rank_ds, label='Layer Rank', color='g')
    plt.plot(epochs_ds, sparsity_ds, label='Layer Sparsity', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Layer Rank')
    file_name = f'layer_{nr_layer}_{layer_type}_rank_sparsity'
    plt.savefig(os.path.join(path_folder_fig, f'{file_name}.png'))
    tikzplotlib.save(os.path.join(path_folder_fig, f'{file_name}.tex')) 

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str,
         nr_layer: int,
         layer_type: str,
         stride: int,
         stride_inner: int) -> None:
        
    path_folder_fig = os.path.join(root, 'data', 'figures', 'learning_curve', MODEL_TYPE)
    mkdir(path_folder_fig)

    layer_name = f'layers.{nr_layer}.{layers_mapping[layer_type]}'
    path_file = os.path.join(root, 'data', 'baseline_fp32', MODEL_TYPE, file)

    info_learning = get_info(path_file)
    loss = info_learning['avg_loss']
    diff = info_learning['avg_diff']

    info_layer = info_learning[layer_name]
    layer_diff = info_layer['loss']
    rank = [x/info_layer['total_rank'][0] for x in info_layer['rank']]
    sparsity = [1 - x/info_layer['total_elements'][0] for x in info_layer['nonzero']]
    
    plot_loss_diff(
        loss=loss,
        diff=diff,
        path_folder_fig=path_folder_fig,
        stride=stride
    )

    plot_loss_layer(
        diff=layer_diff,
        rank=rank,
        sparsity=sparsity,
        path_folder_fig=path_folder_fig,
        nr_layer=nr_layer,
        layer_type=layer_type,
        stride=stride_inner
    )


if __name__ == "__main__":
    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'vanilla_bf16',  
        'baseline_fp32',
    ]


    nr_layer = 3
    layer_type = 'q'  

    file = [
        # '20251204_135646',  # 60m, baseline fp32
        # '20251202_164626',  # 130m, baseline fp32
        '20251203_102315',  # 350m, baseline fp32
        # '20251130_125959',  # 1b, baseline fp32
    ]

    path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                    FOLDERS=FOLDERS,
                                    file=file[0])

    stride_mapping = {
        'llama_60m': 10,
        'llama_130m': 20,
        'llama_350m': 50,
        'llama_1b': 100,
    }

    stide_inner_mapping = {
        'llama_60m': 10, 
        'llama_130m': 10,
        'llama_350m': 20,
        'llama_1b': 30,
    }

    main(MODEL_TYPE=path_part['model_type'],
         FOLDER=path_part['folder'],
         file=file[0],
         nr_layer=nr_layer,
         layer_type=layer_type,
         stride=stride_mapping[path_part['model_type']],
         stride_inner=stride_mapping[path_part['model_type']])   