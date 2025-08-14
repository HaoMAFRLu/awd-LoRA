"""Evaluate the models trained with Salad on the validation set.
"""
import os, sys
import pickle
import io
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

def get_lowspa_layers(pth: str) -> tuple:
    """Load data from the files"""
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        with open(pth, 'rb') as f:
            obj = pickle.load(f) 
    finally:
        torch.storage._load_from_bytes = orig
    return obj['LL'], obj['SS']

def read_layer_info(pth: str) -> list:
    """Load data from the files"""
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        with open(pth, 'rb') as f:
            obj = pickle.load(f) 
    finally:
        torch.storage._load_from_bytes = orig

    ex_list = ['loss', 'loss1', 'loss2', 'num_tokens']
    layer_names = []
    for key in obj.keys():
        if key not in ex_list:
            layer_names.append(key)
    loss = obj['loss']
    loss1 = obj['loss1']
    loss2 = obj['loss2']
    num_tokens = obj['num_tokens']
    return layer_names, loss, loss1, loss2, num_tokens, obj

def plot_loss(loss, loss1, loss2, num_tokens, path_fig):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    set_axes_format(ax[0], r'Iterations', r'Loss')
    ax[0].plot(loss, label='Loss')
    ax[0].plot(loss1, label='Loss1')
    ax[0].plot(loss2, label='Loss2')
    ax[0].grid(True)
    ax[0].legend()

    set_axes_format(ax[1], r'Iterations', r'Number of Tokens')
    ax[1].plot(num_tokens, label='Number of Tokens')
    ax[1].grid(True) 
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path_fig, 'loss.png'))

def plot_layer(layer_info, path_layer):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    set_axes_format(ax[0], r'Iterations', r'Loss')
    ax[0].plot(layer_info['loss'], label='Loss')
    ax[0].grid(True)
    ax[0].legend()  

    set_axes_format(ax[1], r'Iterations', r'rank')
    ax[1].plot(layer_info['rank'], label='Rank')
    ax[1].grid(True)
    ax[1].legend()

    set_axes_format(ax[2], r'Iterations', r'alpha')
    ax[2].plot(layer_info['alpha'], label='Alpha')
    ax[2].grid(True)
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path_layer, 'layer.png'))

def main(cfg_version: str,
         path_folder: str) -> None:
    # load the config
    path_fig = os.path.join(path_folder, 'figures')
    mkdir(path_fig)

    layer_names, loss, loss1, loss2, num_tokens, obj = read_layer_info(os.path.join(path_folder, 'layer_info.pkl'))
    plot_loss(loss, loss1, loss2, num_tokens, path_fig)
    
    for name in layer_names:
        path_layer = os.path.join(path_fig, name)
        mkdir(path_layer) 
        plot_layer(obj[name], path_layer)

if __name__ == "__main__":
    cfg_version = 'llama_60m'
    file = '20250814_105649'
    path_folder = os.path.join(root, 'data', 'salad', cfg_version, file)
    main(cfg_version, path_folder)
    