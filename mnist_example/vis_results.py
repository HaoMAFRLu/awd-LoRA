import os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import pandas as pd
from matplotlib import gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import awave
from awave.utils.evaluate import Validator
from awave.transform2d import DWT2d
from awave.utils.misc import get_2dfilts, get_wavefun
from awave.utils.visualize import plot_1dfilts, plot_2dreconstruct, plot_wavefun

from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def get_params(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_results(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_w_transform(params, path):
    wt = DWT2d(wave=params['wave'], 
               mode=params['mode'], 
               J=params['J'], 
               init_factor=params['init_factor'], 
               noise_factor=params['noise_factor'],
               const_factor=params['const_factor'],
               device=device)
    wt.load_state_dict(torch.load(path))
    return wt

def get_single_result(path):
    params = get_params(os.path.join(path, 'params'))
    results = get_results(os.path.join(path, 'results'))
    wt = get_w_transform(params, os.path.join(path, 'model.pth'))
    return params, results, wt

def get_params_results_models(folders):
    results = []
    wt = []
    params = []
    for folder in folders:
        path_folder = os.path.join(root, 'data', 'awd_training', folder)
        _params, _result, _wt = get_single_result(path_folder)
        params.append(_params)
        results.append(_result)
        wt.append(_wt)

    return params, results, wt

def get_grid(arr):
    return np.unique(arr)

def get_key(key, params):
    return np.array([r[key] for r in params]).astype(float).flatten()

def get_index(lamL1wave, lamL1attr, lamL1wave_grid, lamL1attr_grid):
    index2o = {} 
    index2t = {} 

    for idx, (wval, aval) in enumerate(zip(lamL1wave, lamL1attr)):
        i = np.where(lamL1wave_grid == wval)[0][0]
        j = np.where(lamL1attr_grid == aval)[0][0]

        index2o[(i, j)] = idx
        index2t[idx]    = (i, j)
    
    return index2o, index2t

def get_psi(index2o, index2t, lamL1wave_grid, lamL1attr_grid, models):
    R, C = len(lamL1wave_grid), len(lamL1attr_grid)
    psi_list, wt_list = [], []
    x_list = []
    for r in range(R):
        for c in range(C):
            idx = index2o.get((r, c))
            if idx is None:                 # 该网格点可能没有实验
                psi_list.append(None)
                wt_list.append(None)
                continue

            wt = models[idx]
            wt_list.append(wt)

            phi, psi, x = get_wavefun(wt)
            psi_list.append(psi)
            x_list.append(x)

    return x_list, psi_list

def main(folders):
    params, results, models = get_params_results_models(folders)
    lamL1wave, lamL1attr = get_key('lamL1wave', params), get_key('lamL1attr', params)
    lamL1wave_grid, lamL1attr_grid = get_grid(lamL1wave), get_grid(lamL1attr)
    index2o, index2t = get_index(lamL1wave, lamL1attr, lamL1wave_grid, lamL1attr_grid)
    x_list, psi_list = get_psi(index2o, index2t, lamL1wave_grid, lamL1attr_grid, models)

    R, C = len(lamL1wave_grid), len(lamL1attr_grid)

    plt.figure(figsize=(C + 1, R + 1), dpi=200)
    gs = gridspec.GridSpec(R, C,
                        wspace=0.0, hspace=0.0,
                        top=1. - 0.5 / (R + 1), bottom=0.5 / (R + 1),
                        left=0.5 / (C + 1), right=1 - 0.5 / (C + 1))

    i = 0
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r, c])
            ax.plot(x_list[i], psi_list[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
            if c == 0:
                plt.ylabel(str(lamL1wave_grid[r]))
            if r == 0:
                plt.title(str(lamL1attr_grid[c]))
            i += 1

    plt.show()
    

if __name__ == '__main__':
    folders = ['2025-06-13_14-58-07',
               '2025-06-13_15-06-17']
    main(folders=folders)