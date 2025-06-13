import os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pkl
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

def get_single_result(path):
    params = 


def main(folders):
    for folder in folders:
        path_folder = os.path.join(root, 'data', 'awd_training', folder)
        _result, _model = get_single_result(path_folder)
        path_folder = os.path.join(root, 'data', '')

if __name__ == '__main__':
    folders = ['1']
    main(folders=folders)