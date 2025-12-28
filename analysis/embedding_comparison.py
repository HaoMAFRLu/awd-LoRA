"""This script is used to compare the convergence and sparsity/rank patterns
including and excluding the embedding layers.
"""
import sys, os
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

if __name__ == "__main__":
    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'baseline', 
        'incl_embedding'
    ]

    files = [
        '20251228_201955',
        '20251228_202014',
    ]

    path_parts = []
    for file in files:
        path_parts.append(determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                              FOLDERS=FOLDERS,
                                              file=file))