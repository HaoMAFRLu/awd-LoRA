"""This script is used to generate parameters beta, beta0
for grid search on the cluster.
"""
import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general import *

root = get_parent_path(lvl=1)
file = os.path.join(root, 'params.txt')

wave = ['0.001', '0.01', '0.1']
attr = ['1.0', '2.0', '3.0', '4.0', '5.0']

combinations = [(a, b) for a in wave for b in attr]

with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")