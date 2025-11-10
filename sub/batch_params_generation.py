"""This script is used to generate parameters beta, beta0
for grid search on the cluster.
"""
import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general import *

root = get_parent_path(lvl=1)
file = os.path.join(root, 'batch_params.txt')

rho = ['1e-6']
alpha = ['1e-1', '5e-1', '1']
beta = ['5e-2', '1e-1']

combinations = [(a, b, c) for a in rho for b in alpha for c in beta]

with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")