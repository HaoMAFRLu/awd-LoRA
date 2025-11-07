"""Unified Importance Allocation (UIA); to determine
the number of ranks and sparsity level for each layer automatically.
"""
import sys, os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

class UIA():
    def __init__(self,
                 LL: dict,
                 SS: dict,
                 model: torch.nn.Module,
                 rank: int=0) -> None:
        """Allocate ranks and sparsity levels for each layer given target parameters.
        Args:
            params_tgt (float): target number of parameters (in million)
            LL (dict): low-rank matrices for each layer
            SS (dict): sparse matrices for each layer
            model: the original model
        """
        self.rank = rank
        self.LL = LL
        self.SS = SS
        self.nr_params_model = sum(p.numel() for p in model.parameters())
        self.dim = {}
        self.rank_quantile_energy = {}
        self.rate_density = {}
        self.nr_params_layers = 0
        self.nr_params_L = 0
        self.nr_params_S = 0
        self.intialization()

    def get_rank_quantile(self, L: torch.Tensor, 
                          energy_quantile: float) -> float:
        """Get the rank quantile given energy quantile.
        Args:
            L (torch.Tensor): low-rank matrix
            energy_quantile (float): energy quantile
        Returns:
            rank quantile (float)
        """
        _, s, _ = torch.linalg.svd(L, full_matrices=False)
        energy = torch.cumsum(s, dim=0) / torch.sum(s)
        rank = torch.sum(energy < energy_quantile).item() + 1
        rank_quantile = rank / len(s)
        return rank_quantile, rank

    def intialization(self):
        """Initialize the statistics for each layer.
        Args:
            None
        Returns:
            None
        """
        for key in self.LL:
            L = self.LL[key]
            S = self.SS[key]
            row, col = L.shape
            self.dim[key] = (row, col)
            nr_nonzero = torch.sum(S != 0).item()
            nr_total = row * col

            self.rate_density[key] = nr_nonzero / nr_total
            self.rank_quantile_energy[key], rank = self.get_rank_quantile(L, energy_quantile=0.999)

            # calculate the number of parameters for each layer
            self.nr_params_layers += nr_total
            # calculate the number of parameters for L and S
            self.nr_params_L += int(rank * (row + col))
            self.nr_params_S += int(nr_nonzero)

        # calculate the number of parameters in the rest of the model
        self.nr_params_rest = self.nr_params_model - self.nr_params_layers
        # calculate the total number of parameters with low-rank + sparse
        self.nr_params_total = self.nr_params_rest + self.nr_params_L + self.nr_params_S

    def allocate_L_and_S(self,
                         params_to_reduce: int,
                         component: str='L') -> dict:
        """Allocate ranks or sparsity levels for each layer.
        Args:
            params_to_reduce (int): number of parameters to reduce
            component (str): 'L' for low-rank component, 'S' for sparse component
        Returns:
            rank_quantile_uia (dict): allocated rank quantile or density for each layer
        """        
        uia = {}
        if component == 'L':
            quantiles = self.rank_quantile_energy
            params_capacity = self.nr_params_L
        elif component == 'S':   # component == 'S'
            quantiles = self.rate_density
            params_capacity = self.nr_params_S
        else:
            raise ValueError("component should be either 'L' or 'S'.")
        
        ratio = 1 - params_to_reduce / params_capacity
        ratio = np.clip(ratio, 0.0, 1.0)

        for key, value in quantiles.items():
            uia[key] = value * ratio
        return uia

    def allocate_params(self,
                        params_tgt: float,
                        gamma: float=1.0) -> tuple:
        """Allocate ranks and sparsity levels for each layer.
        Args:
            params_tgt (float): target number of parameters (in million)
            gamma (float): adjustment factor for rank allocation, 1.0 means 
            reducing all low-rank components
        Returns:
            params_diff_L (int): number of parameters to reduce for low-rank components
            params_diff_S (int): number of parameters to reduce for sparse components
        """
        # params target < total params, which means reduction is needed
        params_diff = self.nr_params_total - params_tgt     # how many parameters to reduce
        # allocate the parameters for low-rank components and sparse components
        params_diff_L = int(params_diff * gamma)
        params_diff_S = params_diff - params_diff_L
        # whether it has enough parameters to reduce in L and S
        params_reduce_capacity_L = self.nr_params_L - params_diff_L
        params_reduce_capacity_S = self.nr_params_S - params_diff_S

        if params_reduce_capacity_L < 0 and params_reduce_capacity_S < 0: # not enough parameters to reduce
            # reduce all low-rank and sparse components
            params_diff_L = self.nr_params_L
            params_diff_S = self.nr_params_S
        elif params_reduce_capacity_L < 0 and params_reduce_capacity_S >= 0:
            # reduce all low-rank components
            params_diff_L = self.nr_params_L
            # move the remaining reduction to sparse components
            params_diff_S = min(self.nr_params_S, params_diff_S - params_reduce_capacity_L)
        elif params_reduce_capacity_L >= 0 and params_reduce_capacity_S < 0:
            # reduce all sparse components
            params_diff_S = self.nr_params_S
            # move the remaining reduction to low-rank components
            params_diff_L = min(self.nr_params_L, params_diff_L - params_reduce_capacity_S)
        else:   
            # both have enough parameters to reduce
            params_diff_L = params_diff_L
            params_diff_S = params_diff_S

        return params_diff_L, params_diff_S

    def allocate(self,
                 params_tgt: float,
                 gamma: float=1.0):
        """Allocate ranks and sparsity levels for each layer.
        Args:
            params_tgt (float): target number of parameters (in million)
            gamma (float): adjustment factor for rank allocation, 1.0 means 
            reducing all low-rank components
        Returns:
            rank_quantile_uia (dict): allocated rank quantile for each layer
            rate_density (dict): allocated density for each layer
        """
        params_tgt = int(params_tgt * 1e6)
        assert 0<=gamma<=1.0, "gamma should be between 0 and 1."
        
        if params_tgt >= self.nr_params_total: # no reduction needed
            return self.rank_quantile_energy, self.rate_density
        
        params_diff_L, params_diff_S = self.allocate_params(params_tgt, gamma)

        # allocate ranks for each layer
        return self.allocate_L_and_S(params_diff_L, 'L'), self.allocate_L_and_S(params_diff_S, 'S')

    def check_params(self,
                     rank_quantile: dict,
                     rate_density: dict) -> int:
        """Check the number of parameters given allocated ranks and sparsity levels.
        Args:
            rank_quantile_uia (dict): allocated rank quantile for each layer
            rate_density (dict): allocated density for each layer
        Returns:
            nr_params_total_uia (int): total number of parameters with allocated ranks and sparsity
        """ 
        return cal_nr_params(self.nr_params_model, rank_quantile, rate_density, self.dim)