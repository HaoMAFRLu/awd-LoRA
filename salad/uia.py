"""Unified Importance Allocation (UIA); to determine
the number of ranks and sparsity level for each layer automatically.
"""
import sys, os
import torch

class UIA():
    def __init__(self,
                 LL: dict,
                 SS: dict,
                 model):
        """Allocate ranks and sparsity levels for each layer given target parameters.
        Args:
            params_tgt (float): target number of parameters (in million)
            LL (dict): low-rank matrices for each layer
            SS (dict): sparse matrices for each layer
            model: the original model
        """
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

    def _allocate_rank(self,
                       params_tgt: float):
        """Allocate ranks for each layer to meet the target number of parameters.
        Args:
            params_tgt (float): target number of parameters (in million)
        Returns:
            rank_quantile_uia (dict): allocated rank quantile for each layer
        """        
        # how many parameters to reduce to reach the target
        param_diff = self.nr_params_total - params_tgt * 1e6

    def allocate(self,
                 params_tgt: float,
                 strategy: str):
        """Allocate ranks and sparsity levels for each layer.
        Args:
            params_tgt (float): target number of parameters (in million)
            strategy (str): allocation strategy
        Returns:
            rank_quantile_uia (dict): allocated rank quantile for each layer
            rate_density (dict): allocated density for each layer
        """
        if params_tgt >= self.nr_params_total: # no reduction needed
            rank_quantile_uia = self.rank_quantile_energy
            rate_density = self.rate_density
            return rank_quantile_uia, rate_density
        
        if strategy == None:
            # Reduce all low-rank components
            return _allocate_rank(params_tgt)

