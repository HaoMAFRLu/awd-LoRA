import torch
import torch.distributed as dist
import torch.nn as nn

from lowspa_ddp.utils import *

class ADMMSolver():
    """
    Base interface for per-layer SVD solvers.
    Subclass this to implement customized SVD-based updates.
    """
    def __init__(self, 
                 layer_name: str, 
                 params: dict,
                 X: torch.Tensor) -> None:
        """
        Args:
            layer_name: Name of the layer this solver applies to
            params: Solver-specific hyperparameters
            X: Initial weight matrix for this layer
        """
        self.X_with_grad = X  # Initial weight matrix
        self.X = X.detach()

        _, s, _ = torch.linalg.svd(X, full_matrices=False)
        self.nr_total_rank = len(s)
        self.nr_elements = X.numel()

        self.layer_name = layer_name
        # read the params
        for key, val in params.items():
            setattr(self, key, val)
        # Initialize SVD factors
        self.initialization()

    def get_loss_term(self):
        """
        Compute the loss term for the model.
        """
        if self.loss_version == 'v1':
            loss = self.rho/2 * torch.norm(self.X_with_grad - self.L - self.S + self.Y/self.rho, p='fro') ** 2
        elif self.loss_version == 'v2':
            loss = self.rho/2 * torch.norm(self.X_with_grad - self.L, p='fro') ** 2
        
        self.nr_cals += 1
        self.total_loss += loss.item()
        return loss
    
    def reset(self):
        """
        Reset the solver state for a new training epoch.
        """
        self.total_loss = 0.0
        self.nr_cals = 0

    @staticmethod
    def single_step_PRCA(X: torch.Tensor,
                         L: torch.Tensor,
                         S: torch.Tensor,
                         Y: torch.Tensor,
                         alpha: float,
                         beta: float,
                         rho: float,
                         energy_quantile: float) -> tuple:
        U, s, Vt = torch.linalg.svd(X - S + Y / rho, full_matrices=False)
        _s = soft_threshold(s, alpha/rho)
        L = U @ torch.diag(_s) @ Vt
        S = soft_threshold(X - L + Y/rho, beta/rho)
        Y = Y + rho * (X - L - S)
        return L, S, Y, get_energy_quantile(_s, quantile=energy_quantile)

    def PRCA(self,
             X: torch.Tensor, 
             L: torch.Tensor,  
             S: torch.Tensor,
             Y: torch.Tensor,
             alpha: float,
             beta: float,
             rho: float,
             iter_max: int = 100,
             tol: float = 1e-3,
             energy_quantile: float=0.9) -> tuple:
        """
        Perform the Principal Component Analysis (PCA) using Robust PCA.
        Args:
            X: Input data.
            L: Low-rank component.
            S: Sparse component.
            Y: Dual variable.
            mu: Regularization parameter for the dual variable.
            la: Regularization parameter for the sparse component.
        Returns:
            Updated low-rank and sparse components, and dual variable.
        """
        for it in range(iter_max):
            L, S, Y, nr_rank = self.single_step_PRCA(X, L, S, Y, alpha, beta, rho, energy_quantile)
            if torch.linalg.norm(X - L - S, 'fro') < tol:
                break
        return L, S, Y, nr_rank

    def initialization(self) -> None:
        if self.init_energy <= 0:
            self.L = torch.zeros_like(self.X, device=self.device)
        else:
            U, s, Vt = torch.linalg.svd(self.X, full_matrices=False)
            nr_singular_values = int(len(s) * self.init_energy)
            self.L = U[:, :nr_singular_values] @ torch.diag(s[:nr_singular_values]) @ Vt[:nr_singular_values, :]

        self.S = torch.zeros_like(self.X)
        self.Y = torch.zeros_like(self.X)

    def cal_results(self) -> None:
        """
        Calculate the results after running the solver.
        """
        self.results = {'L': self.L,
                        'S': self.S,
                        'Y': self.Y,
                        'nr_rank': self.nr_rank,
                        'nr_nonzero': int(torch.count_nonzero(self.S)),
                        'nr_total_rank': self.nr_total_rank,
                        'nr_elements': self.nr_elements,
                        'avg_loss': self.total_loss/self.nr_cals}

    def run(self):
        if self.is_cal:
            # calibration the sparse matrix
            self.S = self.X - self.L
        self.L, self.S, self.Y, self.nr_rank = self.PRCA(self.X.clone(),
                                                        self.L.clone(),
                                                        self.S.clone(),
                                                        self.Y.clone(),
                                                        self.alpha,
                                                        self.beta,
                                                        self.rho,
                                                        self.iter_max,
                                                        self.tol,
                                                        self.energy)
        # calculate the results
        self.cal_results()
        
