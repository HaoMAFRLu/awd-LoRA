import torch
import math

from salad.utils import *
from salad.adaptive_rho import RHO
from salad.adaptive_param import PARAM

class SALAD():
    """
    Base interface for per-layer SVD solvers.
    Subclass this to implement customized SVD-based updates.
    """
    def __init__(self, 
                 layer_name: str,
                 params: dict,
                 X: torch.Tensor,
                 nr_layers: int,
                 is_full: bool,
                 precision: str=torch.bfloat16) -> None:
        """
        Args:
            layer_name: Name of the layer this solver applies to
            params: Solver-specific hyperparameters
            X: Initial weight matrix for this layer
        """
        self.X_with_grad = X  # Initial weight matrix
        self.layer_name = layer_name

        self.precision = precision
        self.sum_pre = None
        self.gamma = 0.9
        self.ema_r = None
        self.ema_s = None
        # read the params
        for key, val in params.items():
            setattr(self, key, val)

        self.nr_epoch = 0

        rho_cfg = self.rho_dict
        rho_cfg['row'] = X.shape[0]
        rho_cfg['col'] = X.shape[1]
        rho_cfg['nr_layers'] = nr_layers
        rho_cfg['X_norm'] = torch.norm(X.detach().float(), p='fro').cpu().numpy()
        self.rho_solver = RHO(rho_cfg)
        
        alpha_cfg = self.alpha_dict
        alpha_cfg['target_rate'] = self.rate_rank
        self.alpha_solver = PARAM(alpha_cfg)

        beta_cfg = self.beta_dict
        beta_cfg['target_rate'] = self.rate_sparsity
        self.beta_solver = PARAM(beta_cfg)

        self.rho = self.rho_solver.rho

        # self.rho = 1.0 / (np.sqrt(nr_layers * max(row, col)))
        # self.rho = 1.0 / (5 * nr_layers * np.sqrt(row * col))
        # self.rho = 1.0 / (25 * nr_layers * np.sqrt(row * col))
        # self.rho = 1.0 / (nr_layers * row * col)
        # self.rho = 1.0 / (np.sqrt(max(row, col)))
        # self.rho = 0.1
        # self.rho = 1.0 / (2.0*np.sqrt(max(row, col)))
        
        self.nr_elements = X.numel()

        if is_full:
            _, s, _ = torch.linalg.svd(X.float(), full_matrices=False)
            self.nr_total_rank = len(s)
            
            if self.is_init:   
                k = math.ceil(self.nr_total_rank * self.rate_rank) - 1
                self.alpha_solver.value = float(s[k] * self.rho)

            # Initialize SVD factors
            self.initialization()
        
        self.reset()

    def get_loss_pre_term(self, 
                          L: torch.Tensor, 
                          S: torch.Tensor) -> float:   
        """
        Compute the loss term for the model.
        """
        if self.sum_pre is None:
            self.sum_pre = L.detach() + S.detach()

        loss = self.rho * torch.norm(L + S - self.sum_pre, p='fro')  
        self.sum_pre = L.detach() + S.detach()

        return loss
    
    def get_penalty(self,
                    L: torch.Tensor, 
                    S: torch.Tensor,
                    Y: torch.Tensor) -> float:
        """
        Compute the loss term for the model.
        """
        loss = self.rho/2 * torch.norm(self.X_with_grad - L - S + Y/self.rho, p='fro') ** 2        
        return loss
    
    def _get_diff(self, 
                  X: torch.Tensor,
                  L: torch.Tensor, 
                  S: torch.Tensor) -> float:
        """
        Compute the loss term for the model.
        """   
        loss = torch.norm(X - L - S, p='fro')    
        self.nr_cals += 1
        self.total_loss += loss.item()
        return loss
    
    @staticmethod
    def get_gradient(X: torch.Tensor,
                     L: torch.Tensor,
                     S: torch.Tensor,
                     Y: torch.Tensor,
                     rho: float) -> torch.Tensor:
        return rho * (X - L - S + Y/rho)

    @torch.no_grad()
    def get_diff(self,
                 L: torch.Tensor,
                 S: torch.Tensor,
                 Y: torch.Tensor) -> torch.Tensor:  
        """Get the difference X - L - S for the layer."""
        loss_r = self._get_diff(self.X_with_grad.detach(), L, S)
        
        # loss_s = self.get_loss_pre_term(L, S)
        # if self.ema_r is None:
        #     self.ema_r = loss_r.item()
        #     self.ema_s = loss_s.item()
        # else:
        #     self.ema_r = self.gamma * self.ema_r + (1 - self.gamma) * loss_r.item()
        #     self.ema_s = self.gamma * self.ema_s + (1 - self.gamma) * loss_s.item()
        
        return loss_r
        
    def reset(self):
        """
        Reset the solver state for a new training epoch.
        """
        self.total_loss = 0.0
        self.nr_cals = 0

    def single_step_RPCA(self,
                         X: torch.Tensor,
                         L: torch.Tensor,
                         S: torch.Tensor,
                         Y: torch.Tensor,
                         alpha: float,
                         beta: float,
                         rho: float,
                         energy: float,) -> tuple:
        S = self._update_S(X, L, Y, self.rate_sparsity, rho)
        L, nr_rank = self._update_L(X, S, Y, alpha, rho, energy)
        Y = self._update_Y(X, L, S, rho)
        return L, S, Y, nr_rank
    
    def RPCA(self,
             X: torch.Tensor, 
             L: torch.Tensor,  
             S: torch.Tensor,
             Y: torch.Tensor,
             alpha: float,
             beta: float,
             rho: float,
             iter_max: int = 100,
             tol: float = 1e-3,
             energy: float=0.9) -> tuple:
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
            self.L, self.S, self.Y, self.nr_rank = self.single_step_RPCA(X, L, S, Y, 
                                                                         alpha, beta, rho, 
                                                                         energy)
            self.update_alpha()
            self.update_beta()    

            if torch.linalg.norm(X - L - S, 'fro') < tol:
                break

    def initialization(self) -> None:
        if self.init_energy <= 0:
            self.L = torch.zeros_like(self.X_with_grad.detach().float(), device=self.device).to(self.precision)
        else:
            U, s, Vt = torch.linalg.svd(self.X_with_grad.detach().float(), full_matrices=False)
            nr_singular_values = int(len(s) * self.rate_rank)
            self.L = (U[:, :nr_singular_values] @ torch.diag(s[:nr_singular_values]) @ Vt[:nr_singular_values, :]).to(self.precision)

        self.S = torch.zeros_like(self.X_with_grad.detach()).to(self.precision)
        self.Y = torch.zeros_like(self.X_with_grad.detach()).to(self.precision)

    def init_T(self, l: int, K: int) -> None:
        """[alpha, beta, dalpha, dbeta, rho, 
            rate_decay_alpha, rate_decay_beta, 
            loss, rank, nonzero, total_rank, total_elems]
        """
        self.T = torch.zeros(l, K, dtype=torch.float32, device=self.X_with_grad.device)

    # def cal_weights(self) -> None:
    #     self.results = {'L': self.L,
    #                     'S': self.S,
    #                     'Y': self.Y}

    def cal_results(self) -> None:
        """
        Calculate the results after running the solver.
        """
        self.T[self.layer_idx, 0] = self.alpha_solver.value
        self.T[self.layer_idx, 1] = self.beta_solver.value
        self.T[self.layer_idx, 2] = self.alpha_solver.dvalue
        self.T[self.layer_idx, 3] = self.beta_solver.dvalue
        self.T[self.layer_idx, 4] = self.rho
        self.T[self.layer_idx, 5] = self.alpha_solver.rate_decay
        self.T[self.layer_idx, 6] = self.beta_solver.rate_decay
        self.T[self.layer_idx, 7] = self.total_loss / self.nr_cals
        self.T[self.layer_idx, 8] = self.nr_rank
        self.T[self.layer_idx, 9] = int(torch.count_nonzero(self.S))
        self.T[self.layer_idx, 10] = self.nr_total_rank
        self.T[self.layer_idx, 11] = self.nr_elements

        # self.results = {'L': self.L.to('cpu'),
        #                 'S': self.S.to('cpu'),
        #                 'Y': self.Y.to('cpu'),
        #                 'alpha_mode': self.alpha_solver.mode,
        #                 'beta_mode': self.beta_solver.mode,
        #                 'alpha': self.alpha_solver.value,
        #                 'beta': self.beta_solver.value,
        #                 'dalpha': self.alpha_solver.dvalue,
        #                 'dbeta': self.beta_solver.dvalue,
        #                 'rho': self.rho,
        #                 'rate_decay_alpha': self.alpha_solver.rate_decay,
        #                 'rate_decay_beta': self.beta_solver.rate_decay,
        #                 'nr_rank': self.nr_rank,
        #                 'nr_nonzero': int(torch.count_nonzero(self.S)),
        #                 'nr_total_rank': self.nr_total_rank,
        #                 'nr_elements': self.nr_elements,
        #                 'avg_loss': (self.total_loss/self.nr_cals)}

    def _update_S(self,
                  X: torch.Tensor,
                  L: torch.Tensor,
                  Y: torch.Tensor,
                  rho: float,) -> torch.Tensor:
        if self.beta_solver.mode == 'hard_cut':
            self.beta_solver.update_quantile(X-L+Y/rho, rho)
        return soft_threshold(X - L + Y/rho, self.beta_solver.value/rho)


    def update_S(self) -> None:
        """
        Update the sparse component S. 
        """
        self.S = self._update_S(self.X_with_grad.detach(), 
                                self.L, 
                                self.Y,
                                self.rho).to(self.precision)

    @staticmethod
    def _update_Y(X: torch.Tensor,
                  L: torch.Tensor,
                  S: torch.Tensor,
                  Y: torch.Tensor,
                  rho: float) -> torch.Tensor:
        return Y + rho * (X - L - S)
    
    def update_Y(self) -> None:
        """
        Update the dual variable Y.
        """
        self.Y = self._update_Y(self.X_with_grad.detach(), 
                                self.L, 
                                self.S,
                                self.Y, 
                                self.rho).to(self.precision)

    def update_nr_epoch(self) -> None:
        self.nr_epoch += 1

    def update_rho(self) -> None:
        """update the value of rho based on the loss terms.
        """
        self.update_nr_epoch()
        self.rho = self.rho_solver.get_rho(self.nr_epoch, self.ema_r, self.ema_s)

    @staticmethod
    def _update_L(X: torch.Tensor,
                  S: torch.Tensor,
                  Y: torch.Tensor,
                  alpha: float,
                  rho: float,
                  energy: float) -> torch.Tensor:
        U, s, Vt = torch.linalg.svd(X.float() - S.float() + Y.float() / rho, full_matrices=False)
        _s = soft_threshold(s, alpha/rho)
        L  = U @ torch.diag(_s) @ Vt
        nr_rank = get_energy_quantile(_s, quantile=energy)
        return L, nr_rank

    def update_L(self) -> None:
        """
        Update the low-rank component L.
        """
        L, self.nr_rank = self._update_L(self.X_with_grad.detach(),
                                        self.S,
                                        self.Y,
                                        self.alpha_solver.value,
                                        self.rho,
                                        energy=self.energy)
        self.L = L.to(self.precision)

    def update_alpha(self) -> None:
        """
        Update the alpha parameter based on the rank of singular values.
        """
        self.alpha_solver.update(self.nr_rank/self.nr_total_rank, self.rho)

    def update_beta(self) -> None:
        """
        Update the beta parameter based on the sparsity of the matrix.
        """
        cur_elements = torch.count_nonzero(self.S)
        self.beta_solver.update(cur_elements/self.nr_elements, self.rho)

        
