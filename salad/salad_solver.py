import torch
import math

from salad.utils import *

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
                 is_full: bool) -> None:
        """
        Args:
            layer_name: Name of the layer this solver applies to
            params: Solver-specific hyperparameters
            X: Initial weight matrix for this layer
        """
        self.X_with_grad = X  # Initial weight matrix
        self.layer_name = layer_name

        self.L_pre = None
        self.S_pre = None
        self.gamma = 0.9
        self.ema_r = None
        self.ema_s = None
        # read the params
        for key, val in params.items():
            setattr(self, key, val)

        self.dalpha = 0.0
        self.dbeta = 0.0
        self.nr_epoch = 0

        # overwrite the hyperparameters
        if self.is_adaptive:
            row, col = X.shape  
            # calculate the initial rho
            _rho = torch.norm(self.X_with_grad.detach(), p='fro').cpu().numpy() / (np.sqrt(row * col))
            self.rho = 0.1 * _rho
            self.rho_min = 1e-3 * _rho
            self.rho_max = 20 * _rho
            self.rho_rate = 0.30
            # self.rho = 1.0 / (np.sqrt(nr_layers * max(row, col)))
            # self.rho = 1.0 / (5 * nr_layers * np.sqrt(row * col))
            # self.rho = 1.0 / (25 * nr_layers * np.sqrt(row * col))
            # self.rho = 1.0 / (nr_layers * row * col)
            # self.rho = 1.0 / (np.sqrt(max(row, col)))
            # self.rho = 0.1
            # self.rho = 1.0 / (2.0*np.sqrt(max(row, col)))
        
        self.alpha = self.rho * self.alpha_to_rho
        self.beta = self.rho * self.beta_to_rho
        self.nr_elements = X.numel()

        if is_full:
            self.X = X.detach()
            _, s, _ = torch.linalg.svd(X, full_matrices=False)
            self.nr_total_rank = len(s)
            
            k = math.ceil(self.nr_total_rank * self.rate_rank) - 1
            self.alpha = float(s[k] * self.rho)

            # self.alpha = 1.0e-5
            # Initialize SVD factors
            self.initialization()
        
        self.reset()

    def get_loss_pre_term(self, L, S, Y):   
        """
        Compute the loss term for the model.
        """
        if self.L_pre is None:
            self.L_pre = L.clone().detach()
            self.S_pre = S.clone().detach()

        loss = self.rho * torch.norm(L + S - self.L_pre - self.S_pre, p='fro')  
        return loss
    
    def get_loss_term(self, L, S, Y):
        """
        Compute the loss term for the model.
        """
        # loss = self.rho/2 * torch.norm(self.X_with_grad - L - S + Y/self.rho, p='fro') ** 2        
        loss = torch.norm(self.X_with_grad - L - S, p='fro')    
        self.nr_cals += 1
        self.total_loss += loss.item() / (self.nr_elements)
        return loss
    
    @torch.no_grad()
    def get_loss_info(self, L, S, Y):
        """
        Compute the loss term for the model.
        """
        Z = self.rho * (self.X_with_grad.detach() - L - S + Y/self.rho)
        loss_r = self.get_loss_term(L, S, Y)
        loss_s = self.get_loss_pre_term(L, S, Y)

        if self.ema_r is None:
            self.ema_r = loss_r.item()
            self.ema_s = loss_s.item()
        else:
            self.ema_r = self.gamma * self.ema_r + (1 - self.gamma) * loss_r.item()
            self.ema_s = self.gamma * self.ema_s + (1 - self.gamma) * loss_s.item()

        return self.layer_name, Z, loss_r
    
    def reset(self):
        """
        Reset the solver state for a new training epoch.
        """
        self.total_loss = 0.0
        self.nr_cals = 0

    def update_alpha(self, rate_rank: float, s: torch.Tensor, rho: float) -> float:
        """Return rate_rank-th value of s, then times rho."""
        total_rank = len(s)
        idx = int(total_rank * rate_rank)
        # find idx maximum singular value
        return max(s[idx], 0.0) * rho

    def single_step_PRCA(self,
                         X: torch.Tensor,
                         L: torch.Tensor,
                         S: torch.Tensor,
                         Y: torch.Tensor,
                         alpha: float,
                         beta: float,
                         rho: float) -> tuple:
        beta = self.update_beta(self.rate_sparsity, X - L + Y/rho, rho)
        S = soft_threshold(X - L + Y/rho, beta/rho)
        U, s, Vt = torch.linalg.svd(X - S + Y / rho, full_matrices=False)
        
        # self.alpha = self.update_alpha(self.rate_rank, s, rho)
        # alpha = self.alpha
        
        _s = soft_threshold(s, alpha/rho)
        L = U @ torch.diag(_s) @ Vt
        Y = Y + rho * (X - L - S)
        return L, S, Y, _s

    def update_alpha(self, 
                     rate_rank: float, 
                     s: torch.Tensor, 
                     rho: float,
                     eps: float=1e-4) -> float:
        """Update the alpha parameter based on the rank of singular values."""
        total_rank = len(s)
        idx = int(total_rank * rate_rank)
        # find idx maximum singular value
        return max(s[idx] - eps, 0.0) * rho
    
    def update_beta(self, 
                    rate_sparsity: float, 
                    S: torch.Tensor, 
                    rho: float,
                    scalar: float=1.0,
                    eps: float=1e-4) -> float:
        """Update the beta parameter based on the sparsity of the matrix."""
        vals, _ = torch.sort(S.abs().flatten(), descending=True)
        idx = int(len(vals) * rate_sparsity)
        # return (vals[idx] - eps) * rho * scalar  # in case the same values
        return vals[idx] * rho * scalar  # in case the same values

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
            L, S, Y, singular_values = self.single_step_PRCA(X, L, S, Y, alpha, beta, rho)
            nr_rank = get_energy_quantile(singular_values, quantile=energy_quantile)  # current rank
            nr_sparsity = torch.count_nonzero(S)

            if self.is_adaptive:                
                # self.dalpha = self.rho * (nr_rank / self.nr_total_rank - self.rate_rank) * self.rate_decay  # current rangk - target rank
                # self._rate_decay = tanh_ramp(epoch=self.nr_epoch, total_epochs=2200, a=self.rate_decay/10.0, b=self.rate_decay)
                self._rate_decay = self.rate_decay
                self.dalpha = self.rho * (nr_rank / self.nr_total_rank - self.rate_rank) * self._rate_decay  # current rangk - target rank
                self.dbeta = self.rho * (nr_sparsity / self.nr_elements - self.rate_sparsity) / 500.0 # current sparsity - target sparsity
                self.alpha = self.alpha + self.dalpha  # update alpha
                self.beta = self.beta + self.dbeta  # update beta
            if torch.linalg.norm(X - L - S, 'fro') < tol:
                break
        return L, S, Y, nr_rank

    def initialization(self) -> None:
        if self.init_energy <= 0:
            self.L = torch.zeros_like(self.X, device=self.device)
        else:
            U, s, Vt = torch.linalg.svd(self.X, full_matrices=False)
            nr_singular_values = int(len(s) * self.rate_rank)
            self.L = U[:, :nr_singular_values] @ torch.diag(s[:nr_singular_values]) @ Vt[:nr_singular_values, :]

        self.S = torch.zeros_like(self.X)
        self.Y = torch.zeros_like(self.X)

    def cal_results(self) -> None:
        """
        Calculate the results after running the solver.
        """
        self.results = {'L': self.L.to('cpu'),
                        'S': self.S.to('cpu'),
                        'Y': self.Y.to('cpu'),
                        'alpha': self.alpha,
                        'beta': self.beta,
                        'dalpha': self.dalpha,
                        'dbeta': self.dbeta,
                        'rho': self.rho,
                        'rate_decay': self._rate_decay,
                        'nr_rank': self.nr_rank,
                        'nr_nonzero': int(torch.count_nonzero(self.S)),
                        'nr_total_rank': self.nr_total_rank,
                        'nr_elements': self.nr_elements,
                        'avg_loss': (self.total_loss/self.nr_cals)}

    def clip_rho(self, 
                 rho: float, 
                 ema_r: float, 
                 ema_s: float, 
                 beta: float, 
                 rho_min: float, 
                 rho_max: float,
                 eps: float=1e-6) -> float:
        """Clip the rho value based on the ratio of loss_r and loss_s."""
        return max(min(rho * ((ema_r + eps) / (ema_s + eps))**beta, rho_max), rho_min)

    def run(self):
        if self.is_cal:
            # calibration the sparse matrix
            self.S = self.X - self.L

        self.nr_epoch += 1
        # self.rho = tanh_ramp(self.nr_epoch, 
        #                      total_epochs=1100,
        #                      a=0.01, 
        #                      b=0.1,
        #                      alpha=3.0)
        if self.nr_epoch > 2:
            self.rho = self.clip_rho(self.rho, self.ema_r, self.ema_s, self.rho_rate, self.rho_min, self.rho_max)


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
        
