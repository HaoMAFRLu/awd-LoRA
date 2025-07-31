import torch

from lowspa_ddp.utils import *

class ADMMSolver():
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

        # read the params
        for key, val in params.items():
            setattr(self, key, val)

        self.dalpha = 0.0
        self.dbeta = 0.0

        # overwrite the hyperparameters
        if self.is_adaptive:
            row, col = X.shape  
            # self.rho = 1.0 / (np.sqrt(nr_layers * max(row, col)))
            self.rho = 1.0 / (np.sqrt(max(row, col)))
        
        self.alpha = self.rho * self.alpha_to_rho
        self.beta = self.rho * self.beta_to_rho

        if is_full:
            self.X = X.detach()
            _, s, _ = torch.linalg.svd(X, full_matrices=False)
            self.nr_total_rank = len(s)
            self.nr_elements = X.numel()
            # Initialize SVD factors
            self.initialization()

    def get_loss_term(self, L, S, Y):
        """
        Compute the loss term for the model.
        """
        # def _dev(x): 
        #     return str(x.device) if torch.is_tensor(x) else f"<non-tensor:{type(x)}>"

        # print(
        #     "[debug devices]",
        #     "current:", torch.cuda.current_device(),
        #     "X:", _dev(self.X_with_grad),
        #     "L:", _dev(L),
        #     "S:", _dev(S),
        #     "Y:", _dev(Y),
        #     "rho:", _dev(self.rho) if isinstance(self.rho, torch.Tensor) else type(self.rho)
        # )

        if self.loss_version == 'v1':
            loss = self.rho/2 * torch.norm(self.X_with_grad - L - S + Y/self.rho, p='fro') ** 2
        elif self.loss_version == 'v2':
            loss = self.rho/2 * torch.norm(self.X_with_grad - L, p='fro') ** 2
        
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
                         rho: float) -> tuple:
        U, s, Vt = torch.linalg.svd(X - S + Y / rho, full_matrices=False)
        _s = soft_threshold(s, alpha/rho)
        L = U @ torch.diag(_s) @ Vt
        S = soft_threshold(X - L + Y/rho, beta/rho)
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
        return (vals[idx] - eps) * rho * scalar  # in case the same values

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
            if self.layer_name == 'transformer.wte.weight':
                print('here')
            L, S, Y, singular_values = self.single_step_PRCA(X, L, S, Y, alpha, beta, rho)
            nr_rank = get_energy_quantile(singular_values, quantile=energy_quantile)  # current rank
            nr_sparsity = torch.count_nonzero(S)

            if self.is_adaptive:                
                # self.dalpha = (1 - self.rate_decay)*self.update_alpha(self.rate_rank, singular_values, self.rho)
                # self.dbeta = (1 - self.rate_decay)*self.update_beta(self.rate_sparsity, S, self.rho)
                # self.alpha = self.rate_decay*self.alpha + self.dalpha
                # self.beta = self.rate_decay*self.beta + self.dbeta
                self.dalpha = self.rho * (nr_rank / self.nr_total_rank - self.rate_rank) / 50.0  # current rangk - target rank
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
            nr_singular_values = int(len(s) * self.init_energy)
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
                        'nr_rank': self.nr_rank,
                        'nr_nonzero': int(torch.count_nonzero(self.S)),
                        'nr_total_rank': self.nr_total_rank,
                        'nr_elements': self.nr_elements,
                        'avg_loss': (self.total_loss/self.nr_cals) / (self.rho/2.0)}

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
        
