import numpy as np
import torch

class PARAM():
    def __init__(self, cfg: dict):
        self.mode = cfg.get('mode', 'fixed')  # 'fixed', 'adaptive', 'hard_cut'
        self.nr_updates = 0
        self.pre_rate = 0.0
        self.start_epoch = 1500
        self.initialization(self.mode, cfg)

    def initialization(self, mode: str, cfg: dict):
        self.target_rate = cfg.get('target_rate', 0.1)

        if mode == 'fixed':
            self.value = cfg.get('init', 1e-5)
            self.rate_decay = 0.0
        elif mode == 'hard_cut':
            self.value = 0.0
            self.rate_decay = 0.0
        elif mode == 'adaptive':
            self.value = cfg.get('init', 1e-5)
            self.rate_decay = cfg.get('rate_decay', 0.02) 
            self.drate = cfg.get('drate', 0.01)
        else:
            raise ValueError(f"Unsupported param mode: {mode}")
    
        self.dvalue = 0.0
    
    def update(self, current_rate: float, rho: float, A: torch.Tensor=None):
        """Update the parameter based on the current rate.
        Args:
            current_rate (float): current rate (rank rate or sparsity rate)
            rho (float): rho parameter
            A (torch.Tensor): weight matrix (only needed for 'hard_cut' mode)
        """
        self.nr_updates += 1

        if self.mode == 'fixed':
            return

        elif self.mode == 'adaptive':
            # if self.nr_updates < self.start_epoch:  # apply only the p-controller
            self.dvalue = rho * (current_rate - self.target_rate) * self.rate_decay # current rate - target rate
            self.value = self.value + self.dvalue  # update parameter
            # else:  # apply the p-d controller
            #     rate_diff = max(0.0, current_rate - self.pre_rate)  
            #     self.dvalue = rho * (current_rate - self.target_rate) * self.rate_decay + rho * rate_diff * self.drate
            #     self.value = self.value + self.dvalue  # update parameter
            self.pre_rate = current_rate

        elif self.mode == 'hard_cut':
            return
        
    def update_quantile(self,  
                        S: torch.Tensor, 
                        rho: float,
                        scalar: float=1.0,
                        eps: float=1e-4) -> None:
        """Update the beta parameter based on the sparsity of the matrix."""
        vals, _ = torch.sort(S.abs().flatten(), descending=True)
        idx = int(len(vals) * self.target_rate)
        self.value = vals[idx] * rho * scalar  # in case the same values