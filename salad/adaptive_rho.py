
import numpy as np

class RHO():
    def __init__(self, cfg: dict):
        self.mode = cfg.get('mode', 'fixed')  # 'fixed', 'adaptive'
        self.start_epoch = cfg.get('start_epoch', 0)
        self.initialization(self.mode, cfg)
    
    def tanh_ramp_steps(self, a, b, N, mid_step, W):
        """
        Tanh-shaped transition from a to b over N steps,
        with controllable midpoint and transition speed.

        Args:
            a (float): initial value
            b (float): final value
            N (int): total number of steps
            mid_step (int or float): step index where f = (a+b)/2
            W (float): transition width (controls speed)

        Returns:
            t (np.ndarray): steps [0, ..., N-1]
            y (np.ndarray): values from a to b
        """
        t = np.arange(N)
        k = 2.944439 / W                # controls steepness (≈ 5% → 95%)
        s = 0.5 * (1.0 + np.tanh(k * (t - mid_step)))
        y = a + (b - a) * s
        return t, y

    def initialization(self, mode: str, cfg: dict):
        if mode == 'fixed':
            self.rho = cfg.get('rho', 1e-3)
        elif mode == 'shape_dependent':
            row = cfg.get('row', 512)
            col = cfg.get('col', 512)
            nr_layers = cfg.get('nr_layers', 12)
            self.rho = 1.0 / (nr_layers * np.sqrt(row * col))
        elif mode == 'adaptive':
            # X_norm = cfg.get('X_norm', 1.0)
            # row = cfg.get('row', 512)
            # col = cfg.get('col', 512)
            # coeff_rho = cfg.get('coeff_rho', 1.0)
            # coeff_rho_min = cfg.get('coeff_rho_min', 0.1)
            # coeff_rho_max = cfg.get('coeff_rho_max', 10.0)
            # self.rho_rate = cfg.get('rho_rate', 1.1)

            # _rho = (X_norm / np.sqrt(max(row, col)))
            # self.rho = coeff_rho * _rho
            # self.rho_min = coeff_rho_min * _rho
            # self.rho_max = coeff_rho_max * _rho
            self.rho = cfg.get('rho', 1e-3)
            _, self.rho_list = self.tanh_ramp_steps(self.rho,
                                                    self.rho * 5.0,
                                                    3100,
                                                    mid_step=2000,
                                                    W=100)
        else:
            raise ValueError(f"Unsupported rho mode: {mode}")
        
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

    def update_rho(self, ema_r: float, ema_s: float) -> float:
        self.rho = self.clip_rho(self.rho, ema_r, ema_s, 
                                 self.rho_rate, self.rho_min, self.rho_max)
        return self.rho
    
    def get_rho(self,
                   nr_epoch: int,
                   ema_r: float=None,
                   ema_s: float=None) -> float:
        if nr_epoch < self.start_epoch:
            return self.rho
        else:
            if self.mode == 'fixed' or self.mode == 'shape_dependent':
                return self.rho
            elif self.mode == 'adaptive':
                # if ema_r is None or ema_s is None:
                #     raise ValueError("For adaptive rho, ema_r and ema_s must be provided.")
                # return self.update_rho(ema_r, ema_s)
                return self.rho_list[nr_epoch - 1]
            else:
                raise ValueError(f"Unsupported rho mode: {self.mode}")