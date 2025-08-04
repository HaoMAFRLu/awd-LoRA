"""Custurm learning rate scheduler for distributed data parallel training.
"""
import math

class GPTScheduler:
    def __init__(self,
                 warmup_iters: int = 2000,
                 lr_decay_iters: int = 200_000,
                 min_lr: float = 6e-5,
                 lr: float = 6e-4):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.lr = lr
        self.lr_list = []
    
    def get_lr(self, it) -> float:
        """
        Calculate the learning rate based on the current iteration.
        """
        if it < self.warmup_iters:
            _lr = self.lr * (it / self.warmup_iters)
        elif it > self.lr_decay_iters:
            _lr = self.min_lr
        
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1, "Decay ratio must be between 0 and 1"
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        _lr = self.min_lr + (self.lr - self.min_lr) * coeff

        self.lr_list.append(_lr)
        return _lr
    
    def get_last_lr(self) -> float:
        """
        Get the last calculated learning rate.
        """
        return [self.lr_list[-1]]