"""Collection of utility functions for the lowspa_ddp package."""
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import os
from pathlib import Path
import torch.nn.functional as F
import random
import numpy as np

class GPTCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, T, V)
            targets: (B, T)
        """
        B, T, V = logits.size()
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index
        )
    
def mkdir(path: Path) -> None:
    """Check if the folder exists and create it if it does not."""
    os.makedirs(path, exist_ok=True)
    
def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def soft_threshold(x: torch.Tensor, threshold: float):
    """
    Apply soft thresholding to the input tensor.
    Args:
        x: Input tensor.
        threshold: Threshold value.
    Returns:
        Soft-thresholded tensor.
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))

def get_loss_fn(model_type: str) -> nn.Module:
    """
    Get the loss function based on the provided parameters.
    """
    if model_type == 'CNN':
        loss_fn = nn.CrossEntropyLoss()
    elif model_type == 'GPT':
        loss_fn = GPTCrossEntropyLoss(ignore_index=-1)
    return loss_fn

def get_optimizer(name: str, params: dict, model: nn.Module):
    """
    Get the optimizer based on the provided parameters.
    """
    OptClass = getattr(optim, name, None)
    return OptClass(model.parameters(), **{k: v for k, v in params.items() if v is not None})

def get_scheduler(name: str, params: dict,
                    optimizer: optim.Optimizer):
    """
    Get the learning rate scheduler based on the provided parameters.
    """
    SchedClass = getattr(optim.lr_scheduler, name, None)
    return SchedClass(optimizer, 
                      **{k: v for k, v in params.items() if v is not None})

def get_energy_quantile(s, quantile=0.9) -> int:
    """
    Calculate the index of the energy quantile in the singular values.
    Args:
        s: Singular values tensor.
        quantile: Energy quantile to calculate (default is 0.9).
    Returns:
        idx: Index of the singular value that reaches the specified energy quantile.
    """
    total_energy = torch.sum(s**2)
    if total_energy == 0:
        return 0
    else:
        energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
        return int(torch.where(energy >= quantile)[0][0])

def print_epoch(epoch: int, 
                total_epochs: int, 
                lr: float,
                losses: dict, 
                layer_stats: list):

    header = (f"Epoch {epoch}/{total_epochs} | "
              f"Lr: {lr:.6f} | "
              f"Loss: {losses['loss']:.6f} | "
              f"Loss1: {losses['loss1']:.6f} | "
              f"Loss2: {losses['loss2']:.6f}")
    print(header)

    headers = ["name", "layer loss", "non-zero", "rank", "alpha", "dalpha", "beta", "dbeta", "rho"]
    rows = [
        [s["name"], 
         f"{s['loss']:.6f}", 
         f"{s['non_zero']}/{s['total_elements']} ({100. * s['non_zero']/s['total_elements']:.2f}%)", 
         f"{s['rank']}/{s['total_rank']} ({100. * s['rank']/s['total_rank']:.2f}%)",
         f"{s['alpha']:.8f}", 
         f"{s['dalpha']:.8f}",
         f"{s['beta']:.8f}",
         f"{s['dbeta']:.8f}",
         f"{s['rho']:.8f}"]
        for s in layer_stats
    ]

    print(tabulate(rows, headers=headers, tablefmt="grid"))

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in the model.
    Args:
        model: The model to count parameters for.
    Returns:
        Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())

def set_seed(seed: int):
    # 1) Python built‑ins
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2) Numpy
    np.random.seed(seed)
    # 3) PyTorch CPU
    torch.manual_seed(seed)
    # 4) PyTorch GPU (all devices)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 5) CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False