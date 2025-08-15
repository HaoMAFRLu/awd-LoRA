"""Collection of utility functions for the lowspa_ddp package."""
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
import os
from pathlib import Path
import random
import numpy as np
import yaml
import re, math
import tempfile
import pickle
from matplotlib.axes import Axes
from loguru import logger
import time, random

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

def get_optimizer(name: str, params: dict, model: nn.Module):
    """
    Get the optimizer based on the provided parameters.
    """
    OptClass = getattr(optim, name, None)
    return OptClass(model.parameters(), **{k: v for k, v in params.items() if v is not None})

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
        return int(torch.where(energy >= quantile)[0][0])+1

def print_epoch(epoch: int, 
                total_epochs: int, 
                lr: float,
                num_tokens: int,
                losses: dict, 
                layer_stats: list):

    header = (f"Epoch {epoch}/{total_epochs} | "
              f"Lr: {lr:.6f} | "
              f"Tokens: {num_tokens / 1000000:.3f}M | "
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

def read_cfg(cfg_path: str) -> dict:
    """
    Read a configuration file and return its contents as a dictionary.
    Args:
        cfg_path: Path to the configuration file.
    Returns:
        Dictionary containing the configuration parameters.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_model_layer_names(model: torch.nn.Module):
    """
    Recursively collect all layer names in the model.
    Returns a list of parameter names.
    """
    return {name for name, _ in model.named_parameters()} 

def get_linear_layers_name(model):
    """
    Get the names of linear layers in the model.
    
    Args:
        model: The model to get linear layer names from.
    
    Returns:
        list: A list of names of linear layers in the model.
    """
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

def unwrap(m):
        return m.module if hasattr(m, "module") else m

def grad_norm_by_layer(model):
    m = unwrap(model)
    buckets = {}   # layer_idx -> sum(||grad||^2)
    others = 0.0   # 非 layers（如嵌入、lm_head）

    for name, p in m.named_parameters():
        if p.grad is None: 
            continue
        g = p.grad.detach().float()
        gn2 = g.norm().item() ** 2
        mobj = re.search(r"model\.layers\.(\d+)\.", name)  
        if mobj:
            idx = int(mobj.group(1))
            buckets[idx] = buckets.get(idx, 0.0) + gn2
        else:
            others += gn2

    for i in sorted(buckets):
        print(f"layer {i:2d}: ||g|| = {math.sqrt(buckets[i]):.4e}")
    print(f"others (embed/lm_head/etc): ||g|| = {math.sqrt(others):.4e}")

def find_group_of_param(optimizer, param):
    for g in optimizer.param_groups:
        if param in g["params"]:
            return g
    return None

def preprocess_batched(batch, tokenizer, max_length: int=256):
    batch = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return batch

def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch

def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def atomic_pickle_dump(obj, path):
    """Save an object to a file atomically."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno()) 
        os.replace(tmppath, path)
        try:
            dirfd = os.open(d, os.O_DIRECTORY)
            try: os.fsync(dirfd)
            finally: os.close(dirfd)
        except Exception:
            pass
    except Exception:
        try: os.remove(tmppath)
        except OSError: pass
        raise

def atomic_torch_save(state_dict, path):
    """Save a PyTorch state_dict to a file atomically."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state_dict, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmppath, path)
        try:
            dirfd = os.open(d, os.O_DIRECTORY)
            try: os.fsync(dirfd)
            finally: os.close(dirfd)
        except Exception:
            pass
    except Exception:
        try: os.remove(tmppath)
        except OSError: pass
        raise

def _set_axes_radius_2d(ax, origin, radius) -> None:
    x, y = origin
    ax.set_xlim([x - radius, x + radius])
    ax.set_ylim([y - radius, y + radius])

def set_axes_equal_2d(ax: Axes) -> None:
    """Set equal x, y axes
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius_2d(ax, origin, radius)

def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
    """Format the axes
    """
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

# def _print_setting(cfg: dict) -> None:  

def print_setting(cfg: dict, lvl=0) -> None:
    """Print the settings of the training
    """
    for key, value in cfg.items():
        if key == 'layers':
            pass
        else:
            if isinstance(value, dict):
                logger.info(f"{' ' * lvl}{key}:")
                print_setting(value, lvl + 2)
            elif isinstance(value, list):
                logger.info(f"{' ' * lvl}{key}: {', '.join(map(str, value))}")
            else:
                logger.info(f"{' ' * lvl}{key}: {value}")

def get_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    sub = model.get_submodule(layer_name)
    return sub.weight

def _is_429(err):
    # datasets/hf common 429
    code = getattr(getattr(err, "response", None), "status_code", None)
    return code == 429 or "Too Many Requests" in str(err)

def _backoff_sleep(attempt, base=0.5, cap=30.0, jitter=True):
    delay = min(cap, base * (2 ** attempt))
    if jitter:
        delay = random.uniform(0.0, delay)
    time.sleep(delay)

def resilient_enumerate(loader, start=0, max_retries=10, base=0.5, cap=30.0):
    it = iter(loader)
    idx = start
    while True:
        try:
            batch = next(it)
            yield idx, batch
            idx += 1
        except StopIteration:
            return
        except Exception as e:
            if not _is_429(e) or max_retries <= 0:
                raise
            retry_after = None
            if getattr(e, "response", None) is not None:
                retry_after = e.response.headers.get("Retry-After")
            for attempt in range(max_retries):
                if retry_after:
                    time.sleep(float(retry_after))
                else:
                    _backoff_sleep(attempt, base=base, cap=cap, jitter=True)
                try:
                    batch = next(it)
                    yield idx, batch
                    idx += 1
                    break
                except StopIteration:
                    return
                except Exception as e2:
                    if attempt == max_retries - 1 or not _is_429(e2):
                        raise
                    continue