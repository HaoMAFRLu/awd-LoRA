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
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import io
from salad.register import get_data

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



def print_wandb(
        run,
        *,
        epoch: int,
        total_epochs: int,
        num_freq: int,
        lr: float,
        num_tokens: float,
        losses: dict,
        layer_stats: list,
    ):
    """
    Upload per-layer metrics as scalar time series to W&B (no tables, no local plots).
    Each metric becomes a series like: layer/<name>/<metric>.
    Then you can create charts in the W&B UI by selecting these series.

    Logged (per run epoch):
      Global series:
        - train/loss, train/layer_diff, train/penalty, train/lr, train/tokens_M, epoch
      Per-layer series (for each layer name):
        - layer/<name>/layer_diff
        - layer/<name>/non_zero_ratio
        - layer/<name>/rank_ratio
        - layer/<name>/alpha
        - layer/<name>/dalpha
        - layer/<name>/beta
        - layer/<name>/dbeta
        - layer/<name>/alpha_decay
        - layer/<name>/beta_decay
        - layer/<name>/rho
    """
    # Build one flat payload per epoch (faster & consistent than many small logs)
    payload = {
        # global scalars (optional; remove if you truly only want per-layer series)
        "train/loss": float(losses.get("avg_loss", float("nan"))),
        "train/layer_diff": float(losses.get("avg_diff", float("nan"))),
        "train/penalty": float(losses.get("avg_loss_penalty", float("nan"))),
        "train/lr": float(lr),
        "train/tokens_M": float(num_tokens) / 1e6,
        # DO NOT add "epoch": step already carries this
    }

    for s in layer_stats:
        name = s.get("name")
        if not name:
            continue

        # Ratios
        nz = int(s.get("non_zero", 0))
        tot = int(s.get("total_elements", 0))
        non_zero_ratio = float(nz / tot) if tot else 0.0

        rnk = int(s.get("rank", 0))
        trk = int(s.get("total_rank", 0))
        rank_ratio = float(rnk / trk) if trk else 0.0

        # Scalars for params
        layer_diff = float(s.get("loss", float("nan")))
        alpha  = float(s.get("alpha",  float("nan")))
        dalpha = float(s.get("dalpha", float("nan")))
        beta   = float(s.get("beta",   float("nan")))
        dbeta  = float(s.get("dbeta",  float("nan")))
        rho    = float(s.get("rho",    float("nan")))

        # Grouped metric names (easy to pick in W&B UI)
        prefix = f"layer/{name}"
        payload.update({
            f"{prefix}/diff": layer_diff,                                 
            f"{prefix}/non_zero_ratio": non_zero_ratio,          
            f"{prefix}/rank_ratio": rank_ratio,                   
            f"{prefix}/alpha": alpha,                              
            f"{prefix}/beta": beta,                             
            f"{prefix}/rho": rho,                           
        })

    # Single log call per epoch
    wandb.log(payload, step=epoch)

def print_epoch(epoch: int, 
                total_epochs: int, 
                num_freq: int,
                lr: float,
                num_tokens: int,
                losses: dict, 
                layer_stats: list):

    header = (f"Epoch {epoch}/{total_epochs} | "
              f"It {epoch * num_freq}/{total_epochs * num_freq} | "
              f"Lr: {lr:.6f} | "
              f"Tokens: {num_tokens / 1000000:.3f}M | "
              f"Loss: {losses['avg_loss']:.6f} | "
              f"Layer diff: {losses['avg_diff']:.6f} | "
              f"Penalty: {losses['avg_loss_penalty']:.6f}")
    print(header)

    headers = ["name", "layer diff", "non-zero", "rank", 
               "mode", "alpha", "dalpha", "decay", 
               "mode", "beta", "dbeta", "decay", "rho"]
    rows = [
        [s["name"], 
         f"{s['loss']:.6f}", 
         f"{s['non_zero']}/{s['total_elements']} ({100. * s['non_zero']/s['total_elements']:.2f}%)", 
         f"{s['rank']}/{s['total_rank']} ({100. * s['rank']/s['total_rank']:.2f}%)",
         f"{s['alpha_mode']}",
         f"{s['alpha']:.12f}", 
         f"{s['dalpha']:.12f}",
         f"{s['rate_decay_alpha']:.6f}",
         f"{s['beta_mode']}",
         f"{s['beta']:.8f}",
         f"{s['dbeta']:.8f}",
         f"{s['rate_decay_beta']:.6f}",
         f"{s['rho']:.12f}"
        ]
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
    ll = ['model.embed_tokens']
    return ll + [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

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

def _get_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    sub = model.get_submodule(layer_name)
    return sub.weight

def get_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    candidates = [
        f"module.model.{layer_name}",
        f"module.{layer_name}",
        f"model.{layer_name}",
        layer_name
    ]

    for candidate in candidates:
        try:
            return _get_weight(model, candidate)
        except (KeyError, AttributeError):
            continue

    raise KeyError(f"Weight not found for layer '{layer_name}'. Tried: {candidates}")

def hf_login_once():
    os.environ.setdefault("HF_HOME", "/lustre/home/hma2/hf")
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))

    if not os.environ.get("HF_TOKEN"):
        try:
            with open(os.path.join(os.environ["HF_HOME"], "token"), "r") as f:
                os.environ["HF_TOKEN"] = f.read().strip()
        except FileNotFoundError:
            pass

def tanh_ramp(epoch, total_epochs=1100, a=1e-6, b=1e-4, alpha=3.0, inflect_at=0.3):
    """
    """
    if total_epochs <= 1:
        return float(b)
    e = max(0, min(int(epoch), total_epochs - 1))
    x = 2.0 * e / (total_epochs - 1) - 1.0
    delta = 2.0 * inflect_at - 1.0
    s_raw  = math.tanh(alpha * (x - delta))
    s_min  = math.tanh(alpha * (-1.0 - delta))  # x=-1
    s_max  = math.tanh(alpha * ( 1.0 - delta))  # x=+1
    s = (s_raw - s_min) / (s_max - s_min)

    return a + (b - a) * s

def get_param_tensor(param_dict, name, attr="weight"):
    """
    """
    candidates = [
        f"module.model.{name}.{attr}",
        f"module.{name}.{attr}",
        f"model.{name}.{attr}",
        f"{name}.{attr}",
    ]
    for k in candidates:
        if k in param_dict:
            return param_dict[k]
    raise KeyError(f"Parameter not found for layer '{name}' (attr='{attr}'). Tried: {candidates}")

# def load_model(model: torch.nn.Module, pth: str) -> torch.nn.Module:
#     """
#     Load the model from the given path.
    
#     Args:
#         model (torch.nn.Module): The model to load.
#         pth (str): Path to the model checkpoint.
    
#     Returns:
#         torch.nn.Module: The loaded model.
#     """
#     ckpt = torch.load(pth, map_location="cpu")
#     state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
#     clean_sd = {}
#     for k, v in state_dict.items():
#         while k.startswith("module."):
#             k = k[len("module."):]
#         clean_sd[k] = v

#     model.load_state_dict(clean_sd, strict=True)

def load_model(model, pth):
    # names = get_linear_layers_name(model)
    # p = get_weight(model, names[0]).clone()

    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # DDP?
    is_ddp = isinstance(model, DDP)
    prefix = "module."

    has_module_prefix = all(k.startswith(prefix) for k in sd.keys())

    if has_module_prefix and not is_ddp:
        # from DDP ckpt -> model
        sd = {k[len(prefix):]: v for k, v in sd.items()}
    elif (not has_module_prefix) and is_ddp:
        # from ckpt -> DDP model
        sd = {prefix + k: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_model] missing:", missing)
        print("[load_model] unexpected:", unexpected)

    # pp = get_weight(model, names[0]).clone()

    return model

def get_lowspa_layers(pth: str) -> tuple:
    """Load data from the files"""
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location='cpu', weights_only=False
        )
        with open(pth, 'rb') as f:
            obj = pickle.load(f) 
    finally:
        torch.storage._load_from_bytes = orig
    return obj['LL'], obj['SS']

def get_eval_data(split: str, 
                  seed_for_shuffle: int = 42, 
                  tokenizer=None, 
                  max_length=1024,
                  batch_size: int = 32):
    _data = get_data(seed_for_shuffle, split=split)
    _data_mapped = _data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    _data_mapped.batch = lambda batch_size: batch_fn(_data_mapped, batch_size)
    return _data_mapped

def get_ex_layers(layers: list, model, LL: dict, SS: dict, nr_remove: int) -> list:
    ex_layers = []
    loss = {}
    _list = []
    for layer in layers:
        L = LL[layer]
        S = SS[layer]

        X = model.get_submodule('model.'+layer).weight.data
        loss[layer] = torch.norm(X - L - S, p='fro').item() / X.numel()  # average per element
        _list.append(torch.norm(X - L - S, p='fro').item())

    sorted_layers = sorted(loss.items(), key=lambda item: item[1], reverse=True)
    for i in range(nr_remove):
        ex_layers.append(sorted_layers[i][0])

    return ex_layers

def get_rank(X: torch.Tensor, 
             energy_quantile: float=0.999) -> int:
    """Get the rank of the matrix X based on the energy quantile."""
    _, s, _ = torch.linalg.svd(X, full_matrices=False)
    energy = torch.cumsum(s, dim=0) / torch.sum(s)
    rank = torch.sum(energy < energy_quantile).item() + 1
    return rank