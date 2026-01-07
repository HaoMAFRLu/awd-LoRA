from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Helpers
# -----------------------------
def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    name_to_mod = dict(model.named_modules())
    if name not in name_to_mod:
        raise KeyError(f"Module '{name}' not found in model.named_modules().")
    return name_to_mod[name]


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _module_persistent_bytes(m: nn.Module) -> int:
    """Inference-only persistent memory for a module: parameters + buffers (recursively)."""
    total = 0
    for p in m.parameters(recurse=True):
        total += p.numel() * p.element_size()
    for b in m.buffers(recurse=True):
        total += b.numel() * b.element_size()
    return total


def _model_persistent_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def _is_embedding(m: nn.Module) -> bool:
    return isinstance(m, nn.Embedding)


def _is_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear)


def _rank_from_ratio(rank_ratio: float, base_dim: int) -> int:
    if rank_ratio <= 0:
        raise ValueError(f"rank_ratio must be > 0, got {rank_ratio}")
    return max(1, int(round(rank_ratio * base_dim)))


# -----------------------------
# Memory estimators (inference-only)
# -----------------------------
def estimate_abs_linear_bytes(
    in_features: int,
    out_features: int,
    rank_ratio: float,
    density: float,
    weight_bytes: int,
    index_bytes: int = 4,
    sparse_index_factor: int = 1,
) -> int:
    """
    Linear W[out, in] approximated as AB + S with:
      A[out, r], B[r, in] dense
      S sparse with nnz = density * out * in
    Storage (inference-only):
      bytes(A,B) = (out*r + r*in) * weight_bytes
      bytes(S)   = nnz * (value_bytes + sparse_index_factor * index_bytes)
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0,1], got {density}")

    r = _rank_from_ratio(rank_ratio, min(in_features, out_features))
    nnz = int(round(density * out_features * in_features))

    bytes_ab = (out_features * r + r * in_features) * weight_bytes
    bytes_s = nnz * (weight_bytes + sparse_index_factor * index_bytes)
    return int(bytes_ab + bytes_s)


def estimate_abs_embedding_bytes(
    vocab_size: int,
    hidden_size: int,
    rank_ratio: float,
    density: float,
    weight_bytes: int,
    index_bytes: int = 4,
    sparse_index_factor: int = 1,
) -> int:
    """
    Embedding E[V, H] approximated as AB + S with:
      A[V, r], B[r, H] dense
      S sparse with nnz = density * V * H
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0,1], got {density}")

    r = _rank_from_ratio(rank_ratio, min(vocab_size, hidden_size))
    nnz = int(round(density * vocab_size * hidden_size))

    bytes_ab = (vocab_size * r + r * hidden_size) * weight_bytes
    bytes_s = nnz * (weight_bytes + sparse_index_factor * index_bytes)
    return int(bytes_ab + bytes_s)


@dataclass
class MemoryEstimate:
    original_total_bytes: int
    total_bytes: int
    replaced_bytes: int
    constant_bytes_D: int
    per_layer_bytes: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "replaced_bytes": self.replaced_bytes,
            "constant_bytes_D": self.constant_bytes_D,
            "per_layer_bytes": dict(self.per_layer_bytes),
            "total_GB": self.total_bytes / 1e9,
            "replaced_GB": self.replaced_bytes / 1e9,
            "constant_D_GB": self.constant_bytes_D / 1e9,
        }


def estimate_inference_memory_cost(
    model: nn.Module,
    spec: Dict[str, Dict[str, float]],
    dtype: torch.dtype = torch.bfloat16,
    index_bytes: int = 4,
    sparse_index_factor: int = 2,
    include_buffers_in_D: bool = True,
    strict: bool = True,
) -> MemoryEstimate:
    """
    Estimate inference-only model-state memory cost under AB+S replacement for specified layers.

    Parameters
    ----------
    model:
        Original dense model (used only to read in/out shapes and compute constant D).
    spec:
        dict: layer_name -> {"rank_ratio": float, "density": float}
        layer_name must match a key in model.named_modules().
        Supports nn.Linear and nn.Embedding.
    dtype:
        Weight dtype for the estimated AB+S parameters (bf16 by default).
    index_bytes:
        Bytes for sparse indices (int32=4 recommended, int64=8).
    sparse_index_factor:
        1: row-wise/top-k-like (one index per nnz, e.g., col index; row implied)
        2: COO-like upper bound (two indices per nnz: row+col)
    include_buffers_in_D:
        If True, D includes buffers as well as parameters.
    strict:
        If True, raise on unsupported module types. If False, skip unsupported keys.

    Returns
    -------
    MemoryEstimate:
        total_bytes = replaced_bytes + constant_bytes_D
        constant_bytes_D = (original model persistent bytes) - (original bytes of targeted modules)
    """
    wbytes = _dtype_bytes(dtype)

    # Original persistent bytes (parameters + optionally buffers)
    if include_buffers_in_D:
        original_total = _model_persistent_bytes(model)
    else:
        original_total = sum(p.numel() * p.element_size() for p in model.parameters())

    # Original bytes of targeted modules (to form D)
    original_target = 0
    for name in spec.keys():
        m = _get_module_by_name(model, name)
        if include_buffers_in_D:
            original_target += _module_persistent_bytes(m)
        else:
            original_target += sum(p.numel() * p.element_size() for p in m.parameters(recurse=True))

    constant_D = int(original_total - original_target)

    # Estimated bytes for replaced layers (AB+S storage)
    per_layer: Dict[str, int] = {}
    replaced_total = 0

    for name, cfg in spec.items():
        if "rank_ratio" not in cfg or "density" not in cfg:
            raise ValueError(f"Spec for '{name}' must contain 'rank_ratio' and 'density'.")

        rr = float(cfg["rank_ratio"])
        dens = float(cfg["density"])

        m = _get_module_by_name(model, name)

        if _is_linear(m):
            b = estimate_abs_linear_bytes(
                in_features=int(m.in_features),
                out_features=int(m.out_features),
                rank_ratio=rr,
                density=dens,
                weight_bytes=wbytes,
                index_bytes=index_bytes,
                sparse_index_factor=sparse_index_factor,
            )
        elif _is_embedding(m):
            V, H = int(m.weight.shape[0]), int(m.weight.shape[1])
            b = estimate_abs_embedding_bytes(
                vocab_size=V,
                hidden_size=H,
                rank_ratio=rr,
                density=dens,
                weight_bytes=wbytes,
                index_bytes=index_bytes,
                sparse_index_factor=sparse_index_factor,
            )
        else:
            if strict:
                raise TypeError(
                    f"Module '{name}' has unsupported type {type(m)}. "
                    "Only nn.Linear and nn.Embedding are supported."
                )
            else:
                continue

        per_layer[name] = int(b)
        replaced_total += int(b)

    total = int(constant_D + replaced_total)

    return MemoryEstimate(
        original_total_bytes=original_total,
        total_bytes=total,
        replaced_bytes=int(replaced_total),
        constant_bytes_D=int(constant_D),
        per_layer_bytes=per_layer,
    )


# -----------------------------
# Example
# -----------------------------
# spec = {
#     "model.layers.0.self_attn.q_proj": {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.mlp.up_proj":      {"rank_ratio": 0.05, "density": 0.01},
#     "model.embed_tokens":             {"rank_ratio": 0.01, "density": 0.001},  # embedding (optional)
# }
#
# mem = estimate_inference_memory_cost(
#     model,
#     spec,
#     dtype=torch.bfloat16,
#     index_bytes=4,
#     sparse_index_factor=1,   # 1=row-wise/top-k-like; 2=COO upper bound
# )
# print(mem.as_dict())
