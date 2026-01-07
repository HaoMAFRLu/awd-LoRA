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


def _is_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear)


def _rank_from_ratio(rank_ratio: float, base_dim: int) -> int:
    if rank_ratio <= 0:
        raise ValueError(f"rank_ratio must be > 0, got {rank_ratio}")
    return max(1, int(round(rank_ratio * base_dim)))


# -----------------------------
# Per-token FLOPs for AB+S Linear
# -----------------------------
def per_token_flops_abs_linear(
    in_features: int,
    out_features: int,
    rank_ratio: float,
    density: float,
    macs_to_flops: int = 2,
) -> float:
    """
    per-token FLOPs for AB+S replacing W[out, in], assuming true sparse compute for S:

      FLOPs/token = 2 * (in*r + r*out + nnz)
      nnz = density * in * out

    rank r = round(rank_ratio * min(in, out)), at least 1.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0,1], got {density}")
    r = _rank_from_ratio(rank_ratio, min(in_features, out_features))
    nnz = density * in_features * out_features
    return float(macs_to_flops * (in_features * r + r * out_features + nnz))


# -----------------------------
# Optional: constant background FLOPs/token (unchanged parts)
# -----------------------------
def per_token_flops_background_constant(
    H: int,
    T: int,
    n_heads: int,
    num_layers: int,
    c_softmax: int = 6,
    include_layernorm: bool = False,
    c_layernorm: int = 8,
) -> float:
    """
    Approximate constant per-token FLOPs for parts that are unchanged when you only
    modify linear projections.

    For full-sequence prefill/eval attention (PPL evaluation):
      QK^T FLOPs/token per layer ≈ 2 * T * H
      AV   FLOPs/token per layer ≈ 2 * T * H
      softmax FLOPs/token per layer ≈ c_softmax * n_heads * T

    So attention core per layer:
      ≈ 4*T*H + c_softmax*n_heads*T

    LayerNorm (optional, small):
      2 LayerNorms per block ≈ 2 * c_layernorm * H FLOPs/token per layer
    """
    attn_core = num_layers * (4.0 * T * H + float(c_softmax) * n_heads * T)
    ln = 0.0
    if include_layernorm:
        ln = num_layers * (2.0 * float(c_layernorm) * H)
    return float(attn_core + ln)


# -----------------------------
# Main API
# -----------------------------
@dataclass
class FlopsEstimate:
    flops_per_token_linear: int
    flops_per_token_background: int
    flops_per_token_total: int
    per_layer_linear: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "FLOPs/token (linear replaced)": self.flops_per_token_linear,
            "FLOPs/token (background const)": self.flops_per_token_background,
            "FLOPs/token (total)": self.flops_per_token_total,
            "Per-layer FLOPs/token (linear)": dict(self.per_layer_linear),
        }


def estimate_per_token_flops(
    model: nn.Module,
    spec: Dict[str, Dict[str, float]],
    *,
    # embedding is always ignored (FLOPs=0) by design
    include_background_constant: bool = False,
    # background params (only used if include_background_constant=True)
    T: Optional[int] = None,
    H: Optional[int] = None,
    n_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    c_softmax: int = 6,
    include_layernorm: bool = False,
    c_layernorm: int = 8,
    macs_to_flops: int = 2,
    strict: bool = True,
) -> FlopsEstimate:
    """
    Estimate per-token FLOPs for AB+S replacements specified in `spec`.
    Assumes:
      - only nn.Linear modules in spec contribute to FLOPs changes
      - embedding is ignored / FLOPs=0

    spec: layer_name -> {"rank_ratio": float, "density": float}

    If include_background_constant=True, you must provide T, H, n_heads, num_layers
    to add the constant attention-core (and optional LN) FLOPs/token.
    """
    per_layer: Dict[str, float] = {}
    flops_linear = 0.0

    for name, cfg in spec.items():
        if "rank_ratio" not in cfg or "density" not in cfg:
            raise ValueError(f"Spec for '{name}' must contain 'rank_ratio' and 'density'.")

        m = _get_module_by_name(model, name)

        # Ignore embedding (by requirement); also ignore any non-linear layers if strict=False
        if not _is_linear(m):
            if strict:
                raise TypeError(
                    f"Module '{name}' is type {type(m)}; only nn.Linear is counted for FLOPs. "
                    "Embedding is ignored by design."
                )
            else:
                continue

        rr = float(cfg["rank_ratio"])
        dens = float(cfg["density"])

        f = per_token_flops_abs_linear(
            in_features=int(m.in_features),
            out_features=int(m.out_features),
            rank_ratio=rr,
            density=dens,
            macs_to_flops=macs_to_flops,
        )
        per_layer[name] = float(f)
        flops_linear += float(f)

    flops_bg = 0.0
    if include_background_constant:
        if T is None or H is None or n_heads is None or num_layers is None:
            raise ValueError("To include background constant FLOPs/token, provide T, H, n_heads, num_layers.")
        flops_bg = per_token_flops_background_constant(
            H=int(H),
            T=int(T),
            n_heads=int(n_heads),
            num_layers=int(num_layers),
            c_softmax=c_softmax,
            include_layernorm=include_layernorm,
            c_layernorm=c_layernorm,
        )

    flops_total = float(flops_linear + flops_bg)

    return FlopsEstimate(
        flops_per_token_linear=int(flops_linear),
        flops_per_token_background=int(flops_bg),
        flops_per_token_total=int(flops_total),
        per_layer_linear=per_layer,
    )


# -----------------------------
# Example
# -----------------------------
# spec = {
#     "model.layers.0.self_attn.q_proj": {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.self_attn.k_proj": {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.self_attn.v_proj": {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.self_attn.o_proj": {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.mlp.up_proj":      {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.mlp.gate_proj":    {"rank_ratio": 0.05, "density": 0.01},
#     "model.layers.0.mlp.down_proj":    {"rank_ratio": 0.05, "density": 0.01},
#     # embedding keys can appear but are ignored (FLOPs=0) if strict=False
#     # "model.embed_tokens":             {"rank_ratio": 0.01, "density": 0.001},
# }
#
# fl = estimate_per_token_flops(
#     model,
#     spec,
#     include_background_constant=True,  # or False if you want only replaced linear FLOPs
#     T=2048,
#     H=model.config.hidden_size,
#     n_heads=model.config.num_attention_heads,
#     num_layers=model.config.num_hidden_layers,
#     strict=False,
# )
# print(fl.as_dict())
