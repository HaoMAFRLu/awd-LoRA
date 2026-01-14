from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) The replacement module
# -----------------------------
class SLRBlock(nn.Module):
    """
    Inference-only linear replacement:

        y = x @ (A@B + S)^T
          = (x @ B^T) @ A^T + x @ S^T

    Shapes:
      A: [out, r]
      B: [r, in]
      S: [out, in]   (dense residual)

    No bias. Parameters are stored as buffers (frozen).
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, S: torch.Tensor):
        super().__init__()

        # --- shape checks ---
        if A.dim() != 2 or B.dim() != 2 or S.dim() != 2:
            raise ValueError("A, B, S must be 2D tensors")

        out, r = A.shape
        r2, inn = B.shape
        out2, inn2 = S.shape

        if r2 != r:
            raise ValueError(f"B shape {B.shape} incompatible with A shape {A.shape}: B[0] must equal rank r")
        if out2 != out or inn2 != inn:
            raise ValueError(f"S shape {S.shape} must match (out,in)=({out},{inn}) from A,B")

        self.in_features = int(inn)
        self.out_features = int(out)
        self.rank = int(r)

        # Inference-only: store as buffers (frozen)
        self.register_buffer("A", A, persistent=True)
        self.register_buffer("B", B, persistent=True)
        self.register_buffer("S", S, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        returns: [..., out_features]
        """
        # Low-rank branch
        xB = F.linear(x, self.B)     # weight [r, in]  -> [..., r]
        y_lr = F.linear(xB, self.A)  # weight [out, r] -> [..., out]

        # Dense residual branch
        y_res = F.linear(x, self.S)  # weight [out, in] -> [..., out]

        return y_lr + y_res


# -----------------------------
# 2) Input spec: layer name + A/B/S
# -----------------------------
@dataclass(frozen=True)
class ABSFactor:
    name: str
    A: torch.Tensor
    B: torch.Tensor
    S: torch.Tensor

# -----------------------------
# 3) Replacement utilities
# -----------------------------
def _get_parent_and_child(model: nn.Module, 
                          module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Invalid module path at '{p}' while resolving '{module_name}'")
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_linears(
    model: nn.Module,
    abs_list: Iterable[ABSFactor],
    strict: bool = True,
    cast_to_layer_dtype_device: bool = True,
) -> Dict[str, nn.Module]:
    """
    Replace specified nn.Linear layers by name with LowRankPlusDenseResidualLinearNoBias.

    abs_list: iterable of ABSFactor(name, A, B, S)
    strict:
      - True: error if target name not found, or found module is not nn.Linear
      - False: skip missing names
    cast_to_layer_dtype_device:
      - True: move A/B/S to the same dtype/device as the original layer weight
    Returns: dict of replaced modules by name
    """
    # Build a quick lookup of module objects by name
    name_to_module = dict(model.named_modules())

    replaced: Dict[str, nn.Module] = {}
    for item in abs_list:
        layer_name = 'model.'+item.name

        if layer_name not in name_to_module:
            if strict:
                raise KeyError(f"Target layer '{item.name}' not found in model.named_modules().")
            else:
                continue

        old_mod = name_to_module[layer_name]
        if not isinstance(old_mod, nn.Linear) and 'embed' not in layer_name:
            if strict:
                raise TypeError(f"Target '{layer_name}' is {type(old_mod)}, not nn.Linear.")
            else:
                continue

        A, B, S = item.A, item.B, item.S

        if cast_to_layer_dtype_device:
            device = old_mod.weight.device
            dtype = old_mod.weight.dtype
            A = A.to(device=device, dtype=dtype)
            B = B.to(device=device, dtype=dtype)
            S = S.to(device=device, dtype=dtype)

        new_mod = SLRBlock(A=A, B=B, S=S)

        parent, child = _get_parent_and_child(model, layer_name)
        setattr(parent, child, new_mod)

        replaced[item.name] = new_mod

    return replaced


def list_linear_layer_names(model: nn.Module) -> List[str]:
    """Convenience: print/inspect available nn.Linear names for targeting."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

def check_replaced_modules(model, target_names):
    name_to_module = dict(model.named_modules())
    for name in target_names:
        mod = name_to_module.get(name, None)
        print(
            f"{name:50s} -> "
            f"{type(mod).__name__ if mod is not None else 'NOT FOUND'}"
        )
        
# -----------------------------
# 4) Example usage
# -----------------------------
# from transformers import AutoConfig, LlamaForCausalLM
#
# def get_model(cfg: dict):
#     model_cfg = AutoConfig.from_pretrained(cfg)
#     return LlamaForCausalLM(model_cfg)
#
# model = get_model(cfg)
# model.eval()
#
# # Suppose you have one layer to replace:
# abs_list = [
#     ABSFactor(
#         name="model.layers.0.self_attn.q_proj",
#         A=A0,  # [out, r]
#         B=B0,  # [r, in]
#         S=S0,  # [out, in]
#     ),
#     # ... add more layers here ...
# ]
#
# replaced = replace_linears_with_ABS_inference_only(model, abs_list, strict=True)
#
# # Run inference as usual
# with torch.inference_mode():
#     out = model(input_ids=..., attention_mask=...)
#     gen = model.generate(...)
