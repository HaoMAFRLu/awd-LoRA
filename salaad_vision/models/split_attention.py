"""Explicitly split one DINO attention layer into LL, SL, LS, and SS paths."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from salaad_vision.vendor.dino.vision_transformer import Attention

from .salaad import _files, _load


LOGIT_COMPONENTS = ("LL", "SL", "LS", "SS")


class SplitQKAttention(nn.Module):
    """DINO attention with an explicit four-path Q/K logit expansion.

    The saved qkv bias is assigned to the L projection. Consequently,

        Q = Q_L + Q_S
        K = K_L + K_S
        Q K^T = LL + SL + LS + SS

    where ``SL = Q_S K_L^T`` and ``LS = Q_L K_S^T``. Assigning the bias to
    L is only a bookkeeping convention; the sum of all four paths exactly
    represents the original affine Q/K projections up to floating-point
    evaluation order.

    The original qkv/proj modules are retained, so normal checkpoint keys do
    not change. L/S tensors and intervention scales are non-persistent analysis
    state. This module therefore targets post-hoc evaluation, not checkpoint
    export.
    """

    def __init__(
        self,
        attention: Attention,
        low_rank_weight: Tensor,
        sparse_weight: Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(attention, Attention):
            raise TypeError(
                "SplitQKAttention requires a DINO Attention module, "
                f"got {type(attention).__name__}"
            )
        if low_rank_weight.shape != attention.qkv.weight.shape:
            raise ValueError(
                "L shape does not match qkv weight: "
                f"{tuple(low_rank_weight.shape)} != "
                f"{tuple(attention.qkv.weight.shape)}"
            )
        if sparse_weight.shape != attention.qkv.weight.shape:
            raise ValueError(
                "S shape does not match qkv weight: "
                f"{tuple(sparse_weight.shape)} != "
                f"{tuple(attention.qkv.weight.shape)}"
            )
        if low_rank_weight.shape[0] % 3 != 0:
            raise ValueError("qkv output dimension must be divisible by three")
        if not torch.isfinite(low_rank_weight).all():
            raise ValueError("L contains non-finite values")
        if not torch.isfinite(sparse_weight).all():
            raise ValueError("S contains non-finite values")

        low_q, low_k, _ = low_rank_weight.chunk(3, dim=0)
        sparse_q, sparse_k, _ = sparse_weight.chunk(3, dim=0)
        current_q, current_k, _ = attention.qkv.weight.detach().chunk(3, dim=0)
        expected_q = (low_q.float() + sparse_q.float()).to(
            device=current_q.device,
            dtype=current_q.dtype,
        )
        expected_k = (low_k.float() + sparse_k.float()).to(
            device=current_k.device,
            dtype=current_k.dtype,
        )
        if not torch.allclose(current_q, expected_q, rtol=1e-5, atol=1e-6):
            difference = float((current_q - expected_q).abs().max())
            raise ValueError(
                "current Q weight is not L+S; restore the SALAAD qkv weights "
                f"before splitting (maximum difference={difference:.6g})"
            )
        if not torch.allclose(current_k, expected_k, rtol=1e-5, atol=1e-6):
            difference = float((current_k - expected_k).abs().max())
            raise ValueError(
                "current K weight is not L+S; restore the SALAAD qkv weights "
                f"before splitting (maximum difference={difference:.6g})"
            )

        self.num_heads = attention.num_heads
        self.source_attention_backend = attention.attention_backend
        self.attention_backend = "explicit"
        self.scale = attention.scale
        self.qkv = attention.qkv
        self.attn_drop = attention.attn_drop
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop

        weight_options = {
            "device": self.qkv.weight.device,
            "dtype": self.qkv.weight.dtype,
        }
        self.register_buffer(
            "q_low_rank_weight",
            low_q.detach().to(**weight_options).clone(),
            persistent=False,
        )
        self.register_buffer(
            "q_sparse_weight",
            sparse_q.detach().to(**weight_options).clone(),
            persistent=False,
        )
        self.register_buffer(
            "k_low_rank_weight",
            low_k.detach().to(**weight_options).clone(),
            persistent=False,
        )
        self.register_buffer(
            "k_sparse_weight",
            sparse_k.detach().to(**weight_options).clone(),
            persistent=False,
        )
        self._logit_scales = {name: 1.0 for name in LOGIT_COMPONENTS}
        self.train(attention.training)

    @property
    def logit_scales(self) -> Dict[str, float]:
        """Return a copy of the four intervention scales."""
        return dict(self._logit_scales)

    def set_logit_scales(self, **scales: float) -> None:
        """Set selected LL/SL/LS/SS scales without changing the L/S tensors."""
        unknown = set(scales) - set(LOGIT_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown logit components: {sorted(unknown)}")
        for name, value in scales.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} scale must be a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} scale must be finite")
            self._logit_scales[name] = value

    def reset_logit_scales(self) -> None:
        """Restore the exact full L+S forward path."""
        self._logit_scales = {name: 1.0 for name in LOGIT_COMPONENTS}

    def _heads(self, projection: Tensor) -> Tensor:
        batch, tokens, width = projection.shape
        if width % self.num_heads != 0:
            raise ValueError(
                f"projection width {width} is not divisible by {self.num_heads} heads"
            )
        return projection.reshape(
            batch,
            tokens,
            self.num_heads,
            width // self.num_heads,
        ).permute(0, 2, 1, 3)

    def project_operand(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Project one caller-supplied Q/K weight into per-head activations."""
        expected_shape = self.q_low_rank_weight.shape
        if weight.shape != expected_shape:
            raise ValueError(
                f"operand weight must have shape {tuple(expected_shape)}, "
                f"got {tuple(weight.shape)}"
            )
        if bias is not None and bias.shape != (expected_shape[0],):
            raise ValueError(
                f"operand bias must have shape {(expected_shape[0],)}, "
                f"got {tuple(bias.shape)}"
            )
        return self._heads(F.linear(x, weight, bias))

    def project_qk(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return per-head ``Q_L, Q_S, K_L, K_S`` activations.

        This is public so later experiments can intervene on an operand inside
        SL or LS before rebuilding the corresponding logit component.
        """
        q_bias: Optional[Tensor] = None
        k_bias: Optional[Tensor] = None
        if self.qkv.bias is not None:
            q_bias, k_bias, _ = self.qkv.bias.chunk(3, dim=0)
        q_low = self.project_operand(x, self.q_low_rank_weight, q_bias)
        q_sparse = self.project_operand(x, self.q_sparse_weight)
        k_low = self.project_operand(x, self.k_low_rank_weight, k_bias)
        k_sparse = self.project_operand(x, self.k_sparse_weight)
        return q_low, q_sparse, k_low, k_sparse

    def build_logit_components(
        self,
        q_low: Tensor,
        q_sparse: Tensor,
        k_low: Tensor,
        k_sparse: Tensor,
    ) -> Dict[str, Tensor]:
        """Build the four scaled pre-softmax attention-logit tensors."""
        return {
            "LL": self.build_logit_component(q_low, k_low),
            "SL": self.build_logit_component(q_sparse, k_low),
            "LS": self.build_logit_component(q_low, k_sparse),
            "SS": self.build_logit_component(q_sparse, k_sparse),
        }

    def build_logit_component(self, query: Tensor, key: Tensor) -> Tensor:
        """Build one path, making path-local operand interventions concise."""
        if query.shape != key.shape:
            raise ValueError(
                f"query/key shapes must match, got {tuple(query.shape)} and "
                f"{tuple(key.shape)}"
            )
        return (query @ key.transpose(-2, -1)) * self.scale

    def combine_logit_components(self, components: Mapping[str, Tensor]) -> Tensor:
        """Combine a possibly modified four-path mapping using current scales."""
        names = set(components)
        expected = set(LOGIT_COMPONENTS)
        if names != expected:
            raise ValueError(
                "logit components must be exactly LL, SL, LS, and SS; "
                f"missing={sorted(expected - names)}, "
                f"unexpected={sorted(names - expected)}"
            )
        reference_shape = components["LL"].shape
        for name in LOGIT_COMPONENTS:
            component = components[name]
            if not isinstance(component, Tensor):
                raise TypeError(f"{name} logit component must be a tensor")
            if component.shape != reference_shape:
                raise ValueError(
                    f"{name} shape {tuple(component.shape)} does not match "
                    f"LL shape {tuple(reference_shape)}"
                )
        return sum(
            self._logit_scales[name] * components[name]
            for name in LOGIT_COMPONENTS
        )

    def forward_from_logit_components(
        self,
        x: Tensor,
        components: Mapping[str, Tensor],
        *,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Finish the attention forward from caller-supplied path logits.

        A later ablation can obtain Q/K operands with :meth:`project_qk`,
        rebuild only SL or LS, and pass the modified component mapping here.
        Softmax is deliberately applied once, after all four paths are summed.
        """
        batch, tokens, width = x.shape
        logits = self.combine_logit_components(components)
        expected_shape = (batch, self.num_heads, tokens, tokens)
        if logits.shape != expected_shape:
            raise ValueError(
                f"combined logits must have shape {expected_shape}, "
                f"got {tuple(logits.shape)}"
            )
        attention = self.attn_drop(logits.softmax(dim=-1))

        _, _, value_weight = self.qkv.weight.chunk(3, dim=0)
        value_bias: Optional[Tensor] = None
        if self.qkv.bias is not None:
            _, _, value_bias = self.qkv.bias.chunk(3, dim=0)
        value = self._heads(F.linear(x, value_weight, value_bias))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, width)
        output = self.proj_drop(self.proj(output))
        # The split path is necessarily explicit and has already materialized
        # attention. Match DINO's explicit backend by returning it even when
        # ``return_attention`` is false; the flag remains for call compatibility.
        del return_attention
        return output, attention

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        *,
        return_logit_components: bool = False,
    ):
        q_low, q_sparse, k_low, k_sparse = self.project_qk(x)
        components = self.build_logit_components(
            q_low,
            q_sparse,
            k_low,
            k_sparse,
        )
        output, returned_attention = self.forward_from_logit_components(
            x,
            components,
            return_attention=return_attention,
        )
        if return_logit_components:
            return output, returned_attention, components
        return output, returned_attention


def split_qk_attention(
    model: nn.Module,
    matrix_dir: Path,
    *,
    layer: int = 7,
    restore: bool = False,
) -> SplitQKAttention:
    """Install a four-path Q/K forward at one one-based DINO layer.

    By default, the target qkv weight must already contain the matching full
    ``L+S`` reconstruction. Set ``restore=True`` to reconstruct only this
    layer before installing the split path, leaving all other layers unchanged.
    """
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("layer must be an integer")
    blocks = model.get_submodule("backbone.blocks")
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("model.backbone.blocks must be a ModuleList")
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"layer must be in [1, {len(blocks)}], got {layer}")

    block_index = layer - 1
    target = f"backbone.blocks.{block_index}.attn.qkv"
    matches = []
    for matrix_file in _files(Path(matrix_dir)):
        low_rank, sparse = _load(matrix_file)
        if target in low_rank:
            matches.append((low_rank[target], sparse[target], matrix_file))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one L/S decomposition for {target}, "
            f"found {len(matches)}"
        )

    attention = blocks[block_index].attn
    if isinstance(attention, SplitQKAttention):
        raise ValueError(f"Layer {layer} attention is already split")
    low_rank_weight, sparse_weight, _ = matches[0]
    if not isinstance(low_rank_weight, Tensor) or not isinstance(
        sparse_weight,
        Tensor,
    ):
        raise TypeError(f"SALAAD L and S must be tensors for {target}")
    if restore:
        if (
            low_rank_weight.shape != attention.qkv.weight.shape
            or sparse_weight.shape != attention.qkv.weight.shape
        ):
            raise ValueError(
                f"SALAAD shape mismatch for {target}: "
                f"qkv={tuple(attention.qkv.weight.shape)}, "
                f"L={tuple(low_rank_weight.shape)}, "
                f"S={tuple(sparse_weight.shape)}"
            )
        with torch.no_grad():
            attention.qkv.weight.copy_(
                (low_rank_weight.float() + sparse_weight.float()).to(
                    device=attention.qkv.weight.device,
                    dtype=attention.qkv.weight.dtype,
                )
            )
    split_attention = SplitQKAttention(
        attention,
        low_rank_weight,
        sparse_weight,
    )
    blocks[block_index].attn = split_attention
    return split_attention
