"""Restore trained SALAAD linear weights from rank-local L/S files."""

from __future__ import annotations

import io
import pickle
import re
from pathlib import Path
from typing import Mapping, Set, Tuple

import torch
from torch import Tensor, nn

_ALL_SUFFIXES = (
    "attn.qkv",
    "attn.proj",
    "mlp.fc1",
    "mlp.fc2",
)


def _rank(path: Path) -> int:
    match = re.fullmatch(r"matrix_rank(\d+)\.pkl", path.name)
    if match is None:
        raise ValueError(f"unexpected SALAAD matrix filename: {path.name}")
    return int(match.group(1))


def _files(matrix_dir: Path) -> list[Path]:
    if not matrix_dir.is_dir():
        raise NotADirectoryError(
            f"SALAAD matrix directory does not exist: {matrix_dir}"
        )
    files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=_rank)
    if not files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    ranks = [_rank(path) for path in files]
    if ranks != list(range(len(files))):
        raise ValueError(f"SALAAD matrix ranks must be contiguous from zero, got {ranks}")
    return files


def _load(path: Path) -> Tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    """Load one trusted training output onto CPU."""
    original_loader = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda value: torch.load(
            io.BytesIO(value),
            map_location="cpu",
            weights_only=False,
        )
        with path.open("rb") as matrix_file:
            payload = pickle.load(matrix_file)
    finally:
        torch.storage._load_from_bytes = original_loader

    if not isinstance(payload, Mapping):
        raise TypeError(f"SALAAD matrix file must contain a mapping: {path}")
    low_rank = payload.get("LL")
    sparse = payload.get("SS")
    if not isinstance(low_rank, Mapping) or not isinstance(sparse, Mapping):
        raise TypeError(f"SALAAD matrix file requires LL and SS mappings: {path}")
    if set(low_rank) != set(sparse):
        raise ValueError(f"SALAAD L/S layer names differ: {path}")
    if not all(isinstance(name, str) for name in low_rank):
        raise TypeError(f"SALAAD layer names must be strings: {path}")
    return low_rank, sparse


def _expected(model: nn.Module, variant: str) -> Set[str]:
    linear_layers = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    all_layers = {
        f"backbone.blocks.{block}.{suffix}"
        for block in range(12)
        for suffix in _ALL_SUFFIXES
    }
    missing_model_layers = all_layers - linear_layers
    if missing_model_layers:
        raise ValueError(
            "DINO model does not contain the expected SALAAD layers: "
            f"{sorted(missing_model_layers)}"
        )
    if variant == "salaad_all":
        return all_layers
    if variant == "salaad_qkv":
        return {
            f"backbone.blocks.{block}.attn.qkv"
            for block in range(12)
        }
    raise ValueError(f"unsupported SALAAD variant: {variant!r}")


@torch.no_grad()
def apply_salaad(model: nn.Module, matrix_dir: Path, variant: str) -> Set[str]:
    """Replace the variant's target weights by the saved full L+S matrices."""
    expected = _expected(model, variant)
    seen: Set[str] = set()

    for matrix_file in _files(matrix_dir):
        low_rank, sparse = _load(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate SALAAD layer: {layer_name}")
            if layer_name not in expected:
                raise ValueError(
                    f"{variant} contains an unexpected decomposed layer: {layer_name}"
                )

            layer = model.get_submodule(layer_name)
            low_rank_weight = low_rank[layer_name]
            sparse_weight = sparse[layer_name]
            if not isinstance(low_rank_weight, Tensor) or not isinstance(
                sparse_weight,
                Tensor,
            ):
                raise TypeError(f"SALAAD L and S must be tensors for {layer_name}")
            if (
                low_rank_weight.shape != layer.weight.shape
                or sparse_weight.shape != layer.weight.shape
            ):
                raise ValueError(
                    f"SALAAD shape mismatch for {layer_name}: "
                    f"X={tuple(layer.weight.shape)}, "
                    f"L={tuple(low_rank_weight.shape)}, "
                    f"S={tuple(sparse_weight.shape)}"
                )
            if not torch.isfinite(low_rank_weight).all() or not torch.isfinite(
                sparse_weight
            ).all():
                raise ValueError(f"SALAAD contains non-finite values for {layer_name}")

            replacement = low_rank_weight.float() + sparse_weight.float()
            layer.weight.copy_(
                replacement.to(
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
            )
            seen.add(layer_name)

    missing = expected - seen
    if missing:
        raise ValueError(
            f"{variant} decomposition is incomplete; missing={sorted(missing)}"
        )
    return seen
