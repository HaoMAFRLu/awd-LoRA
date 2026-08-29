"""Restore and transform SALAAD weights from rank-local L/S files."""

from __future__ import annotations

import io
import math
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
_FC_SUFFIXES = (
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


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def _top_magnitude_fraction(sparse: Tensor, fraction: float) -> Tensor:
    flat = sparse.flatten()
    nonzero_indices = torch.nonzero(flat, as_tuple=False).flatten()
    total_nonzero = int(nonzero_indices.numel())
    if total_nonzero == 0:
        return torch.zeros_like(sparse)
    if fraction == 1.0:
        return sparse.clone()

    retained = min(total_nonzero, max(1, round(total_nonzero * fraction)))
    magnitudes = flat[nonzero_indices].abs()
    selected_local = torch.topk(
        magnitudes,
        retained,
        largest=True,
        sorted=False,
    ).indices
    selected_indices = nonzero_indices[selected_local]
    retained_flat = torch.zeros_like(flat)
    retained_flat[selected_indices] = flat[selected_indices]
    return retained_flat.reshape_as(sparse)


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _symmetric_int3_fake_quantize(weight: Tensor) -> Tensor:
    """Fake-quantize a matrix row-wise to signed INT3 codes in [-3, 3]."""
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(weight.shape)}")
    absolute_maximum = weight.abs().amax(dim=1, keepdim=True)
    scale = absolute_maximum / 3.0
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    codes = torch.round(weight / safe_scale).clamp(-3.0, 3.0)
    dequantized = codes * safe_scale
    return torch.where(
        absolute_maximum > 0.0,
        dequantized,
        torch.zeros_like(dequantized),
    )


def _spectral_mask(low_rank: Tensor, relative_sigma_threshold: float) -> Tensor:
    """Remove complete singular components below a relative threshold."""
    u, singular_values, vh = torch.linalg.svd(
        low_rank.float(),
        full_matrices=False,
    )
    if singular_values[0].item() == 0.0:
        return torch.zeros_like(low_rank, dtype=torch.float32)
    retained = singular_values / singular_values[0] >= relative_sigma_threshold
    return (u[:, retained] * singular_values[retained]) @ vh[retained]


def _masked_int3_weight(
    low_rank: Tensor,
    sparse: Tensor,
    *,
    relative_sigma_threshold: float,
    sparse_zero_threshold: float,
) -> Tensor:
    masked_low_rank = _spectral_mask(
        low_rank,
        relative_sigma_threshold,
    )
    masked_sparse = sparse.float().masked_fill(
        sparse.float().abs() <= sparse_zero_threshold,
        0.0,
    )
    return _symmetric_int3_fake_quantize(
        masked_low_rank,
    ) + _symmetric_int3_fake_quantize(masked_sparse)


def _split_low_output_similarity(
    sparse: Tensor,
    cross_low_rank: Tensor,
    *,
    energy_fraction: float,
    reference_rank: int,
) -> tuple[Tensor, Tensor]:
    if int(torch.count_nonzero(sparse)) == 0 or float(sparse.norm()) == 0.0:
        zero = torch.zeros_like(sparse)
        return zero.clone(), zero

    sparse_u, singular_values, sparse_vh = torch.linalg.svd(
        sparse.double(),
        full_matrices=False,
    )
    cross_u = torch.linalg.svd(
        cross_low_rank.double(),
        full_matrices=False,
    ).U
    effective_rank = min(reference_rank, cross_u.shape[1])
    similarities = (
        cross_u[:, :effective_rank].T @ sparse_u
    ).square().sum(dim=0)
    energies = singular_values.square()
    energies /= energies.sum()
    order = torch.argsort(similarities)
    cumulative_energy = energies[order].cumsum(dim=0)
    selected_count = int(
        torch.searchsorted(cumulative_energy, energy_fraction).item()
    ) + 1
    selected_mask = torch.zeros_like(similarities, dtype=torch.bool)
    selected_mask[order[:selected_count]] = True
    selected = (
        (
            sparse_u[:, selected_mask]
            * singular_values[selected_mask].unsqueeze(0)
        )
        @ sparse_vh[selected_mask]
    ).to(dtype=sparse.dtype)
    fixed = sparse - selected
    return fixed, selected


def _s50_alpha_weight(
    low_rank: Tensor,
    sparse: Tensor,
    *,
    sparse_keep_fraction: float,
    selected_energy_fraction: float,
    reference_rank: int,
    alpha: float,
) -> Tensor:
    if low_rank.shape[0] % 3 != 0:
        raise ValueError(
            "qkv output dimension must be divisible by three, "
            f"got {low_rank.shape[0]}"
        )
    low_q, low_k, low_v = low_rank.float().chunk(3, dim=0)
    sparse_q, sparse_k, sparse_v = sparse.float().chunk(3, dim=0)
    sparse_q50 = _top_magnitude_fraction(sparse_q, sparse_keep_fraction)
    sparse_k50 = _top_magnitude_fraction(sparse_k, sparse_keep_fraction)
    fixed_q, selected_q = _split_low_output_similarity(
        sparse_q50,
        low_k,
        energy_fraction=selected_energy_fraction,
        reference_rank=reference_rank,
    )
    fixed_k, selected_k = _split_low_output_similarity(
        sparse_k50,
        low_q,
        energy_fraction=selected_energy_fraction,
        reference_rank=reference_rank,
    )
    enhanced_q = fixed_q + alpha * selected_q
    enhanced_k = fixed_k + alpha * selected_k
    return torch.cat(
        (
            low_q + enhanced_q,
            low_k + enhanced_k,
            low_v + sparse_v,
        ),
        dim=0,
    )


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


@torch.no_grad()
def apply_salaad_all_masked_int3(
    model: nn.Module,
    matrix_dir: Path,
    *,
    relative_sigma_threshold: float = 1e-2,
    sparse_zero_threshold: float = 1e-5,
) -> Set[str]:
    """Restore all 48 layers with mask-then-INT3 fake-quantized L and S.

    L is spectrally masked, while small entries in S are set to zero. Each
    resulting matrix is independently fake-quantized per output row to the
    signed narrow-range INT3 codes ``{-3, ..., 3}``, then their dequantized
    values are summed and copied into the model.
    """
    relative_sigma_threshold = _fraction(
        relative_sigma_threshold,
        "relative_sigma_threshold",
    )
    sparse_zero_threshold = _nonnegative(
        sparse_zero_threshold,
        "sparse_zero_threshold",
    )
    expected = _expected(model, "salaad_all")
    seen: Set[str] = set()

    for matrix_file in _files(matrix_dir):
        low_rank, sparse = _load(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate SALAAD layer: {layer_name}")
            if layer_name not in expected:
                raise ValueError(
                    "salaad_all_masked_int3 contains an unexpected decomposed "
                    f"layer: {layer_name}"
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

            replacement = _masked_int3_weight(
                low_rank_weight,
                sparse_weight,
                relative_sigma_threshold=relative_sigma_threshold,
                sparse_zero_threshold=sparse_zero_threshold,
            )
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
            "salaad_all_masked_int3 decomposition is incomplete; "
            f"missing={sorted(missing)}"
        )
    return seen


@torch.no_grad()
def apply_salaad_fc_s_masked_int3(
    model: nn.Module,
    matrix_dir: Path,
    *,
    sparse_zero_threshold: float = 1e-5,
) -> Set[str]:
    """Restore QKV/proj exactly and mask-then-INT3 only the FC sparse terms.

    All 48 decomposed layers are restored. Attention QKV and projection use
    exact FP32 ``L+S``. MLP FC1/FC2 keep L in FP32 and replace S with its
    epsilon-masked, per-output-row signed INT3 fake-quantized value.
    """
    sparse_zero_threshold = _nonnegative(
        sparse_zero_threshold,
        "sparse_zero_threshold",
    )
    expected = _expected(model, "salaad_all")
    seen: Set[str] = set()

    for matrix_file in _files(matrix_dir):
        low_rank, sparse = _load(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate SALAAD layer: {layer_name}")
            if layer_name not in expected:
                raise ValueError(
                    "salaad_fc_s_masked_int3 contains an unexpected decomposed "
                    f"layer: {layer_name}"
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

            low_rank_fp32 = low_rank_weight.float()
            sparse_fp32 = sparse_weight.float()
            if layer_name.endswith(_FC_SUFFIXES):
                masked_sparse = sparse_fp32.masked_fill(
                    sparse_fp32.abs() <= sparse_zero_threshold,
                    0.0,
                )
                sparse_fp32 = _symmetric_int3_fake_quantize(masked_sparse)
            replacement = low_rank_fp32 + sparse_fp32
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
            "salaad_fc_s_masked_int3 decomposition is incomplete; "
            f"missing={sorted(missing)}"
        )
    return seen


@torch.no_grad()
def apply_salaad_qkv_s50(
    model: nn.Module,
    matrix_dir: Path,
    *,
    sparse_keep_fraction: float,
    selected_energy_fraction: float,
    reference_rank: int,
    alpha: float,
) -> Set[str]:
    """Restore qkv with the all-layer S50 output/output intervention."""
    sparse_keep_fraction = _fraction(
        sparse_keep_fraction,
        "sparse_keep_fraction",
    )
    selected_energy_fraction = _fraction(
        selected_energy_fraction,
        "selected_energy_fraction",
    )
    if isinstance(reference_rank, bool) or not isinstance(reference_rank, int):
        raise TypeError("reference_rank must be an integer")
    if reference_rank <= 0:
        raise ValueError("reference_rank must be positive")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")

    expected = _expected(model, "salaad_qkv")
    seen: Set[str] = set()
    for matrix_file in _files(matrix_dir):
        low_rank, sparse = _load(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate SALAAD layer: {layer_name}")
            if layer_name not in expected:
                raise ValueError(
                    "salaad_qkv_s50 contains an unexpected decomposed layer: "
                    f"{layer_name}"
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

            replacement = _s50_alpha_weight(
                low_rank_weight,
                sparse_weight,
                sparse_keep_fraction=sparse_keep_fraction,
                selected_energy_fraction=selected_energy_fraction,
                reference_rank=reference_rank,
                alpha=alpha,
            )
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
            "salaad_qkv_s50 decomposition is incomplete; "
            f"missing={sorted(missing)}"
        )
    return seen
