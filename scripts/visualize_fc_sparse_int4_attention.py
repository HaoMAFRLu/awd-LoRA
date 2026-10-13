#!/usr/bin/env python3
"""Compare final-block attention before and after quantizing FC sparse weights.

The reconstructed backbone uses the saved SALAAD ``L + S`` weights for every
decomposed linear layer.  QKV and attention-output projections remain full
precision.  Only ``mlp.fc1.S`` and ``mlp.fc2.S`` are fake-quantized, then
dequantized back to float before adding ``L``.  Signed symmetric INT3/INT4,
dense binary INT1, and sparse-mask-preserving INT1 are supported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from salaad_vision.models.dino import DinoViTBase8  # noqa: E402
from scripts.visualize_dino_attention import (  # noqa: E402
    choose_device,
    color_overlay,
    get_last_selfattention,
    load_matrix_file,
    matrix_rank,
    prepare_image,
    top_attention_mask,
)


DEFAULT_RUN_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "vit_b8_all_qkv_rho5e6_fc_rho5e8"
    / "20260828_154729"
)
DEFAULT_IMAGE = (
    REPOSITORY_ROOT
    / "data"
    / "figures"
    / "salaad_vision"
    / "self_attention"
    / "52_green_lizard"
    / "source.jpg"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "figures"
    / "salaad_vision"
    / "fc_sparse_int4_attention"
    / "52_green_lizard"
)
DECOMPOSED_LAYER_PATTERN = re.compile(
    r"^backbone\.blocks\.(?P<block>\d+)\."
    r"(?P<layer>attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$"
)
FC_LAYER_PATTERN = re.compile(
    r"^backbone\.blocks\.(?P<block>\d+)\.mlp\.(?P<fc>fc1|fc2)$"
)
INT4_BITS = 4
INT4_QMIN = -7
INT4_QMAX = 7
INT3_BITS = 3
INT3_QMIN = -3
INT3_QMAX = 3
TRANSFORM_DESCRIPTION = (
    "Resize(256, bicubic) -> CenterCrop(224) -> ImageNet normalization"
)


def parse_args(
    *,
    default_quantization_mode: str = "int4",
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--base-state",
        type=Path,
        default=DEFAULT_RUN_DIR / "non_decomposed_state.pth",
        help=(
            "State dictionary containing every model parameter except the 48 "
            "decomposed linear weights."
        ),
    )
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--quantization-mode",
        choices=(
            "int3",
            "int4",
            "int1_positive_mean",
            "int1_sparse_positive_mean",
        ),
        default=default_quantization_mode,
        help=(
            "INT3/INT4 use signed symmetric quantization. INT1 uses exactly -alpha "
            "and +alpha, with one alpha per FC matrix computed as the mean of "
            "strictly positive thresholded S entries. The sparse INT1 mode "
            "retains thresholded zeros via a separate sparse mask."
        ),
    )
    parser.add_argument(
        "--quantization-granularity",
        choices=("per_output_channel", "per_tensor"),
        default="per_output_channel",
        help=(
            "Scale granularity for signed symmetric INT3/INT4. It is ignored by "
            "INT1, which uses one alpha per FC matrix."
        ),
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-5,
        help=(
            "FC sparse zero threshold. Sparse INT1 uses abs(S) < epsilon as "
            "requested; legacy modes use abs(S) <= epsilon."
        ),
    )
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decomposed_module_names(model: DinoViTBase8) -> set[str]:
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and DECOMPOSED_LAYER_PATTERN.fullmatch(name) is not None
    }
    expected = 12 * 4
    if len(names) != expected:
        raise ValueError(f"expected {expected} decomposed Linear modules, found {len(names)}")
    return names


def load_non_decomposed_state(
    model: DinoViTBase8,
    base_state_path: Path,
    decomposed_modules: set[str],
) -> dict[str, Any]:
    if not base_state_path.is_file():
        raise FileNotFoundError(f"base state does not exist: {base_state_path}")
    state = torch.load(base_state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"base state must be a mapping, got {type(state).__name__}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("every base-state value must be a tensor")
    non_finite = [
        name for name, value in state.items() if not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"base state contains non-finite tensors: {non_finite}")

    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {f"{name}.weight" for name in decomposed_modules}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "partial state does not complement the decomposition exactly: "
            f"missing={sorted(actual_missing)}, "
            f"expected_missing={sorted(expected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return {
        "path": str(base_state_path.resolve()),
        "sha256": sha256_file(base_state_path),
        "retained_state_keys": len(state),
        "retained_parameters": sum(value.numel() for value in state.values()),
        "missing_decomposed_weights": len(actual_missing),
    }


def symmetric_int4_fake_quantize(
    sparse: torch.Tensor,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sparse.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(sparse.shape)}")
    if granularity == "per_output_channel":
        absolute_maximum = sparse.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_tensor":
        absolute_maximum = sparse.abs().amax().reshape(1, 1)
    else:
        raise ValueError(f"unsupported quantization granularity: {granularity}")

    scale = absolute_maximum / INT4_QMAX
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    codes = torch.round(sparse / safe_scale).clamp(INT4_QMIN, INT4_QMAX)
    dequantized = codes * safe_scale
    if granularity == "per_output_channel":
        dequantized = torch.where(
            absolute_maximum > 0.0,
            dequantized,
            torch.zeros_like(dequantized),
        )
    elif absolute_maximum.item() == 0.0:
        dequantized.zero_()
    return dequantized, codes.to(torch.int8), scale


def symmetric_int3_fake_quantize(
    sparse: torch.Tensor,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sparse.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(sparse.shape)}")
    if granularity == "per_output_channel":
        absolute_maximum = sparse.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_tensor":
        absolute_maximum = sparse.abs().amax().reshape(1, 1)
    else:
        raise ValueError(f"unsupported quantization granularity: {granularity}")

    scale = absolute_maximum / INT3_QMAX
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    codes = torch.round(sparse / safe_scale).clamp(INT3_QMIN, INT3_QMAX)
    dequantized = codes * safe_scale
    if granularity == "per_output_channel":
        dequantized = torch.where(
            absolute_maximum > 0.0,
            dequantized,
            torch.zeros_like(dequantized),
        )
    elif absolute_maximum.item() == 0.0:
        dequantized.zero_()
    return dequantized, codes.to(torch.int8), scale


def binary_positive_mean_fake_quantize(
    sparse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map every entry to {-alpha, +alpha} using mean(S[S > 0]).

    Zero is assigned to the positive code so that the result has exactly two
    representable values.  This deliberately removes the explicit zeros in S;
    preserving them would instead define a ternary {-alpha, 0, +alpha} matrix.
    """
    if sparse.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(sparse.shape)}")
    positive = sparse[sparse > 0.0]
    if positive.numel() == 0:
        raise ValueError("binary positive-mean quantization requires positive entries")
    alpha = positive.mean().reshape(1, 1)
    codes = torch.where(
        sparse < 0.0,
        torch.full_like(sparse, -1, dtype=torch.int8),
        torch.full_like(sparse, 1, dtype=torch.int8),
    )
    dequantized = codes.to(dtype=sparse.dtype) * alpha
    return dequantized, codes, alpha


def sparse_binary_positive_mean_fake_quantize(
    sparse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve zeros and binary-quantize retained nonzeros to {-alpha, +alpha}."""
    if sparse.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(sparse.shape)}")
    positive = sparse[sparse > 0.0]
    if positive.numel() == 0:
        raise ValueError("sparse binary quantization requires positive entries")
    alpha = positive.mean().reshape(1, 1)
    codes = torch.sign(sparse).to(torch.int8)
    dequantized = codes.to(dtype=sparse.dtype) * alpha
    return dequantized, codes, alpha


def quantization_spec(mode: str, granularity: str) -> dict[str, Any]:
    if mode == "int3":
        return {
            "mode": mode,
            "display_label": "signed symmetric INT3",
            "short_label": "INT3",
            "output_label": f"fc_s_int3_{granularity}",
            "bits": INT3_BITS,
            "signed_code_range": [INT3_QMIN, INT3_QMAX],
            "code_values": list(range(INT3_QMIN, INT3_QMAX + 1)),
            "scheme": "symmetric",
            "granularity": granularity,
            "axis": (
                "one scale per output channel (weight row)"
                if granularity == "per_output_channel"
                else "one scale per FC matrix"
            ),
            "scale_definition": "max(abs(S)) / 3 over the selected scale group",
            "rounding": "torch.round (nearest, ties-to-even), then clamp to [-3, 3]",
            "zero_assignment": "0 maps to integer code 0",
            "simulation": (
                "INT3 codes are immediately dequantized to FP32 before L+S "
                "inference; packed-kernel storage is not simulated"
            ),
            "weight_expression": "FC=L+dequant(INT3(S))",
            "quantized_symbol": "S_int3",
        }
    if mode == "int4":
        return {
            "mode": mode,
            "display_label": "signed symmetric INT4",
            "short_label": "INT4",
            "output_label": f"fc_s_int4_{granularity}",
            "bits": INT4_BITS,
            "signed_code_range": [INT4_QMIN, INT4_QMAX],
            "code_values": list(range(INT4_QMIN, INT4_QMAX + 1)),
            "scheme": "symmetric",
            "granularity": granularity,
            "axis": (
                "one scale per output channel (weight row)"
                if granularity == "per_output_channel"
                else "one scale per FC matrix"
            ),
            "scale_definition": "max(abs(S)) / 7 over the selected scale group",
            "rounding": "torch.round (nearest, ties-to-even), then clamp to [-7, 7]",
            "zero_assignment": "0 maps to integer code 0",
            "simulation": (
                "INT4 codes are immediately dequantized to FP32 before L+S "
                "inference; packed-kernel storage is not simulated"
            ),
            "weight_expression": "FC=L+dequant(INT4(S))",
            "quantized_symbol": "S_int4",
        }
    if mode == "int1_positive_mean":
        return {
            "mode": mode,
            "display_label": "binary INT1 (±alpha, positive-mean scale)",
            "short_label": "INT1",
            "output_label": "fc_s_int1_positive_mean",
            "bits": 1,
            "signed_code_range": [-1, 1],
            "code_values": [-1, 1],
            "scheme": "binary_sign_positive_elements_mean",
            "granularity": "per_fc_matrix",
            "axis": "one alpha per FC sparse matrix",
            "scale_definition": "alpha = mean(S_ij for strictly positive S_ij)",
            "rounding": "S < 0 maps to -1; S >= 0 maps to +1",
            "zero_assignment": "thresholded and original zeros map to +alpha",
            "simulation": (
                "One-bit sign codes are immediately dequantized to FP32 before "
                "L+S inference; packed-kernel storage is not simulated"
            ),
            "weight_expression": "FC=L+alpha*sign_binary(S)",
            "quantized_symbol": "S_int1",
        }
    if mode == "int1_sparse_positive_mean":
        return {
            "mode": mode,
            "display_label": "sparse-mask INT1 (±alpha, positive-mean scale)",
            "short_label": "INT1-mask",
            "output_label": "fc_s_int1_sparse_positive_mean",
            "bits": 1,
            "signed_code_range": [-1, 1],
            "code_values": [-1, 0, 1],
            "scheme": "sparse_mask_plus_binary_sign_positive_elements_mean",
            "granularity": "per_fc_matrix",
            "axis": "one alpha per FC sparse matrix",
            "scale_definition": (
                "alpha = mean(S_ij for retained positive entries S_ij >= epsilon)"
            ),
            "rounding": "retained S < 0 maps to -1; retained S > 0 maps to +1",
            "zero_assignment": (
                "entries with |S| < epsilon remain 0 via the sparse mask"
            ),
            "simulation": (
                "Retained nonzeros use one-bit sign codes and are immediately "
                "dequantized to FP32 before L+S inference; sparse-mask/index "
                "storage and a packed kernel are not simulated"
            ),
            "weight_expression": "FC=L+alpha*sign(S) on the retained sparse mask",
            "quantized_symbol": "S_int1_masked",
        }
    raise ValueError(f"unsupported quantization mode: {mode}")


def _cosine(numerator: float, norm_a_squared: float, norm_b_squared: float) -> float:
    denominator = math.sqrt(max(norm_a_squared * norm_b_squared, 0.0))
    return numerator / denominator if denominator > 0.0 else 1.0


@torch.no_grad()
def apply_l_plus_s_variant(
    model: DinoViTBase8,
    matrix_dir: Path,
    decomposed_modules: set[str],
    *,
    quantize_fc_sparse: bool,
    granularity: str,
    zero_threshold: float,
    quantization_mode: str = "int4",
) -> dict[str, Any]:
    if not matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {matrix_dir}")
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    if not math.isfinite(zero_threshold) or zero_threshold < 0.0:
        raise ValueError("zero threshold must be a finite non-negative number")
    specification = quantization_spec(quantization_mode, granularity)
    code_values = specification["code_values"]

    seen: set[str] = set()
    original_sparse_nonzero = 0
    thresholded_sparse_nonzero = 0
    sparse_elements = 0
    threshold_removed = 0
    quantization_zeroed = 0
    quantization_activated_zero = 0
    saturation_count = 0
    sparse_error_squared = 0.0
    sparse_norm_squared = 0.0
    dequantized_sparse_norm_squared = 0.0
    sparse_dot = 0.0
    full_error_squared = 0.0
    full_norm_squared = 0.0
    code_histogram = {str(code): 0 for code in code_values}
    layer_statistics: list[dict[str, Any]] = []

    for matrix_file in matrix_files:
        low_rank, sparse = load_matrix_file(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate decomposed layer: {layer_name}")
            if layer_name not in decomposed_modules:
                raise ValueError(f"unexpected decomposed layer: {layer_name}")
            seen.add(layer_name)

            low_rank_weight = low_rank[layer_name]
            sparse_weight = sparse[layer_name]
            if not isinstance(low_rank_weight, torch.Tensor) or not isinstance(
                sparse_weight, torch.Tensor
            ):
                raise TypeError(f"L and S must be tensors for {layer_name}")
            layer = model.get_submodule(layer_name)
            expected_shape = tuple(layer.weight.shape)
            if (
                tuple(low_rank_weight.shape) != expected_shape
                or tuple(sparse_weight.shape) != expected_shape
            ):
                raise ValueError(
                    f"shape mismatch for {layer_name}: expected={expected_shape}, "
                    f"L={tuple(low_rank_weight.shape)}, S={tuple(sparse_weight.shape)}"
                )
            low_rank_weight = low_rank_weight.detach().cpu().float()
            sparse_weight = sparse_weight.detach().cpu().float()
            if not torch.isfinite(low_rank_weight).all() or not torch.isfinite(
                sparse_weight
            ).all():
                raise ValueError(f"non-finite decomposition values for {layer_name}")

            fc_match = FC_LAYER_PATTERN.fullmatch(layer_name)
            if fc_match is None:
                replacement = low_rank_weight + sparse_weight
                layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))
                continue

            original_nonzero_mask = sparse_weight != 0.0
            if quantization_mode == "int1_sparse_positive_mean":
                threshold_mask = original_nonzero_mask & (
                    sparse_weight.abs() < zero_threshold
                )
            else:
                threshold_mask = original_nonzero_mask & (
                    sparse_weight.abs() <= zero_threshold
                )
            thresholded_sparse = sparse_weight.masked_fill(threshold_mask, 0.0)
            thresholded_nonzero_mask = thresholded_sparse != 0.0
            original_nonzero = int(torch.count_nonzero(original_nonzero_mask).item())
            thresholded_nonzero = int(
                torch.count_nonzero(thresholded_nonzero_mask).item()
            )
            removed = int(torch.count_nonzero(threshold_mask).item())

            if quantize_fc_sparse:
                if quantization_mode == "int3":
                    used_sparse, codes, scales = symmetric_int3_fake_quantize(
                        thresholded_sparse,
                        granularity,
                    )
                elif quantization_mode == "int4":
                    used_sparse, codes, scales = symmetric_int4_fake_quantize(
                        thresholded_sparse,
                        granularity,
                    )
                elif quantization_mode == "int1_positive_mean":
                    used_sparse, codes, scales = binary_positive_mean_fake_quantize(
                        thresholded_sparse
                    )
                elif quantization_mode == "int1_sparse_positive_mean":
                    used_sparse, codes, scales = (
                        sparse_binary_positive_mean_fake_quantize(
                            thresholded_sparse
                        )
                    )
                else:
                    raise ValueError(
                        f"unsupported quantization mode: {quantization_mode}"
                    )
                quantized_nonzero_mask = codes != 0
                layer_quantization_zeroed = int(
                    torch.count_nonzero(
                        thresholded_nonzero_mask & ~quantized_nonzero_mask
                    ).item()
                )
                layer_quantization_activated_zero = int(
                    torch.count_nonzero(
                        ~thresholded_nonzero_mask & quantized_nonzero_mask
                    ).item()
                )
                maximum_code = max(abs(int(code)) for code in code_values)
                layer_saturation = int(
                    torch.count_nonzero(codes.abs() == maximum_code).item()
                )
                for code in code_values:
                    code_histogram[str(code)] += int(
                        torch.count_nonzero(codes == code).item()
                    )
                positive_scales = scales[scales > 0.0]
                scale_min = (
                    float(positive_scales.min().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
                scale_median = (
                    float(positive_scales.median().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
                scale_max = (
                    float(positive_scales.max().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
            else:
                used_sparse = thresholded_sparse
                codes = None
                layer_quantization_zeroed = 0
                layer_quantization_activated_zero = 0
                layer_saturation = 0
                scale_min = None
                scale_median = None
                scale_max = None

            full_reference = low_rank_weight + thresholded_sparse
            replacement = low_rank_weight + used_sparse
            sparse_error = used_sparse - thresholded_sparse
            full_error = replacement - full_reference
            layer_sparse_error_squared = float(sparse_error.square().sum().item())
            layer_sparse_norm_squared = float(
                thresholded_sparse.square().sum().item()
            )
            layer_used_sparse_norm_squared = float(used_sparse.square().sum().item())
            layer_sparse_dot = float((thresholded_sparse * used_sparse).sum().item())
            layer_full_error_squared = float(full_error.square().sum().item())
            layer_full_norm_squared = float(full_reference.square().sum().item())

            original_sparse_nonzero += original_nonzero
            thresholded_sparse_nonzero += thresholded_nonzero
            sparse_elements += sparse_weight.numel()
            threshold_removed += removed
            quantization_zeroed += layer_quantization_zeroed
            quantization_activated_zero += layer_quantization_activated_zero
            saturation_count += layer_saturation
            sparse_error_squared += layer_sparse_error_squared
            sparse_norm_squared += layer_sparse_norm_squared
            dequantized_sparse_norm_squared += layer_used_sparse_norm_squared
            sparse_dot += layer_sparse_dot
            full_error_squared += layer_full_error_squared
            full_norm_squared += layer_full_norm_squared

            layer_statistics.append(
                {
                    "layer": layer_name,
                    "block": int(fc_match.group("block")),
                    "fc": fc_match.group("fc"),
                    "shape": list(sparse_weight.shape),
                    "elements": sparse_weight.numel(),
                    "original_sparse_nonzero": original_nonzero,
                    "thresholded_sparse_nonzero": thresholded_nonzero,
                    "threshold_removed": removed,
                    "quantized_sparse_nonzero": int(
                        torch.count_nonzero(used_sparse).item()
                    ),
                    "quantization_zeroed_nonzero": layer_quantization_zeroed,
                    "quantization_activated_zero": (
                        layer_quantization_activated_zero
                    ),
                    "saturation_count": layer_saturation,
                    "scale_min": scale_min,
                    "scale_median": scale_median,
                    "scale_max": scale_max,
                    "alpha": (
                        scale_max
                        if quantization_mode
                        in ("int1_positive_mean", "int1_sparse_positive_mean")
                        and quantize_fc_sparse
                        else None
                    ),
                    "relative_sparse_frobenius_error": math.sqrt(
                        layer_sparse_error_squared
                        / max(layer_sparse_norm_squared, 1e-30)
                    ),
                    "sparse_cosine_similarity": _cosine(
                        layer_sparse_dot,
                        layer_sparse_norm_squared,
                        layer_used_sparse_norm_squared,
                    ),
                    "relative_full_weight_frobenius_error": math.sqrt(
                        layer_full_error_squared
                        / max(layer_full_norm_squared, 1e-30)
                    ),
                    "maximum_absolute_sparse_error": float(
                        sparse_error.abs().max().item()
                    ),
                }
            )
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))

        del low_rank
        del sparse
        gc.collect()

    missing = decomposed_modules - seen
    if missing or len(seen) != len(decomposed_modules):
        raise ValueError(f"decomposition coverage mismatch; missing={sorted(missing)}")

    return {
        "matrix_dir": str(matrix_dir.resolve()),
        "matrix_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
            }
            for path in matrix_files
        ],
        "decomposed_layers": len(seen),
        "weight_definition": (
            f"QKV/proj=L+S(fp32); {specification['weight_expression']}"
            if quantize_fc_sparse
            else "QKV/proj=L+S(fp32); FC=L+S(fp32)"
        ),
        "fc_sparse_zero_threshold": zero_threshold,
        "fc_sparse_zero_rule": (
            "abs(S) < epsilon becomes 0"
            if quantization_mode == "int1_sparse_positive_mean"
            else "abs(S) <= epsilon becomes 0"
        ),
        "fc_sparse_original_nonzero": original_sparse_nonzero,
        "fc_sparse_thresholded_nonzero": thresholded_sparse_nonzero,
        "fc_sparse_elements": sparse_elements,
        "fc_sparse_threshold_removed": threshold_removed,
        "fc_sparse_thresholded_density": (
            thresholded_sparse_nonzero / max(sparse_elements, 1)
        ),
        "quantized": quantize_fc_sparse,
        "quantization": (
            {
                **{
                    key: value
                    for key, value in specification.items()
                    if key != "code_values"
                },
                "quantization_zeroed_nonzero": quantization_zeroed,
                "quantization_zeroed_fraction_of_thresholded_nonzero": (
                    quantization_zeroed / max(thresholded_sparse_nonzero, 1)
                ),
                "quantization_activated_zero": quantization_activated_zero,
                "quantization_activated_zero_fraction_of_thresholded_zeros": (
                    quantization_activated_zero
                    / max(sparse_elements - thresholded_sparse_nonzero, 1)
                ),
                "saturation_count": saturation_count,
                "code_histogram": code_histogram,
                "relative_sparse_frobenius_error": math.sqrt(
                    sparse_error_squared / max(sparse_norm_squared, 1e-30)
                ),
                "sparse_cosine_similarity": _cosine(
                    sparse_dot,
                    sparse_norm_squared,
                    dequantized_sparse_norm_squared,
                ),
                "relative_full_weight_frobenius_error": math.sqrt(
                    full_error_squared / max(full_norm_squared, 1e-30)
                ),
                "layers": sorted(
                    layer_statistics,
                    key=lambda row: (row["block"], row["fc"]),
                ),
            }
            if quantize_fc_sparse
            else None
        ),
    }


def infer_patch_attention(
    model: DinoViTBase8,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int, list[int]]:
    model.eval().to(device)
    with torch.inference_mode():
        attention, source_block = get_last_selfattention(
            model,
            image_tensor.to(device),
            skipped_blocks=set(),
        )
    attention_shape = list(attention.shape)
    cls_to_patch = attention[0, :, 0, 1:].detach().float().cpu()
    cls_to_patch = cls_to_patch / cls_to_patch.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    patch_count = cls_to_patch.shape[1]
    grid_size = int(math.sqrt(patch_count))
    if grid_size * grid_size != patch_count:
        raise RuntimeError(f"patch count is not square: {patch_count}")
    patch_maps = cls_to_patch.reshape(cls_to_patch.shape[0], grid_size, grid_size)
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return patch_maps, source_block, attention_shape


def upsample_patch_map(
    patch_map: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    resized = F.interpolate(
        patch_map[None, None],
        size=(224, 224),
        mode=mode,
        **kwargs,
    )[0, 0]
    return resized.numpy()


def render_variant(
    *,
    original: Image.Image,
    crop: Image.Image,
    patch_maps: torch.Tensor,
    output_dir: Path,
    title: str,
    attention_mass: float,
    attention_source_block: int,
    dpi: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_patch_map = patch_maps.mean(dim=0)
    mean_mask = top_attention_mask(mean_patch_map, attention_mass)
    head_maps = F.interpolate(
        patch_maps.unsqueeze(0),
        size=(224, 224),
        mode="nearest",
    )[0].numpy()
    mean_map = upsample_patch_map(mean_patch_map)
    mean_mask_image = upsample_patch_map(mean_mask.float(), mode="nearest")

    original_array = np.asarray(original, dtype=np.float32) / 255.0
    crop_array = np.asarray(crop, dtype=np.float32) / 255.0
    overlay = color_overlay(crop_array, mean_map, alpha=0.55)
    mask_overlay = crop_array.copy()
    selected = mean_mask_image > 0.5
    mask_overlay[selected] = (
        0.45 * mask_overlay[selected]
        + 0.55 * np.array([1.0, 0.15, 0.0], dtype=np.float32)
    )

    overview_path = output_dir / "overview.png"
    overlay_path = output_dir / "mean_overlay.png"
    raw_path = output_dir / "cls_to_patch_attention.npy"
    plt.imsave(overlay_path, overlay)
    np.save(raw_path, patch_maps.numpy())

    figure, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    flat_axes = axes.flatten()
    flat_axes[0].imshow(original_array)
    flat_axes[0].set_title("Original image")
    flat_axes[1].imshow(crop_array)
    flat_axes[1].set_title("Validation center crop")
    flat_axes[2].imshow(overlay)
    flat_axes[2].set_title("Mean over 12 heads")
    flat_axes[3].imshow(mask_overlay)
    flat_axes[3].set_title(f"Top {attention_mass:.0%} attention mass")
    for head_index, (axis, head_map) in enumerate(zip(flat_axes[4:], head_maps)):
        normalized = head_map / max(float(head_map.max()), 1e-12)
        axis.imshow(crop_array)
        axis.imshow(normalized, cmap="turbo", alpha=0.58, vmin=0.0, vmax=1.0)
        axis.set_title(f"Head {head_index}")
    for axis in flat_axes:
        axis.axis("off")
    figure.suptitle(
        f"{title}: block {attention_source_block} CLS-to-patch self-attention",
        fontsize=16,
    )
    figure.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "overview": str(overview_path.resolve()),
        "mean_overlay": str(overlay_path.resolve()),
        "raw_attention": str(raw_path.resolve()),
    }


def distribution_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    attention_mass: float,
) -> dict[str, float]:
    baseline = baseline.flatten().double()
    candidate = candidate.flatten().double()
    baseline = baseline / baseline.sum().clamp_min(1e-30)
    candidate = candidate / candidate.sum().clamp_min(1e-30)
    midpoint = 0.5 * (baseline + candidate)
    epsilon = 1e-30
    js_divergence = 0.5 * (
        torch.sum(baseline * torch.log((baseline + epsilon) / (midpoint + epsilon)))
        + torch.sum(candidate * torch.log((candidate + epsilon) / (midpoint + epsilon)))
    )
    baseline_mask = top_attention_mask(baseline, attention_mass)
    candidate_mask = top_attention_mask(candidate, attention_mass)
    intersection = torch.count_nonzero(baseline_mask & candidate_mask).item()
    union = torch.count_nonzero(baseline_mask | candidate_mask).item()
    entropy_denominator = math.log(baseline.numel())
    return {
        "cosine_similarity": float(
            F.cosine_similarity(baseline, candidate, dim=0).item()
        ),
        "total_variation_distance": float(
            (0.5 * torch.sum(torch.abs(candidate - baseline))).item()
        ),
        "jensen_shannon_divergence_nats": float(js_divergence.item()),
        "maximum_absolute_probability_change": float(
            torch.max(torch.abs(candidate - baseline)).item()
        ),
        "top_attention_mass_mask_iou": float(intersection / max(union, 1)),
        "baseline_normalized_entropy": float(
            (-torch.sum(baseline * torch.log(baseline + epsilon))).item()
            / entropy_denominator
        ),
        "candidate_normalized_entropy": float(
            (-torch.sum(candidate * torch.log(candidate + epsilon))).item()
            / entropy_denominator
        ),
    }


def within_variant_head_similarity(
    patch_maps: torch.Tensor,
    attention_mass: float,
) -> dict[str, Any]:
    pairwise: list[dict[str, float]] = []
    for first_head in range(patch_maps.shape[0]):
        for second_head in range(first_head + 1, patch_maps.shape[0]):
            row = distribution_metrics(
                patch_maps[first_head],
                patch_maps[second_head],
                attention_mass,
            )
            row["first_head"] = first_head
            row["second_head"] = second_head
            pairwise.append(row)
    return {
        "pair_count": len(pairwise),
        "mean_pairwise_cosine_similarity": float(
            np.mean([row["cosine_similarity"] for row in pairwise])
        ),
        "minimum_pairwise_cosine_similarity": float(
            np.min([row["cosine_similarity"] for row in pairwise])
        ),
        "maximum_pairwise_cosine_similarity": float(
            np.max([row["cosine_similarity"] for row in pairwise])
        ),
        "mean_pairwise_total_variation_distance": float(
            np.mean([row["total_variation_distance"] for row in pairwise])
        ),
        "mean_pairwise_jensen_shannon_divergence_nats": float(
            np.mean([row["jensen_shannon_divergence_nats"] for row in pairwise])
        ),
        "mean_pairwise_top_attention_mass_mask_iou": float(
            np.mean([row["top_attention_mass_mask_iou"] for row in pairwise])
        ),
    }


def attention_metrics(
    baseline_maps: torch.Tensor,
    quantized_maps: torch.Tensor,
    attention_mass: float,
) -> dict[str, Any]:
    if baseline_maps.shape != quantized_maps.shape:
        raise ValueError(
            f"attention shape mismatch: {baseline_maps.shape} vs {quantized_maps.shape}"
        )
    per_head = []
    for head_index in range(baseline_maps.shape[0]):
        row: dict[str, Any] = {"head": head_index}
        row.update(
            distribution_metrics(
                baseline_maps[head_index],
                quantized_maps[head_index],
                attention_mass,
            )
        )
        per_head.append(row)
    mean_metrics = distribution_metrics(
        baseline_maps.mean(dim=0),
        quantized_maps.mean(dim=0),
        attention_mass,
    )
    return {
        "definition": (
            "Each head is normalized over the 28x28 patch keys before comparison."
        ),
        "per_head": per_head,
        "mean_over_heads": mean_metrics,
        "within_variant_head_similarity": {
            "baseline": within_variant_head_similarity(
                baseline_maps,
                attention_mass,
            ),
            "candidate": within_variant_head_similarity(
                quantized_maps,
                attention_mass,
            ),
            "interpretation": (
                "Higher pairwise cosine and lower pairwise divergence indicate "
                "lower attention-head diversity."
            ),
        },
        "per_head_summary": {
            key: {
                "mean": float(np.mean([row[key] for row in per_head])),
                "min": float(np.min([row[key] for row in per_head])),
                "max": float(np.max([row[key] for row in per_head])),
            }
            for key in (
                "cosine_similarity",
                "total_variation_distance",
                "jensen_shannon_divergence_nats",
                "top_attention_mass_mask_iou",
            )
        },
    }


def render_attention_comparison(
    *,
    crop: Image.Image,
    baseline_maps: torch.Tensor,
    quantized_maps: torch.Tensor,
    metrics: dict[str, Any],
    output_dir: Path,
    dpi: int,
    candidate_label: str = "FC-S signed INT4",
    candidate_short_label: str = "INT4",
) -> dict[str, str]:
    crop_array = np.asarray(crop, dtype=np.float32) / 255.0
    baseline_mean = baseline_maps.mean(dim=0)
    quantized_mean = quantized_maps.mean(dim=0)
    mean_difference = quantized_mean - baseline_mean
    baseline_mean_image = upsample_patch_map(baseline_mean)
    quantized_mean_image = upsample_patch_map(quantized_mean)
    difference_image = upsample_patch_map(mean_difference)
    shared_maximum = max(
        float(baseline_mean_image.max()),
        float(quantized_mean_image.max()),
        1e-12,
    )
    difference_limit = max(float(np.max(np.abs(difference_image))), 1e-12)

    mean_comparison_path = output_dir / "mean_attention_comparison.png"
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    axes[0, 0].imshow(crop_array)
    axes[0, 0].set_title("Validation center crop")
    axes[0, 1].imshow(color_overlay(crop_array, baseline_mean_image, alpha=0.55))
    axes[0, 1].set_title("FC-S FP32 baseline")
    axes[0, 2].imshow(color_overlay(crop_array, quantized_mean_image, alpha=0.55))
    axes[0, 2].set_title(candidate_label)
    baseline_artist = axes[1, 0].imshow(
        baseline_mean_image,
        cmap="turbo",
        vmin=0.0,
        vmax=shared_maximum,
    )
    axes[1, 0].set_title("Baseline mean attention")
    axes[1, 1].imshow(
        quantized_mean_image,
        cmap="turbo",
        vmin=0.0,
        vmax=shared_maximum,
    )
    axes[1, 1].set_title(f"{candidate_short_label} mean attention")
    difference_artist = axes[1, 2].imshow(
        difference_image,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[1, 2].set_title(f"{candidate_short_label} - baseline (signed)")
    for axis in axes.flatten():
        axis.axis("off")
    figure.colorbar(
        baseline_artist,
        ax=[axes[1, 0], axes[1, 1]],
        fraction=0.046,
        pad=0.02,
        label="Attention probability",
    )
    figure.colorbar(
        difference_artist,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.02,
        label="Probability change",
    )
    mean_row = metrics["mean_over_heads"]
    figure.suptitle(
        "Green lizard — final-block mean CLS-to-patch attention\n"
        f"cos={mean_row['cosine_similarity']:.6f}, "
        f"TV={mean_row['total_variation_distance']:.6f}, "
        f"JSD={mean_row['jensen_shannon_divergence_nats']:.6g}",
        fontsize=15,
    )
    figure.savefig(mean_comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    head_difference_path = output_dir / "per_head_signed_difference.png"
    head_differences = quantized_maps - baseline_maps
    high_resolution_differences = np.stack(
        [upsample_patch_map(head_difference) for head_difference in head_differences]
    )
    common_limit = max(float(np.max(np.abs(high_resolution_differences))), 1e-12)
    figure, axes = plt.subplots(3, 4, figsize=(13, 9.8), constrained_layout=True)
    difference_artist = None
    for head_index, (axis, difference) in enumerate(
        zip(axes.flatten(), high_resolution_differences)
    ):
        difference_artist = axis.imshow(
            difference,
            cmap="coolwarm",
            vmin=-common_limit,
            vmax=common_limit,
        )
        row = metrics["per_head"][head_index]
        axis.set_title(
            f"Head {head_index}: cos={row['cosine_similarity']:.4f}, "
            f"TV={row['total_variation_distance']:.4f}",
            fontsize=10,
        )
        axis.axis("off")
    if difference_artist is not None:
        figure.colorbar(
            difference_artist,
            ax=axes,
            fraction=0.025,
            pad=0.02,
            label=f"{candidate_short_label} - baseline attention probability",
        )
    figure.suptitle(
        f"Final-block CLS-to-patch attention change from {candidate_label}",
        fontsize=16,
    )
    figure.savefig(head_difference_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    metric_plot_path = output_dir / "per_head_attention_metrics.png"
    per_head = metrics["per_head"]
    heads = np.arange(len(per_head))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    axes[0].bar(heads, [row["cosine_similarity"] for row in per_head])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Cosine similarity")
    axes[1].bar(heads, [row["total_variation_distance"] for row in per_head])
    axes[1].set_title("Total variation distance")
    axes[2].bar(
        heads,
        [row["jensen_shannon_divergence_nats"] for row in per_head],
    )
    axes[2].set_title("Jensen-Shannon divergence")
    for axis in axes:
        axis.set_xlabel("Attention head")
        axis.set_xticks(heads)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"{candidate_label} effect on final-block attention",
        fontsize=15,
    )
    figure.savefig(metric_plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "mean_comparison": str(mean_comparison_path.resolve()),
        "per_head_signed_difference": str(head_difference_path.resolve()),
        "per_head_metrics": str(metric_plot_path.resolve()),
    }


def render_quantization_metrics(
    quantization: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> str:
    layers = quantization["layers"]
    labels = [f"B{row['block']} {row['fc'].upper()}" for row in layers]
    positions = np.arange(len(layers))
    sparse_error = [row["relative_sparse_frobenius_error"] for row in layers]
    full_error = [row["relative_full_weight_frobenius_error"] for row in layers]
    short_label = quantization["short_label"]
    quantized_symbol = quantization["quantized_symbol"]
    if quantization["mode"] in (
        "int1_positive_mean",
        "int1_sparse_positive_mean",
    ):
        third_values = [row["alpha"] for row in layers]
        third_ylabel = "alpha"
        third_title = "Per-matrix alpha = mean of strictly positive S entries"
    else:
        third_values = [
            row["quantization_zeroed_nonzero"]
            / max(row["thresholded_sparse_nonzero"], 1)
            for row in layers
        ]
        third_ylabel = "Fraction"
        third_title = f"Nonzero S entries rounded to {short_label} zero"

    output_path = output_dir / f"fc_sparse_{short_label.lower()}_error_by_layer.png"
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].bar(positions, sparse_error)
    axes[0].set_ylabel("Relative error")
    axes[0].set_title(
        f"S quantization error: ||{quantized_symbol} - S||F / ||S||F"
    )
    axes[1].bar(positions, full_error)
    axes[1].set_ylabel("Relative error")
    axes[1].set_title(
        f"Full FC error: ||(L+{quantized_symbol}) - (L+S)||F / ||L+S||F"
    )
    axes[2].bar(positions, third_values)
    axes[2].set_ylabel(third_ylabel)
    axes[2].set_title(third_title)
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"{quantization['display_label']} quantization of FC sparse matrices",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return str(output_path.resolve())


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"image does not exist: {args.image}")
    if not args.base_state.is_file():
        raise FileNotFoundError(f"base state does not exist: {args.base_state}")
    if not args.matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {args.matrix_dir}")
    if not math.isfinite(args.zero_threshold) or args.zero_threshold < 0.0:
        raise ValueError("zero threshold must be a finite non-negative number")
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0, 1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")


def main(
    *,
    default_quantization_mode: str = "int4",
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    args = parse_args(
        default_quantization_mode=default_quantization_mode,
        default_output_dir=default_output_dir,
    )
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    specification = quantization_spec(
        args.quantization_mode,
        args.quantization_granularity,
    )

    model = DinoViTBase8()
    decomposed_modules = decomposed_module_names(model)
    base_state_metadata = load_non_decomposed_state(
        model,
        args.base_state,
        decomposed_modules,
    )
    original, crop, image_tensor = prepare_image(args.image)

    print(f"device={device}")
    print("building baseline: QKV/proj=L+S(fp32), FC=L+S(fp32)")
    baseline_decomposition = apply_l_plus_s_variant(
        model,
        args.matrix_dir,
        decomposed_modules,
        quantize_fc_sparse=False,
        granularity=args.quantization_granularity,
        zero_threshold=args.zero_threshold,
        quantization_mode=args.quantization_mode,
    )
    baseline_maps, baseline_source_block, baseline_attention_shape = (
        infer_patch_attention(model, image_tensor, device)
    )

    print(
        f"building {specification['short_label']} variant: "
        f"QKV/proj=L+S(fp32), {specification['weight_expression']}"
    )
    quantized_decomposition = apply_l_plus_s_variant(
        model,
        args.matrix_dir,
        decomposed_modules,
        quantize_fc_sparse=True,
        granularity=args.quantization_granularity,
        zero_threshold=args.zero_threshold,
        quantization_mode=args.quantization_mode,
    )
    quantized_maps, quantized_source_block, quantized_attention_shape = (
        infer_patch_attention(model, image_tensor, device)
    )
    if baseline_source_block != quantized_source_block:
        raise RuntimeError("baseline and quantized attention came from different blocks")

    baseline_paths = render_variant(
        original=original,
        crop=crop,
        patch_maps=baseline_maps,
        output_dir=args.output_dir / "fc_s_fp32_baseline",
        title="Mixed-rho L+S, FC-S FP32 baseline",
        attention_mass=args.attention_mass,
        attention_source_block=baseline_source_block,
        dpi=args.dpi,
    )
    quantized_paths = render_variant(
        original=original,
        crop=crop,
        patch_maps=quantized_maps,
        output_dir=args.output_dir / specification["output_label"],
        title=f"Mixed-rho L+S, FC-S {specification['display_label']}",
        attention_mass=args.attention_mass,
        attention_source_block=quantized_source_block,
        dpi=args.dpi,
    )
    metrics = attention_metrics(
        baseline_maps,
        quantized_maps,
        args.attention_mass,
    )
    comparison_paths = render_attention_comparison(
        crop=crop,
        baseline_maps=baseline_maps,
        quantized_maps=quantized_maps,
        metrics=metrics,
        output_dir=args.output_dir,
        dpi=args.dpi,
        candidate_label=f"FC-S {specification['display_label']}",
        candidate_short_label=specification["short_label"],
    )
    quantization = quantized_decomposition["quantization"]
    if not isinstance(quantization, dict):
        raise RuntimeError("missing quantization statistics")
    quantization_plot = render_quantization_metrics(
        quantization,
        args.output_dir,
        args.dpi,
    )

    source_copy = args.output_dir / "source.jpg"
    crop_copy = args.output_dir / "center_crop.jpg"
    original.convert("RGB").save(source_copy, quality=95)
    crop.convert("RGB").save(crop_copy, quality=95)
    metadata = {
        "experiment": (
            "final-block attention with FC sparse matrices quantized using "
            f"{specification['display_label']}"
        ),
        "sample": {
            "sample_id": "52_green_lizard",
            "class_name": "green lizard, Lacerta viridis",
            "source_image": str(args.image.resolve()),
            "source_sha256": sha256_file(args.image),
            "source_copy": str(source_copy.resolve()),
            "center_crop_copy": str(crop_copy.resolve()),
            "transform": TRANSFORM_DESCRIPTION,
        },
        "model_reconstruction": {
            "base_state": base_state_metadata,
            "baseline": baseline_decomposition,
            "quantized": quantized_decomposition,
            "controlled_difference": (
                "Only mlp.fc1.S and mlp.fc2.S differ: thresholded FP32 S in the "
                f"baseline versus {specification['display_label']} S in the candidate."
            ),
        },
        "attention": {
            "source_block": baseline_source_block,
            "baseline_shape": baseline_attention_shape,
            "quantized_shape": quantized_attention_shape,
            "heads": baseline_maps.shape[0],
            "patch_grid": list(baseline_maps.shape[1:]),
            "attention_mass_threshold": args.attention_mass,
            "mechanism_note": (
                "Block 11 FC layers execute after block 11 self-attention, so they "
                "cannot change this block's attention matrix directly. The observed "
                "change is caused by quantized FC-S in blocks 0-10 altering the tokens "
                "entering block 11."
            ),
            "metrics": metrics,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "outputs": {
            "baseline": baseline_paths,
            specification["short_label"].lower(): quantized_paths,
            **comparison_paths,
            "quantization_error_by_layer": quantization_plot,
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    mean_metrics = metrics["mean_over_heads"]
    print(f"output_dir={args.output_dir.resolve()}")
    print(
        f"{specification['short_label'].lower()}_overview="
        f"{quantized_paths['overview']}"
    )
    print(f"mean_comparison={comparison_paths['mean_comparison']}")
    print(
        "mean_attention_metrics="
        f"cosine={mean_metrics['cosine_similarity']:.8f} "
        f"tv={mean_metrics['total_variation_distance']:.8f} "
        f"jsd={mean_metrics['jensen_shannon_divergence_nats']:.8g}"
    )
    print(
        "quantization_metrics="
        f"relative_S_error={quantization['relative_sparse_frobenius_error']:.8f} "
        f"relative_L_plus_S_error="
        f"{quantization['relative_full_weight_frobenius_error']:.8f} "
        f"zeroed_nonzero_fraction="
        f"{quantization['quantization_zeroed_fraction_of_thresholded_nonzero']:.8f} "
        f"activated_zero_fraction="
        f"{quantization['quantization_activated_zero_fraction_of_thresholded_zeros']:.8f}"
    )
    print(f"metadata={metadata_path.resolve()}")


if __name__ == "__main__":
    main()
