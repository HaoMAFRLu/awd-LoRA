#!/usr/bin/env python3
"""Spectrally mask QKV-L and optionally quantize masked QKV-S.

Small singular components of every QKV low-rank matrix are removed using a
relative threshold on ``sigma_i / sigma_1``.  FC and attention-output projection
weights remain the exact checkpoint ``L+S``.  The retained QKV-L is paired with
FP32 S, masked INT3 S, or sparse-mask INT1 S.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from salaad_vision.models.dino import DinoViTBase8  # noqa: E402
from scripts.visualize_dino_attention import (  # noqa: E402
    choose_device,
    load_matrix_file,
    matrix_rank,
    prepare_image,
)
from scripts.visualize_fc_sparse_int4_attention import (  # noqa: E402
    attention_metrics,
    binary_positive_mean_fake_quantize,
    decomposed_module_names,
    infer_patch_attention,
    load_non_decomposed_state,
    sha256_file,
    symmetric_int3_fake_quantize,
    upsample_patch_map,
)
from scripts.visualize_qkv_sparse_masked_quantization_attention import (  # noqa: E402
    QKV_PATTERN,
    apply_qkv_variant,
    fake_quantize_masked_sparse,
    quantization_specification,
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
    / "qkv_low_rank_spectral_mask_attention"
    / "52_green_lizard"
)
VARIANTS = (
    ("fp32", "QKV L+S FP32", None),
    ("lmask_s_fp32", "QKV-L spectral mask\nS FP32", "fp32"),
    ("lmask_s_int3", "QKV-L spectral mask\nS mask→INT3", "int3"),
    ("lmask_s_int1", "QKV-L spectral mask\nS mask→INT1", "int1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--base-state",
        type=Path,
        default=DEFAULT_RUN_DIR / "non_decomposed_state.pth",
    )
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--relative-sigma-threshold",
        type=float,
        default=1e-2,
        help="Remove QKV-L singular components with sigma_i/sigma_1 below this.",
    )
    parser.add_argument("--sparse-zero-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--int3-granularity",
        choices=("per_output_channel", "per_tensor"),
        default="per_output_channel",
    )
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"image does not exist: {args.image}")
    if not args.base_state.is_file():
        raise FileNotFoundError(f"base state does not exist: {args.base_state}")
    if not args.matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {args.matrix_dir}")
    if (
        not math.isfinite(args.relative_sigma_threshold)
        or not 0.0 < args.relative_sigma_threshold <= 1.0
    ):
        raise ValueError("relative sigma threshold must be in (0,1]")
    if (
        not math.isfinite(args.sparse_zero_threshold)
        or args.sparse_zero_threshold < 0.0
    ):
        raise ValueError("sparse zero threshold must be finite and non-negative")
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0,1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0,1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")


@torch.no_grad()
def apply_spectral_mask_variant(
    model: DinoViTBase8,
    matrix_dir: Path,
    decomposed_modules: set[str],
    *,
    sparse_mode: str,
    relative_sigma_threshold: float,
    sparse_zero_threshold: float,
    int3_granularity: str,
    low_rank_quantization_mode: str = "fp32",
) -> dict[str, Any]:
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    if sparse_mode not in ("fp32", "int3", "int1"):
        raise ValueError(f"unsupported sparse mode: {sparse_mode}")
    if low_rank_quantization_mode not in ("fp32", "int3", "int1"):
        raise ValueError(
            f"unsupported low-rank quantization mode: {low_rank_quantization_mode}"
        )

    quantize_sparse = sparse_mode != "fp32"
    sparse_specification = (
        quantization_specification(sparse_mode, int3_granularity)
        if quantize_sparse
        else None
    )
    sparse_code_histogram = (
        {str(code): 0 for code in sparse_specification["codes"]}
        if sparse_specification is not None
        else {}
    )
    low_rank_code_values = {
        "fp32": [],
        "int3": list(range(-3, 4)),
        "int1": [-1, 1],
    }[low_rank_quantization_mode]
    low_rank_code_histogram = {
        str(code): 0 for code in low_rank_code_values
    }
    seen: set[str] = set()
    qkv_layers = 0
    low_rank_norm_squared = 0.0
    masked_low_rank_norm_squared = 0.0
    low_rank_error_squared = 0.0
    low_rank_quantization_error_vs_masked_squared = 0.0
    low_rank_original_nonzero = 0
    low_rank_quantized_nonzero = 0
    low_rank_quantization_zeroed = 0
    low_rank_quantization_activated_zero = 0
    full_qkv_norm_squared = 0.0
    full_qkv_error_squared = 0.0
    sparse_norm_squared = 0.0
    sparse_error_squared = 0.0
    sparse_original_nonzero = 0
    sparse_masked_nonzero = 0
    sparse_threshold_removed = 0
    sparse_quantized_nonzero = 0
    sparse_quantization_zeroed = 0
    layer_statistics: list[dict[str, Any]] = []

    for matrix_file in matrix_files:
        low_rank, sparse = load_matrix_file(matrix_file)
        if set(low_rank) != set(sparse):
            raise ValueError(f"L/S key mismatch in {matrix_file}")
        for layer_name in sorted(low_rank):
            if layer_name in seen:
                raise ValueError(f"duplicate decomposed layer: {layer_name}")
            if layer_name not in decomposed_modules:
                raise ValueError(f"unexpected decomposed layer: {layer_name}")
            seen.add(layer_name)

            low_rank_weight = low_rank[layer_name].detach().cpu().float()
            sparse_weight = sparse[layer_name].detach().cpu().float()
            layer = model.get_submodule(layer_name)
            expected_shape = tuple(layer.weight.shape)
            if tuple(low_rank_weight.shape) != expected_shape:
                raise ValueError(f"L shape mismatch for {layer_name}")
            if tuple(sparse_weight.shape) != expected_shape:
                raise ValueError(f"S shape mismatch for {layer_name}")
            if not torch.isfinite(low_rank_weight).all():
                raise ValueError(f"non-finite L for {layer_name}")
            if not torch.isfinite(sparse_weight).all():
                raise ValueError(f"non-finite S for {layer_name}")

            qkv_match = QKV_PATTERN.fullmatch(layer_name)
            if qkv_match is None:
                layer.weight.copy_(
                    (low_rank_weight + sparse_weight).to(dtype=layer.weight.dtype)
                )
                continue

            qkv_layers += 1
            u, singular_values, vh = torch.linalg.svd(
                low_rank_weight,
                full_matrices=False,
            )
            normalized_singular_values = singular_values / singular_values[0]
            retained_rank = int(
                torch.count_nonzero(
                    normalized_singular_values >= relative_sigma_threshold
                ).item()
            )
            masked_low_rank = (
                u[:, :retained_rank] * singular_values[:retained_rank]
            ) @ vh[:retained_rank]
            singular_energy = singular_values.double().square()
            retained_energy = float(
                (
                    singular_energy[:retained_rank].sum()
                    / singular_energy.sum().clamp_min(1e-30)
                ).item()
            )
            numerical_rank_tolerance = (
                singular_values[0].double()
                * max(low_rank_weight.shape)
                * torch.finfo(low_rank_weight.dtype).eps
            )
            numerical_rank = int(
                torch.count_nonzero(
                    singular_values.double() > numerical_rank_tolerance
                ).item()
            )

            low_rank_quantization_statistics = None
            if low_rank_quantization_mode == "int3":
                used_low_rank, low_rank_codes, low_rank_scales = (
                    symmetric_int3_fake_quantize(
                        masked_low_rank,
                        int3_granularity,
                    )
                )
            elif low_rank_quantization_mode == "int1":
                used_low_rank, low_rank_codes, low_rank_scales = (
                    binary_positive_mean_fake_quantize(masked_low_rank)
                )
            else:
                used_low_rank = masked_low_rank
                low_rank_codes = None
                low_rank_scales = None

            if low_rank_codes is not None and low_rank_scales is not None:
                masked_low_rank_nonzero_mask = masked_low_rank != 0.0
                quantized_low_rank_nonzero_mask = low_rank_codes != 0
                layer_low_rank_original_nonzero = int(
                    torch.count_nonzero(masked_low_rank_nonzero_mask).item()
                )
                layer_low_rank_quantized_nonzero = int(
                    torch.count_nonzero(quantized_low_rank_nonzero_mask).item()
                )
                layer_low_rank_zeroed = int(
                    torch.count_nonzero(
                        masked_low_rank_nonzero_mask
                        & ~quantized_low_rank_nonzero_mask
                    ).item()
                )
                layer_low_rank_activated = int(
                    torch.count_nonzero(
                        ~masked_low_rank_nonzero_mask
                        & quantized_low_rank_nonzero_mask
                    ).item()
                )
                for code in low_rank_code_values:
                    low_rank_code_histogram[str(code)] += int(
                        torch.count_nonzero(low_rank_codes == code).item()
                    )
                positive_low_rank_scales = low_rank_scales[
                    low_rank_scales > 0.0
                ]
                low_rank_scale_min = (
                    float(positive_low_rank_scales.min().item())
                    if positive_low_rank_scales.numel() > 0
                    else 0.0
                )
                low_rank_scale_median = (
                    float(positive_low_rank_scales.median().item())
                    if positive_low_rank_scales.numel() > 0
                    else 0.0
                )
                low_rank_scale_max = (
                    float(positive_low_rank_scales.max().item())
                    if positive_low_rank_scales.numel() > 0
                    else 0.0
                )
                layer_masked_low_rank_norm_squared = float(
                    masked_low_rank.square().sum().item()
                )
                layer_low_rank_quantization_error_squared = float(
                    (used_low_rank - masked_low_rank).square().sum().item()
                )
                low_rank_original_nonzero += layer_low_rank_original_nonzero
                low_rank_quantized_nonzero += layer_low_rank_quantized_nonzero
                low_rank_quantization_zeroed += layer_low_rank_zeroed
                low_rank_quantization_activated_zero += layer_low_rank_activated
                masked_low_rank_norm_squared += (
                    layer_masked_low_rank_norm_squared
                )
                low_rank_quantization_error_vs_masked_squared += (
                    layer_low_rank_quantization_error_squared
                )
                low_rank_quantization_statistics = {
                    "original_nonzero": layer_low_rank_original_nonzero,
                    "quantized_nonzero": layer_low_rank_quantized_nonzero,
                    "quantization_zeroed_nonzero": layer_low_rank_zeroed,
                    "quantization_activated_zero": layer_low_rank_activated,
                    "scale_min": low_rank_scale_min,
                    "scale_median": low_rank_scale_median,
                    "scale_max": low_rank_scale_max,
                    "alpha": (
                        low_rank_scale_max
                        if low_rank_quantization_mode == "int1"
                        else None
                    ),
                    "relative_quantization_error_vs_spectral_masked_L": (
                        math.sqrt(
                            layer_low_rank_quantization_error_squared
                            / max(layer_masked_low_rank_norm_squared, 1e-30)
                        )
                    ),
                }

            sparse_layer_statistics = None
            if quantize_sparse:
                original_nonzero_mask = sparse_weight != 0.0
                threshold_mask = original_nonzero_mask & (
                    sparse_weight.abs() <= sparse_zero_threshold
                )
                masked_sparse = sparse_weight.masked_fill(threshold_mask, 0.0)
                masked_nonzero_mask = masked_sparse != 0.0
                used_sparse, sparse_codes, sparse_scales = (
                    fake_quantize_masked_sparse(
                        masked_sparse,
                        sparse_mode,
                        int3_granularity,
                    )
                )
                quantized_nonzero_mask = sparse_codes != 0
                layer_original_nonzero = int(
                    torch.count_nonzero(original_nonzero_mask).item()
                )
                layer_masked_nonzero = int(
                    torch.count_nonzero(masked_nonzero_mask).item()
                )
                layer_threshold_removed = int(
                    torch.count_nonzero(threshold_mask).item()
                )
                layer_quantized_nonzero = int(
                    torch.count_nonzero(quantized_nonzero_mask).item()
                )
                layer_quantization_zeroed = int(
                    torch.count_nonzero(
                        masked_nonzero_mask & ~quantized_nonzero_mask
                    ).item()
                )
                for code in sparse_specification["codes"]:
                    sparse_code_histogram[str(code)] += int(
                        torch.count_nonzero(sparse_codes == code).item()
                    )
                positive_scales = sparse_scales[sparse_scales > 0.0]
                sparse_scale_min = (
                    float(positive_scales.min().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
                sparse_scale_median = (
                    float(positive_scales.median().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
                sparse_scale_max = (
                    float(positive_scales.max().item())
                    if positive_scales.numel() > 0
                    else 0.0
                )
                layer_sparse_norm_squared = float(
                    sparse_weight.square().sum().item()
                )
                layer_sparse_error_squared = float(
                    (used_sparse - sparse_weight).square().sum().item()
                )
                sparse_norm_squared += layer_sparse_norm_squared
                sparse_error_squared += layer_sparse_error_squared
                sparse_original_nonzero += layer_original_nonzero
                sparse_masked_nonzero += layer_masked_nonzero
                sparse_threshold_removed += layer_threshold_removed
                sparse_quantized_nonzero += layer_quantized_nonzero
                sparse_quantization_zeroed += layer_quantization_zeroed
                sparse_layer_statistics = {
                    "original_nonzero": layer_original_nonzero,
                    "masked_nonzero": layer_masked_nonzero,
                    "threshold_removed": layer_threshold_removed,
                    "quantized_nonzero": layer_quantized_nonzero,
                    "quantization_zeroed_masked_nonzero": (
                        layer_quantization_zeroed
                    ),
                    "scale_min": sparse_scale_min,
                    "scale_median": sparse_scale_median,
                    "scale_max": sparse_scale_max,
                    "relative_S_error_vs_exact": math.sqrt(
                        layer_sparse_error_squared
                        / max(layer_sparse_norm_squared, 1e-30)
                    ),
                }
            else:
                used_sparse = sparse_weight

            reference = low_rank_weight + sparse_weight
            replacement = used_low_rank + used_sparse
            layer_low_rank_norm_squared = float(
                low_rank_weight.square().sum().item()
            )
            layer_low_rank_error_squared = float(
                (used_low_rank - low_rank_weight).square().sum().item()
            )
            layer_full_norm_squared = float(reference.square().sum().item())
            layer_full_error_squared = float(
                (replacement - reference).square().sum().item()
            )
            low_rank_norm_squared += layer_low_rank_norm_squared
            low_rank_error_squared += layer_low_rank_error_squared
            full_qkv_norm_squared += layer_full_norm_squared
            full_qkv_error_squared += layer_full_error_squared

            layer_statistics.append(
                {
                    "layer": layer_name,
                    "block": int(qkv_match.group("block")),
                    "shape": list(low_rank_weight.shape),
                    "original_numerical_rank": numerical_rank,
                    "retained_rank": retained_rank,
                    "removed_rank": numerical_rank - retained_rank,
                    "retained_energy": retained_energy,
                    "relative_L_error": math.sqrt(
                        layer_low_rank_error_squared
                        / max(layer_low_rank_norm_squared, 1e-30)
                    ),
                    "relative_full_QKV_error": math.sqrt(
                        layer_full_error_squared
                        / max(layer_full_norm_squared, 1e-30)
                    ),
                    "L_quantization": low_rank_quantization_statistics,
                    "S_quantization": sparse_layer_statistics,
                }
            )
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))

        del low_rank
        del sparse
        gc.collect()

    missing = decomposed_modules - seen
    if missing:
        raise ValueError(f"decomposition coverage mismatch; missing={sorted(missing)}")
    if qkv_layers != 12:
        raise ValueError(f"expected 12 QKV layers, got {qkv_layers}")

    low_rank_expression = "spectral_mask(L)"
    if low_rank_quantization_mode != "fp32":
        low_rank_expression = (
            f"dequant({low_rank_quantization_mode.upper()}(spectral_mask(L)))"
        )
    sparse_expression = (
        "S(fp32 exact)"
        if sparse_mode == "fp32"
        else f"masked-{sparse_mode.upper()}(S)"
    )
    return {
        "weight_definition": (
            f"QKV={low_rank_expression}+{sparse_expression}; "
            "FC/proj=L+S exact"
        ),
        "relative_sigma_threshold": relative_sigma_threshold,
        "relative_L_error": math.sqrt(
            low_rank_error_squared / max(low_rank_norm_squared, 1e-30)
        ),
        "relative_full_QKV_error": math.sqrt(
            full_qkv_error_squared / max(full_qkv_norm_squared, 1e-30)
        ),
        "retained_rank_min": min(row["retained_rank"] for row in layer_statistics),
        "retained_rank_max": max(row["retained_rank"] for row in layer_statistics),
        "retained_energy_min": min(
            row["retained_energy"] for row in layer_statistics
        ),
        "retained_energy_max": max(
            row["retained_energy"] for row in layer_statistics
        ),
        "low_rank_quantization_mode": low_rank_quantization_mode,
        "low_rank_quantization": (
            {
                "bits": 3 if low_rank_quantization_mode == "int3" else 1,
                "scheme": (
                    "signed symmetric narrow-range"
                    if low_rank_quantization_mode == "int3"
                    else "binary positive-elements-mean scale"
                ),
                "granularity": (
                    int3_granularity
                    if low_rank_quantization_mode == "int3"
                    else "per_qkv_matrix"
                ),
                "codes": low_rank_code_values,
                "code_histogram": low_rank_code_histogram,
                "scale_definition": (
                    "max(abs(spectral_mask(L)))/3 per scale group"
                    if low_rank_quantization_mode == "int3"
                    else "alpha=mean of positive spectral_mask(L) elements"
                ),
                "original_nonzero": low_rank_original_nonzero,
                "quantized_nonzero": low_rank_quantized_nonzero,
                "quantization_zeroed_nonzero": low_rank_quantization_zeroed,
                "quantization_zeroed_fraction_of_original_nonzero": (
                    low_rank_quantization_zeroed
                    / max(low_rank_original_nonzero, 1)
                ),
                "quantization_activated_zero": (
                    low_rank_quantization_activated_zero
                ),
                "relative_quantization_error_vs_spectral_masked_L": math.sqrt(
                    low_rank_quantization_error_vs_masked_squared
                    / max(masked_low_rank_norm_squared, 1e-30)
                ),
            }
            if low_rank_quantization_mode != "fp32"
            else None
        ),
        "sparse_mode": sparse_mode,
        "sparse_quantization": (
            {
                **{
                    key: value
                    for key, value in sparse_specification.items()
                    if key != "codes"
                },
                "zero_threshold": sparse_zero_threshold,
                "code_histogram": sparse_code_histogram,
                "original_nonzero": sparse_original_nonzero,
                "masked_nonzero": sparse_masked_nonzero,
                "threshold_removed": sparse_threshold_removed,
                "quantized_nonzero": sparse_quantized_nonzero,
                "quantization_zeroed_masked_nonzero": (
                    sparse_quantization_zeroed
                ),
                "quantization_zeroed_fraction_of_masked_nonzero": (
                    sparse_quantization_zeroed / max(sparse_masked_nonzero, 1)
                ),
                "relative_S_error_vs_exact": math.sqrt(
                    sparse_error_squared / max(sparse_norm_squared, 1e-30)
                ),
            }
            if quantize_sparse
            else None
        ),
        "layers": sorted(layer_statistics, key=lambda row: row["block"]),
    }


def render_horizontal_matrix(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
    figure_title: str = (
        "Green lizard — QKV-L spectral mask; FC/proj fixed as FP32 L+S\n"
        "Rows: QKV variants; columns: Mean and Head 0–11; each column shares one scale"
    ),
    output_filename: str = "mean_and_heads_horizontal_qkv_L_spectral_mask.png",
) -> Path:
    titles = ["Mean"] + [f"Head {index}" for index in range(12)]
    high_resolution: list[list[np.ndarray]] = []
    for variant in variants:
        attention = variant["attention"]
        row = [upsample_patch_map(attention.mean(dim=0))]
        row.extend(
            upsample_patch_map(attention[index], mode="nearest")
            for index in range(12)
        )
        high_resolution.append(row)
    column_maxima = [
        max(float(row[column].max()) for row in high_resolution)
        for column in range(13)
    ]
    figure, axes = plt.subplots(
        len(variants),
        13,
        figsize=(29.5, 2.25 * len(variants) + 0.3),
        constrained_layout=True,
        squeeze=False,
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12, pad=7)
    for row_index, variant in enumerate(variants):
        for column in range(13):
            axis = axes[row_index, column]
            axis.imshow(crop_array)
            axis.imshow(
                high_resolution[row_index][column],
                cmap="turbo",
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=max(column_maxima[column], 1e-12),
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_ylabel(variant["label"], fontsize=11, labelpad=9)
    figure.suptitle(figure_title, fontsize=17)
    output_path = output_dir / output_filename
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_rank_plot(
    decomposition: dict[str, Any],
    output_dir: Path,
    dpi: int,
    relative_sigma_threshold: float,
) -> Path:
    rows = decomposition["layers"]
    blocks = np.arange(12)
    original = [row["original_numerical_rank"] for row in rows]
    retained = [row["retained_rank"] for row in rows]
    energy_loss = [1.0 - row["retained_energy"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(blocks - width / 2, original, width=width, label="Numerical rank")
    axes[0].bar(blocks + width / 2, retained, width=width, label="Retained rank")
    axes[0].set_ylabel("Rank")
    axes[0].legend()
    axes[1].bar(blocks, energy_loss)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Removed L energy fraction")
    for axis in axes:
        axis.set_xticks(blocks)
        axis.set_xticklabels([f"Block {index}" for index in blocks])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "QKV-L spectral mask: remove singular components with "
        f"sigma_i/sigma_1 < {relative_sigma_threshold:g}",
        fontsize=15,
    )
    output_path = output_dir / "qkv_L_spectral_mask_rank_and_energy.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    model = DinoViTBase8()
    decomposed_modules = decomposed_module_names(model)
    base_state_metadata = load_non_decomposed_state(
        model,
        args.base_state,
        decomposed_modules,
    )
    _, crop, image_tensor = prepare_image(args.image)
    crop_array = np.asarray(crop, dtype=np.float32) / 255.0
    variants: list[dict[str, Any]] = []
    decompositions: dict[str, Any] = {}
    source_block: int | None = None
    attention_shape: list[int] | None = None

    print(f"device={device}", flush=True)
    for key, label, sparse_mode in VARIANTS:
        print(f"building {key}", flush=True)
        if sparse_mode is None:
            decomposition = apply_qkv_variant(
                model,
                args.matrix_dir,
                decomposed_modules,
                mode="fp32",
                zero_threshold=args.sparse_zero_threshold,
                int3_granularity=args.int3_granularity,
            )
        else:
            decomposition = apply_spectral_mask_variant(
                model,
                args.matrix_dir,
                decomposed_modules,
                sparse_mode=sparse_mode,
                relative_sigma_threshold=args.relative_sigma_threshold,
                sparse_zero_threshold=args.sparse_zero_threshold,
                int3_granularity=args.int3_granularity,
            )
        attention, current_source_block, current_shape = infer_patch_attention(
            model,
            image_tensor,
            device,
        )
        if source_block is None:
            source_block = current_source_block
            attention_shape = current_shape
        elif source_block != current_source_block:
            raise RuntimeError("variants used different attention source blocks")
        variant_dir = args.output_dir / key
        variant_dir.mkdir(parents=True, exist_ok=True)
        attention_path = variant_dir / "cls_to_patch_attention.npy"
        np.save(attention_path, attention.numpy())
        variants.append(
            {
                "key": key,
                "label": label,
                "attention": attention,
                "attention_path": attention_path,
            }
        )
        decompositions[key] = decomposition

    attention_by_key = {row["key"]: row["attention"] for row in variants}
    metrics_vs_fp32 = {
        key: (
            None
            if key == "fp32"
            else attention_metrics(
                attention_by_key["fp32"],
                attention_by_key[key],
                args.attention_mass,
            )
        )
        for key, _, _ in VARIANTS
    }
    incremental_metrics = {
        key: attention_metrics(
            attention_by_key["lmask_s_fp32"],
            attention_by_key[key],
            args.attention_mass,
        )
        for key in ("lmask_s_int3", "lmask_s_int1")
    }
    horizontal_path = render_horizontal_matrix(
        variants,
        crop_array,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
    )
    rank_path = render_rank_plot(
        decompositions["lmask_s_fp32"],
        args.output_dir,
        args.dpi,
        args.relative_sigma_threshold,
    )

    metadata = {
        "experiment": (
            "spectrally mask QKV-L, then pair it with FP32, masked INT3, or "
            "masked INT1 QKV-S"
        ),
        "spectral_mask": {
            "definition": (
                "SVD L=U diag(sigma) Vh; retain complete components satisfying "
                "sigma_i/sigma_1 >= threshold"
            ),
            "relative_sigma_threshold": args.relative_sigma_threshold,
            "reason": (
                "This removes weak rank-1 spectral components without applying an "
                "elementwise mask that would destroy low-rank structure."
            ),
        },
        "fixed_components": (
            "All FC and attention-projection weights are exact checkpoint FP32 L+S."
        ),
        "sample": {
            "sample_id": "52_green_lizard",
            "source_image": str(args.image.resolve()),
            "source_sha256": sha256_file(args.image),
        },
        "checkpoint": {
            "matrix_dir": str(args.matrix_dir.resolve()),
            "base_state": base_state_metadata,
        },
        "attention": {
            "source_block": source_block,
            "raw_shape": attention_shape,
            "heads": 12,
            "patch_grid": [28, 28],
        },
        "variants": {
            variant["key"]: {
                "label": variant["label"],
                "attention_path": str(variant["attention_path"].resolve()),
                "attention_sha256": sha256_file(variant["attention_path"]),
                "model_reconstruction": decompositions[variant["key"]],
                "attention_metrics_vs_fp32": metrics_vs_fp32[variant["key"]],
            }
            for variant in variants
        },
        "incremental_S_quantization_vs_Lmask_Sfp32": incremental_metrics,
        "outputs": {
            "horizontal_mean_and_heads": str(horizontal_path.resolve()),
            "rank_and_energy": str(rank_path.resolve()),
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"horizontal_figure={horizontal_path.resolve()}", flush=True)
    print(f"rank_figure={rank_path.resolve()}", flush=True)
    for key in ("lmask_s_fp32", "lmask_s_int3", "lmask_s_int1"):
        decomposition = decompositions[key]
        mean_metrics = metrics_vs_fp32[key]["mean_over_heads"]
        head_similarity = metrics_vs_fp32[key]["within_variant_head_similarity"]
        print(
            f"{key}: rel_L_error={decomposition['relative_L_error']:.8f} "
            f"rel_full_QKV_error={decomposition['relative_full_QKV_error']:.8f} "
            f"rank={decomposition['retained_rank_min']}-"
            f"{decomposition['retained_rank_max']} "
            f"energy={decomposition['retained_energy_min']:.8f}-"
            f"{decomposition['retained_energy_max']:.8f} "
            f"mean_cos={mean_metrics['cosine_similarity']:.8f} "
            f"mean_tv={mean_metrics['total_variation_distance']:.8f} "
            f"top60_iou={mean_metrics['top_attention_mass_mask_iou']:.8f} "
            f"head_pair_cos="
            f"{head_similarity['candidate']['mean_pairwise_cosine_similarity']:.8f}",
            flush=True,
        )
    print(f"metadata={metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
