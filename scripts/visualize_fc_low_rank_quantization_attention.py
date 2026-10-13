#!/usr/bin/env python3
"""Study FC low-rank quantization and a joint masked-sparse INT3 variant.

The mixed-rho SALAAD checkpoint is reconstructed as ``L + S`` for all 48
decomposed ViT linear layers.  QKV and attention-output projections stay FP32.
For the 24 MLP FC matrices only, this script compares the exact FP32 baseline
against INT1, INT3, and INT4 fake quantization of ``L``.  It also tests joint
INT3 where ``S`` is epsilon-masked first and then fake-quantized.  All other
variants copy ``S`` from the checkpoint without thresholding or quantization.
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
    FC_LAYER_PATTERN,
    attention_metrics,
    binary_positive_mean_fake_quantize,
    decomposed_module_names,
    infer_patch_attention,
    load_non_decomposed_state,
    sha256_file,
    symmetric_int3_fake_quantize,
    symmetric_int4_fake_quantize,
    upsample_patch_map,
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
    / "fc_low_rank_quantization_attention"
    / "52_green_lizard"
)
JOINT_INT3_MODE = "int3_l_masked_s_int3"
VARIANT_ORDER = ("fp32", "int4", "int3", JOINT_INT3_MODE, "int1")
VARIANT_LABELS = {
    "fp32": "FP32 FC-L",
    "int4": "INT4 FC-L",
    "int3": "INT3 FC-L",
    JOINT_INT3_MODE: "INT3 FC-L",
    "int1": "INT1 FC-L",
}
VARIANT_S_LABELS = {
    "fp32": "S FP32",
    "int4": "S FP32",
    "int3": "S FP32",
    JOINT_INT3_MODE: "S mask→INT3",
    "int1": "S FP32",
}


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
        "--quantization-granularity",
        choices=("per_output_channel", "per_tensor"),
        default="per_output_channel",
        help="Scale granularity for INT3/INT4. INT1 uses one alpha per FC matrix.",
    )
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument(
        "--sparse-zero-threshold",
        type=float,
        default=1e-5,
        help="For the joint INT3 row, abs(S) <= epsilon is masked to zero first.",
    )
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
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0, 1]")
    if (
        not math.isfinite(args.sparse_zero_threshold)
        or args.sparse_zero_threshold < 0.0
    ):
        raise ValueError("sparse zero threshold must be finite and non-negative")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0, 1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")


def quantization_description(mode: str, granularity: str) -> dict[str, Any]:
    if mode == "int1":
        return {
            "bits": 1,
            "scheme": "binary positive-elements-mean scale",
            "granularity": "per_fc_matrix",
            "codes": [-1, 1],
            "scale_definition": "alpha = mean(L_ij for L_ij > 0)",
            "weight_expression": "FC=dequant(INT1(L))+S(fp32 exact)",
        }
    if mode == "int3":
        qmax = 3
        bits = 3
    elif mode == "int4":
        qmax = 7
        bits = 4
    else:
        raise ValueError(f"unsupported quantization mode: {mode}")
    return {
        "bits": bits,
        "scheme": "signed symmetric narrow-range",
        "granularity": granularity,
        "codes": list(range(-qmax, qmax + 1)),
        "scale_definition": f"max(abs(L))/{qmax} over each scale group",
        "weight_expression": f"FC=dequant({mode.upper()}(L))+S(fp32 exact)",
    }


def fake_quantize_low_rank(
    low_rank: torch.Tensor,
    mode: str,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "int1":
        return binary_positive_mean_fake_quantize(low_rank)
    if mode == "int3":
        return symmetric_int3_fake_quantize(low_rank, granularity)
    if mode == "int4":
        return symmetric_int4_fake_quantize(low_rank, granularity)
    raise ValueError(f"unsupported quantization mode: {mode}")


def cosine_similarity_from_sums(
    dot: float,
    first_norm_squared: float,
    second_norm_squared: float,
) -> float:
    denominator = math.sqrt(max(first_norm_squared * second_norm_squared, 0.0))
    return dot / denominator if denominator > 0.0 else 1.0


@torch.no_grad()
def apply_variant(
    model: DinoViTBase8,
    matrix_dir: Path,
    decomposed_modules: set[str],
    *,
    mode: str,
    granularity: str,
    sparse_zero_threshold: float,
) -> dict[str, Any]:
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")

    quantized = mode != "fp32"
    quantize_masked_sparse = mode == JOINT_INT3_MODE
    low_rank_mode = "int3" if quantize_masked_sparse else mode
    specification = (
        quantization_description(low_rank_mode, granularity) if quantized else None
    )
    code_histogram = (
        {str(code): 0 for code in specification["codes"]}
        if specification is not None
        else {}
    )
    seen: set[str] = set()
    target_elements = 0
    target_nonzero = 0
    quantized_nonzero = 0
    quantization_zeroed_nonzero = 0
    quantization_activated_zero = 0
    saturation_count = 0
    target_norm_squared = 0.0
    used_norm_squared = 0.0
    target_used_dot = 0.0
    target_error_squared = 0.0
    full_weight_norm_squared = 0.0
    full_weight_error_squared = 0.0
    layer_statistics: list[dict[str, Any]] = []
    sparse_code_histogram = {str(code): 0 for code in range(-3, 4)}
    sparse_elements = 0
    sparse_original_nonzero = 0
    sparse_masked_nonzero = 0
    sparse_threshold_removed = 0
    sparse_quantized_nonzero = 0
    sparse_quantization_zeroed_nonzero = 0
    sparse_saturation_count = 0
    sparse_exact_norm_squared = 0.0
    sparse_masked_norm_squared = 0.0
    sparse_quantized_norm_squared = 0.0
    sparse_exact_quantized_dot = 0.0
    sparse_error_vs_exact_squared = 0.0
    sparse_quantization_error_vs_masked_squared = 0.0

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
                raise ValueError(f"non-finite L values for {layer_name}")
            if not torch.isfinite(sparse_weight).all():
                raise ValueError(f"non-finite S values for {layer_name}")

            fc_match = FC_LAYER_PATTERN.fullmatch(layer_name)
            if fc_match is None or not quantized:
                replacement = low_rank_weight + sparse_weight
                layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))
                continue

            used_low_rank, codes, scales = fake_quantize_low_rank(
                low_rank_weight,
                low_rank_mode,
                granularity,
            )
            reference = low_rank_weight + sparse_weight
            sparse_layer_statistics = None
            if quantize_masked_sparse:
                sparse_original_nonzero_mask = sparse_weight != 0.0
                sparse_threshold_mask = sparse_original_nonzero_mask & (
                    sparse_weight.abs() <= sparse_zero_threshold
                )
                masked_sparse = sparse_weight.masked_fill(
                    sparse_threshold_mask,
                    0.0,
                )
                used_sparse, sparse_codes, sparse_scales = (
                    symmetric_int3_fake_quantize(masked_sparse, granularity)
                )
                sparse_masked_nonzero_mask = masked_sparse != 0.0
                sparse_quantized_nonzero_mask = sparse_codes != 0
                layer_sparse_original_nonzero = int(
                    torch.count_nonzero(sparse_original_nonzero_mask).item()
                )
                layer_sparse_masked_nonzero = int(
                    torch.count_nonzero(sparse_masked_nonzero_mask).item()
                )
                layer_sparse_threshold_removed = int(
                    torch.count_nonzero(sparse_threshold_mask).item()
                )
                layer_sparse_quantized_nonzero = int(
                    torch.count_nonzero(sparse_quantized_nonzero_mask).item()
                )
                layer_sparse_quantization_zeroed = int(
                    torch.count_nonzero(
                        sparse_masked_nonzero_mask & ~sparse_quantized_nonzero_mask
                    ).item()
                )
                layer_sparse_saturation = int(
                    torch.count_nonzero(sparse_codes.abs() == 3).item()
                )
                for code in range(-3, 4):
                    sparse_code_histogram[str(code)] += int(
                        torch.count_nonzero(sparse_codes == code).item()
                    )
                sparse_positive_scales = sparse_scales[sparse_scales > 0.0]
                sparse_scale_min = (
                    float(sparse_positive_scales.min().item())
                    if sparse_positive_scales.numel() > 0
                    else 0.0
                )
                sparse_scale_median = (
                    float(sparse_positive_scales.median().item())
                    if sparse_positive_scales.numel() > 0
                    else 0.0
                )
                sparse_scale_max = (
                    float(sparse_positive_scales.max().item())
                    if sparse_positive_scales.numel() > 0
                    else 0.0
                )
                sparse_error_vs_exact = used_sparse - sparse_weight
                sparse_quantization_error_vs_masked = used_sparse - masked_sparse
                layer_sparse_exact_norm_squared = float(
                    sparse_weight.square().sum().item()
                )
                layer_sparse_masked_norm_squared = float(
                    masked_sparse.square().sum().item()
                )
                layer_sparse_quantized_norm_squared = float(
                    used_sparse.square().sum().item()
                )
                layer_sparse_exact_quantized_dot = float(
                    (sparse_weight * used_sparse).sum().item()
                )
                layer_sparse_error_vs_exact_squared = float(
                    sparse_error_vs_exact.square().sum().item()
                )
                layer_sparse_quantization_error_vs_masked_squared = float(
                    sparse_quantization_error_vs_masked.square().sum().item()
                )
                sparse_elements += sparse_weight.numel()
                sparse_original_nonzero += layer_sparse_original_nonzero
                sparse_masked_nonzero += layer_sparse_masked_nonzero
                sparse_threshold_removed += layer_sparse_threshold_removed
                sparse_quantized_nonzero += layer_sparse_quantized_nonzero
                sparse_quantization_zeroed_nonzero += (
                    layer_sparse_quantization_zeroed
                )
                sparse_saturation_count += layer_sparse_saturation
                sparse_exact_norm_squared += layer_sparse_exact_norm_squared
                sparse_masked_norm_squared += layer_sparse_masked_norm_squared
                sparse_quantized_norm_squared += layer_sparse_quantized_norm_squared
                sparse_exact_quantized_dot += layer_sparse_exact_quantized_dot
                sparse_error_vs_exact_squared += (
                    layer_sparse_error_vs_exact_squared
                )
                sparse_quantization_error_vs_masked_squared += (
                    layer_sparse_quantization_error_vs_masked_squared
                )
                sparse_layer_statistics = {
                    "original_nonzero": layer_sparse_original_nonzero,
                    "masked_nonzero": layer_sparse_masked_nonzero,
                    "threshold_removed": layer_sparse_threshold_removed,
                    "quantized_nonzero": layer_sparse_quantized_nonzero,
                    "quantization_zeroed_nonzero": (
                        layer_sparse_quantization_zeroed
                    ),
                    "saturation_count": layer_sparse_saturation,
                    "scale_min": sparse_scale_min,
                    "scale_median": sparse_scale_median,
                    "scale_max": sparse_scale_max,
                    "relative_error_vs_exact_S": math.sqrt(
                        layer_sparse_error_vs_exact_squared
                        / max(layer_sparse_exact_norm_squared, 1e-30)
                    ),
                    "relative_quantization_error_vs_masked_S": math.sqrt(
                        layer_sparse_quantization_error_vs_masked_squared
                        / max(layer_sparse_masked_norm_squared, 1e-30)
                    ),
                }
            else:
                used_sparse = sparse_weight
            replacement = used_low_rank + used_sparse
            target_error = used_low_rank - low_rank_weight
            full_weight_error = replacement - reference
            original_nonzero_mask = low_rank_weight != 0.0
            quantized_nonzero_mask = codes != 0

            layer_target_nonzero = int(torch.count_nonzero(original_nonzero_mask).item())
            layer_quantized_nonzero = int(
                torch.count_nonzero(quantized_nonzero_mask).item()
            )
            layer_zeroed = int(
                torch.count_nonzero(original_nonzero_mask & ~quantized_nonzero_mask).item()
            )
            layer_activated = int(
                torch.count_nonzero(~original_nonzero_mask & quantized_nonzero_mask).item()
            )
            maximum_code = max(abs(int(code)) for code in specification["codes"])
            layer_saturation = int(
                torch.count_nonzero(codes.abs() == maximum_code).item()
            )
            for code in specification["codes"]:
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

            layer_target_norm_squared = float(low_rank_weight.square().sum().item())
            layer_used_norm_squared = float(used_low_rank.square().sum().item())
            layer_dot = float((low_rank_weight * used_low_rank).sum().item())
            layer_target_error_squared = float(target_error.square().sum().item())
            layer_full_norm_squared = float(reference.square().sum().item())
            layer_full_error_squared = float(full_weight_error.square().sum().item())

            target_elements += low_rank_weight.numel()
            target_nonzero += layer_target_nonzero
            quantized_nonzero += layer_quantized_nonzero
            quantization_zeroed_nonzero += layer_zeroed
            quantization_activated_zero += layer_activated
            saturation_count += layer_saturation
            target_norm_squared += layer_target_norm_squared
            used_norm_squared += layer_used_norm_squared
            target_used_dot += layer_dot
            target_error_squared += layer_target_error_squared
            full_weight_norm_squared += layer_full_norm_squared
            full_weight_error_squared += layer_full_error_squared

            layer_statistics.append(
                {
                    "layer": layer_name,
                    "block": int(fc_match.group("block")),
                    "fc": fc_match.group("fc"),
                    "shape": list(low_rank_weight.shape),
                    "elements": low_rank_weight.numel(),
                    "original_L_nonzero": layer_target_nonzero,
                    "quantized_L_nonzero": layer_quantized_nonzero,
                    "quantization_zeroed_nonzero": layer_zeroed,
                    "quantization_activated_zero": layer_activated,
                    "saturation_count": layer_saturation,
                    "scale_min": scale_min,
                    "scale_median": scale_median,
                    "scale_max": scale_max,
                    "alpha": scale_max if low_rank_mode == "int1" else None,
                    "relative_L_frobenius_error": math.sqrt(
                        layer_target_error_squared
                        / max(layer_target_norm_squared, 1e-30)
                    ),
                    "L_cosine_similarity": cosine_similarity_from_sums(
                        layer_dot,
                        layer_target_norm_squared,
                        layer_used_norm_squared,
                    ),
                    "relative_full_weight_frobenius_error": math.sqrt(
                        layer_full_error_squared
                        / max(layer_full_norm_squared, 1e-30)
                    ),
                    "maximum_absolute_L_error": float(target_error.abs().max().item()),
                    "masked_S_INT3": sparse_layer_statistics,
                }
            )
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))

        del low_rank
        del sparse
        gc.collect()

    missing = decomposed_modules - seen
    if missing:
        raise ValueError(f"decomposition coverage mismatch; missing={sorted(missing)}")

    result: dict[str, Any] = {
        "mode": mode,
        "decomposed_layers": len(seen),
        "fc_layers": 24,
        "fixed_component": (
            "QKV/proj S remains exact. FC S is epsilon-masked and INT3-quantized."
            if quantize_masked_sparse
            else "Every S tensor is copied exactly from the checkpoint. No "
            "thresholding or quantization is applied to S."
        ),
        "qkv_and_attention_projection": "L+S, both exact FP32 checkpoint tensors",
        "fc_weight_definition": (
            "L(fp32 exact)+S(fp32 exact)"
            if mode == "fp32"
            else (
                "dequant(INT3(L))+dequant(INT3(mask(abs(S)>epsilon)*S))"
                if quantize_masked_sparse
                else specification["weight_expression"]
            )
        ),
    }
    if quantized:
        result["quantization"] = {
            **{key: value for key, value in specification.items() if key != "codes"},
            "code_histogram": code_histogram,
            "L_elements": target_elements,
            "original_L_nonzero": target_nonzero,
            "quantized_L_nonzero": quantized_nonzero,
            "quantization_zeroed_nonzero": quantization_zeroed_nonzero,
            "quantization_zeroed_fraction_of_original_nonzero": (
                quantization_zeroed_nonzero / max(target_nonzero, 1)
            ),
            "quantization_activated_zero": quantization_activated_zero,
            "saturation_count": saturation_count,
            "relative_L_frobenius_error": math.sqrt(
                target_error_squared / max(target_norm_squared, 1e-30)
            ),
            "L_cosine_similarity": cosine_similarity_from_sums(
                target_used_dot,
                target_norm_squared,
                used_norm_squared,
            ),
            "relative_full_weight_frobenius_error": math.sqrt(
                full_weight_error_squared / max(full_weight_norm_squared, 1e-30)
            ),
            "masked_S_INT3": (
                {
                    "bits": 3,
                    "scheme": "epsilon mask followed by signed symmetric INT3",
                    "granularity": granularity,
                    "zero_threshold": sparse_zero_threshold,
                    "zero_rule": "abs(S) <= epsilon becomes zero before INT3",
                    "codes": list(range(-3, 4)),
                    "scale_definition": "max(abs(masked S))/3 per scale group",
                    "code_histogram": sparse_code_histogram,
                    "S_elements": sparse_elements,
                    "original_S_nonzero": sparse_original_nonzero,
                    "masked_S_nonzero": sparse_masked_nonzero,
                    "threshold_removed": sparse_threshold_removed,
                    "quantized_S_nonzero": sparse_quantized_nonzero,
                    "quantization_zeroed_masked_nonzero": (
                        sparse_quantization_zeroed_nonzero
                    ),
                    "quantization_zeroed_fraction_of_masked_nonzero": (
                        sparse_quantization_zeroed_nonzero
                        / max(sparse_masked_nonzero, 1)
                    ),
                    "saturation_count": sparse_saturation_count,
                    "relative_error_vs_exact_S": math.sqrt(
                        sparse_error_vs_exact_squared
                        / max(sparse_exact_norm_squared, 1e-30)
                    ),
                    "relative_quantization_error_vs_masked_S": math.sqrt(
                        sparse_quantization_error_vs_masked_squared
                        / max(sparse_masked_norm_squared, 1e-30)
                    ),
                    "S_cosine_similarity_vs_exact": cosine_similarity_from_sums(
                        sparse_exact_quantized_dot,
                        sparse_exact_norm_squared,
                        sparse_quantized_norm_squared,
                    ),
                }
                if quantize_masked_sparse
                else None
            ),
            "layers": sorted(
                layer_statistics,
                key=lambda row: (row["block"], row["fc"]),
            ),
        }
    else:
        result["quantization"] = None
    return result


def render_horizontal_attention_matrix(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
) -> Path:
    """Rows are precision variants; columns are Mean and Head 0 through 11."""
    column_titles = ["Mean"] + [f"Head {index}" for index in range(12)]
    high_resolution: list[list[np.ndarray]] = []
    for variant in variants:
        attention = variant["attention"]
        maps = [upsample_patch_map(attention.mean(dim=0))]
        maps.extend(
            upsample_patch_map(attention[index], mode="nearest")
            for index in range(12)
        )
        high_resolution.append(maps)

    shared_column_maxima = [
        max(float(row[column_index].max()) for row in high_resolution)
        for column_index in range(len(column_titles))
    ]
    figure, axes = plt.subplots(
        len(variants),
        len(column_titles),
        figsize=(29.5, 2.25 * len(variants) + 0.3),
        constrained_layout=True,
        squeeze=False,
    )
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title, fontsize=12, pad=7)
    for row_index, variant in enumerate(variants):
        for column_index in range(len(column_titles)):
            axis = axes[row_index, column_index]
            axis.imshow(crop_array)
            axis.imshow(
                high_resolution[row_index][column_index],
                cmap="turbo",
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=max(shared_column_maxima[column_index], 1e-12),
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column_index == 0:
                axis.set_ylabel(
                    variant["label"] + "\n" + VARIANT_S_LABELS[variant["key"]],
                    fontsize=12,
                    rotation=90,
                    labelpad=9,
                )
    figure.suptitle(
        "Green lizard — FC-L quantization and masked FC-S INT3\n"
        "Rows: L precision; columns: Mean and Head 0–11; each column shares one scale",
        fontsize=17,
    )
    output_path = output_dir / "mean_and_heads_horizontal_L_and_masked_S_int3.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_layer_error_plot(
    decompositions: dict[str, dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    modes = ("int4", "int3", JOINT_INT3_MODE, "int1")
    plot_labels = {
        "int4": "L INT4, S FP32",
        "int3": "L INT3, S FP32",
        JOINT_INT3_MODE: "L INT3, masked S INT3",
        "int1": "L INT1, S FP32",
    }
    layers = decompositions["int4"]["quantization"]["layers"]
    labels = [f"B{row['block']} {row['fc'].upper()}" for row in layers]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.20
    figure, axes = plt.subplots(2, 1, figsize=(15, 8.5), constrained_layout=True)
    for mode_index, mode in enumerate(modes):
        quantization = decompositions[mode]["quantization"]
        rows = quantization["layers"]
        offset = (mode_index - 1.5) * width
        axes[0].bar(
            positions + offset,
            [row["relative_L_frobenius_error"] for row in rows],
            width=width,
            label=plot_labels[mode],
        )
        axes[1].bar(
            positions + offset,
            [row["relative_full_weight_frobenius_error"] for row in rows],
            width=width,
            label=plot_labels[mode],
        )
    axes[0].set_title("FC-L quantization error: ||Lq-L||F / ||L||F")
    axes[1].set_title("Full FC error: ||(Lq+Sq)-(L+S)||F / ||L+S||F")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=55, ha="right")
        axis.set_ylabel("Relative error")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle(
        "FC quantization by layer — joint row masks and quantizes S",
        fontsize=16,
    )
    output_path = output_dir / "fc_L_and_masked_S_int3_error_by_layer.png"
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
    decompositions: dict[str, dict[str, Any]] = {}
    attention_source_block: int | None = None
    attention_shape: list[int] | None = None
    print(f"device={device}", flush=True)
    for mode in VARIANT_ORDER:
        sparse_description = (
            f"mask(abs(S)<={args.sparse_zero_threshold:g}) then INT3"
            if mode == JOINT_INT3_MODE
            else "exact FP32"
        )
        low_rank_description = "int3" if mode == JOINT_INT3_MODE else mode
        print(
            f"building {mode}: QKV/proj=L+S exact; "
            f"FC-S={sparse_description}; FC-L={low_rank_description}",
            flush=True,
        )
        decomposition = apply_variant(
            model,
            args.matrix_dir,
            decomposed_modules,
            mode=mode,
            granularity=args.quantization_granularity,
            sparse_zero_threshold=args.sparse_zero_threshold,
        )
        attention, source_block, current_attention_shape = infer_patch_attention(
            model,
            image_tensor,
            device,
        )
        if attention_source_block is None:
            attention_source_block = source_block
            attention_shape = current_attention_shape
        elif source_block != attention_source_block:
            raise RuntimeError("variants used different attention source blocks")

        variant_dir = args.output_dir / mode
        variant_dir.mkdir(parents=True, exist_ok=True)
        attention_path = variant_dir / "cls_to_patch_attention.npy"
        np.save(attention_path, attention.numpy())
        variants.append(
            {
                "key": mode,
                "label": VARIANT_LABELS[mode],
                "attention": attention,
                "attention_path": attention_path,
            }
        )
        decompositions[mode] = decomposition

    baseline_attention = variants[0]["attention"]
    metrics = {
        variant["key"]: (
            None
            if variant["key"] == "fp32"
            else attention_metrics(
                baseline_attention,
                variant["attention"],
                args.attention_mass,
            )
        )
        for variant in variants
    }
    attention_by_key = {
        variant["key"]: variant["attention"] for variant in variants
    }
    joint_incremental_metrics = attention_metrics(
        attention_by_key["int3"],
        attention_by_key[JOINT_INT3_MODE],
        args.attention_mass,
    )
    horizontal_path = render_horizontal_attention_matrix(
        variants,
        crop_array,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
    )
    layer_error_path = render_layer_error_plot(
        decompositions,
        args.output_dir,
        args.dpi,
    )

    metadata = {
        "experiment": (
            "quantize FC low-rank L, including a joint INT3 row where FC-S is "
            "epsilon-masked and then INT3-quantized"
        ),
        "controlled_difference": (
            "Only mlp.fc1/fc2 decomposition components differ. L is quantized in "
            "all low-precision rows; FC-S additionally changes only in the joint "
            "INT3 row. QKV/proj L+S, biases, norms, and all other state remain exact."
        ),
        "important_S_detail": (
            "S is copied exactly in the baseline and L-only rows. In the joint row, "
            f"abs(S) <= {args.sparse_zero_threshold:g} is zeroed before per-output-"
            "channel INT3."
        ),
        "int1_definition": (
            "Each dense FC-L matrix uses exactly {-alpha,+alpha}, where alpha is "
            "the mean of its strictly positive L elements."
        ),
        "int3_int4_definition": (
            "Signed symmetric narrow-range fake quantization, dequantized to FP32; "
            "INT3 codes are -3..3 and INT4 codes are -7..7."
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
            "source_block": attention_source_block,
            "raw_shape": attention_shape,
            "heads": 12,
            "patch_grid": [28, 28],
            "mechanism_note": (
                "Block 11 FC layers occur after block 11 self-attention, so final-block "
                "attention changes are driven by FC component quantization in "
                "blocks 0-10."
            ),
        },
        "variants": {
            variant["key"]: {
                "label": variant["label"],
                "attention_path": str(variant["attention_path"].resolve()),
                "attention_sha256": sha256_file(variant["attention_path"]),
                "model_reconstruction": decompositions[variant["key"]],
                "attention_metrics_vs_fp32": metrics[variant["key"]],
            }
            for variant in variants
        },
        "joint_INT3_increment_vs_L_only_INT3": joint_incremental_metrics,
        "outputs": {
            "horizontal_mean_and_heads": str(horizontal_path.resolve()),
            "layer_quantization_error": str(layer_error_path.resolve()),
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"horizontal_figure={horizontal_path.resolve()}", flush=True)
    print(f"layer_error_figure={layer_error_path.resolve()}", flush=True)
    for mode in ("int4", "int3", JOINT_INT3_MODE, "int1"):
        quantization = decompositions[mode]["quantization"]
        mean_metrics = metrics[mode]["mean_over_heads"]
        head_similarity = metrics[mode]["within_variant_head_similarity"]
        print(
            f"{mode}: rel_L_error={quantization['relative_L_frobenius_error']:.8f} "
            f"rel_full_error={quantization['relative_full_weight_frobenius_error']:.8f} "
            f"mean_cos={mean_metrics['cosine_similarity']:.8f} "
            f"mean_tv={mean_metrics['total_variation_distance']:.8f} "
            f"mean_jsd={mean_metrics['jensen_shannon_divergence_nats']:.8g} "
            f"top60_iou={mean_metrics['top_attention_mass_mask_iou']:.8f} "
            f"head_pair_cos={head_similarity['candidate']['mean_pairwise_cosine_similarity']:.8f}",
            flush=True,
        )
        sparse_quantization = quantization["masked_S_INT3"]
        if sparse_quantization is not None:
            print(
                "joint_S: "
                f"threshold_removed={sparse_quantization['threshold_removed']} "
                f"int3_zeroed_retained="
                f"{sparse_quantization['quantization_zeroed_masked_nonzero']} "
                f"int3_zeroed_fraction="
                f"{sparse_quantization['quantization_zeroed_fraction_of_masked_nonzero']:.8f} "
                f"rel_S_error_vs_exact="
                f"{sparse_quantization['relative_error_vs_exact_S']:.8f}",
                flush=True,
            )
    incremental_mean = joint_incremental_metrics["mean_over_heads"]
    print(
        "joint_vs_L_only_int3: "
        f"mean_cos={incremental_mean['cosine_similarity']:.8f} "
        f"mean_tv={incremental_mean['total_variation_distance']:.8f} "
        f"mean_jsd={incremental_mean['jensen_shannon_divergence_nats']:.8g} "
        f"top60_iou={incremental_mean['top_attention_mass_mask_iou']:.8f}",
        flush=True,
    )
    print(f"metadata={metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
