#!/usr/bin/env python3
"""Quantize masked QKV sparse matrices while keeping FC and projection fixed.

All FC and attention-output projection weights use the exact checkpoint ``L+S``.
For each of the 12 QKV matrices, ``L`` also remains exact FP32 while ``S`` is
epsilon-masked and fake-quantized with sparse-mask INT1 or symmetric INT3.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
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
    decomposed_module_names,
    infer_patch_attention,
    load_non_decomposed_state,
    sha256_file,
    sparse_binary_positive_mean_fake_quantize,
    symmetric_int3_fake_quantize,
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
    / "qkv_sparse_masked_quantization_attention"
    / "52_green_lizard"
)
QKV_PATTERN = re.compile(r"^backbone\.blocks\.(?P<block>\d+)\.attn\.qkv$")
VARIANT_ORDER = ("fp32", "int3", "int1")
VARIANT_LABELS = {
    "fp32": "QKV-S FP32",
    "int3": "QKV-S mask→INT3",
    "int1": "QKV-S mask→INT1",
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
    parser.add_argument("--zero-threshold", type=float, default=1e-5)
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
    if not math.isfinite(args.zero_threshold) or args.zero_threshold < 0.0:
        raise ValueError("zero threshold must be finite and non-negative")
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0, 1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")


def quantization_specification(mode: str, granularity: str) -> dict[str, Any]:
    if mode == "int3":
        return {
            "bits": 3,
            "scheme": "masked signed symmetric narrow-range INT3",
            "granularity": granularity,
            "codes": list(range(-3, 4)),
            "scale_definition": "max(abs(masked S))/3 over each scale group",
            "rounding": "round to nearest, then clamp to [-3,3]",
        }
    if mode == "int1":
        return {
            "bits": 1,
            "scheme": "sparse mask plus binary sign",
            "granularity": "per_qkv_matrix",
            "codes": [-1, 0, 1],
            "scale_definition": (
                "alpha = mean of strictly positive retained S elements"
            ),
            "rounding": "negative retained values→-alpha; positive→+alpha; mask→0",
        }
    raise ValueError(f"unsupported mode: {mode}")


def fake_quantize_masked_sparse(
    masked_sparse: torch.Tensor,
    mode: str,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "int3":
        return symmetric_int3_fake_quantize(masked_sparse, granularity)
    if mode == "int1":
        return sparse_binary_positive_mean_fake_quantize(masked_sparse)
    raise ValueError(f"unsupported mode: {mode}")


def cosine_from_sums(
    dot: float,
    first_norm_squared: float,
    second_norm_squared: float,
) -> float:
    denominator = math.sqrt(max(first_norm_squared * second_norm_squared, 0.0))
    return dot / denominator if denominator > 0.0 else 1.0


@torch.no_grad()
def apply_qkv_variant(
    model: DinoViTBase8,
    matrix_dir: Path,
    decomposed_modules: set[str],
    *,
    mode: str,
    zero_threshold: float,
    int3_granularity: str,
) -> dict[str, Any]:
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")

    quantized = mode != "fp32"
    specification = (
        quantization_specification(mode, int3_granularity) if quantized else None
    )
    code_histogram = (
        {str(code): 0 for code in specification["codes"]}
        if specification is not None
        else {}
    )
    seen: set[str] = set()
    qkv_layers = 0
    sparse_elements = 0
    original_nonzero = 0
    masked_nonzero = 0
    threshold_removed = 0
    quantized_nonzero = 0
    quantization_zeroed_nonzero = 0
    quantization_activated_zero = 0
    saturation_count = 0
    exact_sparse_norm_squared = 0.0
    masked_sparse_norm_squared = 0.0
    used_sparse_norm_squared = 0.0
    exact_used_dot = 0.0
    sparse_error_vs_exact_squared = 0.0
    sparse_quantization_error_vs_masked_squared = 0.0
    full_weight_norm_squared = 0.0
    full_weight_error_squared = 0.0
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
                raise ValueError(f"non-finite L values for {layer_name}")
            if not torch.isfinite(sparse_weight).all():
                raise ValueError(f"non-finite S values for {layer_name}")

            qkv_match = QKV_PATTERN.fullmatch(layer_name)
            if qkv_match is None or not quantized:
                replacement = low_rank_weight + sparse_weight
                layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))
                continue

            qkv_layers += 1
            original_nonzero_mask = sparse_weight != 0.0
            threshold_mask = original_nonzero_mask & (
                sparse_weight.abs() <= zero_threshold
            )
            masked_sparse = sparse_weight.masked_fill(threshold_mask, 0.0)
            masked_nonzero_mask = masked_sparse != 0.0
            used_sparse, codes, scales = fake_quantize_masked_sparse(
                masked_sparse,
                mode,
                int3_granularity,
            )
            quantized_nonzero_mask = codes != 0
            reference = low_rank_weight + sparse_weight
            replacement = low_rank_weight + used_sparse
            sparse_error_vs_exact = used_sparse - sparse_weight
            sparse_quantization_error_vs_masked = used_sparse - masked_sparse
            full_weight_error = replacement - reference

            layer_original_nonzero = int(
                torch.count_nonzero(original_nonzero_mask).item()
            )
            layer_masked_nonzero = int(
                torch.count_nonzero(masked_nonzero_mask).item()
            )
            layer_threshold_removed = int(torch.count_nonzero(threshold_mask).item())
            layer_quantized_nonzero = int(
                torch.count_nonzero(quantized_nonzero_mask).item()
            )
            layer_zeroed = int(
                torch.count_nonzero(
                    masked_nonzero_mask & ~quantized_nonzero_mask
                ).item()
            )
            layer_activated = int(
                torch.count_nonzero(
                    ~masked_nonzero_mask & quantized_nonzero_mask
                ).item()
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

            layer_exact_norm_squared = float(sparse_weight.square().sum().item())
            layer_masked_norm_squared = float(masked_sparse.square().sum().item())
            layer_used_norm_squared = float(used_sparse.square().sum().item())
            layer_dot = float((sparse_weight * used_sparse).sum().item())
            layer_error_vs_exact_squared = float(
                sparse_error_vs_exact.square().sum().item()
            )
            layer_quantization_error_vs_masked_squared = float(
                sparse_quantization_error_vs_masked.square().sum().item()
            )
            layer_full_norm_squared = float(reference.square().sum().item())
            layer_full_error_squared = float(full_weight_error.square().sum().item())

            sparse_elements += sparse_weight.numel()
            original_nonzero += layer_original_nonzero
            masked_nonzero += layer_masked_nonzero
            threshold_removed += layer_threshold_removed
            quantized_nonzero += layer_quantized_nonzero
            quantization_zeroed_nonzero += layer_zeroed
            quantization_activated_zero += layer_activated
            saturation_count += layer_saturation
            exact_sparse_norm_squared += layer_exact_norm_squared
            masked_sparse_norm_squared += layer_masked_norm_squared
            used_sparse_norm_squared += layer_used_norm_squared
            exact_used_dot += layer_dot
            sparse_error_vs_exact_squared += layer_error_vs_exact_squared
            sparse_quantization_error_vs_masked_squared += (
                layer_quantization_error_vs_masked_squared
            )
            full_weight_norm_squared += layer_full_norm_squared
            full_weight_error_squared += layer_full_error_squared

            layer_statistics.append(
                {
                    "layer": layer_name,
                    "block": int(qkv_match.group("block")),
                    "shape": list(sparse_weight.shape),
                    "elements": sparse_weight.numel(),
                    "original_S_nonzero": layer_original_nonzero,
                    "masked_S_nonzero": layer_masked_nonzero,
                    "threshold_removed": layer_threshold_removed,
                    "quantized_S_nonzero": layer_quantized_nonzero,
                    "quantization_zeroed_masked_nonzero": layer_zeroed,
                    "quantization_activated_zero": layer_activated,
                    "saturation_count": layer_saturation,
                    "scale_min": scale_min,
                    "scale_median": scale_median,
                    "scale_max": scale_max,
                    "alpha": scale_max if mode == "int1" else None,
                    "relative_S_error_vs_exact": math.sqrt(
                        layer_error_vs_exact_squared
                        / max(layer_exact_norm_squared, 1e-30)
                    ),
                    "relative_S_quantization_error_vs_masked": math.sqrt(
                        layer_quantization_error_vs_masked_squared
                        / max(layer_masked_norm_squared, 1e-30)
                    ),
                    "relative_full_QKV_weight_error": math.sqrt(
                        layer_full_error_squared
                        / max(layer_full_norm_squared, 1e-30)
                    ),
                }
            )
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))

        del low_rank
        del sparse
        gc.collect()

    missing = decomposed_modules - seen
    if missing:
        raise ValueError(f"decomposition coverage mismatch; missing={sorted(missing)}")
    if quantized and qkv_layers != 12:
        raise ValueError(f"expected 12 quantized QKV layers, got {qkv_layers}")

    result: dict[str, Any] = {
        "mode": mode,
        "decomposed_layers": len(seen),
        "weight_definition": (
            "All decomposed layers use exact FP32 L+S"
            if mode == "fp32"
            else f"QKV=L(fp32)+masked-{mode.upper()}(S); FC/proj=L+S exact FP32"
        ),
        "controlled_components": {
            "FC": "exact checkpoint L+S FP32",
            "attention_projection": "exact checkpoint L+S FP32",
            "QKV_L": "exact checkpoint FP32",
            "QKV_S": (
                "exact checkpoint FP32"
                if mode == "fp32"
                else f"abs(S)<={zero_threshold:g} masked, then {mode.upper()}"
            ),
        },
    }
    if not quantized:
        result["quantization"] = None
        return result

    result["quantization"] = {
        **{key: value for key, value in specification.items() if key != "codes"},
        "zero_threshold": zero_threshold,
        "zero_rule": "abs(S) <= epsilon becomes zero before quantization",
        "code_histogram": code_histogram,
        "QKV_layers": qkv_layers,
        "S_elements": sparse_elements,
        "original_S_nonzero": original_nonzero,
        "masked_S_nonzero": masked_nonzero,
        "threshold_removed": threshold_removed,
        "quantized_S_nonzero": quantized_nonzero,
        "quantization_zeroed_masked_nonzero": quantization_zeroed_nonzero,
        "quantization_zeroed_fraction_of_masked_nonzero": (
            quantization_zeroed_nonzero / max(masked_nonzero, 1)
        ),
        "quantization_activated_zero": quantization_activated_zero,
        "saturation_count": saturation_count,
        "relative_S_error_vs_exact": math.sqrt(
            sparse_error_vs_exact_squared / max(exact_sparse_norm_squared, 1e-30)
        ),
        "relative_S_quantization_error_vs_masked": math.sqrt(
            sparse_quantization_error_vs_masked_squared
            / max(masked_sparse_norm_squared, 1e-30)
        ),
        "S_cosine_similarity_vs_exact": cosine_from_sums(
            exact_used_dot,
            exact_sparse_norm_squared,
            used_sparse_norm_squared,
        ),
        "relative_full_QKV_weight_error": math.sqrt(
            full_weight_error_squared / max(full_weight_norm_squared, 1e-30)
        ),
        "layers": sorted(layer_statistics, key=lambda row: row["block"]),
    }
    return result


def render_horizontal_attention_matrix(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
) -> Path:
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
                    variant["label"],
                    fontsize=12,
                    rotation=90,
                    labelpad=10,
                )
    figure.suptitle(
        "Green lizard — QKV sparse-component quantization; FC fixed as FP32 L+S\n"
        "Rows: QKV-S variants; columns: Mean and Head 0–11; each column shares one scale",
        fontsize=17,
    )
    output_path = output_dir / "mean_and_heads_horizontal_qkv_S_masked_int1_int3.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_layer_error_plot(
    decompositions: dict[str, dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    modes = ("int3", "int1")
    labels = [f"Block {index}" for index in range(12)]
    positions = np.arange(12, dtype=np.float64)
    width = 0.34
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8), constrained_layout=True)
    for mode_index, mode in enumerate(modes):
        rows = decompositions[mode]["quantization"]["layers"]
        offset = (mode_index - 0.5) * width
        axes[0].bar(
            positions + offset,
            [row["relative_S_error_vs_exact"] for row in rows],
            width=width,
            label=f"masked {mode.upper()}",
        )
        axes[1].bar(
            positions + offset,
            [row["relative_full_QKV_weight_error"] for row in rows],
            width=width,
            label=f"masked {mode.upper()}",
        )
    axes[0].set_title("QKV-S error: ||Sq-S||F / ||S||F")
    axes[1].set_title("Full QKV error: ||(L+Sq)-(L+S)||F / ||L+S||F")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel("Relative error")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Masked QKV-S quantization error by block", fontsize=16)
    output_path = output_dir / "qkv_S_masked_quantization_error_by_block.png"
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
        qkv_s = (
            "exact FP32"
            if mode == "fp32"
            else f"mask(abs(S)<={args.zero_threshold:g}) then {mode.upper()}"
        )
        print(
            f"building {mode}: FC/proj=L+S exact; QKV-L exact; QKV-S={qkv_s}",
            flush=True,
        )
        decomposition = apply_qkv_variant(
            model,
            args.matrix_dir,
            decomposed_modules,
            mode=mode,
            zero_threshold=args.zero_threshold,
            int3_granularity=args.int3_granularity,
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
            "mask and quantize only QKV sparse S while FC and attention projection "
            "remain exact FP32 L+S"
        ),
        "controlled_difference": (
            "Only backbone.blocks.*.attn.qkv sparse S changes. QKV L, all FC L+S, "
            "attention-projection L+S, biases, norms, and other state are exact."
        ),
        "mask": {
            "epsilon": args.zero_threshold,
            "rule": "abs(S) <= epsilon becomes zero before INT1/INT3",
        },
        "int1_definition": (
            "The sparse mask remains zero. Retained signs use {-alpha,+alpha}; "
            "each QKV matrix has alpha=mean(retained positive S)."
        ),
        "int3_definition": (
            "Per-output-channel signed symmetric narrow-range fake quantization "
            "with codes {-3,-2,-1,0,1,2,3}."
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
                "Block 11 Q/K directly form the displayed block-11 attention logits. "
                "Therefore quantizing block-11 QKV-S can directly alter this map; "
                "earlier QKV layers also change the incoming tokens."
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
    for mode in ("int3", "int1"):
        quantization = decompositions[mode]["quantization"]
        mean_metrics = metrics[mode]["mean_over_heads"]
        head_similarity = metrics[mode]["within_variant_head_similarity"]
        print(
            f"{mode}: rel_S_error={quantization['relative_S_error_vs_exact']:.8f} "
            f"rel_full_QKV_error={quantization['relative_full_QKV_weight_error']:.8f} "
            f"threshold_removed={quantization['threshold_removed']} "
            f"zeroed_retained_fraction="
            f"{quantization['quantization_zeroed_fraction_of_masked_nonzero']:.8f} "
            f"mean_cos={mean_metrics['cosine_similarity']:.8f} "
            f"mean_tv={mean_metrics['total_variation_distance']:.8f} "
            f"mean_jsd={mean_metrics['jensen_shannon_divergence_nats']:.8g} "
            f"top60_iou={mean_metrics['top_attention_mass_mask_iou']:.8f} "
            f"head_pair_cos={head_similarity['candidate']['mean_pairwise_cosine_similarity']:.8f}",
            flush=True,
        )
    print(f"metadata={metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
