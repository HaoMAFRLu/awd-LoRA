#!/usr/bin/env python3
"""Spectrally mask QKV-L, then compare FP32, INT3, and INT1 L with S fixed."""

from __future__ import annotations

import argparse
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
from scripts.visualize_dino_attention import choose_device, prepare_image  # noqa: E402
from scripts.visualize_fc_sparse_int4_attention import (  # noqa: E402
    attention_metrics,
    decomposed_module_names,
    infer_patch_attention,
    load_non_decomposed_state,
    sha256_file,
)
from scripts.visualize_qkv_low_rank_spectral_mask_attention import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_RUN_DIR,
    apply_spectral_mask_variant,
    render_horizontal_matrix,
    render_rank_plot,
)
from scripts.visualize_qkv_sparse_masked_quantization_attention import (  # noqa: E402
    apply_qkv_variant,
)


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "figures"
    / "salaad_vision"
    / "qkv_low_rank_spectral_mask_quantization_attention"
    / "52_green_lizard"
)
VARIANTS = (
    ("fp32", "QKV-L FP32\nS FP32", None),
    ("lmask_fp32", "QKV-L spectral mask\nS FP32", "fp32"),
    ("lmask_int3", "QKV-L mask→INT3\nS FP32", "int3"),
    ("lmask_int1", "QKV-L mask→INT1\nS FP32", "int1"),
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
    parser.add_argument("--relative-sigma-threshold", type=float, default=1e-2)
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
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0,1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0,1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")


def render_quantization_error_by_block(
    decompositions: dict[str, dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    keys = ("lmask_fp32", "lmask_int3", "lmask_int1")
    labels = {
        "lmask_fp32": "Spectral mask only",
        "lmask_int3": "Spectral mask → INT3",
        "lmask_int1": "Spectral mask → INT1",
    }
    blocks = np.arange(12, dtype=np.float64)
    width = 0.25
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8), constrained_layout=True)
    for index, key in enumerate(keys):
        rows = decompositions[key]["layers"]
        offset = (index - 1) * width
        axes[0].bar(
            blocks + offset,
            [row["relative_L_error"] for row in rows],
            width=width,
            label=labels[key],
        )
        axes[1].bar(
            blocks + offset,
            [row["relative_full_QKV_error"] for row in rows],
            width=width,
            label=labels[key],
        )
    axes[0].set_title("QKV-L total error versus original L")
    axes[1].set_title("Full QKV error; S remains exact FP32")
    for axis in axes:
        axis.set_xticks(blocks)
        axis.set_xticklabels([f"Block {index}" for index in range(12)])
        axis.set_ylabel("Relative Frobenius error")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("QKV-L spectral mask followed by low-bit quantization", fontsize=16)
    output_path = output_dir / "qkv_L_masked_quantization_error_by_block.png"
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
    source_block: int | None = None
    attention_shape: list[int] | None = None

    print(f"device={device}", flush=True)
    for key, label, low_rank_mode in VARIANTS:
        print(f"building {key}: QKV-S exact FP32", flush=True)
        if low_rank_mode is None:
            decomposition = apply_qkv_variant(
                model,
                args.matrix_dir,
                decomposed_modules,
                mode="fp32",
                zero_threshold=1e-5,
                int3_granularity="per_output_channel",
            )
        else:
            decomposition = apply_spectral_mask_variant(
                model,
                args.matrix_dir,
                decomposed_modules,
                sparse_mode="fp32",
                relative_sigma_threshold=args.relative_sigma_threshold,
                sparse_zero_threshold=1e-5,
                int3_granularity="per_output_channel",
                low_rank_quantization_mode=low_rank_mode,
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
            attention_by_key["lmask_fp32"],
            attention_by_key[key],
            args.attention_mass,
        )
        for key in ("lmask_int3", "lmask_int1")
    }
    horizontal_path = render_horizontal_matrix(
        variants,
        crop_array,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
        figure_title=(
            "Green lizard — QKV-L spectral mask then INT3/INT1; QKV-S fixed FP32\n"
            "Rows: L variants; columns: Mean and Head 0–11; each column shares one scale"
        ),
        output_filename="mean_and_heads_horizontal_qkv_L_masked_int1_int3.png",
    )
    rank_path = render_rank_plot(
        decompositions["lmask_fp32"],
        args.output_dir,
        args.dpi,
        args.relative_sigma_threshold,
    )
    error_path = render_quantization_error_by_block(
        decompositions,
        args.output_dir,
        args.dpi,
    )

    metadata = {
        "experiment": (
            "spectrally mask QKV-L, then quantize masked L to INT3 or INT1 while "
            "keeping QKV-S exact FP32"
        ),
        "controlled_components": {
            "QKV_S": "exact checkpoint FP32, no mask and no quantization",
            "FC": "exact checkpoint L+S FP32",
            "attention_projection": "exact checkpoint L+S FP32",
        },
        "spectral_mask": {
            "rule": "remove complete components with sigma_i/sigma_1 < threshold",
            "relative_sigma_threshold": args.relative_sigma_threshold,
        },
        "int3": (
            "per-output-channel signed symmetric narrow-range codes {-3,...,3}"
        ),
        "int1": (
            "per-QKV-matrix {-alpha,+alpha}, alpha=mean of positive masked-L entries"
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
        "incremental_quantization_vs_spectral_mask_only": incremental_metrics,
        "outputs": {
            "horizontal_mean_and_heads": str(horizontal_path.resolve()),
            "rank_and_energy": str(rank_path.resolve()),
            "quantization_error_by_block": str(error_path.resolve()),
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"horizontal_figure={horizontal_path.resolve()}", flush=True)
    print(f"error_figure={error_path.resolve()}", flush=True)
    for key in ("lmask_fp32", "lmask_int3", "lmask_int1"):
        decomposition = decompositions[key]
        mean_metrics = metrics_vs_fp32[key]["mean_over_heads"]
        head_similarity = metrics_vs_fp32[key]["within_variant_head_similarity"]
        print(
            f"{key}: rel_L_error={decomposition['relative_L_error']:.8f} "
            f"rel_full_QKV_error={decomposition['relative_full_QKV_error']:.8f} "
            f"mean_cos={mean_metrics['cosine_similarity']:.8f} "
            f"mean_tv={mean_metrics['total_variation_distance']:.8f} "
            f"mean_jsd={mean_metrics['jensen_shannon_divergence_nats']:.8g} "
            f"top60_iou={mean_metrics['top_attention_mass_mask_iou']:.8f} "
            f"head_pair_cos="
            f"{head_similarity['candidate']['mean_pairwise_cosine_similarity']:.8f}",
            flush=True,
        )
        quantization = decomposition.get("low_rank_quantization")
        if quantization is not None:
            print(
                f"{key}_quant: rel_error_vs_masked_L="
                f"{quantization['relative_quantization_error_vs_spectral_masked_L']:.8f} "
                f"zeroed_fraction="
                f"{quantization['quantization_zeroed_fraction_of_original_nonzero']:.8f}",
                flush=True,
            )
    print(f"metadata={metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
