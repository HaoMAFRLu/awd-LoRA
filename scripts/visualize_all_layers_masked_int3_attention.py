#!/usr/bin/env python3
"""Apply mask-then-INT3 to every decomposed L and S matrix in DINO ViT-B/8.

For all 48 decomposed linear layers (QKV, attention projection, FC1, and FC2),
L is spectrally masked and S is elementwise masked before per-output-channel
signed symmetric INT3 fake quantization.  The exact FP32 L+S reconstruction is
used as the controlled baseline.
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
    DECOMPOSED_LAYER_PATTERN,
    attention_metrics,
    decomposed_module_names,
    infer_patch_attention,
    load_non_decomposed_state,
    sha256_file,
    symmetric_int3_fake_quantize,
)
from scripts.visualize_qkv_low_rank_spectral_mask_attention import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_RUN_DIR,
    render_horizontal_matrix,
)
from scripts.visualize_qkv_sparse_masked_quantization_attention import (  # noqa: E402
    apply_qkv_variant,
)


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "figures"
    / "salaad_vision"
    / "all_layers_l_s_masked_int3_attention"
    / "52_green_lizard"
)
LAYER_GROUP_ORDER = {
    "attn.qkv": 0,
    "attn.proj": 1,
    "mlp.fc1": 2,
    "mlp.fc2": 3,
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
    parser.add_argument("--relative-sigma-threshold", type=float, default=1e-2)
    parser.add_argument("--sparse-zero-threshold", type=float, default=1e-5)
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


def empty_accumulator() -> dict[str, float | int]:
    return {
        "layers": 0,
        "elements": 0,
        "L_norm_squared": 0.0,
        "L_mask_error_squared": 0.0,
        "L_quantization_error_vs_masked_squared": 0.0,
        "L_total_error_squared": 0.0,
        "L_masked_norm_squared": 0.0,
        "L_original_nonzero": 0,
        "L_quantized_nonzero": 0,
        "L_quantization_zeroed": 0,
        "L_numerical_rank": 0,
        "L_retained_rank": 0,
        "L_energy_total": 0.0,
        "L_energy_retained": 0.0,
        "S_norm_squared": 0.0,
        "S_masked_norm_squared": 0.0,
        "S_quantization_error_vs_masked_squared": 0.0,
        "S_total_error_squared": 0.0,
        "S_original_nonzero": 0,
        "S_masked_nonzero": 0,
        "S_threshold_removed": 0,
        "S_quantized_nonzero": 0,
        "S_quantization_zeroed": 0,
        "full_norm_squared": 0.0,
        "full_error_squared": 0.0,
    }


def add_accumulator(
    accumulator: dict[str, float | int],
    values: dict[str, float | int],
) -> None:
    for key, value in values.items():
        accumulator[key] += value


def finalize_accumulator(
    accumulator: dict[str, float | int],
) -> dict[str, Any]:
    def ratio_sqrt(numerator: str, denominator: str) -> float:
        return math.sqrt(
            float(accumulator[numerator])
            / max(float(accumulator[denominator]), 1e-30)
        )

    return {
        **accumulator,
        "relative_L_mask_error": ratio_sqrt(
            "L_mask_error_squared",
            "L_norm_squared",
        ),
        "relative_L_quantization_error_vs_masked": ratio_sqrt(
            "L_quantization_error_vs_masked_squared",
            "L_masked_norm_squared",
        ),
        "relative_L_total_error": ratio_sqrt(
            "L_total_error_squared",
            "L_norm_squared",
        ),
        "L_quantization_zeroed_fraction": (
            int(accumulator["L_quantization_zeroed"])
            / max(int(accumulator["L_original_nonzero"]), 1)
        ),
        "L_retained_rank_fraction": (
            int(accumulator["L_retained_rank"])
            / max(int(accumulator["L_numerical_rank"]), 1)
        ),
        "L_retained_energy_fraction": (
            float(accumulator["L_energy_retained"])
            / max(float(accumulator["L_energy_total"]), 1e-30)
        ),
        "relative_S_quantization_error_vs_masked": ratio_sqrt(
            "S_quantization_error_vs_masked_squared",
            "S_masked_norm_squared",
        ),
        "relative_S_total_error": ratio_sqrt(
            "S_total_error_squared",
            "S_norm_squared",
        ),
        "S_quantization_zeroed_fraction_of_masked_nonzero": (
            int(accumulator["S_quantization_zeroed"])
            / max(int(accumulator["S_masked_nonzero"]), 1)
        ),
        "relative_full_weight_error": ratio_sqrt(
            "full_error_squared",
            "full_norm_squared",
        ),
    }


@torch.no_grad()
def apply_all_layers_masked_int3(
    model: DinoViTBase8,
    matrix_dir: Path,
    decomposed_modules: set[str],
    *,
    relative_sigma_threshold: float,
    sparse_zero_threshold: float,
) -> dict[str, Any]:
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")

    global_statistics = empty_accumulator()
    group_statistics = {
        group: empty_accumulator() for group in LAYER_GROUP_ORDER
    }
    low_rank_code_histogram = {str(code): 0 for code in range(-3, 4)}
    sparse_code_histogram = {str(code): 0 for code in range(-3, 4)}
    layer_statistics: list[dict[str, Any]] = []
    seen: set[str] = set()

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
            match = DECOMPOSED_LAYER_PATTERN.fullmatch(layer_name)
            if match is None:
                raise ValueError(f"cannot classify decomposed layer: {layer_name}")
            block = int(match.group("block"))
            group = match.group("layer")

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

            u, singular_values, vh = torch.linalg.svd(
                low_rank_weight,
                full_matrices=False,
            )
            if singular_values[0].item() == 0.0:
                retained_rank = 0
                numerical_rank = 0
                masked_low_rank = torch.zeros_like(low_rank_weight)
            else:
                normalized_singular_values = singular_values / singular_values[0]
                retained_rank = int(
                    torch.count_nonzero(
                        normalized_singular_values >= relative_sigma_threshold
                    ).item()
                )
                masked_low_rank = (
                    u[:, :retained_rank] * singular_values[:retained_rank]
                ) @ vh[:retained_rank]
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
            used_low_rank, low_rank_codes, low_rank_scales = (
                symmetric_int3_fake_quantize(
                    masked_low_rank,
                    "per_output_channel",
                )
            )

            original_sparse_nonzero_mask = sparse_weight != 0.0
            sparse_threshold_mask = original_sparse_nonzero_mask & (
                sparse_weight.abs() <= sparse_zero_threshold
            )
            masked_sparse = sparse_weight.masked_fill(sparse_threshold_mask, 0.0)
            used_sparse, sparse_codes, sparse_scales = symmetric_int3_fake_quantize(
                masked_sparse,
                "per_output_channel",
            )

            reference = low_rank_weight + sparse_weight
            replacement = used_low_rank + used_sparse
            low_rank_mask_error = masked_low_rank - low_rank_weight
            low_rank_quantization_error = used_low_rank - masked_low_rank
            low_rank_total_error = used_low_rank - low_rank_weight
            sparse_quantization_error = used_sparse - masked_sparse
            sparse_total_error = used_sparse - sparse_weight
            full_error = replacement - reference

            low_rank_original_nonzero_mask = masked_low_rank != 0.0
            low_rank_quantized_nonzero_mask = low_rank_codes != 0
            masked_sparse_nonzero_mask = masked_sparse != 0.0
            sparse_quantized_nonzero_mask = sparse_codes != 0
            singular_energy = singular_values.double().square()
            total_singular_energy = float(singular_energy.sum().item())
            retained_singular_energy = float(
                singular_energy[:retained_rank].sum().item()
            )

            layer_values: dict[str, float | int] = {
                "layers": 1,
                "elements": low_rank_weight.numel(),
                "L_norm_squared": float(low_rank_weight.square().sum().item()),
                "L_mask_error_squared": float(
                    low_rank_mask_error.square().sum().item()
                ),
                "L_quantization_error_vs_masked_squared": float(
                    low_rank_quantization_error.square().sum().item()
                ),
                "L_total_error_squared": float(
                    low_rank_total_error.square().sum().item()
                ),
                "L_masked_norm_squared": float(
                    masked_low_rank.square().sum().item()
                ),
                "L_original_nonzero": int(
                    torch.count_nonzero(low_rank_original_nonzero_mask).item()
                ),
                "L_quantized_nonzero": int(
                    torch.count_nonzero(low_rank_quantized_nonzero_mask).item()
                ),
                "L_quantization_zeroed": int(
                    torch.count_nonzero(
                        low_rank_original_nonzero_mask
                        & ~low_rank_quantized_nonzero_mask
                    ).item()
                ),
                "L_numerical_rank": numerical_rank,
                "L_retained_rank": retained_rank,
                "L_energy_total": total_singular_energy,
                "L_energy_retained": retained_singular_energy,
                "S_norm_squared": float(sparse_weight.square().sum().item()),
                "S_masked_norm_squared": float(masked_sparse.square().sum().item()),
                "S_quantization_error_vs_masked_squared": float(
                    sparse_quantization_error.square().sum().item()
                ),
                "S_total_error_squared": float(
                    sparse_total_error.square().sum().item()
                ),
                "S_original_nonzero": int(
                    torch.count_nonzero(original_sparse_nonzero_mask).item()
                ),
                "S_masked_nonzero": int(
                    torch.count_nonzero(masked_sparse_nonzero_mask).item()
                ),
                "S_threshold_removed": int(
                    torch.count_nonzero(sparse_threshold_mask).item()
                ),
                "S_quantized_nonzero": int(
                    torch.count_nonzero(sparse_quantized_nonzero_mask).item()
                ),
                "S_quantization_zeroed": int(
                    torch.count_nonzero(
                        masked_sparse_nonzero_mask
                        & ~sparse_quantized_nonzero_mask
                    ).item()
                ),
                "full_norm_squared": float(reference.square().sum().item()),
                "full_error_squared": float(full_error.square().sum().item()),
            }
            add_accumulator(global_statistics, layer_values)
            add_accumulator(group_statistics[group], layer_values)
            for code in range(-3, 4):
                low_rank_code_histogram[str(code)] += int(
                    torch.count_nonzero(low_rank_codes == code).item()
                )
                sparse_code_histogram[str(code)] += int(
                    torch.count_nonzero(sparse_codes == code).item()
                )

            positive_low_rank_scales = low_rank_scales[low_rank_scales > 0.0]
            positive_sparse_scales = sparse_scales[sparse_scales > 0.0]
            layer_summary = finalize_accumulator(layer_values)
            layer_statistics.append(
                {
                    "layer": layer_name,
                    "block": block,
                    "group": group,
                    "shape": list(low_rank_weight.shape),
                    "original_numerical_rank": numerical_rank,
                    "retained_rank": retained_rank,
                    "removed_rank": numerical_rank - retained_rank,
                    "retained_L_energy": (
                        retained_singular_energy / max(total_singular_energy, 1e-30)
                    ),
                    "L_scale_min": float(positive_low_rank_scales.min().item()),
                    "L_scale_median": float(
                        positive_low_rank_scales.median().item()
                    ),
                    "L_scale_max": float(positive_low_rank_scales.max().item()),
                    "S_scale_min": float(positive_sparse_scales.min().item()),
                    "S_scale_median": float(positive_sparse_scales.median().item()),
                    "S_scale_max": float(positive_sparse_scales.max().item()),
                    "statistics": layer_summary,
                }
            )
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))

        del low_rank
        del sparse
        gc.collect()

    missing = decomposed_modules - seen
    if missing or len(seen) != 48:
        raise ValueError(
            f"decomposition coverage mismatch: seen={len(seen)}, missing={sorted(missing)}"
        )
    return {
        "weight_definition": (
            "For every QKV/proj/FC1/FC2: "
            "W=dequant(INT3(spectral_mask(L)))+"
            "dequant(INT3(epsilon_mask(S)))"
        ),
        "decomposed_layers": len(seen),
        "L": {
            "mask": (
                "retain complete SVD components satisfying "
                "sigma_i/sigma_1 >= threshold"
            ),
            "relative_sigma_threshold": relative_sigma_threshold,
            "quantization": "per-output-channel signed symmetric INT3 [-3,3]",
            "code_histogram": low_rank_code_histogram,
        },
        "S": {
            "mask": "abs(S) <= epsilon becomes zero",
            "zero_threshold": sparse_zero_threshold,
            "quantization": "per-output-channel signed symmetric INT3 [-3,3]",
            "code_histogram": sparse_code_histogram,
        },
        "global": finalize_accumulator(global_statistics),
        "by_group": {
            group: finalize_accumulator(values)
            for group, values in group_statistics.items()
        },
        "layers": sorted(
            layer_statistics,
            key=lambda row: (row["block"], LAYER_GROUP_ORDER[row["group"]]),
        ),
    }


def render_error_summary(
    decomposition: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> Path:
    group_names = list(LAYER_GROUP_ORDER)
    display_names = ["QKV", "Proj", "FC1", "FC2"]
    group_rows = [decomposition["by_group"][name] for name in group_names]
    positions = np.arange(len(group_names), dtype=np.float64)
    width = 0.23
    figure, axes = plt.subplots(2, 1, figsize=(15, 8.5), constrained_layout=True)
    axes[0].bar(
        positions - width,
        [row["relative_L_total_error"] for row in group_rows],
        width=width,
        label="L total error",
    )
    axes[0].bar(
        positions,
        [row["relative_S_total_error"] for row in group_rows],
        width=width,
        label="S total error",
    )
    axes[0].bar(
        positions + width,
        [row["relative_full_weight_error"] for row in group_rows],
        width=width,
        label="Full L+S error",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(display_names)
    axes[0].set_ylabel("Relative Frobenius error")
    axes[0].set_title("Aggregate error by layer type")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    layers = decomposition["layers"]
    labels = [
        f"B{row['block']} {row['group'].split('.')[-1].upper()}" for row in layers
    ]
    axes[1].bar(
        np.arange(len(layers)),
        [row["statistics"]["relative_full_weight_error"] for row in layers],
    )
    axes[1].set_xticks(np.arange(len(layers)))
    axes[1].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_ylabel("Relative full-weight error")
    axes[1].set_title("Full L+S error for all 48 decomposed layers")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("All-layer L/S mask→INT3 quantization error", fontsize=16)
    output_path = output_dir / "all_layers_masked_int3_error.png"
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

    print(f"device={device}", flush=True)
    print("building exact FP32 L+S baseline", flush=True)
    baseline_decomposition = apply_qkv_variant(
        model,
        args.matrix_dir,
        decomposed_modules,
        mode="fp32",
        zero_threshold=args.sparse_zero_threshold,
        int3_granularity="per_output_channel",
    )
    baseline_attention, source_block, attention_shape = infer_patch_attention(
        model,
        image_tensor,
        device,
    )

    print("building all 48 layers: L mask→INT3 + S mask→INT3", flush=True)
    candidate_decomposition = apply_all_layers_masked_int3(
        model,
        args.matrix_dir,
        decomposed_modules,
        relative_sigma_threshold=args.relative_sigma_threshold,
        sparse_zero_threshold=args.sparse_zero_threshold,
    )
    candidate_attention, candidate_source_block, candidate_attention_shape = (
        infer_patch_attention(model, image_tensor, device)
    )
    if candidate_source_block != source_block:
        raise RuntimeError("baseline and candidate used different attention blocks")

    baseline_dir = args.output_dir / "fp32"
    candidate_dir = args.output_dir / "all_l_s_masked_int3"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_dir / "cls_to_patch_attention.npy"
    candidate_path = candidate_dir / "cls_to_patch_attention.npy"
    np.save(baseline_path, baseline_attention.numpy())
    np.save(candidate_path, candidate_attention.numpy())
    variants = [
        {
            "key": "fp32",
            "label": "All L+S FP32",
            "attention": baseline_attention,
            "attention_path": baseline_path,
        },
        {
            "key": "all_l_s_masked_int3",
            "label": "All L spectral mask→INT3\nAll S mask→INT3",
            "attention": candidate_attention,
            "attention_path": candidate_path,
        },
    ]
    metrics = attention_metrics(
        baseline_attention,
        candidate_attention,
        args.attention_mass,
    )
    horizontal_path = render_horizontal_matrix(
        variants,
        crop_array,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
        figure_title=(
            "Green lizard — all 48 decomposed layers: L/S mask→INT3\n"
            "Rows: FP32 and all-layer INT3; columns: Mean and Head 0–11; "
            "each column shares one scale"
        ),
        output_filename="mean_and_heads_horizontal_all_L_S_masked_int3.png",
    )
    error_path = render_error_summary(
        candidate_decomposition,
        args.output_dir,
        args.dpi,
    )

    metadata = {
        "experiment": "all 48 decomposed L and S matrices mask then INT3",
        "controlled_difference": (
            "Every decomposed QKV, attention projection, FC1, and FC2 weight "
            "changes from exact FP32 L+S to masked-INT3 L + masked-INT3 S. "
            "Biases, norms, embeddings, and other state remain exact."
        ),
        "L_definition": {
            "mask": "remove singular components with sigma_i/sigma_1 < 1e-2",
            "quantization": "per-output-channel symmetric INT3 codes -3..3",
        },
        "S_definition": {
            "mask": "abs(S) <= 1e-5 becomes zero",
            "quantization": "per-output-channel symmetric INT3 codes -3..3",
        },
        "sample": {
            "sample_id": "52_green_lizard",
            "source_image": str(args.image.resolve()),
            "source_sha256": sha256_file(args.image),
        },
        "checkpoint": {
            "matrix_dir": str(args.matrix_dir.resolve()),
            "base_state": base_state_metadata,
        },
        "model_reconstruction": {
            "baseline": baseline_decomposition,
            "candidate": candidate_decomposition,
        },
        "attention": {
            "source_block": source_block,
            "baseline_shape": attention_shape,
            "candidate_shape": candidate_attention_shape,
            "heads": 12,
            "patch_grid": [28, 28],
            "metrics_vs_fp32": metrics,
        },
        "outputs": {
            "baseline_attention": str(baseline_path.resolve()),
            "candidate_attention": str(candidate_path.resolve()),
            "horizontal_mean_and_heads": str(horizontal_path.resolve()),
            "quantization_error": str(error_path.resolve()),
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    global_statistics = candidate_decomposition["global"]
    mean_metrics = metrics["mean_over_heads"]
    head_similarity = metrics["within_variant_head_similarity"]
    print(f"horizontal_figure={horizontal_path.resolve()}", flush=True)
    print(f"error_figure={error_path.resolve()}", flush=True)
    print(
        "quantization: "
        f"rel_L_error={global_statistics['relative_L_total_error']:.8f} "
        f"rel_S_error={global_statistics['relative_S_total_error']:.8f} "
        f"rel_full_error={global_statistics['relative_full_weight_error']:.8f} "
        f"L_zeroed_fraction="
        f"{global_statistics['L_quantization_zeroed_fraction']:.8f} "
        f"S_zeroed_fraction="
        f"{global_statistics['S_quantization_zeroed_fraction_of_masked_nonzero']:.8f}",
        flush=True,
    )
    print(
        "attention: "
        f"mean_cos={mean_metrics['cosine_similarity']:.8f} "
        f"mean_tv={mean_metrics['total_variation_distance']:.8f} "
        f"mean_jsd={mean_metrics['jensen_shannon_divergence_nats']:.8g} "
        f"top60_iou={mean_metrics['top_attention_mass_mask_iou']:.8f} "
        f"head_pair_cos="
        f"{head_similarity['candidate']['mean_pairwise_cosine_similarity']:.8f}",
        flush=True,
    )
    print(f"metadata={metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
