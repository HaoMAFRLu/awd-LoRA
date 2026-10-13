#!/usr/bin/env python3
"""Place corresponding DINO attention heads side by side across FC-S variants."""

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
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visualize_fc_sparse_int4_attention import (  # noqa: E402
    distribution_metrics,
    prepare_image,
    sha256_file,
    upsample_patch_map,
    within_variant_head_similarity,
)


FIGURE_ROOT = (
    REPOSITORY_ROOT / "data" / "figures" / "salaad_vision"
)
SAMPLE_ID = "52_green_lizard"
DEFAULT_VARIANTS = (
    (
        "fp32_fc_s",
        "FP32 FC-S baseline",
        FIGURE_ROOT
        / "fc_sparse_masked_int1_attention"
        / SAMPLE_ID
        / "fc_s_fp32_baseline"
        / "cls_to_patch_attention.npy",
    ),
    (
        "int4_per_output_channel",
        "INT4 per-output-channel",
        FIGURE_ROOT
        / "fc_sparse_int4_attention"
        / SAMPLE_ID
        / "fc_s_int4_per_output_channel"
        / "cls_to_patch_attention.npy",
    ),
    (
        "int3_per_output_channel",
        "INT3 per-output-channel",
        FIGURE_ROOT
        / "fc_sparse_int3_attention"
        / SAMPLE_ID
        / "fc_s_int3_per_output_channel"
        / "cls_to_patch_attention.npy",
    ),
    (
        "dense_int1_positive_mean",
        "Dense INT1",
        FIGURE_ROOT
        / "fc_sparse_int1_attention"
        / SAMPLE_ID
        / "fc_s_int1_positive_mean"
        / "cls_to_patch_attention.npy",
    ),
    (
        "masked_int1_positive_mean",
        "Masked INT1, epsilon=1e-5",
        FIGURE_ROOT
        / "fc_sparse_masked_int1_attention"
        / SAMPLE_ID
        / "fc_s_int1_sparse_positive_mean"
        / "cls_to_patch_attention.npy",
    ),
)
DEFAULT_SOURCE = (
    FIGURE_ROOT / "self_attention" / SAMPLE_ID / "source.jpg"
)
DEFAULT_OUTPUT_DIR = (
    FIGURE_ROOT / "fc_quantization_attention_head_comparison" / SAMPLE_ID
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--only-horizontal-matrix",
        action="store_true",
        help="Write only the backbone-row, Mean-plus-head-columns matrix.",
    )
    return parser.parse_args()


def load_attention(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"attention array does not exist: {path}")
    values = np.load(path)
    if values.shape != (12, 28, 28):
        raise ValueError(f"expected [12, 28, 28], got {values.shape}: {path}")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"attention must be finite and non-negative: {path}")
    head_sums = values.sum(axis=(1, 2))
    if not np.allclose(head_sums, 1.0, atol=1e-6):
        raise ValueError(f"attention heads are not normalized: {path}")
    return torch.from_numpy(values).float()


def hotspot(patch_map: torch.Tensor) -> dict[str, Any]:
    flat_index = int(torch.argmax(patch_map).item())
    width = patch_map.shape[1]
    patch_y, patch_x = divmod(flat_index, width)
    return {
        "patch_x": patch_x,
        "patch_y": patch_y,
        "pixel_center_x": (patch_x + 0.5) * (224 / width),
        "pixel_center_y": (patch_y + 0.5) * (224 / patch_map.shape[0]),
        "peak_probability": float(patch_map[patch_y, patch_x].item()),
    }


def build_metrics(
    variants: list[dict[str, Any]],
    attention_mass: float,
) -> dict[str, Any]:
    baseline = variants[0]["attention"]
    output: dict[str, Any] = {
        "baseline": variants[0]["key"],
        "per_variant": {},
    }
    for variant in variants:
        attention = variant["attention"]
        per_head = []
        for head_index in range(attention.shape[0]):
            row: dict[str, Any] = {
                "head": head_index,
                "hotspot": hotspot(attention[head_index]),
            }
            row.update(
                distribution_metrics(
                    baseline[head_index],
                    attention[head_index],
                    attention_mass,
                )
            )
            baseline_hotspot = hotspot(baseline[head_index])
            row["hotspot_shift_patches"] = math.hypot(
                row["hotspot"]["patch_x"] - baseline_hotspot["patch_x"],
                row["hotspot"]["patch_y"] - baseline_hotspot["patch_y"],
            )
            per_head.append(row)
        output["per_variant"][variant["key"]] = {
            "label": variant["label"],
            "mean_attention_vs_baseline": distribution_metrics(
                baseline.mean(dim=0),
                attention.mean(dim=0),
                attention_mass,
            ),
            "within_variant_head_similarity": within_variant_head_similarity(
                attention,
                attention_mass,
            ),
            "per_head_vs_baseline": per_head,
        }
    return output


def render_shared_head_scale_overlays(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
) -> Path:
    head_count = variants[0]["attention"].shape[0]
    figure, axes = plt.subplots(
        head_count,
        len(variants),
        figsize=(3.7 * len(variants), 37.0),
        constrained_layout=True,
        squeeze=False,
    )
    for variant_index, variant in enumerate(variants):
        axes[0, variant_index].set_title(variant["label"], fontsize=14, pad=10)

    for head_index in range(head_count):
        high_resolution_maps = [
            upsample_patch_map(
                variant["attention"][head_index],
                mode="nearest",
            )
            for variant in variants
        ]
        shared_maximum = max(
            max(float(attention_map.max()) for attention_map in high_resolution_maps),
            1e-12,
        )
        for variant_index, (variant, attention_map) in enumerate(
            zip(variants, high_resolution_maps)
        ):
            axis = axes[head_index, variant_index]
            axis.imshow(crop_array)
            axis.imshow(
                attention_map,
                cmap="turbo",
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=shared_maximum,
            )
            peak = hotspot(variant["attention"][head_index])
            axis.scatter(
                [peak["pixel_center_x"]],
                [peak["pixel_center_y"]],
                s=30,
                facecolors="none",
                edgecolors="white",
                linewidths=1.0,
            )
            if variant_index == 0:
                annotation = f"Head {head_index}\npeak={peak['peak_probability']:.4f}"
            else:
                row = metrics["per_variant"][variant["key"]][
                    "per_head_vs_baseline"
                ][head_index]
                annotation = (
                    f"Head {head_index}  cos={row['cosine_similarity']:.3f}\n"
                    f"TV={row['total_variation_distance']:.3f}, "
                    f"shift={row['hotspot_shift_patches']:.1f} patches"
                )
            axis.text(
                0.015,
                0.985,
                annotation,
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.50,
                    "edgecolor": "none",
                    "pad": 2.0,
                },
            )
            axis.axis("off")

    figure.suptitle(
        "Green lizard: corresponding final-block attention heads side by side\n"
        "Each row uses a shared scale across variants; white circle marks the hotspot",
        fontsize=18,
    )
    output_path = output_dir / "heads_side_by_side_shared_head_scale.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_global_scale_heatmaps(
    variants: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    head_count = variants[0]["attention"].shape[0]
    global_maximum = max(
        float(variant["attention"].max().item()) for variant in variants
    )
    figure, axes = plt.subplots(
        head_count,
        len(variants),
        figsize=(3.35 * len(variants), 34.0),
        constrained_layout=True,
        squeeze=False,
    )
    artist = None
    for variant_index, variant in enumerate(variants):
        axes[0, variant_index].set_title(variant["label"], fontsize=14, pad=10)
    for head_index in range(head_count):
        for variant_index, variant in enumerate(variants):
            axis = axes[head_index, variant_index]
            artist = axis.imshow(
                variant["attention"][head_index].numpy(),
                cmap="turbo",
                interpolation="nearest",
                vmin=0.0,
                vmax=global_maximum,
            )
            if variant_index == 0:
                axis.set_ylabel(f"Head {head_index}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    if artist is not None:
        figure.colorbar(
            artist,
            ax=axes,
            fraction=0.012,
            pad=0.012,
            label="CLS-to-patch attention probability",
        )
    figure.suptitle(
        "Final-block attention heads with one global probability scale",
        fontsize=18,
    )
    output_path = output_dir / "heads_side_by_side_global_scale.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_mean_attention_comparison(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
) -> Path:
    mean_maps = [variant["attention"].mean(dim=0) for variant in variants]
    high_resolution_maps = [upsample_patch_map(mean_map) for mean_map in mean_maps]
    shared_maximum = max(
        max(float(attention_map.max()) for attention_map in high_resolution_maps),
        1e-12,
    )
    figure, axes = plt.subplots(
        1,
        len(variants),
        figsize=(4.0 * len(variants), 4.3),
        constrained_layout=True,
    )
    for variant_index, (axis, variant, attention_map) in enumerate(
        zip(axes, variants, high_resolution_maps)
    ):
        axis.imshow(crop_array)
        axis.imshow(
            attention_map,
            cmap="turbo",
            alpha=overlay_alpha,
            vmin=0.0,
            vmax=shared_maximum,
        )
        mean_row = metrics["per_variant"][variant["key"]][
            "mean_attention_vs_baseline"
        ]
        title = variant["label"]
        if variant_index > 0:
            title += (
                f"\ncos={mean_row['cosine_similarity']:.4f}, "
                f"TV={mean_row['total_variation_distance']:.4f}"
            )
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(
        "Mean over 12 heads — shared scale across all variants",
        fontsize=16,
    )
    output_path = output_dir / "mean_attention_four_way_comparison.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_backbone_rows_mean_and_heads_columns(
    variants: list[dict[str, Any]],
    crop_array: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
    dpi: int,
) -> Path:
    """Render exactly one row per variant and Mean + Head 0..11 as columns."""
    column_titles = ["Mean"] + [f"Head {index}" for index in range(12)]
    row_labels = {
        "fp32_fc_s": "FP32 FC-S",
        "int4_per_output_channel": "INT4 FC-S",
        "int3_per_output_channel": "INT3 FC-S",
        "dense_int1_positive_mean": "Dense INT1 FC-S",
        "masked_int1_positive_mean": "Masked INT1 FC-S",
    }
    high_resolution: list[list[np.ndarray]] = []
    for variant in variants:
        attention = variant["attention"]
        maps = [upsample_patch_map(attention.mean(dim=0))]
        maps.extend(
            upsample_patch_map(attention[head_index], mode="nearest")
            for head_index in range(12)
        )
        high_resolution.append(maps)

    shared_column_maxima = [
        max(
            float(high_resolution[row_index][column_index].max())
            for row_index in range(len(variants))
        )
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
                    row_labels[variant["key"]],
                    fontsize=13,
                    rotation=90,
                    labelpad=10,
                )
    figure.suptitle(
        "Green lizard — final-block CLS-to-patch attention\n"
        "Rows: backbone variants; columns: Mean and Head 0–11; "
        "each column shares one scale",
        fontsize=17,
    )
    output_path = output_dir / "mean_and_heads_horizontal_backbones_vertical.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_overview_panels_side_by_side(
    variants: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    overview_paths = [variant["path"].parent / "overview.png" for variant in variants]
    missing = [str(path) for path in overview_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing 16-panel overview images: {missing}")
    figure, axes = plt.subplots(
        1,
        len(variants),
        figsize=(7.9 * len(variants), 8.4),
        constrained_layout=True,
    )
    for axis, variant, overview_path in zip(axes, variants, overview_paths):
        with Image.open(overview_path) as overview:
            axis.imshow(overview.convert("RGB"))
        axis.set_title(variant["label"], fontsize=15)
        axis.axis("off")
    figure.suptitle(
        f"{len(variants)} complete 16-panel overviews side by side\n"
        "Each overview contains 4 context panels and 12 attention heads",
        fontsize=18,
    )
    output_path = output_dir / "complete_16_panel_overviews_side_by_side.png"
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"source image does not exist: {args.source}")
    if not 0.0 < args.attention_mass <= 1.0:
        raise ValueError("attention mass must be in (0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0, 1]")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")

    variants = [
        {
            "key": key,
            "label": label,
            "path": path,
            "attention": load_attention(path),
        }
        for key, label, path in DEFAULT_VARIANTS
    ]
    head_counts = {variant["attention"].shape[0] for variant in variants}
    if head_counts != {12}:
        raise ValueError(f"expected exactly 12 attention heads, got {head_counts}")

    _, crop, _ = prepare_image(args.source)
    crop_array = np.asarray(crop, dtype=np.float32) / 255.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only_horizontal_matrix:
        requested_layout_path = render_backbone_rows_mean_and_heads_columns(
            variants,
            crop_array,
            args.output_dir,
            args.overlay_alpha,
            args.dpi,
        )
        print(f"requested_layout={requested_layout_path.resolve()}")
        return

    metrics = build_metrics(variants, args.attention_mass)
    overlay_path = render_shared_head_scale_overlays(
        variants,
        crop_array,
        metrics,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
    )
    global_path = render_global_scale_heatmaps(
        variants,
        args.output_dir,
        args.dpi,
    )
    mean_path = render_mean_attention_comparison(
        variants,
        crop_array,
        metrics,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
    )
    requested_layout_path = render_backbone_rows_mean_and_heads_columns(
        variants,
        crop_array,
        args.output_dir,
        args.overlay_alpha,
        args.dpi,
    )
    overview_path = render_overview_panels_side_by_side(
        variants,
        args.output_dir,
        args.dpi,
    )

    metadata = {
        "sample_id": SAMPLE_ID,
        "class_name": "green lizard, Lacerta viridis",
        "source_image": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "attention_heads": 12,
        "clarification": (
            "DINO ViT-B/8 has 12 attention heads. The earlier 4x4 overview has "
            "16 panels, but its first four panels are image/mean/mask context."
        ),
        "comparison_design": {
            "overlay_figure": (
                "Each row is one corresponding head. All four variants in a row "
                "share the same vmin/vmax. Scales may differ between rows."
            ),
            "global_figure": (
                "All 48 maps share one vmin/vmax, allowing both cross-head and "
                "cross-variant magnitude comparison."
            ),
            "hotspot_marker": "White circle marks the maximum-probability patch.",
        },
        "variants": [
            {
                "key": variant["key"],
                "label": variant["label"],
                "attention_path": str(variant["path"].resolve()),
                "attention_sha256": sha256_file(variant["path"]),
            }
            for variant in variants
        ],
        "metrics": metrics,
        "outputs": {
            "heads_shared_head_scale": str(overlay_path.resolve()),
            "heads_global_scale": str(global_path.resolve()),
            "mean_attention_four_way": str(mean_path.resolve()),
            "backbone_rows_mean_and_heads_columns": str(
                requested_layout_path.resolve()
            ),
            "complete_16_panel_overviews": str(overview_path.resolve()),
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"head_count=12")
    print(f"shared_head_scale={overlay_path.resolve()}")
    print(f"global_scale={global_path.resolve()}")
    print(f"mean_comparison={mean_path.resolve()}")
    print(f"requested_layout={requested_layout_path.resolve()}")
    print(f"complete_overviews={overview_path.resolve()}")
    print(f"metadata={metadata_path.resolve()}")


if __name__ == "__main__":
    main()
