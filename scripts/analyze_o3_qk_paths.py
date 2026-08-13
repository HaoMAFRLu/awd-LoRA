"""Analyze where the LL, SL, LS, and SS Q--K paths attend on O3.

For each DINO ViT-B/8 transformer block, the trained dense-X token stream is
held fixed and only the current block's Q--K interaction is changed.  With

    Q = Q_L + Q_S,  K = K_L + K_S,

the bias-free attention logits decompose exactly as

    Z = Z_LL + Z_SL + Z_LS + Z_SS,

where SL means Q_S K_L^T and LS means Q_L K_S^T.  Two complementary outputs
are reported:

* softmax(Z_path), which is an isolated-path behavioral ablation answering
  where that path attends when used alone; and
* signed region-logit contrasts, which are exactly additive and describe how
  each path promotes or suppresses O3 regions in the joint interaction.

The four isolated softmax maps are not interpreted as additive probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_o3_shallow_ll import (  # noqa: E402
    NUM_PATCHES,
    PATCH_GRID,
    QUERY_NAMES,
    REGION_NAMES,
    O3Dataset,
    _annotated_image,
    _attention_overlay,
    _bootstrap_mean_ci,
    _group_values,
    _load_display_image,
    _project_attention,
    choose_device,
    choose_representatives,
    load_decompositions,
)
from salaad_vision.models.dino import DinoViTBase8  # noqa: E402


PATH_NAMES = ("ll", "sl", "ls", "ss")
PATH_LABELS = {
    "ll": r"LL: $Q_LK_L^T$",
    "sl": r"SL: $Q_SK_L^T$",
    "ls": r"LS: $Q_LK_S^T$",
    "ss": r"SS: $Q_SK_S^T$",
}
CONDITION_NAMES = (
    "dense_x",
    "full_l_plus_s",
    "joint_no_bias",
    *PATH_NAMES,
)
CONDITION_LABELS = {
    "dense_x": "Dense X\n(+ bias)",
    "full_l_plus_s": "Full L+S\n(+ bias)",
    "joint_no_bias": "LL+SL+LS+SS\n(no bias)",
    "ll": "LL",
    "sl": "SL",
    "ls": "LS",
    "ss": "SS",
}
PATH_COLORS = {
    "ll": "#277da1",
    "sl": "#f8961e",
    "ls": "#43aa8b",
    "ss": "#d62828",
}
GROUP_NAMES = ("same", "different", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/salaad_vision/o3/raw",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/salaad_vision/vit_b8_qkv/20260804_111549/model.pth",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=ROOT / "data/salaad_vision/vit_b8_qkv/20260804_111549",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/figures/salaad_vision/o3_qk_paths",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=tuple(range(12)))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--representative-count", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _project_qk(
    patches: Tensor,
    query_indices: Tensor,
    qkv_weight: Tensor,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """Project selected queries and all patch keys without qkv bias."""
    batch, num_patches, width = patches.shape
    if num_patches != NUM_PATCHES:
        raise ValueError(f"expected {NUM_PATCHES} patch tokens, got {num_patches}")
    if qkv_weight.shape != (3 * width, width):
        raise ValueError(f"unexpected qkv shape: {tuple(qkv_weight.shape)}")
    if width % num_heads:
        raise ValueError("embedding width must be divisible by the number of heads")
    batch_indices = torch.arange(batch, device=patches.device)[:, None]
    query_features = patches[batch_indices, query_indices]
    q_weight, k_weight, _ = qkv_weight.chunk(3, dim=0)
    head_dim = width // num_heads
    queries = F.linear(query_features, q_weight).reshape(
        batch,
        len(QUERY_NAMES),
        num_heads,
        head_dim,
    )
    keys = F.linear(patches, k_weight).reshape(
        batch,
        num_patches,
        num_heads,
        head_dim,
    )
    return queries, keys


def _path_logits(
    patches: Tensor,
    query_indices: Tensor,
    low_rank: Tensor,
    sparse: Tensor,
    *,
    num_heads: int,
    scale: float,
) -> dict[str, Tensor]:
    q_low, k_low = _project_qk(patches, query_indices, low_rank, num_heads)
    q_sparse, k_sparse = _project_qk(patches, query_indices, sparse, num_heads)

    def interaction(queries: Tensor, keys: Tensor) -> Tensor:
        return torch.einsum("bqhd,bnhd->bqhn", queries, keys) * scale

    return {
        "ll": interaction(q_low, k_low),
        "sl": interaction(q_sparse, k_low),
        "ls": interaction(q_low, k_sparse),
        "ss": interaction(q_sparse, k_sparse),
    }


def _region_mean(values: Tensor, masks: Tensor) -> Tensor:
    """Average [B,Q,H,N] values over [B,R,N] or [B,Q,R,N] masks."""
    if masks.ndim == 3:
        numerator = torch.einsum("bqhn,brn->bqhr", values, masks)
        denominator = masks.sum(dim=-1)[:, None, None, :]
    elif masks.ndim == 4:
        numerator = torch.einsum("bqhn,bqrn->bqhr", values, masks)
        denominator = masks.sum(dim=-1)[:, :, None, :]
    else:
        raise ValueError(f"unexpected mask shape: {tuple(masks.shape)}")
    return numerator / denominator.clamp_min(1e-12)


def _attention_per_image_rows(
    records: Sequence[object],
    blocks: Sequence[int],
    masses: np.ndarray,
    enrichments: np.ndarray,
    nonself_enrichments: np.ndarray,
) -> list[dict[str, object]]:
    grouped = {
        "mass": _group_values(masses).mean(axis=3),
        "enrichment": _group_values(enrichments).mean(axis=3),
        "nonself_enrichment": _group_values(nonself_enrichments).mean(axis=3),
    }
    directional = {
        "mass": masses.mean(axis=4),
        "enrichment": enrichments.mean(axis=4),
        "nonself_enrichment": nonself_enrichments.mean(axis=4),
    }
    rows: list[dict[str, object]] = []
    for image_index, record in enumerate(records):
        for block_offset, block in enumerate(blocks):
            for condition_offset, condition in enumerate(CONDITION_NAMES):
                row: dict[str, object] = {
                    "image_name": record.image_name,
                    "target_type": record.target_type,
                    "attributes": ",".join(record.attributes),
                    "num_distractors": record.num_distractors,
                    "block": block,
                    "condition": condition,
                }
                for measure, values in grouped.items():
                    same, different, background = values[
                        image_index,
                        block_offset,
                        condition_offset,
                    ]
                    row[f"{measure}_same"] = float(same)
                    row[f"{measure}_different"] = float(different)
                    row[f"{measure}_background"] = float(background)
                    row[f"{measure}_grouping_margin"] = float(same - different)
                for measure, values in directional.items():
                    regions = values[image_index, block_offset, condition_offset]
                    for query_offset, query in enumerate(QUERY_NAMES):
                        for region_offset, region in enumerate(REGION_NAMES):
                            row[f"{measure}_{query}_to_{region}"] = float(
                                regions[query_offset, region_offset]
                            )
                    row[f"{measure}_distractor_grouping_margin"] = float(
                        regions[1, 1] - regions[1, 0]
                    )
                rows.append(row)
    return rows


def _path_logit_per_image_rows(
    records: Sequence[object],
    blocks: Sequence[int],
    region_means: np.ndarray,
    nonself_region_means: np.ndarray,
) -> list[dict[str, object]]:
    grouped = {
        "logit": _group_values(region_means).mean(axis=3),
        "nonself_logit": _group_values(nonself_region_means).mean(axis=3),
    }
    directional = {
        "logit": region_means.mean(axis=4),
        "nonself_logit": nonself_region_means.mean(axis=4),
    }
    rows: list[dict[str, object]] = []
    for image_index, record in enumerate(records):
        for block_offset, block in enumerate(blocks):
            for path_offset, path in enumerate(PATH_NAMES):
                row: dict[str, object] = {
                    "image_name": record.image_name,
                    "target_type": record.target_type,
                    "attributes": ",".join(record.attributes),
                    "num_distractors": record.num_distractors,
                    "block": block,
                    "path": path,
                }
                for measure, values in grouped.items():
                    same, different, background = values[
                        image_index,
                        block_offset,
                        path_offset,
                    ]
                    row[f"{measure}_same"] = float(same)
                    row[f"{measure}_different"] = float(different)
                    row[f"{measure}_background"] = float(background)
                    row[f"{measure}_grouping_contrast"] = float(same - different)
                for measure, values in directional.items():
                    regions = values[image_index, block_offset, path_offset]
                    for query_offset, query in enumerate(QUERY_NAMES):
                        for region_offset, region in enumerate(REGION_NAMES):
                            row[f"{measure}_{query}_to_{region}"] = float(
                                regions[query_offset, region_offset]
                            )
                    row[f"{measure}_distractor_contrast"] = float(
                        regions[1, 1] - regions[1, 0]
                    )
                rows.append(row)
    return rows


def _summarize_dataframe(
    frame: pd.DataFrame,
    keys: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = list(keys)
    for group_values, subset in frame.groupby(group_key, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        prefix = dict(zip(keys, group_values))
        for metric in metrics:
            mean, low, high = _bootstrap_mean_ci(
                subset[metric].to_numpy(),
                bootstrap_samples,
                rng,
            )
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _plot_attention_margins(
    per_image: pd.DataFrame,
    blocks: Sequence[int],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    bootstrap_samples: int,
    seed: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    groups = (
        ("References", CONDITION_NAMES[:3]),
        ("Isolated paths", PATH_NAMES),
    )
    rng = np.random.default_rng(seed)
    for axis, (panel_title, conditions) in zip(axes, groups):
        for condition in conditions:
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            subset = per_image[per_image["condition"] == condition]
            for block in blocks:
                values = subset[subset["block"] == block][metric].to_numpy()
                mean, low, high = _bootstrap_mean_ci(
                    values,
                    bootstrap_samples,
                    rng,
                )
                means.append(mean)
                lows.append(low)
                highs.append(high)
            means_array = np.asarray(means)
            low_array = np.asarray(lows)
            high_array = np.asarray(highs)
            axis.errorbar(
                blocks,
                means_array,
                yerr=np.vstack((means_array - low_array, high_array - means_array)),
                marker="o",
                markersize=4,
                linewidth=1.7,
                capsize=2,
                label=CONDITION_LABELS[condition].replace("\n", " "),
                color=PATH_COLORS.get(condition),
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(panel_title)
        axis.set_xticks(blocks)
        axis.set_xlabel("Transformer block")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(ylabel)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_path_region_heatmap(
    values: np.ndarray,
    blocks: Sequence[int],
    query_offset: int,
    output_path: Path,
    measure_label: str,
) -> None:
    path_offsets = [CONDITION_NAMES.index(path) for path in PATH_NAMES]
    mean = values[:, :, path_offsets, query_offset].mean(axis=(0, 3))
    # [block, path, region] -> one [path, block] matrix per region.
    figure, axes = plt.subplots(
        1,
        len(REGION_NAMES),
        figsize=(17.2, 4.2),
        layout="constrained",
    )
    positive = np.all(mean >= 0)
    if positive:
        minimum = 0.0
        maximum = max(float(np.percentile(mean, 99)), 1e-6)
        cmap = "viridis"
    else:
        maximum = max(float(np.percentile(np.abs(mean), 99)), 1e-6)
        minimum = -maximum
        cmap = "coolwarm"
    for region_offset, (axis, region) in enumerate(zip(axes, REGION_NAMES)):
        matrix = mean[:, :, region_offset].T
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(region.capitalize())
        axis.set_xticks(np.arange(len(blocks)), blocks)
        axis.set_xlabel("Block")
        axis.set_yticks(np.arange(len(PATH_NAMES)), [name.upper() for name in PATH_NAMES])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(matrix[row, column]) > 0.55 * maximum else "black",
                )
    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.025,
    )
    colorbar.set_label(measure_label)
    figure.suptitle(
        f"Isolated-path {measure_label.lower()} | {QUERY_NAMES[query_offset]} query"
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_additive_logit_contributions(
    nonself_region_means: np.ndarray,
    blocks: Sequence[int],
    output_path: Path,
) -> None:
    grouped = _group_values(nonself_region_means).mean(axis=(0, 3))
    combined = grouped[..., 0] - grouped[..., 1]  # [block,path]
    directional = nonself_region_means.mean(axis=(0, 4))
    distractor = directional[..., 1, 1] - directional[..., 1, 0]
    figure, axes = plt.subplots(2, 1, figsize=(12.2, 8.0), sharex=True)
    for axis, values, title in (
        (axes[0], combined, "Combined same − different region-logit contrast"),
        (
            axes[1],
            distractor,
            "Distractor query: distractor − target region-logit contrast",
        ),
    ):
        positive_bottom = np.zeros(len(blocks))
        negative_bottom = np.zeros(len(blocks))
        for path_offset, path in enumerate(PATH_NAMES):
            path_values = values[:, path_offset]
            bottom = np.where(path_values >= 0, positive_bottom, negative_bottom)
            axis.bar(
                blocks,
                path_values,
                bottom=bottom,
                color=PATH_COLORS[path],
                label=PATH_LABELS[path],
                width=0.72,
            )
            positive_bottom += np.where(path_values >= 0, path_values, 0.0)
            negative_bottom += np.where(path_values < 0, path_values, 0.0)
        axis.plot(
            blocks,
            values.sum(axis=1),
            color="black",
            marker="o",
            linewidth=1.4,
            label="Sum",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("Signed logit contrast")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=5)
    axes[1].set_xticks(blocks)
    axes[1].set_xlabel("Transformer block")
    figure.suptitle("Exact additive Q–K path contributions (query patch excluded)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_path_logit_heatmap(
    nonself_region_means: np.ndarray,
    blocks: Sequence[int],
    output_path: Path,
) -> None:
    grouped = _group_values(nonself_region_means).mean(axis=(0, 3))
    combined = (grouped[..., 0] - grouped[..., 1]).T
    directional = nonself_region_means.mean(axis=(0, 4))
    distractor = (directional[..., 1, 1] - directional[..., 1, 0]).T
    maximum = max(
        float(np.percentile(np.abs(np.concatenate((combined, distractor))), 99)),
        1e-6,
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 5.9),
        sharex=True,
        layout="constrained",
    )
    for axis, matrix, title in (
        (axes[0], combined, "Combined same − different"),
        (axes[1], distractor, "Distractor query: distractor − target"),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
        )
        axis.set_yticks(np.arange(len(PATH_NAMES)), [name.upper() for name in PATH_NAMES])
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(matrix[row, column]) > 0.55 * maximum else "black",
                )
    axes[1].set_xticks(np.arange(len(blocks)), blocks)
    axes[1].set_xlabel("Transformer block")
    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.025,
    )
    colorbar.set_label("Signed region-logit contrast")
    figure.suptitle("Exact additive path contrasts (query patch excluded)")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_headwise_paths(
    nonself_enrichments: np.ndarray,
    blocks: Sequence[int],
    output_path: Path,
) -> None:
    path_offsets = [CONDITION_NAMES.index(path) for path in PATH_NAMES]
    values = nonself_enrichments[:, :, path_offsets]
    direct = values[..., 1, :, 1] - values[..., 1, :, 0]
    matrix = direct.mean(axis=0).transpose(1, 0, 2).reshape(len(PATH_NAMES), -1)
    maximum = max(float(np.percentile(np.abs(matrix), 99)), 1e-6)
    figure, axis = plt.subplots(figsize=(16.5, 3.6))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set_yticks(np.arange(len(PATH_NAMES)), [name.upper() for name in PATH_NAMES])
    head_count = direct.shape[-1]
    axis.set_xticks(
        np.arange(len(blocks) * head_count),
        [str(head + 1) for _ in blocks for head in range(head_count)],
        fontsize=5.5,
    )
    for block_offset in range(1, len(blocks)):
        axis.axvline(block_offset * head_count - 0.5, color="black", linewidth=1.0)
    for block_offset, block in enumerate(blocks):
        axis.text(
            block_offset * head_count + (head_count - 1) / 2,
            -0.8,
            f"B{block}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlabel("Attention head (one-based)")
    axis.set_title(
        "Head-wise non-self distractor-region margin for isolated paths"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    colorbar.set_label("Enrichment margin")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_decomposition_stats(
    decomposition_stats: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    blocks = [int(row["block"]) for row in decomposition_stats]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for component, marker in (("q", "o"), ("k", "s")):
        axes[0].plot(
            blocks,
            [float(row[f"{component}_sparse_density"]) for row in decomposition_stats],
            marker=marker,
            label=component.upper(),
        )
        axes[1].plot(
            blocks,
            [float(row[f"{component}_relative_s_norm"]) for row in decomposition_stats],
            marker=marker,
            label=component.upper(),
        )
    axes[0].set_ylabel("S nonzero fraction")
    axes[1].set_ylabel(r"$\|S\|_F / \|X\|_F$")
    for axis in axes:
        axis.set_xticks(blocks)
        axis.set_xlabel("Transformer block")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_title("Q/K sparse density")
    axes[1].set_title("Q/K sparse magnitude")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_representatives(
    dataset: O3Dataset,
    representative_indices: Sequence[int],
    blocks: Sequence[int],
    maps: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    query_indices: np.ndarray,
    patch_masks: np.ndarray,
    output_dir: Path,
) -> None:
    block_chunks = [blocks[start : start + 4] for start in range(0, len(blocks), 4)]
    for index in representative_indices:
        record = dataset.records[index]
        image = _load_display_image(dataset, index)
        stem = Path(record.image_name).stem
        for query_offset, query_name in enumerate(QUERY_NAMES):
            query_index = int(query_indices[index, query_offset])
            for chunk in block_chunks:
                figure, axes = plt.subplots(
                    len(chunk),
                    len(CONDITION_NAMES) + 1,
                    figsize=(2.55 * (len(CONDITION_NAMES) + 1), 2.55 * len(chunk)),
                    squeeze=False,
                )
                for row, block in enumerate(chunk):
                    axes[row, 0].imshow(
                        _annotated_image(
                            image,
                            patch_masks[index],
                            query_index,
                            query_name,
                        )
                    )
                    axes[row, 0].set_ylabel(f"Block {block}")
                    for column, condition in enumerate(CONDITION_NAMES, start=1):
                        attention = maps[index][block][condition][query_offset].mean(axis=0)
                        axes[row, column].imshow(_attention_overlay(image, attention))
                    for axis in axes[row]:
                        axis.set_xticks([])
                        axis.set_yticks([])
                axes[0, 0].set_title("Mask + query")
                for column, condition in enumerate(CONDITION_NAMES, start=1):
                    axes[0, column].set_title(CONDITION_LABELS[condition], fontsize=9)
                attributes = ", ".join(record.attributes) or "unspecified"
                figure.suptitle(
                    f"{record.image_name} | {record.target_type} | {attributes} | "
                    f"{query_name} query | head mean",
                    fontsize=11,
                )
                figure.tight_layout()
                figure.savefig(
                    output_dir
                    / (
                        f"sample_{stem}_{query_name}_blocks"
                        f"{chunk[0]:02d}-{chunk[-1]:02d}.png"
                    ),
                    dpi=170,
                    bbox_inches="tight",
                )
                plt.close(figure)


def _write_readme(
    output_dir: Path,
    image_count: int,
    blocks: Sequence[int],
    representative_names: Sequence[str],
    additivity_errors: Mapping[int, float],
    path_logit_summary: pd.DataFrame,
) -> None:
    lines = [
        "# O3 Q--K LL/SL/LS/SS path analysis",
        "",
        f"Images: {image_count}; blocks: {', '.join(map(str, blocks))}.",
        "",
        "For every block, upstream tokens come from the trained dense-X model. "
        "Only the current block's Q--K interaction changes. Paths are bias-free: "
        "LL = Q_L K_L^T, SL = Q_S K_L^T, LS = Q_L K_S^T, and SS = Q_S K_S^T.",
        "",
        "`softmax(Z_path)` is an isolated-path intervention and answers where a "
        "path attends on its own. These four probability maps are not additive. "
        "The signed region-logit contrasts are additive, and their sum equals the "
        "bias-free full L+S contrast.",
        "",
        "All headline distractor margins remove the selected query patch from "
        "the measurement masks. O3's distractor mask still aggregates the query "
        "object and other distractor instances, so this is a region-grouping test, "
        "not by itself a proof of cross-instance semantic correspondence.",
        "",
        "Heatmap overlays normalize each panel independently and should be used "
        "to compare spatial location, not absolute attention strength.",
        "",
        "Dense X and Full L+S retain the learned qkv bias. The four paths and "
        "their joint reference exclude it. Their exact sum therefore explains "
        "the weight-mediated interaction, but not the additional query-bias--key "
        "interaction present in the full model.",
        "",
        "## Main full-dataset result",
        "",
        "The table reports the signed, non-self distractor-region logit contrast: "
        "mean logit on the distractor region minus mean logit on the target "
        "region, averaged over 2001 images and 12 heads. Positive values promote "
        "same-region grouping; negative values promote the different target "
        "region. These four columns are exactly additive.",
        "",
        "| Block | LL | SL | LS | SS | Sum | Largest-magnitude path |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    headline = path_logit_summary[
        path_logit_summary["metric"] == "nonself_logit_distractor_contrast"
    ].pivot(index="block", columns="path", values="mean")
    for block in blocks:
        values = {path: float(headline.loc[block, path]) for path in PATH_NAMES}
        dominant = max(values, key=lambda path: abs(values[path])).upper()
        lines.append(
            "| {block} | {ll:+.3f} | {sl:+.3f} | {ls:+.3f} | {ss:+.3f} | "
            "{total:+.3f} | {dominant} |".format(
                block=block,
                total=sum(values.values()),
                dominant=dominant,
                **values,
            )
        )
    dominant_paths = [
        max(
            PATH_NAMES,
            key=lambda path: abs(float(headline.loc[block, path])),
        ).upper()
        for block in blocks
    ]
    lines.extend(
        [
            "",
            "Dominant path by absolute signed contrast, from the first to the "
            f"last analyzed block: {', '.join(dominant_paths)}. This is a "
            "descriptive result for the selected checkpoint and decomposition; "
            "the signs and magnitudes in the table carry the semantic comparison.",
            "",
        "## Additivity verification",
        "",
        "Maximum absolute logit error between direct Full(L+S, no bias) and "
        "LL+SL+LS+SS:",
        "",
        "| Block | Max absolute error |",
        "|---:|---:|",
        ]
    )
    for block in blocks:
        lines.append(f"| {block} | {additivity_errors[block]:.6g} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `isolated_path_*_query_{mass,enrichment}.png`: where each path "
            "attends when independently passed through softmax.",
            "- `isolated_path_*_query_nonself_enrichment.png`: the same regional "
            "allocation after removing the selected query patch.",
            "- `nonself_{distractor,combined}_margin.png`: reference and isolated-"
            "path attention behavior over all blocks.",
            "- `additive_logit_contributions.png`: signed, exactly additive path "
            "contributions.",
            "- `path_logit_contrast_heatmap.png`: the same contributions as a "
            "path-by-block matrix.",
            "- `headwise_path_distractor_margin.png`: every block and attention head.",
            "- `qk_sparse_stats.png`: layer-wise Q/K sparse density and magnitude.",
            "- `sample_*.png`: deterministic representative attention overlays.",
            "- `attention_summary.csv` and `path_logit_summary.csv`: bootstrap "
            "means and confidence intervals.",
            "- `attention_per_image.csv.gz`, `path_logit_per_image.csv.gz`, "
            "`raw_metrics.npz`, and `metadata.json`: complete numerical outputs.",
            "",
            "## Representative images",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in representative_names)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    blocks = tuple(sorted(set(args.blocks)))
    if not blocks or blocks[0] < 0 or blocks[-1] >= 12:
        raise ValueError("blocks must be non-empty and lie in [0, 11]")
    if blocks != tuple(range(blocks[-1] + 1)):
        raise ValueError("controlled dense forwarding requires consecutive blocks from 0")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap samples must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    dataset = O3Dataset(args.data_root, args.max_images)
    representative_indices = choose_representatives(
        dataset,
        min(args.representative_count, len(dataset)),
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    model = DinoViTBase8(attention_backend="sdpa")
    if all(key.startswith("backbone.") for key in state):
        model.load_state_dict(state, strict=True)
    else:
        model.backbone.load_state_dict(state, strict=True)
    decomposition_cpu = load_decompositions(matrix_dir, blocks)

    decomposition_stats: list[dict[str, object]] = []
    for block_index in blocks:
        x_weight = (
            model.backbone.blocks[block_index]
            .attn.qkv.weight.detach().cpu().float()
        )
        low_rank, sparse = decomposition_cpu[block_index]
        q_x, k_x, _ = x_weight.chunk(3, dim=0)
        q_sparse, k_sparse, _ = sparse.chunk(3, dim=0)
        row: dict[str, object] = {
            "block": block_index,
            "relative_l_plus_s_minus_x": float(
                (low_rank + sparse - x_weight).norm()
                / x_weight.norm().clamp_min(1e-12)
            ),
        }
        for name_part, sparse_part, x_part in (
            ("q", q_sparse, q_x),
            ("k", k_sparse, k_x),
        ):
            nonzero = int(torch.count_nonzero(sparse_part).item())
            row[f"{name_part}_sparse_nonzero"] = nonzero
            row[f"{name_part}_sparse_density"] = nonzero / sparse_part.numel()
            row[f"{name_part}_relative_s_norm"] = float(
                sparse_part.norm() / x_part.norm().clamp_min(1e-12)
            )
        decomposition_stats.append(row)

    model.eval().to(device)
    decomposition = {
        block: (low.to(device), sparse.to(device))
        for block, (low, sparse) in decomposition_cpu.items()
    }

    image_count = len(dataset)
    block_count = len(blocks)
    condition_count = len(CONDITION_NAMES)
    path_count = len(PATH_NAMES)
    head_count = model.backbone.blocks[0].attn.num_heads
    attention_shape = (
        image_count,
        block_count,
        condition_count,
        len(QUERY_NAMES),
        head_count,
        len(REGION_NAMES),
    )
    path_shape = (
        image_count,
        block_count,
        path_count,
        len(QUERY_NAMES),
        head_count,
        len(REGION_NAMES),
    )
    masses = np.full(attention_shape, np.nan, dtype=np.float32)
    nonself_masses = np.full(attention_shape, np.nan, dtype=np.float32)
    path_region_logits = np.full(path_shape, np.nan, dtype=np.float32)
    path_nonself_region_logits = np.full(path_shape, np.nan, dtype=np.float32)
    areas = np.full((image_count, len(REGION_NAMES)), np.nan, dtype=np.float32)
    nonself_areas = np.full(
        (image_count, len(QUERY_NAMES), len(REGION_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    query_indices = np.full((image_count, len(QUERY_NAMES)), -1, dtype=np.int64)
    patch_masks = np.full(
        (image_count, len(REGION_NAMES), PATCH_GRID, PATCH_GRID),
        np.nan,
        dtype=np.float32,
    )
    representative_maps: dict[int, dict[int, dict[str, np.ndarray]]] = {
        index: {} for index in representative_indices
    }
    additivity_errors = {block: 0.0 for block in blocks}
    dense_consistency_errors = {block: 0.0 for block in blocks}
    checked_blocks: set[int] = set()

    print(
        f"O3 Q-K paths | images={image_count} | blocks={blocks} | "
        f"batch={args.batch_size} | device={device}",
        flush=True,
    )
    for batch_number, batch in enumerate(loader, start=1):
        indices = batch["index"].numpy()
        images = batch["image"].to(device, non_blocking=True)
        batch_masks = batch["masks"].to(device, non_blocking=True)
        mask_flat = batch_masks.flatten(2)
        batch_queries = torch.stack(
            (mask_flat[:, 0].argmax(dim=-1), mask_flat[:, 1].argmax(dim=-1)),
            dim=1,
        )
        query_masks = mask_flat[:, None].expand(
            -1,
            len(QUERY_NAMES),
            -1,
            -1,
        ).clone()
        batch_indices = torch.arange(images.shape[0], device=device)[:, None]
        query_offsets = torch.arange(len(QUERY_NAMES), device=device)[None, :]
        query_masks[batch_indices, query_offsets, :, batch_queries] = 0.0

        areas[indices] = mask_flat.mean(dim=-1).cpu().numpy()
        nonself_areas[indices] = query_masks.mean(dim=-1).cpu().numpy()
        query_indices[indices] = batch_queries.cpu().numpy()
        patch_masks[indices] = batch_masks.cpu().numpy()

        tokens = model.backbone.prepare_tokens(images)
        for block_offset, block_index in enumerate(blocks):
            block = model.backbone.blocks[block_index]
            normalized = block.norm1(tokens)
            patches = normalized[:, 1:]
            low_rank, sparse = decomposition[block_index]
            logits_by_path = _path_logits(
                patches,
                batch_queries,
                low_rank,
                sparse,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            joint_logits = sum(logits_by_path.values())
            dense_logits, dense_attention = _project_attention(
                patches,
                batch_queries,
                block.attn.qkv.weight,
                block.attn.qkv.bias,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            full_logits, full_attention = _project_attention(
                patches,
                batch_queries,
                low_rank + sparse,
                block.attn.qkv.bias,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            conditions = {
                "dense_x": dense_attention,
                "full_l_plus_s": full_attention,
                "joint_no_bias": joint_logits.softmax(dim=-1),
                **{
                    path: logits_by_path[path].softmax(dim=-1)
                    for path in PATH_NAMES
                },
            }

            for condition_offset, condition in enumerate(CONDITION_NAMES):
                attention = conditions[condition]
                masses[indices, block_offset, condition_offset] = torch.einsum(
                    "bqhn,brn->bqhr",
                    attention,
                    mask_flat,
                ).cpu().numpy()
                nonself_masses[
                    indices,
                    block_offset,
                    condition_offset,
                ] = torch.einsum(
                    "bqhn,bqrn->bqhr",
                    attention,
                    query_masks,
                ).cpu().numpy()

            for path_offset, path in enumerate(PATH_NAMES):
                path_region_logits[
                    indices,
                    block_offset,
                    path_offset,
                ] = _region_mean(logits_by_path[path], mask_flat).cpu().numpy()
                path_nonself_region_logits[
                    indices,
                    block_offset,
                    path_offset,
                ] = _region_mean(logits_by_path[path], query_masks).cpu().numpy()

            if block_index not in checked_blocks:
                direct_no_bias_logits, _ = _project_attention(
                    patches,
                    batch_queries,
                    low_rank + sparse,
                    None,
                    num_heads=head_count,
                    scale=block.attn.scale,
                )
                additivity_error = float((direct_no_bias_logits - joint_logits).abs().max())
                additivity_errors[block_index] = additivity_error
                tolerance = 2e-4 + 2e-5 * float(direct_no_bias_logits.abs().max())
                if additivity_error > tolerance:
                    raise AssertionError(
                        f"Block {block_index} path logits are not additive: "
                        f"error={additivity_error:.6g}, tolerance={tolerance:.6g}"
                    )
                _, actual_attention = block.attn(normalized, return_attention=True)
                actual_patch_attention = actual_attention[:, :, 1:, 1:]
                actual_patch_attention = actual_patch_attention / actual_patch_attention.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                gathered = torch.gather(
                    actual_patch_attention,
                    dim=2,
                    index=batch_queries[:, None, :, None].expand(
                        -1,
                        head_count,
                        -1,
                        NUM_PATCHES,
                    ),
                ).permute(0, 2, 1, 3)
                consistency_error = float((gathered - dense_attention).abs().max())
                dense_consistency_errors[block_index] = consistency_error
                if consistency_error > 3e-5:
                    raise AssertionError(
                        f"Block {block_index} dense attention mismatch: "
                        f"{consistency_error:.6g}"
                    )
                checked_blocks.add(block_index)

            for local_offset, global_index in enumerate(indices.tolist()):
                if global_index not in representative_maps:
                    continue
                representative_maps[global_index][block_index] = {
                    condition: conditions[condition][local_offset]
                    .reshape(len(QUERY_NAMES), head_count, PATCH_GRID, PATCH_GRID)
                    .cpu()
                    .numpy()
                    for condition in CONDITION_NAMES
                }
            tokens = block(tokens)

        if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(loader):
            completed = min(batch_number * args.batch_size, image_count)
            print(
                f"  [{completed:4d}/{image_count}] "
                f"{100.0 * completed / image_count:5.1f}%",
                flush=True,
            )

    if np.isnan(masses).any() or np.isnan(path_region_logits).any():
        raise AssertionError("not every O3 sample received all path metrics")
    if not np.allclose(masses.sum(axis=-1), 1.0, atol=2e-4):
        raise AssertionError("attention region masses do not sum to one")

    enrichments = masses / np.maximum(
        areas[:, None, None, None, None, :],
        1e-12,
    )
    nonself_enrichments = nonself_masses / np.maximum(
        nonself_areas[:, None, None, :, None, :],
        1e-12,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "raw_metrics.npz",
        masses=masses,
        nonself_masses=nonself_masses,
        enrichments=enrichments,
        nonself_enrichments=nonself_enrichments,
        path_region_logits=path_region_logits,
        path_nonself_region_logits=path_nonself_region_logits,
        areas=areas,
        nonself_areas=nonself_areas,
        query_indices=query_indices,
        image_names=np.asarray([record.image_name for record in dataset.records]),
        blocks=np.asarray(blocks),
        conditions=np.asarray(CONDITION_NAMES),
        paths=np.asarray(PATH_NAMES),
        queries=np.asarray(QUERY_NAMES),
        regions=np.asarray(REGION_NAMES),
    )

    attention_per_image = pd.DataFrame(
        _attention_per_image_rows(
            dataset.records,
            blocks,
            masses,
            enrichments,
            nonself_enrichments,
        )
    )
    attention_per_image.to_csv(
        output_dir / "attention_per_image.csv.gz",
        index=False,
        compression="gzip",
    )
    attention_metrics = (
        "enrichment_grouping_margin",
        "enrichment_distractor_grouping_margin",
        "nonself_enrichment_grouping_margin",
        "nonself_enrichment_distractor_grouping_margin",
    )
    attention_summary = _summarize_dataframe(
        attention_per_image,
        ("block", "condition"),
        attention_metrics,
        args.bootstrap_samples,
        args.seed,
    )
    attention_summary.to_csv(output_dir / "attention_summary.csv", index=False)

    path_logit_per_image = pd.DataFrame(
        _path_logit_per_image_rows(
            dataset.records,
            blocks,
            path_region_logits,
            path_nonself_region_logits,
        )
    )
    path_logit_per_image.to_csv(
        output_dir / "path_logit_per_image.csv.gz",
        index=False,
        compression="gzip",
    )
    logit_metrics = (
        "logit_grouping_contrast",
        "logit_distractor_contrast",
        "nonself_logit_grouping_contrast",
        "nonself_logit_distractor_contrast",
    )
    path_logit_summary = _summarize_dataframe(
        path_logit_per_image,
        ("block", "path"),
        logit_metrics,
        args.bootstrap_samples,
        args.seed + 1,
    )
    path_logit_summary.to_csv(output_dir / "path_logit_summary.csv", index=False)

    _plot_attention_margins(
        attention_per_image,
        blocks,
        "nonself_enrichment_distractor_grouping_margin",
        output_dir / "nonself_distractor_margin.png",
        "O3 distractor-region grouping by current-block Q–K interaction",
        "Distractor→distractor − distractor→target enrichment",
        args.bootstrap_samples,
        args.seed,
    )
    _plot_attention_margins(
        attention_per_image,
        blocks,
        "nonself_enrichment_grouping_margin",
        output_dir / "nonself_combined_margin.png",
        "O3 combined same-versus-different grouping",
        "Same − different enrichment",
        args.bootstrap_samples,
        args.seed + 1,
    )
    for query_offset, query in enumerate(QUERY_NAMES):
        _plot_path_region_heatmap(
            masses,
            blocks,
            query_offset,
            output_dir / f"isolated_path_{query}_query_mass.png",
            "attention mass",
        )
        _plot_path_region_heatmap(
            enrichments,
            blocks,
            query_offset,
            output_dir / f"isolated_path_{query}_query_enrichment.png",
            "area-normalized enrichment",
        )
        _plot_path_region_heatmap(
            nonself_enrichments,
            blocks,
            query_offset,
            output_dir / f"isolated_path_{query}_query_nonself_enrichment.png",
            "non-self area-normalized enrichment",
        )
    _plot_additive_logit_contributions(
        path_nonself_region_logits,
        blocks,
        output_dir / "additive_logit_contributions.png",
    )
    _plot_path_logit_heatmap(
        path_nonself_region_logits,
        blocks,
        output_dir / "path_logit_contrast_heatmap.png",
    )
    _plot_headwise_paths(
        nonself_enrichments,
        blocks,
        output_dir / "headwise_path_distractor_margin.png",
    )
    _plot_decomposition_stats(
        decomposition_stats,
        output_dir / "qk_sparse_stats.png",
    )
    _plot_representatives(
        dataset,
        representative_indices,
        blocks,
        representative_maps,
        query_indices,
        patch_masks,
        output_dir,
    )

    metadata = {
        "experiment": "O3 Q-K LL/SL/LS/SS path analysis",
        "image_count": image_count,
        "blocks": list(blocks),
        "conditions": list(CONDITION_NAMES),
        "paths": {
            "ll": "Q_L K_L^T",
            "sl": "Q_S K_L^T",
            "ls": "Q_L K_S^T",
            "ss": "Q_S K_S^T",
        },
        "checkpoint": str(checkpoint),
        "matrix_dir": str(matrix_dir),
        "data_root": str(dataset.root),
        "device": str(device),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "upstream_tokens": "trained dense-X path",
        "path_bias": "excluded",
        "attention_scope": "selected patch queries to patch keys; CLS key excluded",
        "image_preprocessing": "direct bicubic resize to 224x224; ImageNet normalization",
        "mask_preprocessing": "threshold at 128 and area resize to 28x28",
        "nonself_control": "selected query patch removed from every measurement region",
        "additivity_max_absolute_error": additivity_errors,
        "dense_attention_consistency_max_absolute_error": dense_consistency_errors,
        "decomposition": decomposition_stats,
        "representative_indices": representative_indices,
        "representative_images": [
            dataset.records[index].image_name for index in representative_indices
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir,
        image_count,
        blocks,
        metadata["representative_images"],
        additivity_errors,
        path_logit_summary,
    )

    print("\nMean non-self distractor-region enrichment margin", flush=True)
    headline = attention_summary[
        attention_summary["metric"]
        == "nonself_enrichment_distractor_grouping_margin"
    ]
    pivot = headline.pivot(index="block", columns="condition", values="mean")
    print(pivot.loc[list(blocks), list(CONDITION_NAMES)].to_string(), flush=True)
    print("\nMean signed non-self distractor logit contrast", flush=True)
    logit_headline = path_logit_summary[
        path_logit_summary["metric"] == "nonself_logit_distractor_contrast"
    ]
    logit_pivot = logit_headline.pivot(index="block", columns="path", values="mean")
    print(logit_pivot.loc[list(blocks), list(PATH_NAMES)].to_string(), flush=True)
    print(f"\nResults written to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
