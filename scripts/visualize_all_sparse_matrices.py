#!/usr/bin/env python3
"""Render every SALAAD ViT sparse matrix after an explicit zero mask.

The script reads the trusted ``matrix_rank<N>.pkl`` shards produced by the
SALAAD trainer and extracts the ``S`` matrix for the four decomposed Linear
layers in every requested transformer block:

* ``attn.qkv``
* ``attn.proj``
* ``mlp.fc1``
* ``mlp.fc2``

Heatmaps use the native PyTorch weight orientation ``[output, input]``.  Q, K,
and V therefore appear as three vertically stacked regions in the QKV panel.
The source pickle files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch
from torch import Tensor


LAYER_TYPES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
LAYER_TITLES = {
    "attn.qkv": "Attention QKV S (Q/K/V stacked)",
    "attn.proj": "Attention projection S",
    "mlp.fc1": "MLP FC1 S",
    "mlp.fc2": "MLP FC2 S",
}
LAYER_SLUGS = {
    "attn.qkv": "attn_qkv",
    "attn.proj": "attn_proj",
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
}
LAYER_PATTERN = re.compile(
    r"^backbone\.blocks\.(?P<block>\d+)\."
    r"(?P<layer>attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$"
)
RANK_FILE_PATTERN = re.compile(r"^matrix_rank(?P<rank>\d+)\.pkl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Directory containing matrix_rank<N>.pkl shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to write the heatmaps and summaries.",
    )
    parser.add_argument(
        "--run-label",
        default="SALAAD",
        help="Short experiment label shown in titles and metadata.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Zero-based transformer blocks to render (default: 0..11).",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-4,
        help="Values satisfying abs(S) < threshold are displayed as zero.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.9,
        choices=(99.0, 99.9, 100.0),
        help="Per-matrix |S| percentile used for column-wise color clipping.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def matrix_rank(path: Path) -> int:
    match = RANK_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected matrix shard name: {path.name}")
    return int(match.group("rank"))


def load_pickle_on_cpu(path: Path) -> Mapping[str, object]:
    """Load one trusted training pickle while remapping tensors to the CPU."""
    original_loader = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda value: torch.load(
            io.BytesIO(value), map_location="cpu", weights_only=False
        )
        with path.open("rb") as input_file:
            payload = pickle.load(input_file)
    finally:
        torch.storage._load_from_bytes = original_loader
    if not isinstance(payload, Mapping):
        raise TypeError(f"matrix shard must contain a mapping: {path}")
    return payload


def load_sparse_matrices(
    matrix_dir: Path,
    blocks: Sequence[int],
) -> dict[tuple[int, str], Tensor]:
    matrix_dir = matrix_dir.expanduser().resolve()
    if not matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {matrix_dir}")
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    ranks = [matrix_rank(path) for path in matrix_files]
    if ranks != list(range(len(ranks))):
        raise ValueError(f"matrix rank files must be contiguous from zero, got {ranks}")

    block_set = set(blocks)
    matrices: dict[tuple[int, str], Tensor] = {}
    for path in matrix_files:
        print(f"loading {path.name}", flush=True)
        payload = load_pickle_on_cpu(path)
        sparse = payload.get("SS")
        if not isinstance(sparse, Mapping):
            raise TypeError(f"matrix shard requires an SS mapping: {path}")
        for layer_name, value in sparse.items():
            if not isinstance(layer_name, str):
                raise TypeError(f"SS keys must be strings: {path}")
            match = LAYER_PATTERN.fullmatch(layer_name)
            if match is None:
                continue
            block = int(match.group("block"))
            layer_type = match.group("layer")
            if block not in block_set:
                continue
            key = (block, layer_type)
            if key in matrices:
                raise ValueError(f"duplicate sparse matrix: {layer_name}")
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise TypeError(f"SS value must be a 2-D tensor: {layer_name}")
            matrix = value.detach().cpu().float().contiguous()
            if not torch.isfinite(matrix).all():
                raise ValueError(f"non-finite S values in {layer_name}")
            matrices[key] = matrix
        del sparse
        del payload
        gc.collect()

    expected = {
        (block, layer_type) for block in block_set for layer_type in LAYER_TYPES
    }
    missing = sorted(expected - set(matrices))
    unexpected = sorted(set(matrices) - expected)
    if missing:
        raise ValueError(f"missing sparse matrices: {missing}")
    if unexpected:
        raise ValueError(f"unexpected sparse matrices: {unexpected}")
    return matrices


def apply_zero_threshold(
    matrices: Mapping[tuple[int, str], Tensor], threshold: float
) -> dict[tuple[int, str], int]:
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("zero threshold must be finite and non-negative")
    removed: dict[tuple[int, str], int] = {}
    for key, matrix in matrices.items():
        mask = (matrix != 0.0) & (matrix.abs() < threshold)
        removed[key] = int(mask.count_nonzero().item())
        matrix[mask] = 0.0
    return removed


def matrix_statistics(
    block: int,
    layer_type: str,
    matrix: Tensor,
    removed: int,
) -> dict[str, object]:
    values = matrix.numpy().reshape(-1)
    nonzero = values[values != 0.0]
    absolute = np.abs(nonzero)
    elements = int(values.size)
    nonzero_count = int(nonzero.size)
    if nonzero_count:
        quantiles = np.quantile(absolute, (0.01, 0.50, 0.95, 0.99, 0.999, 1.0))
        signed_min = float(nonzero.min())
        signed_max = float(nonzero.max())
    else:
        quantiles = np.full(6, np.nan)
        signed_min = math.nan
        signed_max = math.nan
    return {
        "block": block,
        "layer_type": layer_type,
        "layer": f"backbone.blocks.{block}.{layer_type}",
        "shape_out": int(matrix.shape[0]),
        "shape_in": int(matrix.shape[1]),
        "elements": elements,
        "nonzero": nonzero_count,
        "zeros": elements - nonzero_count,
        "density": nonzero_count / elements,
        "removed_by_threshold": removed,
        "positive": int(np.count_nonzero(nonzero > 0.0)),
        "negative": int(np.count_nonzero(nonzero < 0.0)),
        "mean_nonzero": (
            float(np.mean(nonzero, dtype=np.float64)) if nonzero_count else math.nan
        ),
        "std_nonzero": (
            float(np.std(nonzero, dtype=np.float64)) if nonzero_count else math.nan
        ),
        "abs_q01": float(quantiles[0]),
        "abs_q50": float(quantiles[1]),
        "abs_q95": float(quantiles[2]),
        "abs_q99": float(quantiles[3]),
        "abs_q99_9": float(quantiles[4]),
        "abs_max": float(quantiles[5]),
        "min": signed_min,
        "max": signed_max,
    }


def color_scales(
    statistics: Sequence[dict[str, object]], percentile: float, threshold: float
) -> dict[str, dict[str, float]]:
    percentile_key = {99.0: "abs_q99", 99.9: "abs_q99_9", 100.0: "abs_max"}[
        percentile
    ]
    scales: dict[str, dict[str, float]] = {}
    for layer_type in LAYER_TYPES:
        rows = [row for row in statistics if row["layer_type"] == layer_type]
        vmax_candidates = [
            float(row[percentile_key])
            for row in rows
            if math.isfinite(float(row[percentile_key]))
            and float(row[percentile_key]) > 0.0
        ]
        q01 = [
            float(row["abs_q01"])
            for row in rows
            if math.isfinite(float(row["abs_q01"])) and float(row["abs_q01"]) > 0
        ]
        if not vmax_candidates:
            # An aggressive display threshold can legitimately remove every
            # saved value of one layer type.  Keep those panels renderable so
            # the all-zero support remains visible instead of aborting the run.
            vmax = max(threshold, float(np.finfo(np.float32).eps))
            scales[layer_type] = {
                "vmax": vmax,
                "linthresh": vmax * 0.1,
                "all_zero": True,
            }
            continue
        vmax = max(vmax_candidates)
        linthresh = float(np.median(q01)) if q01 else max(threshold, vmax * 1e-3)
        linthresh = min(max(linthresh, vmax * 1e-6), vmax * 0.1)
        scales[layer_type] = {
            "vmax": vmax,
            "linthresh": linthresh,
            "all_zero": False,
        }
    return scales


def make_norm(scale: Mapping[str, float]) -> SymLogNorm:
    vmax = float(scale["vmax"])
    return SymLogNorm(
        linthresh=float(scale["linthresh"]),
        linscale=1.0,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )


def add_qkv_boundaries(axis: plt.Axes, matrix: Tensor) -> None:
    if matrix.shape[0] % 3 != 0:
        return
    section = matrix.shape[0] // 3
    axis.axhline(section - 0.5, color="black", linewidth=0.35, alpha=0.7)
    axis.axhline(2 * section - 0.5, color="black", linewidth=0.35, alpha=0.7)
    for label, fraction in zip(("Q", "K", "V"), (1 / 6, 1 / 2, 5 / 6)):
        axis.text(
            0.01,
            1.0 - fraction,
            label,
            transform=axis.transAxes,
            fontsize=7,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )


def plot_overview(
    path: Path,
    matrices: Mapping[tuple[int, str], Tensor],
    statistics: Sequence[dict[str, object]],
    scales: Mapping[str, Mapping[str, float]],
    blocks: Sequence[int],
    threshold: float,
    percentile: float,
    dpi: int,
    run_label: str,
) -> None:
    stat_by_key = {
        (int(row["block"]), str(row["layer_type"])): row for row in statistics
    }
    figure, axes = plt.subplots(
        len(blocks),
        len(LAYER_TYPES),
        figsize=(24, max(8.0, 2.2 * len(blocks))),
        squeeze=False,
        constrained_layout=True,
    )
    images: dict[str, object] = {}
    for row_index, block in enumerate(blocks):
        for column_index, layer_type in enumerate(LAYER_TYPES):
            axis = axes[row_index, column_index]
            matrix = matrices[(block, layer_type)]
            image = axis.imshow(
                matrix.numpy(),
                cmap="RdBu_r",
                norm=make_norm(scales[layer_type]),
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            images[layer_type] = image
            if layer_type == "attn.qkv":
                add_qkv_boundaries(axis, matrix)
            stats = stat_by_key[(block, layer_type)]
            axis.text(
                0.99,
                0.02,
                f"density {100.0 * float(stats['density']):.2f}%",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
            if row_index == 0:
                axis.set_title(LAYER_TITLES[layer_type], fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"Block {block}\noutput channel", fontsize=8)
            if row_index == len(blocks) - 1:
                axis.set_xlabel("input channel", fontsize=8)
            axis.tick_params(labelsize=5, length=2)
    for column_index, layer_type in enumerate(LAYER_TYPES):
        figure.colorbar(
            images[layer_type],
            ax=axes[:, column_index],
            shrink=0.55,
            pad=0.01,
            label=f"S ({percentile:g}th-percentile clip per layer type)",
        )
    figure.suptitle(
        f"{run_label}: every thresholded sparse matrix S\n"
        f"native [output,input] orientation; |S| < {threshold:g} shown as exact zero",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_block_panels(
    output_dir: Path,
    matrices: Mapping[tuple[int, str], Tensor],
    statistics: Sequence[dict[str, object]],
    scales: Mapping[str, Mapping[str, float]],
    blocks: Sequence[int],
    threshold: float,
    dpi: int,
    run_label: str,
) -> None:
    stat_by_key = {
        (int(row["block"]), str(row["layer_type"])): row for row in statistics
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        figure, axes = plt.subplots(
            2, 4, figsize=(24, 9), squeeze=False, constrained_layout=True
        )
        for column_index, layer_type in enumerate(LAYER_TYPES):
            matrix = matrices[(block, layer_type)]
            stats = stat_by_key[(block, layer_type)]
            image = axes[0, column_index].imshow(
                matrix.numpy(),
                cmap="RdBu_r",
                norm=make_norm(scales[layer_type]),
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            if layer_type == "attn.qkv":
                add_qkv_boundaries(axes[0, column_index], matrix)
            axes[0, column_index].set_title(
                f"{LAYER_TITLES[layer_type]}\n"
                f"density={100.0 * float(stats['density']):.3f}%, "
                f"median |S|={float(stats['abs_q50']):.3g}",
                fontsize=9,
            )
            axes[1, column_index].imshow(
                matrix.numpy() != 0.0,
                cmap="Greys",
                vmin=0,
                vmax=1,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            if layer_type == "attn.qkv":
                add_qkv_boundaries(axes[1, column_index], matrix)
            axes[1, column_index].set_title("support (black = nonzero)", fontsize=9)
            axes[1, column_index].set_xlabel("input channel")
            for row_index in range(2):
                axes[row_index, column_index].set_ylabel("output channel")
                axes[row_index, column_index].tick_params(labelsize=6)
            figure.colorbar(
                image,
                ax=axes[0, column_index],
                shrink=0.72,
                pad=0.01,
                label="S",
            )
        figure.suptitle(
            f"{run_label} block {block}: S values and support | "
            f"|S| < {threshold:g} is zero",
            fontsize=13,
        )
        figure.savefig(
            output_dir / f"block{block:02d}_all_s_heatmaps.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_individual_matrices(
    output_dir: Path,
    matrices: Mapping[tuple[int, str], Tensor],
    statistics: Sequence[dict[str, object]],
    scales: Mapping[str, Mapping[str, float]],
    blocks: Sequence[int],
    threshold: float,
    dpi: int,
    run_label: str,
) -> None:
    stat_by_key = {
        (int(row["block"]), str(row["layer_type"])): row for row in statistics
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        for layer_type in LAYER_TYPES:
            matrix = matrices[(block, layer_type)]
            stats = stat_by_key[(block, layer_type)]
            figure, axes = plt.subplots(
                1, 2, figsize=(15, 6.5), constrained_layout=True
            )
            image = axes[0].imshow(
                matrix.numpy(),
                cmap="RdBu_r",
                norm=make_norm(scales[layer_type]),
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            axes[1].imshow(
                matrix.numpy() != 0.0,
                cmap="Greys",
                vmin=0,
                vmax=1,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            if layer_type == "attn.qkv":
                add_qkv_boundaries(axes[0], matrix)
                add_qkv_boundaries(axes[1], matrix)
            axes[0].set_title("signed S values")
            axes[1].set_title("support (black = nonzero)")
            for axis in axes:
                axis.set_xlabel("input channel")
                axis.set_ylabel("output channel")
                axis.tick_params(labelsize=7)
            figure.colorbar(image, ax=axes[0], shrink=0.8, label="S")
            figure.suptitle(
                f"{run_label} | block {block} {LAYER_TITLES[layer_type]} | "
                f"shape={matrix.shape[0]}×{matrix.shape[1]}, "
                f"density={100.0 * float(stats['density']):.4f}% | "
                f"|S| < {threshold:g} is zero",
                fontsize=12,
            )
            figure.savefig(
                output_dir
                / f"block{block:02d}_{LAYER_SLUGS[layer_type]}_s_heatmap.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    blocks = sorted(set(args.blocks))
    if not blocks or blocks[0] < 0 or blocks[-1] > 11:
        raise ValueError("blocks must be a non-empty subset of 0..11")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label.replace("_", " ")

    matrices = load_sparse_matrices(args.matrix_dir, blocks)
    removed = apply_zero_threshold(matrices, args.zero_threshold)
    statistics = [
        matrix_statistics(
            block,
            layer_type,
            matrices[(block, layer_type)],
            removed[(block, layer_type)],
        )
        for block in blocks
        for layer_type in LAYER_TYPES
    ]
    scales = color_scales(statistics, args.clip_percentile, args.zero_threshold)

    write_csv(output_dir / "s_matrix_statistics.csv", statistics)
    print("rendering all-block overview", flush=True)
    plot_overview(
        output_dir / "all_blocks_all_s_heatmaps.png",
        matrices,
        statistics,
        scales,
        blocks,
        args.zero_threshold,
        args.clip_percentile,
        args.dpi,
        run_label,
    )
    print("rendering per-block panels", flush=True)
    plot_block_panels(
        output_dir / "blocks",
        matrices,
        statistics,
        scales,
        blocks,
        args.zero_threshold,
        args.dpi,
        run_label,
    )
    print("rendering individual matrix panels", flush=True)
    plot_individual_matrices(
        output_dir / "layers",
        matrices,
        statistics,
        scales,
        blocks,
        args.zero_threshold,
        args.dpi,
        run_label,
    )

    metadata = {
        "run_label": run_label,
        "source_matrix_dir": str(args.matrix_dir.expanduser().resolve()),
        "blocks": blocks,
        "layer_types": list(LAYER_TYPES),
        "matrix_count": len(matrices),
        "zero_rule": "abs(S) < threshold is set to zero; equality is retained",
        "zero_threshold": args.zero_threshold,
        "removed_saved_nonzero_values_total": sum(removed.values()),
        "removed_saved_nonzero_values": {
            f"block{block:02d}_{LAYER_SLUGS[layer_type]}": count
            for (block, layer_type), count in sorted(removed.items())
        },
        "orientation": "native weight layout [output channel, input channel]",
        "clip_percentile": args.clip_percentile,
        "color_scales_by_layer_type": scales,
        "statistics": statistics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"# {run_label} sparse-matrix heatmaps\n\n"
        f"All 48 sparse matrices use the strict rule `|S| < {args.zero_threshold:g}` "
        "→ zero. Values exactly equal to the threshold are retained. Source shards "
        "are unchanged.\n\n"
        "- `all_blocks_all_s_heatmaps.png`: 12 blocks × 4 Linear layers.\n"
        "- `blocks/`: one signed-value/support comparison per block.\n"
        "- `layers/`: one signed-value/support comparison per matrix.\n"
        "- `s_matrix_statistics.csv`: exact statistics after thresholding.\n"
        "- `metadata.json`: source, threshold, scales, and removal counts.\n",
        encoding="utf-8",
    )
    print(
        f"rendered {len(matrices)} matrices; removed "
        f"{sum(removed.values()):,} saved nonzero values; output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
