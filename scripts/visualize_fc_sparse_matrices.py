#!/usr/bin/env python3
"""Visualize and quantify the saved SALAAD sparse FC matrices.

The script reads the trusted ``matrix_rank<N>.pkl`` files written by SALAAD
training, extracts ``mlp.fc1.S`` and ``mlp.fc2.S`` for every selected ViT
block, and writes heatmaps plus machine-readable distribution summaries.

For visualization, FC1 is transposed from ``[hidden, embedding]`` to
``[embedding, hidden]``. FC2 is already ``[embedding, hidden]``. The aligned
plots therefore share an x-axis of MLP hidden units and a y-axis of embedding
channels.
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


LAYER_PATTERN = re.compile(
    r"^backbone\.blocks\.(?P<block>\d+)\.mlp\.(?P<fc>fc1|fc2)$"
)
RANK_FILE_PATTERN = re.compile(r"^matrix_rank(?P<rank>\d+)\.pkl$")
FC_TYPES = ("fc1", "fc2")
SIGNED_QUANTILES = (
    ("q0", 0.0),
    ("q1", 0.01),
    ("q5", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
    ("q99_9", 0.999),
    ("q100", 1.0),
)
ABS_QUANTILES = SIGNED_QUANTILES[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Directory containing matrix_rank<N>.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to write plots and tables.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Zero-based transformer blocks to include (default: all 0..11).",
    )
    parser.add_argument(
        "--histogram-bins-per-sign",
        type=int,
        default=96,
        help="Log-spaced histogram bins on each side of zero.",
    )
    parser.add_argument(
        "--heatmap-percentile",
        type=float,
        default=99.9,
        help="Per-matrix absolute percentile used to derive the shared clip.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Treat values with abs(S) <= this threshold as exact zero.",
    )
    return parser.parse_args()


def _matrix_rank(path: Path) -> int:
    match = RANK_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected matrix filename: {path.name}")
    return int(match.group("rank"))


def _load_pickle_on_cpu(path: Path) -> Mapping[str, object]:
    """Load one trusted SALAAD training pickle while remapping tensors to CPU."""
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
        raise TypeError(f"matrix file must contain a mapping: {path}")
    return payload


def load_fc_sparse_matrices(
    matrix_dir: Path,
    blocks: Sequence[int],
) -> dict[tuple[int, str], Tensor]:
    matrix_dir = matrix_dir.expanduser().resolve()
    if not matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {matrix_dir}")
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=_matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    ranks = [_matrix_rank(path) for path in matrix_files]
    if ranks != list(range(len(ranks))):
        raise ValueError(f"matrix rank files must be contiguous from zero, got {ranks}")

    block_set = set(blocks)
    matrices: dict[tuple[int, str], Tensor] = {}
    for matrix_file in matrix_files:
        payload = _load_pickle_on_cpu(matrix_file)
        sparse = payload.get("SS")
        if not isinstance(sparse, Mapping):
            raise TypeError(f"matrix file requires an SS mapping: {matrix_file}")
        for layer_name, value in sparse.items():
            if not isinstance(layer_name, str):
                raise TypeError(f"SS layer names must be strings: {matrix_file}")
            match = LAYER_PATTERN.fullmatch(layer_name)
            if match is None:
                continue
            block = int(match.group("block"))
            fc_type = match.group("fc")
            if block not in block_set:
                continue
            key = (block, fc_type)
            if key in matrices:
                raise ValueError(f"duplicate sparse matrix for {layer_name}")
            if not isinstance(value, Tensor):
                raise TypeError(f"SS value must be a tensor for {layer_name}")
            value = value.detach().cpu().float().contiguous()
            if value.ndim != 2:
                raise ValueError(
                    f"FC sparse matrix must be 2-D for {layer_name}, got {value.shape}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite sparse values in {layer_name}")
            matrices[key] = value
        del sparse
        del payload
        gc.collect()

    expected = {(block, fc_type) for block in block_set for fc_type in FC_TYPES}
    missing = sorted(expected - set(matrices))
    if missing:
        raise ValueError(f"missing FC sparse matrices: {missing}")
    return matrices


def apply_zero_threshold(
    matrices: Mapping[tuple[int, str], Tensor],
    threshold: float,
) -> dict[tuple[int, str], int]:
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("zero threshold must be a finite non-negative number")
    removed_counts: dict[tuple[int, str], int] = {}
    for key, matrix in matrices.items():
        removed = (matrix != 0.0) & (matrix.abs() <= threshold)
        removed_counts[key] = int(torch.count_nonzero(removed).item())
        matrix[removed] = 0.0
    return removed_counts


def _safe_quantiles(values: np.ndarray, definitions: Sequence[tuple[str, float]]) -> dict[str, float]:
    if values.size == 0:
        return {name: math.nan for name, _ in definitions}
    probabilities = [probability for _, probability in definitions]
    quantiles = np.quantile(values, probabilities)
    return {
        name: float(value)
        for (name, _), value in zip(definitions, quantiles)
    }


def matrix_statistics(block: int, fc_type: str, matrix: Tensor) -> dict[str, object]:
    values = matrix.numpy().reshape(-1)
    nonzero = values[values != 0.0]
    absolute_nonzero = np.abs(nonzero)
    elements = int(values.size)
    nonzero_count = int(nonzero.size)
    positive_count = int(np.count_nonzero(nonzero > 0.0))
    negative_count = int(np.count_nonzero(nonzero < 0.0))

    row: dict[str, object] = {
        "block": block,
        "fc": fc_type,
        "layer": f"backbone.blocks.{block}.mlp.{fc_type}",
        "shape_out": int(matrix.shape[0]),
        "shape_in": int(matrix.shape[1]),
        "elements": elements,
        "nonzero": nonzero_count,
        "zeros": elements - nonzero_count,
        "density": nonzero_count / elements,
        "zero_fraction": 1.0 - nonzero_count / elements,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction_all": positive_count / elements,
        "negative_fraction_all": negative_count / elements,
        "positive_fraction_nonzero": (
            positive_count / nonzero_count if nonzero_count else math.nan
        ),
        "negative_fraction_nonzero": (
            negative_count / nonzero_count if nonzero_count else math.nan
        ),
        "mean_all": float(np.mean(values, dtype=np.float64)),
        "std_all": float(np.std(values, dtype=np.float64)),
        "rms_all": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "min_all": float(values.min()),
        "max_all": float(values.max()),
        "mean_nonzero": (
            float(np.mean(nonzero, dtype=np.float64)) if nonzero_count else math.nan
        ),
        "std_nonzero": (
            float(np.std(nonzero, dtype=np.float64)) if nonzero_count else math.nan
        ),
        "rms_nonzero": (
            float(np.sqrt(np.mean(np.square(nonzero), dtype=np.float64)))
            if nonzero_count
            else math.nan
        ),
        "abs_mean_nonzero": (
            float(np.mean(absolute_nonzero, dtype=np.float64))
            if nonzero_count
            else math.nan
        ),
    }
    for name, value in _safe_quantiles(nonzero, SIGNED_QUANTILES).items():
        row[f"value_{name}"] = value
    for name, value in _safe_quantiles(absolute_nonzero, ABS_QUANTILES).items():
        row[f"abs_{name}"] = value
    return row


def write_statistics_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aligned_matrix(matrix: Tensor, fc_type: str) -> np.ndarray:
    values = matrix.numpy()
    return values.T if fc_type == "fc1" else values


def heatmap_scale(
    statistics: Sequence[dict[str, object]],
    percentile: float,
) -> tuple[float, float]:
    percentile_key = {
        99.0: "abs_q99",
        99.9: "abs_q99_9",
        100.0: "abs_q100",
    }.get(float(percentile))
    if percentile_key is None:
        raise ValueError("heatmap percentile must be one of 99, 99.9, or 100")
    vmax = max(float(row[percentile_key]) for row in statistics)
    if not math.isfinite(vmax) or vmax <= 0.0:
        raise ValueError("cannot create a heatmap scale from all-zero matrices")
    q1_values = [
        float(row["abs_q1"])
        for row in statistics
        if math.isfinite(float(row["abs_q1"])) and float(row["abs_q1"]) > 0.0
    ]
    linthresh = float(np.median(q1_values)) if q1_values else vmax * 1e-3
    linthresh = min(max(linthresh, vmax * 1e-6), vmax * 0.1)
    return vmax, linthresh


def plot_heatmaps(
    output_dir: Path,
    matrices: Mapping[tuple[int, str], Tensor],
    statistics: Sequence[dict[str, object]],
    blocks: Sequence[int],
    percentile: float,
    zero_threshold: float,
    dpi: int,
) -> dict[str, float]:
    stat_by_key = {(int(row["block"]), str(row["fc"])): row for row in statistics}
    vmax, linthresh = heatmap_scale(statistics, percentile)
    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    figure, axes = plt.subplots(
        len(blocks),
        2,
        figsize=(18, max(4.0, 2.25 * len(blocks))),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for row_index, block in enumerate(blocks):
        for column_index, fc_type in enumerate(FC_TYPES):
            axis = axes[row_index, column_index]
            values = aligned_matrix(matrices[(block, fc_type)], fc_type)
            last_image = axis.imshow(
                values,
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            stats = stat_by_key[(block, fc_type)]
            axis.set_title(
                f"Block {block} {fc_type.upper()} S"
                + ("ᵀ" if fc_type == "fc1" else "")
                + f" | density={100.0 * float(stats['density']):.2f}%"
                + f" | median |S|={float(stats['abs_q50']):.3g}",
                fontsize=8,
            )
            axis.set_ylabel("embedding channel", fontsize=7)
            if row_index == len(blocks) - 1:
                axis.set_xlabel("MLP hidden unit", fontsize=7)
            axis.tick_params(labelsize=6)
    if last_image is not None:
        figure.colorbar(
            last_image,
            ax=axes,
            shrink=0.6,
            label=f"S value (symmetric log; shared {percentile:g}th-percentile clip)",
        )
    figure.suptitle(
        "Mixed-rho SALAAD FC sparse matrices\n"
        "FC1 is transposed so both columns share embedding × MLP-hidden coordinates"
        f" | |S| ≤ {zero_threshold:.3g} treated as zero",
        fontsize=13,
    )
    overview_path = output_dir / "fc_sparse_heatmaps_all_blocks.png"
    figure.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    block_dir = output_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        figure, axes = plt.subplots(2, 2, figsize=(18, 7), constrained_layout=True)
        last_image = None
        for column_index, fc_type in enumerate(FC_TYPES):
            values = aligned_matrix(matrices[(block, fc_type)], fc_type)
            stats = stat_by_key[(block, fc_type)]
            last_image = axes[0, column_index].imshow(
                values,
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            axes[0, column_index].set_title(
                f"{fc_type.upper()} S"
                + ("ᵀ" if fc_type == "fc1" else "")
                + f" | density={100.0 * float(stats['density']):.3f}%"
                + f" | |S| q50/q99={float(stats['abs_q50']):.3g}/"
                + f"{float(stats['abs_q99']):.3g}",
                fontsize=10,
            )
            axes[1, column_index].imshow(
                values != 0.0,
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            axes[1, column_index].set_title(
                f"{fc_type.upper()} support (black = nonzero)", fontsize=10
            )
            axes[1, column_index].set_xlabel("MLP hidden unit")
            for row_index in range(2):
                axes[row_index, column_index].set_ylabel("embedding channel")
                axes[row_index, column_index].tick_params(labelsize=7)
        if last_image is not None:
            figure.colorbar(
                last_image,
                ax=axes[0, :],
                shrink=0.8,
                label="S value (shared symmetric-log scale)",
            )
        figure.suptitle(
            f"Transformer block {block}: FC sparse values and support"
            f" | |S| ≤ {zero_threshold:.3g} treated as zero"
        )
        figure.savefig(
            block_dir / f"block{block:02d}_fc_sparse_heatmaps.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)

    return {"vmax": vmax, "linthresh": linthresh}


def histogram_edges(
    matrices: Mapping[tuple[int, str], Tensor],
    bins_per_sign: int,
) -> np.ndarray:
    if bins_per_sign < 8:
        raise ValueError("histogram bins per sign must be at least 8")
    absolute_min = math.inf
    absolute_max = 0.0
    for matrix in matrices.values():
        values = matrix.numpy().reshape(-1)
        absolute = np.abs(values[values != 0.0])
        if absolute.size:
            absolute_min = min(absolute_min, float(absolute.min()))
            absolute_max = max(absolute_max, float(absolute.max()))
    if not math.isfinite(absolute_min) or absolute_max <= 0.0:
        raise ValueError("cannot construct nonzero histograms from all-zero matrices")
    if absolute_min == absolute_max:
        absolute_min *= 0.5
        absolute_max *= 1.5
    positive = np.geomspace(absolute_min, absolute_max, bins_per_sign + 1)
    return np.concatenate((-positive[::-1], np.asarray([0.0]), positive))


def histogram_rows(
    matrices: Mapping[tuple[int, str], Tensor],
    blocks: Sequence[int],
    edges: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    counts_by_key: dict[tuple[int, str], np.ndarray] = {}
    for block in blocks:
        for fc_type in FC_TYPES:
            values = matrices[(block, fc_type)].numpy().reshape(-1)
            nonzero = values[values != 0.0]
            counts, _ = np.histogram(nonzero, bins=edges)
            if int(counts.sum()) != int(nonzero.size):
                raise RuntimeError(f"histogram lost values for block {block} {fc_type}")
            counts_by_key[(block, fc_type)] = counts
            denominator = max(int(nonzero.size), 1)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "block": block,
                        "fc": fc_type,
                        "bin_index": index,
                        "left": float(edges[index]),
                        "right": float(edges[index + 1]),
                        "count": int(count),
                        "probability_nonzero": int(count) / denominator,
                    }
                )
    return rows, counts_by_key


def write_histogram_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _histogram_centers(edges: np.ndarray) -> np.ndarray:
    centers = np.empty(edges.size - 1, dtype=np.float64)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if left < 0.0 and right < 0.0:
            centers[index] = -math.sqrt(abs(left * right))
        elif left > 0.0 and right > 0.0:
            centers[index] = math.sqrt(left * right)
        else:
            centers[index] = 0.5 * (left + right)
    return centers


def plot_distributions(
    output_dir: Path,
    statistics: Sequence[dict[str, object]],
    blocks: Sequence[int],
    edges: np.ndarray,
    counts_by_key: Mapping[tuple[int, str], np.ndarray],
    zero_threshold: float,
    dpi: int,
) -> None:
    stat_by_key = {(int(row["block"]), str(row["fc"])): row for row in statistics}
    centers = _histogram_centers(edges)
    linthresh = min(abs(float(edges[len(edges) // 2 - 1])), float(edges[len(edges) // 2 + 1]))

    figure, axes = plt.subplots(
        len(blocks),
        2,
        figsize=(17, max(4.0, 2.2 * len(blocks))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, block in enumerate(blocks):
        for column_index, fc_type in enumerate(FC_TYPES):
            axis = axes[row_index, column_index]
            counts = counts_by_key[(block, fc_type)]
            total = max(int(counts.sum()), 1)
            probability = counts.astype(np.float64) / total
            present = counts > 0
            axis.plot(centers[present], probability[present], linewidth=0.9)
            axis.fill_between(
                centers[present], probability[present], alpha=0.2, linewidth=0.0
            )
            axis.set_xscale("symlog", linthresh=linthresh)
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
            stats = stat_by_key[(block, fc_type)]
            axis.set_title(
                f"Block {block} {fc_type.upper()} nonzero S"
                f" | zeros={100.0 * float(stats['zero_fraction']):.2f}%",
                fontsize=8,
            )
            if row_index == len(blocks) - 1:
                axis.set_xlabel("S value (symmetric-log bins)")
            axis.set_ylabel("probability / bin", fontsize=7)
            axis.tick_params(labelsize=6)
    figure.suptitle(
        "Signed nonzero-value distributions of FC sparse matrices"
        f" | |S| ≤ {zero_threshold:.3g} treated as zero"
    )
    figure.savefig(
        output_dir / "fc_sparse_signed_distributions_all_blocks.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    colors = {"fc1": "#3569b5", "fc2": "#d55e46"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for fc_type in FC_TYPES:
        rows = [stat_by_key[(block, fc_type)] for block in blocks]
        x = np.asarray(blocks)
        density = np.asarray([100.0 * float(row["density"]) for row in rows])
        positive = np.asarray(
            [100.0 * float(row["positive_fraction_nonzero"]) for row in rows]
        )
        q50 = np.asarray([float(row["abs_q50"]) for row in rows])
        q95 = np.asarray([float(row["abs_q95"]) for row in rows])
        q99 = np.asarray([float(row["abs_q99"]) for row in rows])
        rms = np.asarray([float(row["rms_all"]) for row in rows])
        axes[0, 0].plot(x, density, "o-", label=fc_type.upper(), color=colors[fc_type])
        axes[0, 1].plot(x, positive, "o-", label=fc_type.upper(), color=colors[fc_type])
        axes[1, 0].plot(x, q50, "o-", label=f"{fc_type.upper()} q50", color=colors[fc_type])
        axes[1, 0].plot(x, q95, "s--", label=f"{fc_type.upper()} q95", color=colors[fc_type])
        axes[1, 0].plot(x, q99, "^:", label=f"{fc_type.upper()} q99", color=colors[fc_type])
        axes[1, 1].plot(x, rms, "o-", label=fc_type.upper(), color=colors[fc_type])

    axes[0, 0].set_title("Sparse density")
    axes[0, 0].set_ylabel("nonzero elements (%)")
    axes[0, 1].set_title("Sign balance among nonzeros")
    axes[0, 1].set_ylabel("positive values (%)")
    axes[0, 1].axhline(50.0, color="black", linestyle=":", linewidth=0.8)
    axes[1, 0].set_title("Absolute nonzero-value quantiles")
    axes[1, 0].set_ylabel("|S|")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("Whole-matrix RMS (zeros included)")
    axes[1, 1].set_ylabel("RMS(S)")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xlabel("transformer block")
        axis.set_xticks(blocks)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Layer-wise FC sparse-matrix statistics"
        f" | |S| ≤ {zero_threshold:.3g} treated as zero"
    )
    figure.savefig(
        output_dir / "fc_sparse_layer_statistics.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_report(
    path: Path,
    matrix_dir: Path,
    statistics: Sequence[dict[str, object]],
    blocks: Sequence[int],
    scale: Mapping[str, float],
    zero_threshold: float,
    removed_counts: Mapping[tuple[int, str], int],
) -> None:
    stat_by_key = {(int(row["block"]), str(row["fc"])): row for row in statistics}
    lines = [
        "# FC sparse-matrix heatmaps and distributions",
        "",
        f"Source matrix directory: `{matrix_dir.resolve()}`",
        "",
        f"All entries with `|S| <= {zero_threshold:.8g}` are treated as exact zero. "
        f"This removed {sum(removed_counts.values()):,} values that were nonzero in "
        "the saved matrices. The source files were not modified.",
        "",
        "FC1 is displayed transposed, so both FC1 and FC2 heatmaps have MLP hidden unit "
        "on the x-axis and embedding channel on the y-axis. Heatmaps share one "
        f"symmetric-log color scale (`vmax={scale['vmax']:.8g}`, "
        f"`linthresh={scale['linthresh']:.8g}`). Values outside the visualization "
        "range are clipped only in the PNG; CSV statistics use every exact value.",
        "",
        "| Block | FC | Shape [out,in] | Density | Positive / nonzero | Mean(nonzero) | "
        "Std(nonzero) | Median |S| | q95 |S| | q99 |S| | Min | Max |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for block in blocks:
        for fc_type in FC_TYPES:
            row = stat_by_key[(block, fc_type)]
            lines.append(
                f"| {block} | {fc_type} | {int(row['shape_out'])}×{int(row['shape_in'])} "
                f"| {100.0 * float(row['density']):.4f}% "
                f"| {100.0 * float(row['positive_fraction_nonzero']):.2f}% "
                f"| {float(row['mean_nonzero']):.5g} "
                f"| {float(row['std_nonzero']):.5g} "
                f"| {float(row['abs_q50']):.5g} "
                f"| {float(row['abs_q95']):.5g} "
                f"| {float(row['abs_q99']):.5g} "
                f"| {float(row['value_q0']):.5g} "
                f"| {float(row['value_q100']):.5g} |"
            )

    lines.extend(["", "## Layer summaries", ""])
    for fc_type in FC_TYPES:
        rows = [stat_by_key[(block, fc_type)] for block in blocks]
        density_values = np.asarray([float(row["density"]) for row in rows])
        median_values = np.asarray([float(row["abs_q50"]) for row in rows])
        density_correlation = (
            float(np.corrcoef(np.asarray(blocks), density_values)[0, 1])
            if len(blocks) > 1 and np.std(density_values) > 0.0
            else math.nan
        )
        magnitude_correlation = (
            float(np.corrcoef(np.asarray(blocks), median_values)[0, 1])
            if len(blocks) > 1 and np.std(median_values) > 0.0
            else math.nan
        )
        densest_index = int(np.argmax(density_values))
        largest_index = int(np.argmax(median_values))
        lines.append(
            f"- {fc_type.upper()}: density range "
            f"{100.0 * density_values.min():.4f}%–{100.0 * density_values.max():.4f}% "
            f"(densest block {blocks[densest_index]}); median |S| range "
            f"{median_values.min():.5g}–{median_values.max():.5g} "
            f"(largest block {blocks[largest_index]}). Pearson depth correlations: "
            f"density `{density_correlation:.4f}`, median |S| `{magnitude_correlation:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `fc_sparse_heatmaps_all_blocks.png`: signed S values for every block.",
            "- `blocks/`: per-block signed-value and binary-support heatmaps.",
            "- `fc_sparse_signed_distributions_all_blocks.png`: signed nonzero histograms.",
            "- `fc_sparse_layer_statistics.png`: density, sign, magnitude, and RMS trends.",
            "- `fc_sparse_statistics.csv`: exact per-matrix descriptive statistics.",
            "- `fc_sparse_histograms.csv`: exact log-spaced histogram counts.",
            "- `metadata.json`: source and plotting metadata.",
            "",
            "These are input-independent parameter statistics. A lizard-conditioned FC study "
            "must additionally measure the token-wise activation contribution `S @ h`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    blocks = sorted(set(args.blocks))
    if not blocks or blocks[0] < 0 or blocks[-1] > 11:
        raise ValueError("blocks must be a non-empty subset of 0..11")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading FC sparse matrices from {args.matrix_dir}", flush=True)
    matrices = load_fc_sparse_matrices(args.matrix_dir, blocks)
    removed_counts = apply_zero_threshold(matrices, args.zero_threshold)
    print(
        f"zero threshold={args.zero_threshold:.8g} removed "
        f"{sum(removed_counts.values()):,} saved nonzero values",
        flush=True,
    )
    statistics = [
        matrix_statistics(block, fc_type, matrices[(block, fc_type)])
        for block in blocks
        for fc_type in FC_TYPES
    ]
    write_statistics_csv(output_dir / "fc_sparse_statistics.csv", statistics)

    edges = histogram_edges(matrices, args.histogram_bins_per_sign)
    histogram_data, counts_by_key = histogram_rows(matrices, blocks, edges)
    write_histogram_csv(output_dir / "fc_sparse_histograms.csv", histogram_data)

    print("rendering heatmaps", flush=True)
    scale = plot_heatmaps(
        output_dir,
        matrices,
        statistics,
        blocks,
        args.heatmap_percentile,
        args.zero_threshold,
        args.dpi,
    )
    print("rendering distributions", flush=True)
    plot_distributions(
        output_dir,
        statistics,
        blocks,
        edges,
        counts_by_key,
        args.zero_threshold,
        args.dpi,
    )
    write_report(
        output_dir / "report.md",
        args.matrix_dir,
        statistics,
        blocks,
        scale,
        args.zero_threshold,
        removed_counts,
    )
    metadata = {
        "matrix_dir": str(args.matrix_dir.expanduser().resolve()),
        "blocks": blocks,
        "fc_types": list(FC_TYPES),
        "zero_threshold": args.zero_threshold,
        "removed_saved_nonzero_values": {
            f"block{block}_{fc_type}": count
            for (block, fc_type), count in sorted(removed_counts.items())
        },
        "removed_saved_nonzero_values_total": sum(removed_counts.values()),
        "orientation": {
            "fc1": "S.T -> [embedding input channel, MLP hidden unit]",
            "fc2": "S -> [embedding output channel, MLP hidden unit]",
        },
        "heatmap": {
            "normalization": "symmetric_log",
            "shared_across_all_matrices": True,
            "percentile": args.heatmap_percentile,
            **scale,
        },
        "histogram": {
            "values": "nonzero signed S values",
            "bins_per_sign": args.histogram_bins_per_sign,
            "absolute_min_edge": float(min(abs(edges[edges != 0.0]))),
            "absolute_max_edge": float(max(abs(edges))),
        },
        "statistics": statistics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
