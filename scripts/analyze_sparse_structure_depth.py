#!/usr/bin/env python3
"""Quantify how SALAAD sparse-matrix structure changes with ViT depth."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visualize_dino_attention import load_matrix_file, matrix_rank


LAYER_PATTERN = re.compile(
    r"^backbone\.blocks\.(?P<block>\d+)\."
    r"(?P<layer>attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)$"
)
COMPONENTS = ("Q", "K", "V", "Proj", "FC1", "FC2")
COLORS = {
    "Q": "#0072B2",
    "K": "#56B4E9",
    "V": "#009E73",
    "Proj": "#D55E00",
    "FC1": "#CC79A7",
    "FC2": "#E69F00",
}
MARKERS = {"Q": "o", "K": "s", "V": "^", "Proj": "D", "FC1": "P", "FC2": "X"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(1e-4, 1e-2)
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def split_components(
    layer_type: str, low_rank: Tensor, sparse: Tensor
) -> list[tuple[str, Tensor, Tensor]]:
    if layer_type == "attn.qkv":
        if sparse.shape[0] % 3 != 0:
            raise ValueError(f"QKV output dimension is not divisible by three: {sparse.shape}")
        return list(zip(("Q", "K", "V"), low_rank.chunk(3, dim=0), sparse.chunk(3, dim=0)))
    name = {"attn.proj": "Proj", "mlp.fc1": "FC1", "mlp.fc2": "FC2"}[
        layer_type
    ]
    return [(name, low_rank, sparse)]


def lag_one_correlation(profile: np.ndarray) -> float:
    if profile.size < 3 or float(profile.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(profile[:-1], profile[1:])[0, 1])


def low_frequency_fraction(profile: np.ndarray, fraction: float = 0.10) -> float:
    centered = profile.astype(np.float64) - float(profile.mean())
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 1 or float(spectrum[1:].sum()) == 0.0:
        return math.nan
    cutoff = max(2, int(math.ceil(fraction * spectrum.size)))
    return float(spectrum[1:cutoff].sum() / spectrum[1:].sum())


def participation_concentration(profile: np.ndarray) -> float:
    values = profile.astype(np.float64)
    denominator = values.size * float(np.square(values).sum())
    if denominator == 0.0:
        return math.nan
    participation = float(values.sum()) ** 2 / denominator
    return 1.0 - participation


def element_fraction_for_energy(sparse: Tensor, fraction: float) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("energy fraction must be in (0, 1]")
    energy = sparse.detach().cpu().float().square().reshape(-1)
    total = energy.sum()
    if float(total.item()) == 0.0:
        return 0.0
    ordered = torch.sort(energy, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0) / total
    count = int(torch.searchsorted(cumulative, torch.tensor(fraction)).item()) + 1
    return count / energy.numel()


def support_statistics(support: np.ndarray) -> dict[str, float]:
    if support.ndim != 2:
        raise ValueError(f"support must be two-dimensional, got {support.shape}")
    density = float(support.mean())
    row_profile = support.mean(axis=1)
    column_profile = support.mean(axis=0)
    row_std = float(row_profile.std())
    column_std = float(column_profile.std())
    if density in (0.0, 1.0):
        row_excess = column_excess = direction_log2 = math.nan
    else:
        row_iid_std = math.sqrt(density * (1.0 - density) / support.shape[1])
        column_iid_std = math.sqrt(density * (1.0 - density) / support.shape[0])
        row_excess = row_std / row_iid_std
        column_excess = column_std / column_iid_std
        if row_excess > 0.0 and column_excess > 0.0:
            direction_log2 = math.log2(row_excess / column_excess)
        else:
            direction_log2 = math.nan
    return {
        "density": density,
        "row_profile_std": row_std,
        "column_profile_std": column_std,
        "row_dispersion_over_iid": row_excess,
        "column_dispersion_over_iid": column_excess,
        "direction_log2_row_over_column": direction_log2,
        "row_lag1": lag_one_correlation(row_profile),
        "column_lag1": lag_one_correlation(column_profile),
        "row_low_frequency_fraction": low_frequency_fraction(row_profile),
        "column_low_frequency_fraction": low_frequency_fraction(column_profile),
        "row_concentration": participation_concentration(row_profile),
        "column_concentration": participation_concentration(column_profile),
    }


def base_statistics(
    block: int, component: str, low_rank: Tensor, sparse: Tensor
) -> dict[str, object]:
    low_rank = low_rank.detach().cpu().float()
    sparse = sparse.detach().cpu().float()
    reconstruction = low_rank + sparse
    sparse_abs = sparse.abs()
    nonzero_abs = sparse_abs[sparse_abs > 0.0]
    if nonzero_abs.numel():
        quantiles = torch.quantile(
            nonzero_abs,
            torch.tensor((0.50, 0.90, 0.95, 0.99, 0.999, 1.0)),
        )
    else:
        quantiles = torch.full((6,), torch.nan)
    sparse_energy = float(sparse.square().sum().item())
    reconstruction_energy = float(reconstruction.square().sum().item())
    low_rank_energy = float(low_rank.square().sum().item())
    cross = float((low_rank * sparse).sum().item())
    return {
        "block": block,
        "component": component,
        "shape_out": int(sparse.shape[0]),
        "shape_in": int(sparse.shape[1]),
        "elements": sparse.numel(),
        "exact_nonzero_density": float((sparse != 0.0).float().mean().item()),
        "sparse_l2": math.sqrt(sparse_energy),
        "low_rank_l2": math.sqrt(low_rank_energy),
        "reconstruction_l2": math.sqrt(reconstruction_energy),
        "sparse_l2_over_reconstruction_l2": (
            math.sqrt(sparse_energy / reconstruction_energy)
            if reconstruction_energy
            else math.nan
        ),
        "sparse_energy_over_reconstruction_energy": (
            sparse_energy / reconstruction_energy if reconstruction_energy else math.nan
        ),
        "low_rank_sparse_cosine": (
            cross / math.sqrt(low_rank_energy * sparse_energy)
            if low_rank_energy and sparse_energy
            else math.nan
        ),
        "abs_nonzero_q50": float(quantiles[0]),
        "abs_nonzero_q90": float(quantiles[1]),
        "abs_nonzero_q95": float(quantiles[2]),
        "abs_nonzero_q99": float(quantiles[3]),
        "abs_nonzero_q99_9": float(quantiles[4]),
        "abs_nonzero_max": float(quantiles[5]),
        "element_fraction_for_90pct_s_energy": element_fraction_for_energy(
            sparse, 0.90
        ),
        "element_fraction_for_95pct_s_energy": element_fraction_for_energy(
            sparse, 0.95
        ),
        "element_fraction_for_99pct_s_energy": element_fraction_for_energy(
            sparse, 0.99
        ),
        "sparse_energy": sparse_energy,
        "sparse_l1": float(sparse_abs.sum().item()),
    }


def threshold_statistics(
    base: Mapping[str, object], sparse: Tensor, threshold: float
) -> dict[str, object]:
    sparse = sparse.detach().cpu().float()
    absolute = sparse.abs()
    support_tensor = absolute >= threshold
    support = support_tensor.numpy()
    result: dict[str, object] = {
        "block": int(base["block"]),
        "component": str(base["component"]),
        "threshold": threshold,
        "kept": int(support_tensor.count_nonzero().item()),
        "removed_saved_nonzero": int(
            ((sparse != 0.0) & ~support_tensor).count_nonzero().item()
        ),
    }
    result.update(support_statistics(support))
    sparse_energy = float(base["sparse_energy"])
    sparse_l1 = float(base["sparse_l1"])
    kept_energy = float(sparse.square().masked_select(support_tensor).sum().item())
    kept_l1 = float(absolute.masked_select(support_tensor).sum().item())
    result["energy_retained"] = kept_energy / sparse_energy if sparse_energy else 0.0
    result["l1_retained"] = kept_l1 / sparse_l1 if sparse_l1 else 0.0
    result["kept_fraction_of_saved_nonzero"] = (
        float(result["density"]) / float(base["exact_nonzero_density"])
        if float(base["exact_nonzero_density"])
        else 0.0
    )
    return result


def collect_statistics(
    matrix_dir: Path, blocks: Sequence[int], thresholds: Sequence[float]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files in {matrix_dir}")
    block_set = set(blocks)
    base_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for path in files:
        print(f"loading {path.name}", flush=True)
        low_rank_mapping, sparse_mapping = load_matrix_file(path)
        for layer_name in sorted(sparse_mapping):
            match = LAYER_PATTERN.fullmatch(layer_name)
            if match is None:
                continue
            block = int(match.group("block"))
            if block not in block_set:
                continue
            layer_type = match.group("layer")
            low_rank = low_rank_mapping[layer_name].detach().cpu().float()
            sparse = sparse_mapping[layer_name].detach().cpu().float()
            for component, component_l, component_s in split_components(
                layer_type, low_rank, sparse
            ):
                key = (block, component)
                if key in seen:
                    raise ValueError(f"duplicate component {key}")
                seen.add(key)
                base = base_statistics(block, component, component_l, component_s)
                base_rows.append(base)
                for threshold in thresholds:
                    threshold_rows.append(
                        threshold_statistics(base, component_s, threshold)
                    )
        del low_rank_mapping
        del sparse_mapping
        gc.collect()
    expected = {(block, component) for block in block_set for component in COMPONENTS}
    if seen != expected:
        raise ValueError(f"component mismatch: missing={sorted(expected - seen)}")
    base_rows.sort(key=lambda row: (int(row["block"]), COMPONENTS.index(str(row["component"]))))
    threshold_rows.sort(
        key=lambda row: (
            float(row["threshold"]),
            int(row["block"]),
            COMPONENTS.index(str(row["component"])),
        )
    )
    return base_rows, threshold_rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def indexed_rows(
    rows: Sequence[Mapping[str, object]], threshold: float | None = None
) -> dict[tuple[int, str], Mapping[str, object]]:
    selected = rows
    if threshold is not None:
        selected = [row for row in rows if float(row["threshold"]) == threshold]
    return {(int(row["block"]), str(row["component"])): row for row in selected}


def plot_density_and_energy(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    blocks: Sequence[int],
    thresholds: Sequence[float],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(thresholds), 2, figsize=(15, 5.0 * len(thresholds)), squeeze=False,
        constrained_layout=True,
    )
    for row_index, threshold in enumerate(thresholds):
        lookup = indexed_rows(rows, threshold)
        for component in COMPONENTS:
            density = [100.0 * float(lookup[(block, component)]["density"]) for block in blocks]
            energy = [100.0 * float(lookup[(block, component)]["energy_retained"]) for block in blocks]
            style = {
                "color": COLORS[component],
                "marker": MARKERS[component],
                "linewidth": 1.8,
                "markersize": 4.5,
                "label": component,
            }
            axes[row_index, 0].plot(blocks, density, **style)
            axes[row_index, 1].plot(blocks, energy, **style)
        axes[row_index, 0].set_title(f"Support density after |S| < {threshold:g} → 0")
        axes[row_index, 0].set_ylabel("kept elements (%)")
        axes[row_index, 1].set_title(f"S energy retained at threshold {threshold:g}")
        axes[row_index, 1].set_ylabel("retained $L_2^2$ energy (%)")
        axes[row_index, 1].set_ylim(-2, 102)
        for axis in axes[row_index]:
            axis.set_xlabel("transformer block")
            axis.set_xticks(blocks)
            axis.grid(alpha=0.25)
            axis.legend(ncol=3, fontsize=8)
    figure.suptitle("Sparse residual density and energy redistribution with depth")
    figure.savefig(output_dir / "density_and_energy_vs_depth.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_directionality(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    blocks: Sequence[int],
    thresholds: Sequence[float],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(thresholds), 2, figsize=(15, 5.0 * len(thresholds)), squeeze=False,
        constrained_layout=True,
    )
    for row_index, threshold in enumerate(thresholds):
        lookup = indexed_rows(rows, threshold)
        for component in COMPONENTS:
            direction = [
                float(lookup[(block, component)]["direction_log2_row_over_column"])
                for block in blocks
            ]
            row_lag = [float(lookup[(block, component)]["row_lag1"]) for block in blocks]
            column_lag = [
                float(lookup[(block, component)]["column_lag1"]) for block in blocks
            ]
            axes[row_index, 0].plot(
                blocks,
                direction,
                color=COLORS[component],
                marker=MARKERS[component],
                linewidth=1.8,
                markersize=4.5,
                label=component,
            )
            target_lag = column_lag if component == "Proj" else row_lag
            axes[row_index, 1].plot(
                blocks,
                target_lag,
                color=COLORS[component],
                marker=MARKERS[component],
                linewidth=1.8,
                markersize=4.5,
                label=(f"{component} column" if component == "Proj" else f"{component} row"),
            )
        axes[row_index, 0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axes[row_index, 0].set_title(
            f"Directional support anisotropy at threshold {threshold:g}"
        )
        axes[row_index, 0].set_ylabel("log₂(row dispersion / column dispersion)")
        axes[row_index, 1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        axes[row_index, 1].set_title(
            "Adjacent-profile coherence (row for Q/K/V/FC; column for Proj)"
        )
        axes[row_index, 1].set_ylabel("lag-1 correlation")
        for axis in axes[row_index]:
            axis.set_xlabel("transformer block")
            axis.set_xticks(blocks)
            axis.grid(alpha=0.25)
            axis.legend(ncol=3, fontsize=8)
    figure.suptitle("Direction and continuity of thresholded S stripes")
    figure.savefig(output_dir / "direction_and_coherence_vs_depth.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_magnitude_and_role(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    blocks: Sequence[int],
    dpi: int,
) -> None:
    lookup = indexed_rows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for component in COMPONENTS:
        q99 = [float(lookup[(block, component)]["abs_nonzero_q99"]) for block in blocks]
        relative_norm = [
            float(lookup[(block, component)]["sparse_l2_over_reconstruction_l2"])
            for block in blocks
        ]
        style = {
            "color": COLORS[component],
            "marker": MARKERS[component],
            "linewidth": 1.8,
            "markersize": 4.5,
            "label": component,
        }
        axes[0].plot(blocks, q99, **style)
        axes[1].plot(blocks, relative_norm, **style)
    axes[0].set_title("99th percentile of nonzero |S|")
    axes[0].set_ylabel("|S| q99")
    axes[0].set_yscale("log")
    axes[1].set_title("Sparse-residual norm relative to deployed L+S")
    axes[1].set_ylabel(r"$\|S\|_F / \|L+S\|_F$")
    for axis in axes:
        axis.set_xlabel("transformer block")
        axis.set_xticks(blocks)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    figure.suptitle("Sparse-residual scale and relative norm with depth")
    figure.savefig(output_dir / "magnitude_and_relative_norm_vs_depth.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_energy_concentration(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    blocks: Sequence[int],
    dpi: int,
) -> None:
    lookup = indexed_rows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for component in COMPONENTS:
        fractions_95 = [
            100.0
            * float(lookup[(block, component)]["element_fraction_for_95pct_s_energy"])
            for block in blocks
        ]
        fractions_99 = [
            100.0
            * float(lookup[(block, component)]["element_fraction_for_99pct_s_energy"])
            for block in blocks
        ]
        fractions_95 = [value if value > 0.0 else math.nan for value in fractions_95]
        fractions_99 = [value if value > 0.0 else math.nan for value in fractions_99]
        style = {
            "color": COLORS[component],
            "marker": MARKERS[component],
            "linewidth": 1.8,
            "markersize": 4.5,
            "label": component,
        }
        axes[0].plot(blocks, fractions_95, **style)
        axes[1].plot(blocks, fractions_99, **style)
    axes[0].set_title("Elements required to retain 95% of S energy")
    axes[1].set_title("Elements required to retain 99% of S energy")
    for axis in axes:
        axis.set_xlabel("transformer block")
        axis.set_ylabel("matrix elements required (%)")
        axis.set_xticks(blocks)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    figure.suptitle("Depth-wise diffusion of sparse-residual energy")
    figure.savefig(
        output_dir / "energy_concentration_vs_depth.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close(figure)


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else math.nan


def write_report(
    path: Path,
    base_rows: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]],
    blocks: Sequence[int],
    thresholds: Sequence[float],
) -> None:
    base = indexed_rows(base_rows)
    early = [block for block in blocks if 1 <= block <= 3]
    late = [block for block in blocks if 9 <= block <= 11]
    lines = [
        "# Depth analysis of mixed-rho SALAAD sparse matrices",
        "",
        "Q, K, and V are analyzed as separate 768×768 matrices. Thresholding uses "
        "the strict rule `|S| < threshold → 0`. Horizontal/vertical direction is "
        "measured from row/column support-density dispersion normalized by an IID "
        "Bernoulli support with the same matrix shape and density.",
        "",
        "## Early-versus-late summary",
        "",
    ]
    for threshold in thresholds:
        lookup = indexed_rows(threshold_rows, threshold)
        lines.extend(
            [
                f"### Threshold {threshold:g}",
                "",
                "| Component | Density early → late | Energy retained early → late | "
                "Direction log2 early → late | Target coherence early → late |",
                "|:---|---:|---:|---:|---:|",
            ]
        )
        for component in COMPONENTS:
            def average(metric: str, selected: Sequence[int]) -> float:
                return finite_mean(
                    [float(lookup[(block, component)][metric]) for block in selected]
                )

            target = "column_lag1" if component == "Proj" else "row_lag1"
            lines.append(
                f"| {component} | {100 * average('density', early):.3f}% → "
                f"{100 * average('density', late):.3f}% | "
                f"{100 * average('energy_retained', early):.2f}% → "
                f"{100 * average('energy_retained', late):.2f}% | "
                f"{average('direction_log2_row_over_column', early):.3f} → "
                f"{average('direction_log2_row_over_column', late):.3f} | "
                f"{average(target, early):.3f} → {average(target, late):.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Relative S norm",
            "",
            "| Component | Early mean ||S||/||L+S|| | Late mean ||S||/||L+S|| | "
            "Elements for 95% S energy, early → late |",
            "|:---|---:|---:|---:|",
        ]
    )
    for component in COMPONENTS:
        early_value = finite_mean(
            [
                float(base[(block, component)]["sparse_l2_over_reconstruction_l2"])
                for block in early
            ]
        )
        late_value = finite_mean(
            [
                float(base[(block, component)]["sparse_l2_over_reconstruction_l2"])
                for block in late
            ]
        )
        early_energy_density = finite_mean(
            [
                float(base[(block, component)]["element_fraction_for_95pct_s_energy"])
                for block in early
            ]
        )
        late_energy_density = finite_mean(
            [
                float(base[(block, component)]["element_fraction_for_95pct_s_energy"])
                for block in late
            ]
        )
        lines.append(
            f"| {component} | {early_value:.4f} | {late_value:.4f} | "
            f"{100 * early_energy_density:.4f}% → {100 * late_energy_density:.4f}% |"
        )
    lines.extend(
        [
            "",
            "`direction_log2 > 0` means row/horizontal anisotropy; `< 0` means "
            "column/vertical anisotropy. Coherence is lag-1 correlation of adjacent "
            "row-density profiles for Q/K/V/FC and adjacent column-density profiles "
            "for projection. Density alone must not be interpreted as structural "
            "degradation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    blocks = sorted(set(args.blocks))
    thresholds = sorted(set(args.thresholds))
    if not blocks or blocks[0] < 0 or blocks[-1] > 11:
        raise ValueError("blocks must be a non-empty subset of 0..11")
    if not thresholds or any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("thresholds must be finite and non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows, threshold_rows = collect_statistics(matrix_dir, blocks, thresholds)
    write_csv(output_dir / "component_base_statistics.csv", base_rows)
    write_csv(output_dir / "component_threshold_statistics.csv", threshold_rows)
    plot_density_and_energy(output_dir, threshold_rows, blocks, thresholds, args.dpi)
    plot_directionality(output_dir, threshold_rows, blocks, thresholds, args.dpi)
    plot_magnitude_and_role(output_dir, base_rows, blocks, args.dpi)
    plot_energy_concentration(output_dir, base_rows, blocks, args.dpi)
    write_report(
        output_dir / "report.md", base_rows, threshold_rows, blocks, thresholds
    )
    metadata = {
        "matrix_dir": str(matrix_dir),
        "blocks": blocks,
        "components": list(COMPONENTS),
        "thresholds": thresholds,
        "zero_rule": "abs(S) < threshold is zero; equality is retained",
        "direction_metric": "log2(row dispersion over IID / column dispersion over IID)",
        "coherence_metric": "lag-1 Pearson correlation of adjacent support-density profile",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
