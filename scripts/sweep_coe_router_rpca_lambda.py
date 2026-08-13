#!/usr/bin/env python3
"""Sweep the RPCA sparse penalty for the extracted CoE routers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salad.ialm import fit_torch  # noqa: E402


DATA_ROOT = ROOT / "data" / "moe_router_analysis" / "chain_of_experts"
MODEL_KEYS = ("no_shared", "one_shared")
BASE_LAMBDA = 1.0 / math.sqrt(1024)
DEFAULT_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=DEFAULT_MULTIPLIERS,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if any(multiplier <= 0 for multiplier in args.multipliers):
        parser.error("all multipliers must be positive")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def spectral_metrics(matrix: torch.Tensor) -> tuple[int, float, int]:
    singular_values = torch.linalg.svdvals(matrix.double())
    tolerance = (
        singular_values[0]
        * max(matrix.shape)
        * torch.finfo(torch.float32).eps
    )
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    energy = singular_values.square()
    probabilities = energy / energy.sum()
    nonzero_probabilities = probabilities[probabilities > 0]
    effective_rank = float(
        torch.exp(
            -(nonzero_probabilities * torch.log(nonzero_probabilities)).sum()
        ).item()
    )
    cumulative = torch.cumsum(probabilities, dim=0)
    rank_90 = int(torch.searchsorted(cumulative, 0.90).item() + 1)
    return rank, effective_rank, rank_90


def evaluate_lambda(
    routers: dict[str, torch.Tensor],
    lambda_value: float,
    device: torch.device,
) -> dict[str, float]:
    total_elements = 0
    total_sparse_nonzero = 0
    total_reference_energy = 0.0
    total_residual_energy = 0.0
    ranks = []
    rank_ratios = []
    effective_ranks = []
    rank_90_values = []
    relative_l_energy = []
    relative_s_energy = []

    for reference in routers.values():
        reference = reference.float()
        low_rank, sparse = fit_torch(
            reference.to(device),
            lambda_=lambda_value,
            epsilon1=1e-2,
            epsilon2=1e-2,
            rho=1.6,
            max_iter=1000,
            verbose=False,
            device=device,
            dtype=torch.float32,
            approx_svd=False,
        )
        low_rank = low_rank.cpu()
        sparse = sparse.cpu()
        residual = reference - low_rank - sparse

        rank, effective_rank, rank_90 = spectral_metrics(low_rank)
        ranks.append(rank)
        rank_ratios.append(rank / min(reference.shape))
        effective_ranks.append(effective_rank)
        rank_90_values.append(rank_90)
        reference_energy = reference.double().square().sum().item()
        relative_l_energy.append(
            low_rank.double().square().sum().item() / reference_energy
        )
        relative_s_energy.append(
            sparse.double().square().sum().item() / reference_energy
        )
        total_elements += sparse.numel()
        total_sparse_nonzero += int(torch.count_nonzero(sparse).item())
        total_reference_energy += reference_energy
        total_residual_energy += residual.double().square().sum().item()

    return {
        "mean_rank_L": float(np.mean(ranks)),
        "mean_rank_ratio_L": float(np.mean(rank_ratios)),
        "mean_effective_rank_L": float(np.mean(effective_ranks)),
        "mean_rank_90_energy_L": float(np.mean(rank_90_values)),
        "sparse_density_S": total_sparse_nonzero / total_elements,
        "mean_relative_energy_L": float(np.mean(relative_l_energy)),
        "mean_relative_energy_S": float(np.mean(relative_s_energy)),
        "relative_reconstruction_error": math.sqrt(
            total_residual_energy / total_reference_energy
        ),
    }


def save_outputs(rows: list[dict[str, float | str]]) -> None:
    analysis_dir = DATA_ROOT / "analysis"
    figures_dir = DATA_ROOT / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = analysis_dir / "router_rpca_lambda_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (analysis_dir / "router_rpca_lambda_sweep.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.8))
    colors = {"no_shared": "#2474b5", "one_shared": "#d95f02"}
    labels = {"no_shared": "0 shared", "one_shared": "1 shared"}
    for model_key in MODEL_KEYS:
        model_rows = [row for row in rows if row["model"] == model_key]
        lambdas = [float(row["lambda_value"]) for row in model_rows]
        axes[0].plot(
            lambdas,
            [100.0 * float(row["sparse_density_S"]) for row in model_rows],
            marker="o",
            color=colors[model_key],
            label=labels[model_key],
        )
        axes[1].plot(
            lambdas,
            [100.0 * float(row["mean_rank_ratio_L"]) for row in model_rows],
            marker="o",
            color=colors[model_key],
            label=labels[model_key],
        )
        axes[2].plot(
            lambdas,
            [float(row["mean_effective_rank_L"]) for row in model_rows],
            marker="o",
            color=colors[model_key],
            label=labels[model_key],
        )
        axes[3].plot(
            lambdas,
            [
                100.0 * float(row["relative_reconstruction_error"])
                for row in model_rows
            ],
            marker="o",
            color=colors[model_key],
            label=labels[model_key],
        )

    axes[0].set_ylabel("S nonzero density (%)")
    axes[1].set_ylabel("Mean rank(L) / min(shape) (%)")
    axes[2].set_ylabel("Mean effective rank(L)")
    axes[3].set_ylabel("Relative reconstruction error (%)")
    for axis in axes:
        axis.set_xlabel("RPCA lambda")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures_dir / f"router_rpca_lambda_sweep.{suffix}",
            dpi=220,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    routers_by_model = {
        model_key: load_file(
            str(DATA_ROOT / model_key / "router_only.safetensors"),
            device="cpu",
        )
        for model_key in MODEL_KEYS
    }
    rows: list[dict[str, float | str]] = []
    print(f"device={device}; base_lambda={BASE_LAMBDA}", flush=True)
    for multiplier in args.multipliers:
        lambda_value = BASE_LAMBDA * multiplier
        for model_key in MODEL_KEYS:
            metrics = evaluate_lambda(
                routers_by_model[model_key],
                lambda_value,
                device,
            )
            row: dict[str, float | str] = {
                "model": model_key,
                "lambda_multiplier": multiplier,
                "lambda_value": lambda_value,
                **metrics,
            }
            rows.append(row)
            print(
                f"lambda={lambda_value:.8f} ({multiplier:g}x) "
                f"model={model_key} S_density={metrics['sparse_density_S']:.4f} "
                f"rank_ratio_L={metrics['mean_rank_ratio_L']:.4f} "
                f"effective_rank_L={metrics['mean_effective_rank_L']:.3f} "
                f"error={metrics['relative_reconstruction_error']:.6g}",
                flush=True,
            )
    save_outputs(rows)


if __name__ == "__main__":
    main()
