#!/usr/bin/env python3
"""Decompose the extracted Chain-of-Experts routers into RPCA L + S."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salad.ialm import fit_torch  # noqa: E402


DATA_ROOT = ROOT / "data" / "moe_router_analysis" / "chain_of_experts"
MODEL_KEYS = ("no_shared", "one_shared")
ROUTER_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\."
    r"(?P<iteration>\d+)\.weight$"
)

EPSILON1 = 1e-2
EPSILON2 = 1e-2
RHO = 1.6
MAX_ITER = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(MODEL_KEYS),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=None,
        help=(
            "RPCA sparse penalty (default: 1/sqrt(max(rows, columns))). "
            "Larger values make S sparser."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="rpca",
        help="Output subdirectory below each model directory.",
    )
    args = parser.parse_args()
    if args.lambda_value is not None and args.lambda_value <= 0:
        parser.error("--lambda-value must be positive")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_name) is None:
        parser.error("--output-name may contain only letters, digits, _, ., and -")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_safetensors_save(
    tensors: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.double().square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


def effective_rank(singular_values: torch.Tensor) -> float:
    energy = singular_values.double().square()
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * torch.log(probabilities)).sum()).item())


def component_statistics(
    reference: torch.Tensor,
    low_rank: torch.Tensor,
    sparse: torch.Tensor,
    elapsed_seconds: float,
) -> dict[str, Any]:
    reference64 = reference.double()
    low_rank64 = low_rank.double()
    sparse64 = sparse.double()
    reconstruction = low_rank64 + sparse64
    residual = reference64 - reconstruction

    reference_energy = float(reference64.square().sum().item())
    low_rank_energy = float(low_rank64.square().sum().item())
    sparse_energy = float(sparse64.square().sum().item())
    residual_energy = float(residual.square().sum().item())
    cross_energy = float((2.0 * low_rank64 * sparse64).sum().item())

    singular_values = torch.linalg.svdvals(low_rank64)
    tolerance = (
        singular_values[0]
        * max(reference.shape)
        * torch.finfo(torch.float32).eps
    )
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    sparse_nonzero = int(torch.count_nonzero(sparse).item())

    return {
        "shape": list(reference.shape),
        "lambda": 1.0 / math.sqrt(max(reference.shape)),
        "rank_L": rank,
        "rank_ratio_L": rank / min(reference.shape),
        "effective_rank_L": effective_rank(singular_values),
        "rank_90_energy_L": energy_rank(singular_values, 0.90),
        "rank_99_energy_L": energy_rank(singular_values, 0.99),
        "sparse_nonzero_S": sparse_nonzero,
        "sparse_density_S": sparse_nonzero / sparse.numel(),
        "relative_reconstruction_error": math.sqrt(
            residual_energy / max(reference_energy, 1e-30)
        ),
        "relative_energy_L": low_rank_energy / max(reference_energy, 1e-30),
        "relative_energy_S": sparse_energy / max(reference_energy, 1e-30),
        "relative_cross_energy_2LS": cross_energy
        / max(reference_energy, 1e-30),
        "reference_squared_frobenius_norm": reference_energy,
        "reconstruction_squared_error": residual_energy,
        "elapsed_seconds": elapsed_seconds,
    }


def validate_router_file(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"router file does not exist: {path}; run "
            "scripts/analyze_coe_shared_router_spectra.py first"
        )
    tensors = load_file(str(path), device="cpu")
    if len(tensors) != 8:
        raise ValueError(f"expected 8 routers in {path}, found {len(tensors)}")
    for name, tensor in tensors.items():
        if ROUTER_PATTERN.fullmatch(name) is None:
            raise ValueError(f"unexpected tensor in router file: {name}")
        if tensor.ndim != 2 or tensor.shape[1] != 1024:
            raise ValueError(f"unexpected shape for {name}: {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"router must be finite float32: {name}")
    return tensors


def decompose_model(
    model_key: str,
    device: torch.device,
    lambda_value: float | None,
    output_name: str,
) -> dict[str, Any]:
    source_path = DATA_ROOT / model_key / "router_only.safetensors"
    output_dir = DATA_ROOT / model_key / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    routers = validate_router_file(source_path)
    source_digest = sha256(source_path)
    if lambda_value is None:
        lambda_value = 1.0 / math.sqrt(1024)

    low_rank_tensors: dict[str, torch.Tensor] = {}
    sparse_tensors: dict[str, torch.Tensor] = {}
    reconstructed_tensors: dict[str, torch.Tensor] = {}
    router_statistics: dict[str, dict[str, Any]] = {}

    ordered_names = sorted(
        routers,
        key=lambda name: tuple(
            int(value) for value in ROUTER_PATTERN.fullmatch(name).groups()
        ),
    )
    for index, name in enumerate(ordered_names, start=1):
        reference = routers[name].detach().float()
        started = time.perf_counter()
        low_rank, sparse = fit_torch(
            reference.to(device),
            lambda_=lambda_value,
            epsilon1=EPSILON1,
            epsilon2=EPSILON2,
            rho=RHO,
            max_iter=MAX_ITER,
            verbose=False,
            device=device,
            dtype=torch.float32,
            approx_svd=False,
        )
        low_rank = low_rank.cpu().contiguous()
        sparse = sparse.cpu().contiguous()
        reconstruction = (low_rank + sparse).contiguous()
        elapsed = time.perf_counter() - started

        statistics = component_statistics(
            reference,
            low_rank,
            sparse,
            elapsed,
        )
        router_statistics[name] = statistics
        low_rank_tensors[name] = low_rank
        sparse_tensors[name] = sparse
        reconstructed_tensors[name] = reconstruction
        print(
            f"[{model_key} {index}/8] {name}; "
            f"rank(L)={statistics['rank_L']}; "
            f"density(S)={statistics['sparse_density_S']:.4f}; "
            f"relative_error={statistics['relative_reconstruction_error']:.6g}; "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    shared_metadata = {
        "source_file": str(source_path.relative_to(ROOT)),
        "source_sha256": source_digest,
        "algorithm": "salad.ialm.fit_torch",
        "lambda": str(lambda_value),
    }
    output_paths = {
        "L": output_dir / "router_L.safetensors",
        "S": output_dir / "router_S.safetensors",
        "L_plus_S": output_dir / "router_L_plus_S.safetensors",
    }
    atomic_safetensors_save(
        low_rank_tensors,
        output_paths["L"],
        {**shared_metadata, "component": "L"},
    )
    atomic_safetensors_save(
        sparse_tensors,
        output_paths["S"],
        {**shared_metadata, "component": "S"},
    )
    atomic_safetensors_save(
        reconstructed_tensors,
        output_paths["L_plus_S"],
        {**shared_metadata, "component": "L_plus_S"},
    )

    total_reference_energy = sum(
        values["reference_squared_frobenius_norm"]
        for values in router_statistics.values()
    )
    total_residual_energy = sum(
        values["reconstruction_squared_error"]
        for values in router_statistics.values()
    )
    total_sparse_nonzero = sum(
        values["sparse_nonzero_S"] for values in router_statistics.values()
    )
    total_elements = sum(
        math.prod(values["shape"]) for values in router_statistics.values()
    )
    ranks = np.asarray(
        [values["rank_L"] for values in router_statistics.values()],
        dtype=np.float64,
    )
    rank_ratios = np.asarray(
        [values["rank_ratio_L"] for values in router_statistics.values()],
        dtype=np.float64,
    )
    effective_ranks = np.asarray(
        [values["effective_rank_L"] for values in router_statistics.values()],
        dtype=np.float64,
    )
    rank_90_values = np.asarray(
        [values["rank_90_energy_L"] for values in router_statistics.values()],
        dtype=np.float64,
    )
    relative_energy_l = np.asarray(
        [values["relative_energy_L"] for values in router_statistics.values()],
        dtype=np.float64,
    )
    relative_energy_s = np.asarray(
        [values["relative_energy_S"] for values in router_statistics.values()],
        dtype=np.float64,
    )

    metadata = {
        "schema_version": 1,
        "model": model_key,
        "source_router_file": str(source_path.relative_to(ROOT)),
        "source_router_sha256": source_digest,
        "algorithm": "salad.ialm.fit_torch",
        "algorithm_parameters": {
            "lambda": "1 / sqrt(max(rows, columns))",
            "lambda_value": lambda_value,
            "epsilon1": EPSILON1,
            "epsilon2": EPSILON2,
            "rho": RHO,
            "max_iter": MAX_ITER,
            "dtype": "torch.float32",
            "approx_svd": False,
            "device": str(device),
        },
        "outputs": {
            component: str(path.relative_to(ROOT))
            for component, path in output_paths.items()
        },
        "output_sha256": {
            component: sha256(path) for component, path in output_paths.items()
        },
        "aggregate": {
            "number_of_routers": len(routers),
            "mean_rank_L": float(ranks.mean()),
            "min_rank_L": int(ranks.min()),
            "max_rank_L": int(ranks.max()),
            "mean_rank_ratio_L": float(rank_ratios.mean()),
            "mean_effective_rank_L": float(effective_ranks.mean()),
            "mean_rank_90_energy_L": float(rank_90_values.mean()),
            "mean_relative_energy_L": float(relative_energy_l.mean()),
            "mean_relative_energy_S": float(relative_energy_s.mean()),
            "sparse_density_S": total_sparse_nonzero / total_elements,
            "relative_reconstruction_error": math.sqrt(
                total_residual_energy / max(total_reference_energy, 1e-30)
            ),
        },
        "routers": router_statistics,
    }
    atomic_json_save(metadata, output_dir / "metadata.json")
    return metadata


def write_comparison_csv(
    results: dict[str, dict[str, Any]], output_name: str
) -> Path:
    suffix = "" if output_name == "rpca" else f"_{output_name}"
    output_path = DATA_ROOT / "analysis" / f"router_rpca_summary{suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "layer",
        "iteration",
        "rows",
        "columns",
        "rank_L",
        "rank_ratio_L",
        "effective_rank_L",
        "rank_90_energy_L",
        "rank_99_energy_L",
        "sparse_nonzero_S",
        "sparse_density_S",
        "relative_energy_L",
        "relative_energy_S",
        "relative_reconstruction_error",
        "elapsed_seconds",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_key in MODEL_KEYS:
            if model_key not in results:
                continue
            for name, statistics in results[model_key]["routers"].items():
                match = ROUTER_PATTERN.fullmatch(name)
                assert match is not None
                writer.writerow(
                    {
                        "model": model_key,
                        "layer": int(match.group("layer")),
                        "iteration": int(match.group("iteration")),
                        "rows": statistics["shape"][0],
                        "columns": statistics["shape"][1],
                        **{
                            field: statistics[field]
                            for field in fields
                            if field in statistics
                        },
                    }
                )
    return output_path


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    results = {
        model_key: decompose_model(
            model_key,
            device,
            lambda_value=args.lambda_value,
            output_name=args.output_name,
        )
        for model_key in args.models
    }
    comparison_path = write_comparison_csv(results, args.output_name)
    print(
        json.dumps(
            {
                "comparison_csv": str(comparison_path.relative_to(ROOT)),
                "aggregate": {
                    model_key: result["aggregate"]
                    for model_key, result in results.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
