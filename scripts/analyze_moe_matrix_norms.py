#!/usr/bin/env python3
"""Inspect X, mapped X, X_e, W_e and P_e in a dense-residual MoE checkpoint."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from salaad_moe.alignment import validate_permutation
from salaad_moe.export import checkpoint_states
from salaad_moe.solver import DenseResidualState


def stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None,
                "max": None, "std_population": None}
    return {"count": len(values), "min": min(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "max": max(values),
            "std_population": statistics.pstdev(values)}


def fro(matrices):
    return torch.linalg.vector_norm(matrices.double(), dim=(-2, -1))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def permutation_rows(layer, permutation):
    """Report matrix norms, never the norm of a hard permutation's index vector."""
    experts, channels = permutation.shape[:2]
    identity = torch.arange(channels, device=permutation.device)
    hard = permutation.ndim == 2
    if hard:
        validate_permutation(permutation, experts, channels)
    rows = []
    for expert, value in enumerate(permutation):
        if hard:
            moved = int((value != identity).sum())
            row = {"permutation_fro": math.sqrt(channels), "is_identity": moved == 0,
                   "moved_channels": moved, "moved_fraction": moved / channels,
                   "row_sum_error_max": 0.0, "column_sum_error_max": 0.0,
                   "identity_distance_fro": math.sqrt(2 * moved),
                   "relative_identity_distance_fro": math.sqrt(2 * moved / channels),
                   "diagonal_mean": 1 - moved / channels}
        else:
            value = value.double()
            difference = value - torch.eye(channels, device=value.device, dtype=value.dtype)
            distance = float(fro(difference))
            row = {"permutation_fro": float(fro(value)),
                   "is_identity": torch.equal(value, torch.eye(channels, device=value.device)),
                   "moved_channels": None, "moved_fraction": None,
                   "row_sum_error_max": float((value.sum(-1) - 1).abs().max()),
                   "column_sum_error_max": float((value.sum(-2) - 1).abs().max()),
                   "identity_distance_fro": distance,
                   "relative_identity_distance_fro": distance / math.sqrt(channels),
                   "diagonal_mean": float(value.diagonal().mean())}
        rows.append({"layer": layer, "expert": expert, "channels": channels,
                     "representation": "hard" if hard else "soft", **row})
    return rows


def analyze_group(name, weight, state, rho):
    if not isinstance(state, DenseResidualState):
        raise ValueError("Matrix norm inspection currently requires dense expert-specific X_e")
    if weight.shape != state.residual.shape or not torch.isfinite(weight).all():
        raise ValueError(f"Invalid raw expert weights: {name}")
    layer, projection = int(name.split(".")[1]), name.rsplit(".", 1)[1]
    mapped = state.native_shared().expand_as(weight)
    # Match saved FP32 mapping/addition semantics; accumulate norms and dot products in FP64.
    reconstruction_error = fro(weight - (mapped + state.residual))
    mapped_norm, specific_norm, full_norm = fro(mapped), fro(state.residual), fro(weight)
    dual_norm = fro(state.dual)
    shared_norm = float(fro(state.shared))
    dots = (mapped.double() * state.residual.double()).sum(dim=(-2, -1))
    rows = []
    for expert in range(len(weight)):
        mapped_f, specific_f, full_f = (float(values[expert]) for values in
                                       (mapped_norm, specific_norm, full_norm))
        rows.append({
            "layer": layer, "projection": projection, "expert": expert,
            "matrix_rows": weight.shape[-2], "matrix_cols": weight.shape[-1],
            "consensus_fro": shared_norm, "mapped_consensus_fro": mapped_f,
            "expert_specific_fro": specific_f, "full_expert_fro": full_f,
            "specific_over_mapped_consensus": ratio(specific_f, mapped_f),
            "specific_over_full": ratio(specific_f, full_f),
            "mapped_consensus_over_full": ratio(mapped_f, full_f),
            "shared_specific_cosine": ratio(float(dots[expert]), mapped_f * specific_f),
            "reconstruction_fro": float(reconstruction_error[expert]),
            "reconstruction_relative_fro": ratio(float(reconstruction_error[expert]), full_f),
            "multiplier_fro": rho * float(dual_norm[expert]),
        })
    metric_keys = [key for key in rows[0] if key not in
                   ("layer", "projection", "expert", "matrix_rows", "matrix_cols", "consensus_fro")]
    group = {"layer": layer, "projection": projection, "matrix_shape": list(weight.shape[1:]),
             "num_experts": len(weight), "consensus_fro": shared_norm,
             **{key: stats(row[key] for row in rows) for key in metric_keys}}
    return group, rows, metric_keys


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def analyze_checkpoint(checkpoint, output, *, expected_config_hash=None, expected_step=None):
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty norm output directory: {output}")
    payload, states = checkpoint_states(checkpoint, device="cpu")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ValueError("Norm checkpoint differs from the evaluated configuration")
    if expected_step is not None and payload["step"] != expected_step:
        raise ValueError("Norm checkpoint differs from the evaluated step")
    config = payload["config"]
    groups, experts, permutations = [], [], []
    seen_layers = set()
    for name, state in sorted(states.items()):
        group, rows, metric_keys = analyze_group(
            name, payload["model"][name], state, config["salaad"]["rho"]
        )
        groups.append(group)
        experts.extend(rows)
        if state.permutation is not None and group["layer"] not in seen_layers:
            # checkpoint_states validates that gate/up/down use the same P_e.
            permutations.extend(permutation_rows(group["layer"], state.permutation))
            seen_layers.add(group["layer"])
    summary = {}
    for projection in ("gate", "up", "down"):
        selected = [row for row in experts if row["projection"] == projection]
        if not selected:
            continue
        summary[projection] = {
            "consensus_fro_per_group": stats(group["consensus_fro"] for group in groups
                                              if group["projection"] == projection),
            **{key: stats(row[key] for row in selected) for key in metric_keys},
            **{"stacked_" + key: math.sqrt(math.fsum(row[key] ** 2 for row in selected))
               for key in ("mapped_consensus_fro", "expert_specific_fro", "full_expert_fro")},
        }
    report = {
        "format": "salaad_moe.matrix_norms.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": payload["step"],
        "config_hash": payload["config_hash"], "weight_decay": config["training"]["weight_decay"],
        "norm": "Frobenius with FP64 accumulation over saved FP32 tensors; FP32 mapping and reconstruction",
        "mapping": "Document convention: gate/up X @ P_e; down P_e.T @ X. Saved weights are transposed.",
        "notation": {"consensus_fro": "X", "expert_specific_fro": "X_e", "full_expert_fro": "W_e",
                     "permutation_fro": "P_e matrix (not its index vector)",
                     "multiplier_fro": "Y_hat_e = rho * saved scaled dual"},
        "ratio_interpretation": "Ratios of norms are not additive knowledge or energy fractions.",
        "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "num_layers": config["model"]["num_layers"],
        "num_experts_per_layer": config["model"]["num_experts"],
        "groups_count": len(groups), "per_expert_rows": len(experts),
        "summary": summary, "groups": groups, "experts": experts,
        "permutation_summary": {
            "count": len(permutations), "identity_count": sum(row["is_identity"] for row in permutations),
            "permutation_fro": stats(row["permutation_fro"] for row in permutations),
            **{key: stats(row[key] for row in permutations) for key in
               ("identity_distance_fro", "relative_identity_distance_fro", "diagonal_mean")},
            "moved_fraction": stats(row["moved_fraction"] for row in permutations),
            "hard_permutation_singular_values": "All 1, implied by validated bijections" if
                permutations and all(row["representation"] == "hard" for row in permutations) else None,
        },
        "permutations": permutations, "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "matrix_norms.json", report)
    write_csv(output / "per_expert_norms.csv", experts)
    write_csv(output / "permutation_norms.csv", permutations)
    group_rows = []
    for group in groups:
        row = {key: group[key] for key in ("layer", "projection", "num_experts", "consensus_fro")}
        for key in metric_keys:
            row.update({f"{key}_{stat}": value for stat, value in group[key].items()})
        group_rows.append(row)
    write_csv(output / "per_layer_projection_norms.csv", group_rows)
    for projection, values in summary.items():
        print(f"  {projection}: mean ||X||F={values['consensus_fro_per_group']['mean']:.6f}, "
              f"||mapped X||F={values['mapped_consensus_fro']['mean']:.6f}, "
              f"||X_e||F={values['expert_specific_fro']['mean']:.6f}, "
              f"||W_e||F={values['full_expert_fro']['mean']:.6f}", flush=True)
    print(f"Saved {output / 'matrix_norms.json'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be >= 1")
    torch.set_num_threads(args.cpu_threads)
    return analyze_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
