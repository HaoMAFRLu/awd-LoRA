#!/usr/bin/env python3
"""Check whether each saved soft P_e has its row maxima on the diagonal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from salaad_moe.alignment import validate_permutation
from salaad_moe.export import checkpoint_states
from salaad_moe.sinkhorn import project_transport_to_permutation
from scripts.analyze_moe_matrix_norms import permutation_rows, stats, write_csv, write_json


def row_maxima(layer, permutation):
    """Use exact saved values; inclusive maxima and strict maxima handle ties separately."""
    if permutation.ndim != 3 or permutation.shape[-2] != permutation.shape[-1]:
        raise ValueError("Expected dense soft P_e with shape [experts, channels, channels]")
    if permutation.shape[-1] < 2 or not torch.isfinite(permutation).all():
        raise ValueError("Expected finite matrices with at least two channels")
    value = permutation.detach().double().cpu()
    diagonal = value.diagonal(dim1=-2, dim2=-1)
    maxima, columns = value.max(dim=-1)
    ties = (value == maxima.unsqueeze(-1)).sum(-1)
    off_diagonal = value.clone()
    off_diagonal.diagonal(dim1=-2, dim2=-1).fill_(-torch.inf)
    off_maxima, off_columns = off_diagonal.max(dim=-1)
    ranks = (value > diagonal.unsqueeze(-1)).sum(-1) + 1
    fields = [diagonal, maxima, columns, ties, off_maxima, off_columns, ranks]
    diagonal, maxima, columns, ties, off_maxima, off_columns, ranks = [v.tolist() for v in fields]
    rows = []
    for expert in range(len(value)):
        for row in range(value.shape[-1]):
            diag, maximum = diagonal[expert][row], maxima[expert][row]
            rows.append({
                "layer": layer, "expert": expert, "row": row,
                "argmax_column": columns[expert][row], "diagonal_value": diag,
                "row_max_value": maximum, "row_max_ties": ties[expert][row],
                "diagonal_is_max": diag == maximum,
                "diagonal_is_strict_max": diag > off_maxima[expert][row],
                "largest_off_diagonal_column": off_columns[expert][row],
                "largest_off_diagonal_value": off_maxima[expert][row],
                "diagonal_minus_off_diagonal_max": diag - off_maxima[expert][row],
                "diagonal_rank": ranks[expert][row],
            })
    return rows


def summarize(rows):
    count = len(rows)
    inclusive = sum(row["diagonal_is_max"] for row in rows)
    strict = sum(row["diagonal_is_strict_max"] for row in rows)
    return {
        "row_count": count, "diagonal_max_rows": inclusive,
        "diagonal_strict_max_rows": strict, "diagonal_max_tie_rows": inclusive - strict,
        "off_diagonal_wins_rows": count - inclusive,
        "any_max_tie_rows": sum(row["row_max_ties"] > 1 for row in rows),
        "diagonal_max_fraction": inclusive / count,
        "diagonal_strict_max_fraction": strict / count,
        "mean_diagonal_value": stats(row["diagonal_value"] for row in rows)["mean"],
        "mean_row_max_value": stats(row["row_max_value"] for row in rows)["mean"],
    }


@torch.no_grad()
def project_hungarian(layer, permutation, *, tie_rule="identity"):
    """Project soft entries, optionally accepting the solver's choice on exact ties."""
    experts, channels, _ = permutation.shape
    identity = torch.arange(channels, device=permutation.device).expand(experts, -1).clone()
    if tie_rule == "solver":
        from scripts.evaluate_moe_hard_projection import direct_hungarian

        indices = direct_hungarian(permutation).to(permutation.device)
    elif tie_rule == "identity":
        indices = project_transport_to_permutation(permutation, identity)
    else:
        raise ValueError(f"Unknown Hungarian tie rule: {tie_rule}")
    validate_permutation(indices, experts, channels)
    scores = permutation.double()
    expert_ids = torch.arange(experts, device=permutation.device)[:, None]
    native_ids = torch.arange(channels, device=permutation.device)[None, :]
    optimal = scores[expert_ids, indices, native_ids].sum(-1)
    identity_score = scores.diagonal(dim1=-2, dim2=-1).sum(-1)
    independent_row_max = scores.amax(-1).sum(-1)
    moved = (indices != identity).sum(-1)
    rows = []
    for expert in range(experts):
        gain = float(optimal[expert] - identity_score[expert])
        if gain < 0:
            raise AssertionError("Projected assignment scores below identity")
        rows.append({
            "layer": layer, "expert": expert, "channels": channels,
            "is_identity": int(moved[expert]) == 0,
            "moved_channels": int(moved[expert]),
            "moved_fraction": float(moved[expert]) / channels,
            "identity_score": float(identity_score[expert]),
            "optimal_score": float(optimal[expert]), "score_gain_over_identity": gain,
            "independent_row_max_upper_bound": float(independent_row_max[expert]),
        })
    return rows, indices.cpu()


def projection_summary(matrices):
    return {
        "matrix_count": len(matrices),
        "identity_count": sum(row["is_identity"] for row in matrices),
        "non_identity_count": sum(not row["is_identity"] for row in matrices),
        "total_channels": sum(row["channels"] for row in matrices),
        "total_moved_channels": sum(row["moved_channels"] for row in matrices),
        **{key: stats(row[key] for row in matrices) for key in
           ("moved_channels", "moved_fraction", "identity_score", "optimal_score", "score_gain_over_identity")},
    }


@torch.no_grad()
def analyze(checkpoint, output, *, expected_step=None, expected_config_hash=None,
            with_hungarian=False, hungarian_tie_rule="identity"):
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    payload, states = checkpoint_states(checkpoint, device="cpu")
    if expected_step is not None and payload["step"] != expected_step:
        raise ValueError("Unexpected checkpoint step")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ValueError("Unexpected checkpoint configuration")
    rows, matrices, layers, soft_matrices = [], [], [], []
    hard_rows, hard_indices = [], []
    for layer in range(payload["config"]["model"]["num_layers"]):
        # checkpoint_states checks that gate/up/down share exactly the same P_e.
        state = states[f"layers.{layer}.moe.experts.gate"]
        soft_matrices.extend(permutation_rows(layer, state.permutation))
        if with_hungarian:
            projected_rows, projected_indices = project_hungarian(
                layer, state.permutation, tie_rule=hungarian_tie_rule)
            hard_rows.extend(projected_rows)
            hard_indices.append(projected_indices)
        layer_rows = row_maxima(layer, state.permutation)
        rows.extend(layer_rows)
        layer_matrices = []
        for expert in range(state.permutation.shape[0]):
            expert_rows = [row for row in layer_rows if row["expert"] == expert]
            layer_matrices.append({"layer": layer, "expert": expert, **summarize(expert_rows)})
        matrices.extend(layer_matrices)
        layers.append({"layer": layer, **summarize(layer_rows),
                       "all_rows_diagonal_max_matrices": sum(m["off_diagonal_wins_rows"] == 0 for m in layer_matrices)})
    exceptions = sorted((row for row in rows if not row["diagonal_is_max"]),
                        key=lambda row: row["diagonal_minus_off_diagonal_max"])[:16]
    report = {
        "format": "salaad_moe.pe_row_maxima.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": payload["step"],
        "config_hash": payload["config_hash"],
        "orientation": "P_e rows are shared channels, columns are native channels; no transpose",
        "comparison": "Exact saved FP32 values promoted to FP64; ties counted separately",
        "matrix_count": len(matrices), "summary": summarize(rows),
        "soft_matrix_summary": {
            "identity_count": sum(row["is_identity"] for row in soft_matrices),
            **{key: stats(row[key] for row in soft_matrices) for key in
               ("permutation_fro", "identity_distance_fro", "row_sum_error_max", "column_sum_error_max")},
        },
        "all_rows_diagonal_max_matrices": sum(m["off_diagonal_wins_rows"] == 0 for m in matrices),
        "all_rows_diagonal_strict_max_matrices": sum(m["diagonal_strict_max_rows"] == m["row_count"] for m in matrices),
        "diagonal_max_fraction_per_matrix": stats(m["diagonal_max_fraction"] for m in matrices),
        "layers": layers, "matrices": matrices, "largest_off_diagonal_wins": exceptions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    if with_hungarian:
        import numpy as np

        indices = torch.stack(hard_indices).numpy()
        hard_matrices = np.eye(indices.shape[-1], dtype=np.uint8)[indices].swapaxes(-1, -2)
        if not ((hard_matrices.sum(-1) == 1).all() and (hard_matrices.sum(-2) == 1).all()):
            raise AssertionError("Hard matrices must have exactly one 1 per row and column")
        projection = {
            "objective": "Maximize sum of selected saved soft P_e entries; no logarithms",
            "tie_rule": ("Accept solver result directly; no identity fallback" if hungarian_tie_rule == "solver"
                         else "Keep identity if it is also optimal"),
            "indices_convention": "P_e[native_to_shared[b], b] = 1; all other entries are zero",
            "matrix_array": "hard_permutations.npz:P_e; uint8 [layer, expert, shared_row, native_column]",
            "summary": projection_summary(hard_rows),
            "layers": [{"layer": layer["layer"], **projection_summary(
                [row for row in hard_rows if row["layer"] == layer["layer"]])} for layer in layers],
            "matrices": hard_rows,
        }
        report["hungarian_projection"] = projection
        np.savez_compressed(output / "hard_permutations.npz",
                            P_e=hard_matrices, native_to_shared=indices,
                            shared_to_native=indices.argsort(axis=-1))
        write_json(output / "hungarian_projection.json", projection)
        write_csv(output / "hungarian_per_matrix.csv", hard_rows)
        print("Hungarian projection:", projection["summary"], flush=True)
    write_json(output / "pe_row_maxima.json", report)
    write_csv(output / "per_matrix.csv", matrices)
    write_csv(output / "soft_matrix_norms.csv", soft_matrices)
    write_csv(output / "per_layer.csv", layers)
    write_csv(output / "per_row.csv", rows)
    print(report["summary"], flush=True)
    print(f"Saved {output / 'pe_row_maxima.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--project-hungarian", action="store_true",
                        help="Also save the closest hard permutations and compare them with identity")
    parser.add_argument("--hungarian-tie-rule", choices=("identity", "solver"), default="identity",
                        help="Preserve the existing identity tie preference or accept the direct solver result")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be >= 1")
    torch.set_num_threads(args.cpu_threads)
    analyze(args.checkpoint, args.output, expected_step=args.expected_step,
            expected_config_hash=args.expected_config_hash, with_hungarian=args.project_hungarian,
            hungarian_tie_rule=args.hungarian_tie_rule)


if __name__ == "__main__":
    main()
