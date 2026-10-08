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

from salaad_moe.export import checkpoint_states
from scripts.analyze_moe_matrix_norms import stats, write_csv, write_json


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
def analyze(checkpoint, output, *, expected_step=None, expected_config_hash=None):
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    payload, states = checkpoint_states(checkpoint, device="cpu")
    if expected_step is not None and payload["step"] != expected_step:
        raise ValueError("Unexpected checkpoint step")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ValueError("Unexpected checkpoint configuration")
    rows, matrices, layers = [], [], []
    for layer in range(payload["config"]["model"]["num_layers"]):
        # checkpoint_states checks that gate/up/down share exactly the same P_e.
        state = states[f"layers.{layer}.moe.experts.gate"]
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
        "all_rows_diagonal_max_matrices": sum(m["off_diagonal_wins_rows"] == 0 for m in matrices),
        "all_rows_diagonal_strict_max_matrices": sum(m["diagonal_strict_max_rows"] == m["row_count"] for m in matrices),
        "diagonal_max_fraction_per_matrix": stats(m["diagonal_max_fraction"] for m in matrices),
        "layers": layers, "matrices": matrices, "largest_off_diagonal_wins": exceptions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pe_row_maxima.json", report)
    write_csv(output / "per_matrix.csv", matrices)
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
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be >= 1")
    torch.set_num_threads(args.cpu_threads)
    analyze(args.checkpoint, args.output, expected_step=args.expected_step,
            expected_config_hash=args.expected_config_hash)


if __name__ == "__main__":
    main()
