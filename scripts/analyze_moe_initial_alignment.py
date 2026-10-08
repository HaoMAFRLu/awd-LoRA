#!/usr/bin/env python3
"""Inspect direct channel-similarity assignments on the original step-zero weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from salaad_moe.alignment import PROJECTIONS, validate_permutation
from salaad_moe.config import fingerprint, load_config, validate_config
from salaad_moe.model import MoELanguageModel
from salaad_moe.trainer import seed_everything
from scripts.analyze_moe_matrix_norms import stats, write_csv, write_json


def descriptors(weights):
    """A channel contains its gate row, up row, and down column in stored coordinates."""
    gate, up, down = (weights[p] for p in PROJECTIONS)
    if gate.ndim != 3 or up.shape != gate.shape or down.shape != gate.mT.shape:
        raise ValueError("Expected matching [expert, channel, hidden] SwiGLU triplets")
    value = torch.cat((gate, up, down.mT), dim=-1).double()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite channel descriptors")
    return value


@torch.no_grad()
def match_similarity(value, template, metric):
    """Return native-to-shared indices, using the solver result without a tie fallback."""
    if value.ndim != 3 or template.shape != value.shape[1:]:
        raise ValueError("Expected [expert, native_channel, features] and [shared_channel, features]")
    value, template = value.double(), template.double()
    if metric == "cosine":
        norms, template_norms = value.norm(dim=-1, keepdim=True), template.norm(dim=-1, keepdim=True)
        if (norms == 0).any() or (template_norms == 0).any():
            raise ValueError("Cosine similarity is undefined for a zero channel")
        value, template = value / norms, template / template_norms
    elif metric != "squared_l2":
        raise ValueError(f"Unknown similarity metric: {metric}")
    # For a complete one-to-one assignment, the sum of squared norms is
    # constant: minimizing squared L2 distance is equivalent to maximizing dot products.
    scores = (value @ template.T).cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite channel similarity scores")
    experts, channels, _ = scores.shape
    indices = np.empty((experts, channels), dtype=np.int64)
    records = []
    for expert, score in enumerate(scores):
        native, shared = linear_sum_assignment(score, maximize=True)
        indices[expert, native] = shared
        identity_score = np.trace(score)
        optimal_score = score[native, shared].sum()
        moved = int(np.count_nonzero(indices[expert] != np.arange(channels)))
        records.append({
            "expert": expert, "channels": channels, "is_identity": moved == 0,
            "moved_channels": moved, "moved_fraction": moved / channels,
            "identity_score": float(identity_score), "optimal_score": float(optimal_score),
            "score_gain_over_identity": float(optimal_score - identity_score),
        })
    validate_permutation(torch.from_numpy(indices), experts, channels)
    return indices, records


def summarize(records):
    count = len(records)
    total = sum(row["channels"] for row in records)
    moved = sum(row["moved_channels"] for row in records)
    identities = sum(row["is_identity"] for row in records)
    return {
        "matrix_count": count, "identity_count": identities,
        "non_identity_count": count - identities, "total_channels": total,
        "moved_channels": moved, "moved_fraction": moved / total if total else None,
        "moved_channels_per_matrix": stats(row["moved_channels"] for row in records),
        "score_gain_over_identity": stats(row["score_gain_over_identity"] for row in records),
    }


@torch.no_grad()
def analyze(config, output, *, device="cpu", reference_expert=0):
    validate_config(config)
    if not 0 <= reference_expert < config["model"]["num_experts"]:
        raise ValueError("Reference expert is out of range")
    output, device = Path(output), torch.device(device)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a GPU worker")
    started = time.perf_counter()
    # Match Trainer.__init__: seed, construct and initialize the whole model on
    # CPU, then transfer it to the device. No optimizer, residual, or W&B run is created.
    seed_everything(config["seed"])
    model = MoELanguageModel(config)
    initial_hashes = {
        f"layer_{layer}_{projection}": hashlib.sha256(
            getattr(block.moe.experts, projection).detach().numpy().tobytes()).hexdigest()
        for layer, block in enumerate(model.layers) for projection in PROJECTIONS
    }
    model = model.to(device)
    cases = {f"{template}_{metric}": {"records": [], "indices": []}
             for template in ("unaligned_mean", "reference_expert") for metric in ("cosine", "squared_l2")}
    for layer, block in enumerate(model.layers):
        weights = {p: getattr(block.moe.experts, p).detach() for p in PROJECTIONS}
        value = descriptors(weights)
        # Compute the current consensus in FP32, as the real initializer does.
        mean = descriptors({p: w.mean(0, keepdim=True) for p, w in weights.items()})[0]
        templates = {"unaligned_mean": mean, "reference_expert": value[reference_expert]}
        for template_name, template in templates.items():
            for metric in ("cosine", "squared_l2"):
                key = f"{template_name}_{metric}"
                indices, records = match_similarity(value, template, metric)
                cases[key]["indices"].append(indices)
                cases[key]["records"].extend({"layer": layer, **row} for row in records)
        print(f"Matched initial channels in layer {layer + 1}/{len(model.layers)}", flush=True)
    # This is an observation of initialized weights, not a training intervention.
    for layer, block in enumerate(model.layers):
        for projection in PROJECTIONS:
            digest = hashlib.sha256(getattr(block.moe.experts, projection).detach().cpu().numpy().tobytes()).hexdigest()
            if digest != initial_hashes[f"layer_{layer}_{projection}"]:
                raise AssertionError("Initialization analysis modified expert weights")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "salaad_moe.initial_channel_alignment.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "step": 0,
        "seed": config["seed"], "config_hash": fingerprint(config),
        "model_initialization": "Same seed and CPU model constructor as Trainer; no training steps",
        "descriptor": "Concatenate stored gate row, up row, down column; no separate projection normalization",
        "score_orientation": "Rows are native expert channels; columns are template/shared channels",
        "assignment": "Direct scipy linear_sum_assignment(maximize=True); no identity fallback or reference exemption",
        "scope": "One matching pass on raw W_e before constructing X_e; no alternating consensus refinements",
        "reference_expert": reference_expert, "wandb_initialized": False,
        "expert_weights_unchanged": True, "initial_expert_sha256": initial_hashes,
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "cases": {},
    }
    arrays = {}
    for name, case in cases.items():
        indices = np.stack(case["indices"])
        hard = np.eye(indices.shape[-1], dtype=np.uint8)[indices].swapaxes(-1, -2)
        if not ((hard.sum(-1) == 1).all() and (hard.sum(-2) == 1).all()):
            raise AssertionError("Invalid binary permutation matrices")
        arrays[f"{name}_P_e"] = hard
        arrays[f"{name}_native_to_shared"] = indices
        rows = case["records"]
        result = {
            "summary": summarize(rows),
            "excluding_reference_expert": summarize([row for row in rows if row["expert"] != reference_expert]),
            "layers": [{"layer": layer, **summarize([row for row in rows if row["layer"] == layer])}
                       for layer in range(len(model.layers))],
        }
        report["cases"][name] = result
        write_csv(output / f"{name}_per_matrix.csv", rows)
        print(f"{name}: {result['summary']}", flush=True)
    np.savez_compressed(output / "initial_permutations.npz", **arrays)
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", report)
    write_json(output / "config.resolved.json", config)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-expert", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    analyze(load_config(args.config), args.output, device=args.device, reference_expert=args.reference_expert)


if __name__ == "__main__":
    main()
