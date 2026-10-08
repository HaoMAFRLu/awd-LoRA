#!/usr/bin/env python3
"""Evaluate a saved soft-Sinkhorn model after direct Hungarian projection of P_e."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from salaad_moe.alignment import native_shared, validate_permutation
from salaad_moe.data import TokenCorpus
from salaad_moe.export import checkpoint_states
from salaad_moe.model import MoELanguageModel
from salaad_moe.sinkhorn import validate_transport
from salaad_moe.solver import DenseResidualState
from salaad_moe.trainer import evaluate_model, seed_everything
from scripts.analyze_moe_matrix_norms import write_csv, write_json


@torch.no_grad()
def direct_hungarian(soft):
    """Direct maximum-weight assignment; accept the solver result without an identity comparison."""
    experts, channels, _ = soft.shape
    validate_transport(soft, experts, channels)
    scores = soft.detach().double().cpu().numpy()
    indices = torch.empty((experts, channels), dtype=torch.int64)
    for expert, score in enumerate(scores):
        shared_rows, native_columns = linear_sum_assignment(score, maximize=True)
        indices[expert, torch.from_numpy(native_columns)] = torch.from_numpy(shared_rows)
    validate_permutation(indices, experts, channels)
    return indices


@torch.no_grad()
def projected_weights(payload, states):
    """Keep X and X_e fixed, replacing only the map in native_shared(X, P_e) + X_e."""
    config = payload["config"]
    weights = dict(payload["model"])
    projected = []
    for layer in range(config["model"]["num_layers"]):
        prefix = f"layers.{layer}.moe.experts."
        indices = direct_hungarian(states[prefix + "gate"].permutation)
        projected.append(indices)
        for projection in ("gate", "up", "down"):
            name = prefix + projection
            state = states[name]
            if not isinstance(state, DenseResidualState):
                raise ValueError("This evaluation requires dense expert-specific X_e")
            if not torch.equal(state.permutation, states[prefix + "gate"].permutation):
                raise ValueError("The three projections must share the same P_e")
            mapped = native_shared(state.shared, indices.to(state.shared.device), state.channel_axis)
            weights[name] = mapped + state.residual
            if weights[name].shape != payload["model"][name].shape or not torch.isfinite(weights[name]).all():
                raise ValueError(f"Invalid projected expert weight: {name}")
    return weights, torch.stack(projected)


def evaluation_model(payload, weights, device):
    config = payload["config"]
    if (payload["format"] == "salaad_moe.megatron_evaluation.v1"
            and config["training"]["task_precision"] == "bfloat16"):
        weights = {name: value.bfloat16() for name, value in weights.items()}
    model = MoELanguageModel(config, initialize=False)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


class ProgressCorpus:
    """Preserve the evaluation API and sample order while logging completed input batches."""
    def __init__(self, corpus, label, count):
        self.corpus, self.label, self.count = corpus, label, count
        self.lengths = corpus.lengths
        self.processed = 0

    def batch(self, split, indices, device):
        batch = self.corpus.batch(split, indices, device)
        self.processed += len(indices)
        if self.processed % 1024 == 0 or self.processed == self.count:
            print(f"{self.label}: evaluating sequences {self.processed}/{self.count}", flush=True)
        return batch


def run(args):
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use a GPU worker")
    torch.set_num_threads(args.cpu_threads)
    payload, states = checkpoint_states(args.checkpoint, device="cpu")
    config = payload["config"]
    if config["salaad"]["channel_alignment"]["method"] != "sinkhorn":
        raise ValueError("Expected a saved soft Sinkhorn checkpoint")
    if args.expected_step is not None and payload["step"] != args.expected_step:
        raise ValueError("Unexpected checkpoint step")
    if args.expected_config_hash and payload["config_hash"] != args.expected_config_hash:
        raise ValueError("Unexpected checkpoint configuration")
    print("Checking corpus identity and token shard hashes...", flush=True)
    corpus = TokenCorpus(args.data_manifest, config)
    if args.expected_corpus_identity and corpus.identity != args.expected_corpus_identity:
        raise ValueError("Unexpected evaluation corpus")
    if payload.get("reader", {}).get("corpus_identity") not in (None, corpus.identity):
        raise ValueError("Training and evaluation corpus differ")
    count = min(args.sequences, corpus.lengths[args.split]) if args.sequences else corpus.lengths[args.split]
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "format": "salaad_moe.hard_projection_evaluation.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": payload["step"],
        "config_hash": payload["config_hash"], "corpus_identity": corpus.identity,
        "data_manifest": str(corpus.path), "corpus_provenance": corpus.manifest["provenance"],
        "split": args.split, "sequences": count, "debug": bool(args.sequences),
        "sequence_length": config["data"]["seq_length"], "batch_size": args.batch_size,
        "task_precision": config["training"]["task_precision"],
        "projection": "Direct maximum-weight assignment on saved P_e entries; no identity comparison or fallback",
        "intervention": "Freeze X, X_e and all other weights; reconstruct expert weights with hard P_e",
        "baseline": "Original trained raw W_e from the same checkpoint",
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "results": [],
    }
    write_json(output / "summary.json", result)
    started = time.perf_counter()
    try:
        print("Running direct Hungarian projection of all saved P_e...", flush=True)
        hard_weights, indices = projected_weights(payload, states)
        native_to_shared = indices.numpy()
        hard = np.eye(indices.shape[-1], dtype=np.uint8)[native_to_shared].swapaxes(-1, -2)
        if not ((hard.sum(-1) == 1).all() and (hard.sum(-2) == 1).all()):
            raise AssertionError("Invalid binary permutation matrices")
        np.savez_compressed(output / "hard_permutations.npz", P_e=hard,
                            native_to_shared=native_to_shared,
                            shared_to_native=native_to_shared.argsort(axis=-1))
        result["projected_matrix_count"] = indices.shape[0] * indices.shape[1]
        result["hard_matrix_shape"] = list(hard.shape)
        result["hard_matrix_convention"] = "P_e[layer, expert, shared, native]; P_e[native_to_shared[b], b]=1"
        write_json(output / "summary.json", result)
        for label, weights in (("hungarian_projected", hard_weights), ("original_raw", payload["model"])):
            seed_everything(config["seed"])
            print(f"Evaluating {label} on {count} {args.split} sequences...", flush=True)
            model = evaluation_model(payload, weights, device)
            evaluation_started = time.perf_counter()
            try:
                metrics = evaluate_model(model, ProgressCorpus(corpus, label, count), args.split,
                                         args.sequences, args.batch_size, config, device, distributed=False)
            finally:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            record = {"name": label, **metrics, "elapsed_seconds": time.perf_counter() - evaluation_started}
            result["results"].append(record)
            write_json(output / (label + ".json"), record)
            write_csv(output / "summary.csv", result["results"])
            write_json(output / "summary.json", result)
            print(f"{label}: NLL={metrics['nll']:.6f}, PPL={metrics['perplexity']:.6f}", flush=True)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sequences", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--expected-corpus-identity")
    args = parser.parse_args(argv)
    if args.sequences < 0 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("Sequences must be nonnegative; batch size and CPU threads must be positive")
    return run(args)


if __name__ == "__main__":
    main()
