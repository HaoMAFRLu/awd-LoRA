"""Evaluate Consensus SALAAD structure variants on identical fixed C4 batches."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

import datasets.distributed
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salad.register import get_data, get_model, get_preprocessed_dataset, get_tokenizer
from salad.consensus import compose_decomposition, magnitude_prune_sparse
from models.consensus import ConsensusLinear
from models.sparse_loop import SparseLoopLinear
from salad.sparse_loop import apply_sparse_residuals
from salad.utils import hf_login_once, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "llama_consensus_structures_c4_eval.yaml",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _init_distributed() -> Tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("C4 evaluation requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {path}")
    if state_dict and all(name.startswith("module.") for name in state_dict):
        state_dict = {
            name.removeprefix("module."): value for name, value in state_dict.items()
        }
    return state_dict


def _load_consensus_states(directory: Path) -> Dict[str, Dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"consensus state directory does not exist: {directory}")
    paths = sorted(directory.glob("consensus_rank*.pth"))
    if not paths:
        raise FileNotFoundError(f"no consensus_rank*.pth files found in {directory}")

    states: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        try:
            rank_states = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            rank_states = torch.load(path, map_location="cpu")
        if not isinstance(rank_states, dict):
            raise TypeError(f"consensus state file is not a dictionary: {path}")
        duplicate_names = set(states).intersection(rank_states)
        if duplicate_names:
            raise KeyError(
                f"duplicate consensus states in {path}: {sorted(duplicate_names)}"
            )
        states.update(rank_states)
    return states


def _load_sparse_loop_states(directory: Path) -> Dict[str, Dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"sparse-loop state directory does not exist: {directory}"
        )
    paths = sorted(directory.glob("sparse_loop_rank*.pth"))
    if not paths:
        raise FileNotFoundError(
            f"no sparse_loop_rank*.pth files found in {directory}"
        )

    states: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        try:
            rank_states = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            rank_states = torch.load(path, map_location="cpu")
        if not isinstance(rank_states, dict):
            raise TypeError(f"sparse-loop state file is not a dictionary: {path}")
        duplicate_names = set(states).intersection(rank_states)
        if duplicate_names:
            raise KeyError(
                f"duplicate sparse-loop states in {path}: "
                f"{sorted(duplicate_names)}"
            )
        states.update(rank_states)
    return states


@torch.no_grad()
def _apply_reconstruction(
    model: torch.nn.Module,
    states: Dict[str, Dict[str, Any]],
    reference_weights: Dict[str, torch.Tensor],
    components: Tuple[str, ...],
    sparse_density: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    modules = dict(model.named_modules())
    expected = {
        name
        for name, module in modules.items()
        if isinstance(module, ConsensusLinear)
    }
    provided = set(states)
    if provided != expected:
        raise KeyError(
            "consensus state mismatch; "
            f"missing={sorted(expected - provided)}, "
            f"unexpected={sorted(provided - expected)}"
        )

    squared_error = torch.zeros(
        (),
        dtype=torch.float64,
        device=next(model.parameters()).device,
    )
    squared_weight = torch.zeros_like(squared_error)
    sparse_nonzero = 0
    sparse_elements = 0
    for name, state in states.items():
        module = modules.get(name)
        if not isinstance(module, ConsensusLinear):
            raise KeyError(f"model has no ConsensusLinear named {name!r}")

        state_for_composition = state
        if sparse_density is not None:
            state_for_composition = dict(state)
            state_for_composition["sparse"] = magnitude_prune_sparse(
                state["sparse"],
                sparse_density,
            )
        if "sparse" in components:
            sparse = state_for_composition["sparse"]
            sparse_nonzero += int(torch.count_nonzero(sparse).item())
            sparse_elements += sparse.numel()

        reconstructed = compose_decomposition(
            state_for_composition,
            components,
        ).to(
            device=module.weight.device,
            dtype=torch.float32,
        )
        original = reference_weights[name]
        squared_error += (original - reconstructed).square().sum(dtype=torch.float64)
        squared_weight += original.square().sum(dtype=torch.float64)
        module.weight.copy_(reconstructed.to(dtype=module.weight.dtype))

    relative_error = torch.sqrt(
        squared_error / squared_weight.clamp_min(1.0e-24)
    ).item()
    actual_sparse_density = (
        sparse_nonzero / sparse_elements if sparse_elements else None
    )
    return relative_error, actual_sparse_density


@torch.no_grad()
def _copy_consensus_weights(
    model: torch.nn.Module,
    states: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    weights: Dict[str, torch.Tensor] = {}
    for name in states:
        module = modules.get(name)
        if not isinstance(module, ConsensusLinear):
            raise KeyError(f"model has no ConsensusLinear named {name!r}")
        weights[name] = module.weight.detach().to(dtype=torch.float32).clone()
    return weights


@torch.no_grad()
def _copy_sparse_loop_effective_weights(
    model: torch.nn.Module,
    states: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    expected = {
        name
        for name, module in modules.items()
        if isinstance(module, SparseLoopLinear)
    }
    provided = set(states)
    if provided != expected:
        raise KeyError(
            "specific sparsity state mismatch; "
            f"missing={sorted(expected - provided)}, "
            f"unexpected={sorted(provided - expected)}"
        )

    return {
        name: (
            modules[name].weight.detach().float().unsqueeze(0)
            + modules[name].specific_weight.detach().float()
        ).clone()
        for name in states
    }


@torch.no_grad()
def _apply_sparse_loop_reconstruction(
    model: torch.nn.Module,
    states: Dict[str, Dict[str, Any]],
    reference_weights: Dict[str, torch.Tensor],
    sparse_density: Optional[float] = None,
) -> Tuple[float, float, Optional[float]]:
    modules = dict(model.named_modules())
    squared_error = torch.zeros(
        (),
        dtype=torch.float64,
        device=next(model.parameters()).device,
    )
    squared_weight = torch.zeros_like(squared_error)
    sparse_nonzero = 0
    sparse_elements = 0
    target_rates = set()
    materialized_states: Dict[str, Dict[str, Any]] = {}

    for name, state in states.items():
        module = modules.get(name)
        if not isinstance(module, SparseLoopLinear):
            raise KeyError(f"model has no SparseLoopLinear named {name!r}")
        sparse = state.get("sparse")
        if not isinstance(sparse, torch.Tensor):
            raise TypeError(f"saved sparse residual for {name!r} must be a tensor")
        if sparse.shape != module.specific_weight.shape:
            raise ValueError(
                f"{name} shape mismatch: {tuple(sparse.shape)} != "
                f"{tuple(module.specific_weight.shape)}"
            )

        if sparse_density is not None:
            sparse = magnitude_prune_sparse(sparse, sparse_density)
        materialized_state = dict(state)
        materialized_state["sparse"] = sparse
        materialized_states[name] = materialized_state
        sparse = sparse.to(device=module.weight.device, dtype=torch.float32)
        reconstructed = module.weight.detach().float().unsqueeze(0) + sparse
        original = reference_weights[name]
        squared_error += (original - reconstructed).square().sum(dtype=torch.float64)
        squared_weight += original.square().sum(dtype=torch.float64)
        sparse_nonzero += int(torch.count_nonzero(sparse).item())
        sparse_elements += sparse.numel()
        if "rate_sparsity" in state:
            target_rates.add(float(state["rate_sparsity"]))

    if len(target_rates) > 1:
        raise ValueError(
            "saved sparse-loop states disagree on rate_sparsity: "
            f"{sorted(target_rates)}"
        )
    apply_sparse_residuals(model, materialized_states, strict=True)
    relative_error = torch.sqrt(
        squared_error / squared_weight.clamp_min(1.0e-24)
    ).item()
    actual_density = sparse_nonzero / sparse_elements
    target_density = (
        sparse_density
        if sparse_density is not None
        else next(iter(target_rates)) if target_rates else None
    )
    return relative_error, actual_density, target_density


def _get_sparse_loop_variants(
    specification: Dict[str, Any],
) -> List[Dict[str, Any]]:
    variants = specification.get("sparse_loop_variants")
    if variants is None:
        return [{"name": "x_plus_s", "sparse_density": None}]
    if not isinstance(variants, list) or not variants:
        raise ValueError("sparse_loop_variants must be a non-empty list")

    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise TypeError(
                "each sparse-loop reconstruction variant must be a dictionary"
            )
        name = variant.get("name")
        sparse_density = variant.get("sparse_density")
        if not isinstance(name, str) or not name:
            raise ValueError("each sparse-loop variant requires a name")
        if name in seen_names:
            raise ValueError(f"duplicate sparse-loop variant name: {name!r}")
        if sparse_density is not None:
            if isinstance(sparse_density, bool) or not isinstance(
                sparse_density, (int, float)
            ):
                raise TypeError(
                    f"sparse-loop variant {name!r} sparse_density must be "
                    "a real number"
                )
            sparse_density = float(sparse_density)
            if not 0.0 <= sparse_density <= 1.0:
                raise ValueError(
                    f"sparse-loop variant {name!r} sparse_density must be "
                    f"in [0, 1], got {sparse_density}"
                )
        seen_names.add(name)
        normalized.append({
            "name": name,
            "sparse_density": sparse_density,
        })
    return normalized


def _get_decomposition_variants(
    specification: Dict[str, Any],
    states: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    variants = specification.get("decomposition_variants")
    if variants is None:
        components = ("shared", "low_rank", "sparse")
        if states:
            saved_components = {
                tuple(state.get("components", components))
                for state in states.values()
            }
            if len(saved_components) != 1:
                raise ValueError(
                    "all consensus states must use the same components"
                )
            components = next(iter(saved_components))
        return [
            {
                "name": "reconstructed",
                "components": components,
                "sparse_density": None,
            }
        ]
    if not isinstance(variants, list) or not variants:
        raise ValueError("decomposition_variants must be a non-empty list")

    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise TypeError("each decomposition variant must be a dictionary")
        name = variant.get("name")
        components = variant.get("components")
        sparse_density = variant.get("sparse_density")
        if not isinstance(name, str) or not name:
            raise ValueError("each decomposition variant requires a name")
        if name in seen_names:
            raise ValueError(f"duplicate decomposition variant name: {name!r}")
        if (
            not isinstance(components, list)
            or not components
            or not all(isinstance(component, str) for component in components)
        ):
            raise ValueError(
                f"decomposition variant {name!r} requires a non-empty "
                "component list"
            )
        # Validate component names and state-independent behavior immediately.
        allowed = {"shared", "low_rank", "sparse"}
        unknown = set(components) - allowed
        if unknown:
            raise ValueError(
                f"decomposition variant {name!r} has unknown components: "
                f"{sorted(unknown)}"
            )
        if sparse_density is not None:
            if isinstance(sparse_density, bool) or not isinstance(
                sparse_density, (int, float)
            ):
                raise TypeError(
                    f"decomposition variant {name!r} sparse_density must be "
                    "a real number"
                )
            sparse_density = float(sparse_density)
            if not 0.0 <= sparse_density <= 1.0:
                raise ValueError(
                    f"decomposition variant {name!r} sparse_density must be "
                    f"in [0, 1], got {sparse_density}"
                )
            if "sparse" not in components:
                raise ValueError(
                    f"decomposition variant {name!r} specifies sparse_density "
                    "without selecting the sparse component"
                )
        seen_names.add(name)
        normalized.append({
            "name": name,
            "components": tuple(components),
            "sparse_density": sparse_density,
        })
    return normalized


def _build_model(
    specification: Dict[str, Any],
    device: torch.device,
    precision: torch.dtype,
) -> torch.nn.Module:
    model = get_model(str(_resolve_path(specification["model_config"])))
    checkpoint = _resolve_path(specification["checkpoint"])
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)

    decoder = model.model
    observed = {
        "physical_depth": len(decoder.layers),
        "logical_depth": decoder.logical_num_layers,
        "num_loops": decoder.num_loops,
        "blocks_per_loop": len(decoder.loop_layers),
    }
    for key, value in observed.items():
        expected = specification.get(key)
        if expected is not None and int(expected) != value:
            raise ValueError(
                f"{specification['name']} expected {key}={expected}, observed {value}"
            )

    model.to(device=device, dtype=precision)
    model.eval()
    return model


def _cache_eval_batches(
    config: Dict[str, Any],
    tokenizer,
    rank: int,
    world_size: int,
) -> Tuple[List[Dict[str, torch.Tensor]], str]:
    global_batch_size = int(config["batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by "
            f"world_size ({world_size})"
        )
    local_batch_size = global_batch_size // world_size
    num_eval_batches = int(config["num_eval_batches"])
    if num_eval_batches <= 0:
        raise ValueError("num_eval_batches must be positive")

    data = get_data(config)
    data = datasets.distributed.split_dataset_by_node(
        data,
        rank=rank,
        world_size=world_size,
    )
    dataset = get_preprocessed_dataset(data, tokenizer, config, local_batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(config.get("num_workers", 0)),
    )

    batches: List[Dict[str, torch.Tensor]] = []
    fingerprint = hashlib.sha256()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_eval_batches:
            break
        cached_batch = {
            "input_ids": batch["input_ids"].to(dtype=torch.int32).contiguous(),
            "attention_mask": batch["attention_mask"]
            .to(dtype=torch.uint8)
            .contiguous(),
        }
        batches.append(cached_batch)
        fingerprint.update(cached_batch["input_ids"].numpy().tobytes())

    if len(batches) != num_eval_batches:
        raise RuntimeError(
            f"rank {rank} received only {len(batches)} C4 batches; "
            f"expected {num_eval_batches}"
        )
    return batches, fingerprint.hexdigest()


@torch.inference_mode()
def _evaluate_model(
    model: torch.nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    dist.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for cached_batch in batches:
        input_ids = cached_batch["input_ids"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        attention_mask = cached_batch["attention_mask"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        valid_tokens = attention_mask[:, 1:].sum()
        if valid_tokens.item() == 0:
            continue

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        totals[0] += loss.detach().to(torch.float64) * valid_tokens
        totals[1] += valid_tokens
        totals[2] += input_ids.shape[0]
        totals[3] += 1

    torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        time.perf_counter() - started,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    token_count = int(totals[1].item())
    if token_count == 0:
        raise RuntimeError("evaluation produced no valid next-token targets")
    average_loss = (totals[0] / totals[1]).item()
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
        "tokens": token_count,
        "examples": int(totals[2].item()),
        "batches": int(totals[3].item()) // dist.get_world_size(),
        "seconds": elapsed.item(),
        "tokens_per_second": token_count / elapsed.item(),
    }


def _write_results(
    output_directory: Path,
    config: Dict[str, Any],
    fingerprints: List[str],
    results: List[Dict[str, Any]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config["data"],
        "seed": int(config["seed"]),
        "seed_for_shuffle": int(config["seed_for_shuffle"]),
        "global_batch_size": int(config["batch_size"]),
        "num_eval_batches": int(config["num_eval_batches"]),
        "rank_batch_fingerprints": fingerprints,
        "results": results,
    }
    with (output_directory / "results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    with (output_directory / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output_directory / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def _format_density(value: Optional[float]) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _print_results(results: List[Dict[str, Any]]) -> None:
    print("\nConsensus structure evaluation")
    print("=" * 178)
    print(
        f"{'condition':<58} {'source':>28} {'loops':>5} {'blocks':>6} {'physical':>8} "
        f"{'logical':>7} {'loss':>10} {'delta':>10} {'ppl':>10} "
        f"{'S target':>10} {'S actual':>10} {'tokens':>12} {'tokens/s':>12}"
    )
    print("-" * 178)
    for result in results:
        print(
            f"{result['condition']:<58} {result['weight_source']:>28} "
            f"{result['num_loops']:>5d} "
            f"{result['blocks_per_loop']:>6d} {result['physical_depth']:>8d} "
            f"{result['logical_depth']:>7d} {result['loss']:>10.6f} "
            f"{result['loss_delta_vs_vanilla']:>10.6f} "
            f"{result['perplexity']:>10.4f} "
            f"{_format_density(result['sparse_density_target']):>10} "
            f"{_format_density(result['sparse_density_actual']):>10} "
            f"{result['tokens']:>12d} "
            f"{result['tokens_per_second']:>12.0f}"
        )
    print("=" * 178)


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(str(args.config))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    rank, world_size, device = _init_distributed()
    set_seed(int(config["seed"]))
    hf_login_once()

    precisions = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    precision_name = config.get("precision", "bfloat16")
    if precision_name not in precisions:
        raise ValueError(f"unsupported precision: {precision_name!r}")
    precision = precisions[precision_name]

    tokenizer = get_tokenizer(int(config["max_length"]), config)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")

    if rank == 0:
        print(
            f"C4 evaluation | world_size={world_size} | seed={config['seed']} | "
            f"shuffle_seed={config['seed_for_shuffle']} | "
            f"global_batch={config['batch_size']} | "
            f"eval_batches={config['num_eval_batches']}"
        )
    batches, fingerprint = _cache_eval_batches(
        config,
        tokenizer,
        rank,
        world_size,
    )
    fingerprints: List[str] = [""] * world_size
    dist.all_gather_object(fingerprints, fingerprint)

    results: List[Dict[str, Any]] = []
    for specification in config["checkpoints"]:
        if rank == 0:
            print(f"\nEvaluating {specification['name']}...")
        model = _build_model(specification, device, precision)
        decoder = model.model
        has_consensus_reconstruction = (
            "consensus_state_directory" in specification
        )
        has_sparse_loop_reconstruction = (
            "sparse_loop_state_directory" in specification
        )
        if has_consensus_reconstruction and has_sparse_loop_reconstruction:
            raise ValueError(
                f"{specification['name']} cannot select both consensus and "
                "sparse-loop state directories"
            )
        has_reconstruction = (
            has_consensus_reconstruction or has_sparse_loop_reconstruction
        )
        if specification.get("evaluate_raw", True):
            metrics = _evaluate_model(model, batches, device)
            raw_condition = (
                f"{specification['name']}_raw"
                if has_reconstruction
                else specification["name"]
            )
            result = {
                "condition": raw_condition,
                "weight_source": "raw",
                "decomposition_components": None,
                "num_loops": decoder.num_loops,
                "blocks_per_loop": len(decoder.loop_layers),
                "physical_depth": len(decoder.layers),
                "logical_depth": decoder.logical_num_layers,
                "reconstruction_relative_frobenius": None,
                "sparse_density_target": None,
                "sparse_density_actual": None,
                **metrics,
            }
            results.append(result)
            if rank == 0:
                print(
                    f"{specification['name']}: loss={metrics['loss']:.6f}, "
                    f"ppl={metrics['perplexity']:.4f}"
                )

        if has_consensus_reconstruction:
            states = _load_consensus_states(
                _resolve_path(specification["consensus_state_directory"])
            )
            reference_weights = _copy_consensus_weights(model, states)
            for variant in _get_decomposition_variants(specification, states):
                components = variant["components"]
                sparse_density = variant["sparse_density"]
                relative_error, actual_sparse_density = _apply_reconstruction(
                    model,
                    states,
                    reference_weights,
                    components,
                    sparse_density,
                )
                reconstructed_metrics = _evaluate_model(model, batches, device)
                reconstructed_result = {
                    "condition": f"{specification['name']}_{variant['name']}",
                    "weight_source": variant["name"],
                    "decomposition_components": list(components),
                    "num_loops": decoder.num_loops,
                    "blocks_per_loop": len(decoder.loop_layers),
                    "physical_depth": len(decoder.layers),
                    "logical_depth": decoder.logical_num_layers,
                    "reconstruction_relative_frobenius": relative_error,
                    "sparse_density_target": sparse_density,
                    "sparse_density_actual": actual_sparse_density,
                    **reconstructed_metrics,
                }
                results.append(reconstructed_result)
                if rank == 0:
                    print(
                        f"{specification['name']} {variant['name']}: "
                        f"loss={reconstructed_metrics['loss']:.6f}, "
                        f"ppl={reconstructed_metrics['perplexity']:.4f}, "
                        f"relative_frobenius={relative_error:.6f}, "
                        f"sparse_density="
                        f"{_format_density(actual_sparse_density)}"
                    )
            del reference_weights, states
        elif has_sparse_loop_reconstruction:
            states = _load_sparse_loop_states(
                _resolve_path(specification["sparse_loop_state_directory"])
            )
            reference_weights = _copy_sparse_loop_effective_weights(
                model,
                states,
            )
            for variant in _get_sparse_loop_variants(specification):
                (
                    relative_error,
                    actual_sparse_density,
                    target_sparse_density,
                ) = _apply_sparse_loop_reconstruction(
                    model,
                    states,
                    reference_weights,
                    variant["sparse_density"],
                )
                reconstructed_metrics = _evaluate_model(model, batches, device)
                reconstructed_result = {
                    "condition": (
                        f"{specification['name']}_{variant['name']}"
                    ),
                    "weight_source": variant["name"],
                    "decomposition_components": [
                        "shared_weight",
                        "sparse_residual",
                    ],
                    "num_loops": decoder.num_loops,
                    "blocks_per_loop": len(decoder.loop_layers),
                    "physical_depth": len(decoder.layers),
                    "logical_depth": decoder.logical_num_layers,
                    "reconstruction_relative_frobenius": relative_error,
                    "sparse_density_target": target_sparse_density,
                    "sparse_density_actual": actual_sparse_density,
                    **reconstructed_metrics,
                }
                results.append(reconstructed_result)
                if rank == 0:
                    print(
                        f"{specification['name']} {variant['name']}: "
                        f"loss={reconstructed_metrics['loss']:.6f}, "
                        f"ppl={reconstructed_metrics['perplexity']:.4f}, "
                        f"relative_frobenius={relative_error:.6f}, "
                        f"sparse_density="
                        f"{_format_density(actual_sparse_density)}"
                    )
            del reference_weights, states
        del model
        torch.cuda.empty_cache()

    baseline_loss = results[0]["loss"]
    for result in results:
        result["loss_delta_vs_vanilla"] = result["loss"] - baseline_loss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    timestamp_values = [timestamp]
    dist.broadcast_object_list(timestamp_values, src=0)
    output_directory = _resolve_path(config["output_directory"]) / timestamp_values[0]
    if rank == 0:
        _write_results(output_directory, config, fingerprints, results)
        _print_results(results)
        print(f"Results written to {output_directory}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
