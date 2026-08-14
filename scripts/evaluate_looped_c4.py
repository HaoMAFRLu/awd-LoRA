"""Evaluate vanilla and variable-depth Llama execution paths on fixed C4 batches."""

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
from typing import Any, Dict, List, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

import datasets.distributed
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salad.register import get_data, get_model, get_preprocessed_dataset, get_tokenizer
from salad.utils import hf_login_once, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "llama_looped_c4_eval.yaml",
    )
    return parser.parse_args()


def _init_distributed() -> Tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("C4 evaluation requires CUDA")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def _build_model(
    model_config: str,
    checkpoint: str,
    device: torch.device,
    precision: torch.dtype,
) -> torch.nn.Module:
    model_path = _resolve_path(model_config)
    checkpoint_path = _resolve_path(checkpoint)
    model = get_model(str(model_path))
    model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
    model.to(device=device, dtype=precision)
    model.eval()
    return model


def _configure_vanilla_middle_loop(model: torch.nn.Module) -> None:
    """Loop all vanilla decoder blocks except its first and last blocks.

    This changes only the forward execution order. The model parameters and
    checkpoint values remain untouched.
    """
    decoder = model.model
    num_layers = len(decoder.layers)
    if num_layers != 8:
        raise ValueError(
            "vanilla middle-loop evaluation requires exactly 8 physical layers; "
            f"got {num_layers}"
        )

    decoder.entry_layers = (0,)
    decoder.loop_layers = tuple(range(1, num_layers - 1))
    decoder.exit_layers = (num_layers - 1,)


def _run_vanilla_middle_loop_evaluation(
    config: Dict[str, Any],
    batches: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    device: torch.device,
    precision: torch.dtype,
    rank: int,
) -> List[Dict[str, Any]]:
    checkpoint_config = config["checkpoints"]["vanilla"]
    if rank == 0:
        print("\nLoading vanilla checkpoint and looping its middle six layers...")
    model = _build_model(
        checkpoint_config["model_config"],
        checkpoint_config["checkpoint"],
        device,
        precision,
    )
    _configure_vanilla_middle_loop(model)

    loop_values = [int(value) for value in config["loop_values"]]
    if loop_values != [2, 4, 6, 8, 10]:
        raise ValueError("vanilla middle-loop values must be exactly [2, 4, 6, 8, 10]")

    # The trained looped model uses a three-block recurrent region. One pass
    # through vanilla's six-block middle region therefore represents two of
    # those reported loop units.
    loop_units_per_middle_pass = int(config.get("loop_units_per_middle_pass", 2))
    if loop_units_per_middle_pass != 2:
        raise ValueError("loop_units_per_middle_pass must be 2")

    results: List[Dict[str, Any]] = []
    for num_loops in loop_values:
        if num_loops % loop_units_per_middle_pass != 0:
            raise ValueError(
                f"num_loop={num_loops} is not divisible by "
                f"loop_units_per_middle_pass={loop_units_per_middle_pass}"
            )
        middle_passes = num_loops // loop_units_per_middle_pass
        model.model.set_num_loops(middle_passes)

        if num_loops == 2 and model.model.layer_order != tuple(range(8)):
            raise RuntimeError(
                "num_loop=2 must reproduce the original vanilla execution order; "
                f"got {model.model.layer_order}"
            )

        metrics = _evaluate_condition(model, batches, pad_token_id, device)
        result = {
            "condition": f"vanilla_loop_{num_loops}",
            "num_loops": num_loops,
            "middle_passes": middle_passes,
            "logical_depth": len(model.model.layer_order),
            **metrics,
        }
        results.append(result)
        if rank == 0:
            print(
                f"num_loop={num_loops:>2d}, middle_passes={middle_passes}, "
                f"depth={result['logical_depth']:>2d}: "
                f"loss={result['loss']:.6f}, ppl={result['perplexity']:.4f}"
            )

    return results


def _cache_eval_batches(
    config: Dict[str, Any],
    tokenizer,
    rank: int,
    world_size: int,
) -> Tuple[List[Dict[str, torch.Tensor]], str]:
    global_batch_size = int(config["batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by world_size "
            f"({world_size})"
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
        # Compact CPU copies are kept once and reused by every condition. This
        # guarantees that all models see exactly the same tokenized examples.
        cached_batch = {
            "input_ids": batch["input_ids"].to(dtype=torch.int32).contiguous(),
            "attention_mask": batch["attention_mask"].to(dtype=torch.uint8).contiguous(),
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
def _evaluate_condition(
    model: torch.nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
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
        labels[labels == pad_token_id] = -100
        valid_tokens = (labels[:, 1:] != -100).sum()
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
    local_seconds = time.perf_counter() - started
    elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=device)
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


def _print_results(results: List[Dict[str, Any]]) -> None:
    print("\nEvaluation results")
    print("=" * 125)
    print(
        f"{'condition':<18} {'loops':>5} {'passes':>7} {'depth':>6} {'loss':>10} "
        f"{'ppl':>10} {'tokens':>12} {'seconds':>10} {'tokens/s':>12}"
    )
    print("-" * 125)
    for result in results:
        loops = "-" if result["num_loops"] is None else str(result["num_loops"])
        passes = str(result.get("middle_passes", "-"))
        print(
            f"{result['condition']:<18} {loops:>5} {passes:>7} "
            f"{result['logical_depth']:>6d} {result['loss']:>10.6f} "
            f"{result['perplexity']:>10.4f} {result['tokens']:>12d} "
            f"{result['seconds']:>10.2f} {result['tokens_per_second']:>12.0f}"
        )
    print("=" * 125)


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(str(args.config))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    rank, world_size, local_rank, device = _init_distributed()
    set_seed(int(config["seed"]))
    hf_login_once()

    precision_name = config.get("precision", "bfloat16")
    precisions = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if precision_name not in precisions:
        raise ValueError(f"unsupported precision: {precision_name!r}")
    precision = precisions[precision_name]

    tokenizer = get_tokenizer(int(config["max_length"]), config)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
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
    if rank == 0:
        print("Fixed evaluation batches cached; per-rank fingerprints:")
        for fingerprint_rank, value in enumerate(fingerprints):
            print(f"  rank {fingerprint_rank}: {value}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    timestamp_values = [timestamp]
    dist.broadcast_object_list(timestamp_values, src=0)
    timestamp = timestamp_values[0]
    output_directory = _resolve_path(config["output_directory"]) / timestamp

    run = None
    if rank == 0 and config.get("is_wandb", False):
        import wandb

        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False)
        run = wandb.init(
            project=config.get("wandb_project", "LOOPED_TRANSFORMER"),
            entity=config.get("wandb_entity", "hao-ma-eth-z-rich"),
            group=config.get("wandb_group", "c4_evaluation"),
            config=config,
            name=timestamp,
        )
        run.define_metric("condition_index")
        run.define_metric("eval/*", step_metric="condition_index")

    evaluation_mode = config.get("evaluation_mode", "vanilla_and_looped")
    if evaluation_mode == "vanilla_middle_loop":
        results = _run_vanilla_middle_loop_evaluation(
            config,
            batches,
            pad_token_id,
            device,
            precision,
            rank,
        )
    elif evaluation_mode == "vanilla_and_looped":
        results = []
        checkpoint_config = config["checkpoints"]

        if rank == 0:
            print("\nLoading vanilla checkpoint...")
        vanilla = _build_model(
            checkpoint_config["vanilla"]["model_config"],
            checkpoint_config["vanilla"]["checkpoint"],
            device,
            precision,
        )
        vanilla_metrics = _evaluate_condition(vanilla, batches, pad_token_id, device)
        results.append({
            "condition": "vanilla",
            "num_loops": None,
            "logical_depth": len(vanilla.model.layer_order),
            **vanilla_metrics,
        })
        del vanilla
        torch.cuda.empty_cache()

        if rank == 0:
            print(
                f"vanilla: loss={vanilla_metrics['loss']:.6f}, "
                f"ppl={vanilla_metrics['perplexity']:.4f}"
            )
            print("\nLoading looped checkpoint...")
        looped = _build_model(
            checkpoint_config["looped"]["model_config"],
            checkpoint_config["looped"]["checkpoint"],
            device,
            precision,
        )
        loop_values = [int(value) for value in config["loop_values"]]
        if not loop_values or loop_values != list(range(1, loop_values[-1] + 1)):
            raise ValueError(
                "loop_values must be consecutive positive integers starting at 1"
            )

        for num_loops in loop_values:
            looped.model.set_num_loops(num_loops)
            loop_metrics = _evaluate_condition(looped, batches, pad_token_id, device)
            result = {
                "condition": f"loop_{num_loops}",
                "num_loops": num_loops,
                "logical_depth": len(looped.model.layer_order),
                **loop_metrics,
            }
            results.append(result)
            if rank == 0:
                print(
                    f"loop={num_loops:>2d}, depth={result['logical_depth']:>2d}: "
                    f"loss={result['loss']:.6f}, ppl={result['perplexity']:.4f}"
                )
    else:
        raise ValueError(f"unsupported evaluation_mode: {evaluation_mode!r}")

    if rank == 0:
        vanilla_loss = results[0]["loss"]
        for condition_index, result in enumerate(results):
            result["loss_delta_vs_vanilla"] = result["loss"] - vanilla_loss
            if run is not None:
                run.log({
                    "condition_index": condition_index,
                    "eval/loss": result["loss"],
                    "eval/perplexity": result["perplexity"],
                    "eval/loss_delta_vs_vanilla": result["loss_delta_vs_vanilla"],
                    "eval/num_loops": (
                        -1 if result["num_loops"] is None else result["num_loops"]
                    ),
                    "eval/middle_passes": result.get("middle_passes", -1),
                    "eval/logical_depth": result["logical_depth"],
                    "eval/tokens": result["tokens"],
                    "eval/seconds": result["seconds"],
                    "eval/tokens_per_second": result["tokens_per_second"],
                })
        _write_results(output_directory, config, fingerprints, results)
        _print_results(results)
        print(f"Results written to {output_directory}")
        if run is not None:
            run.summary["results_path"] = str(output_directory / "results.json")
            run.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
