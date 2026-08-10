"""Train a config-selected downstream task on a vision backbone."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salaad_vision.build import build_data, build_model, build_task
from salaad_vision.trainer import VisionTrainer


class Runtime(NamedTuple):
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def print_run_summary(config: Mapping[str, Any], runtime: Runtime) -> None:
    task = config["task"]
    model = config["model"]
    data = config["data"]
    training = config["training"]
    optimizer = config["optimizer"]
    scheduler = config.get("scheduler", {"name": "none"})
    output = config.get("output", {})
    wandb_config = config.get("wandb", {})

    train_limit = training.get("max_steps_per_epoch")
    validation_limit = config.get("validation", {}).get("max_steps")
    global_batch_size = training["batch_size"] * runtime.world_size

    if output.get("save", False):
        output_dir = Path(
            output.get("dir", "data/salaad_vision/downstream")
        ).expanduser()
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        checkpoint = str(output_dir / "checkpoint.pth")
    else:
        checkpoint = "disabled"

    if wandb_config.get("enabled", False):
        wandb_status = wandb_config.get("project", "SALAAD_VISION_DOWNSTREAM")
        if wandb_config.get("group"):
            wandb_status += f" / {wandb_config['group']}"
        wandb_status += " / <YYYYMMDD_HHMMSS>"
    else:
        wandb_status = "disabled"

    rows = [
        ("Task", f"{task['name']} ({task.get('head', 'default')})"),
        ("Model", f"{model['name']} [{model['variant']}]"),
        ("Attention", str(model.get("attention_backend", "explicit"))),
        ("Data", f"{data['name']} ({data['train']['split']})"),
        ("Device", str(runtime.device)),
        ("World size", str(runtime.world_size)),
        ("Epochs", str(training["epochs"])),
        (
            "Batch size",
            f"{training['batch_size']} per rank / {global_batch_size} global",
        ),
        ("Train steps/epoch", str(train_limit or "all")),
        ("Validation steps", str(validation_limit or "all")),
        ("Precision", str(training.get("precision", "float32"))),
        (
            "Optimizer",
            f"{optimizer['name']} (lr={float(optimizer['lr']):.6g})",
        ),
        ("Scheduler", str(scheduler.get("name", "none"))),
        ("W&B", wandb_status),
        ("Checkpoint", checkpoint),
    ]
    width = max(len(label) for label, _ in rows)
    line = "=" * 88
    print(f"\n{line}")
    print("Vision downstream training")
    print(line)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")
    print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "vision_imagenet_smoke.yaml",
    )
    return parser.parse_args()


def read_config(path: Path) -> Mapping[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"config must contain a mapping: {path}")
    return config


def choose_device(config: Mapping[str, Any], local_rank: int = 0) -> torch.device:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("config 'runtime' must be a mapping")
    requested = runtime.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} cannot address "
                f"{torch.cuda.device_count()} visible CUDA devices"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def init_runtime(config: Mapping[str, Any]) -> Runtime:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("config 'runtime' must be a mapping")

    try:
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE and LOCAL_RANK must be integers") from error
    if env_world_size <= 0 or local_rank < 0:
        raise ValueError("WORLD_SIZE must be positive and LOCAL_RANK non-negative")

    setting = runtime.get("distributed", "auto")
    if setting == "auto":
        distributed = env_world_size > 1
    elif isinstance(setting, bool):
        distributed = setting
    else:
        raise ValueError("runtime.distributed must be 'auto', true, or false")

    if not distributed and env_world_size > 1:
        raise RuntimeError(
            "torchrun started multiple processes but runtime.distributed is false"
        )
    if distributed and not all(
        name in os.environ for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    ):
        raise RuntimeError(
            "distributed execution must be launched with torch.distributed.run"
        )

    device = choose_device(config, local_rank)
    if distributed:
        backend = runtime.get(
            "backend",
            "nccl" if device.type == "cuda" else "gloo",
        )
        if not isinstance(backend, str) or not backend:
            raise ValueError("runtime.backend must be a non-empty string")
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return Runtime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def close_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    runtime = init_runtime(config)
    try:
        seed = config.get("seed", 0)
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        random.seed(seed + runtime.rank)
        torch.manual_seed(seed + runtime.rank)

        model = build_model(config)
        task = build_task(config)
        train_data = build_data(
            config,
            "train",
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
        validation_data = build_data(
            config,
            "validation",
            rank=runtime.rank,
            world_size=runtime.world_size,
        )

        if runtime.rank == 0:
            print_run_summary(config, runtime)
        trainer = VisionTrainer(
            model,
            task,
            train_data,
            validation_data,
            config,
            runtime.device,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
        trainer.fit()
    finally:
        close_runtime()


if __name__ == "__main__":
    main()
