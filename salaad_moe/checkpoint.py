"""Synchronous, complete checkpoints with owner state and per-rank RNG."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint
from .distributed import agree_or_raise, rank, world_size


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_torch(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(trainer, root):
    root = Path(root)
    name = f"step_{trainer.step:08d}"
    temporary, target = root / (name + ".incomplete"), root / name
    error = None
    if rank() == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if target.exists() or temporary.exists():
                raise FileExistsError(f"Checkpoint already exists: {target}")
            temporary.mkdir()
        except Exception as exc:
            error = exc
    agree_or_raise(error, trainer.device, "Checkpoint directory creation")
    error = None
    try:
        atomic_save(
            {
                "rng": rng_state(),
                "salaad": trainer.manager.local_state_dict() if trainer.manager else None,
            },
            temporary / f"rank_{rank():05d}.pt",
        )
        if rank() == 0:
            atomic_save(
                {
                    "format": "salaad_moe.training.v1",
                    "config": trainer.config,
                    "config_hash": fingerprint(trainer.config),
                    "step": trainer.step,
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "reader": trainer.reader.state_dict(),
                },
                temporary / "training.pt",
            )
    except Exception as exc:
        error = exc
    agree_or_raise(error, trainer.device, "Checkpoint write")
    error = None
    if rank() == 0:
        try:
            marker = {
                "format": "salaad_moe.checkpoint.v1",
                "step": trainer.step,
                "world_size": world_size(),
                "config_hash": fingerprint(trainer.config),
            }
            (temporary / "complete.json").write_text(json.dumps(marker, indent=2) + "\n")
            os.replace(temporary, target)
        except Exception as exc:
            error = exc
    agree_or_raise(error, trainer.device, "Checkpoint publication")
    return target


def checkpoint_metadata(path):
    path = Path(path)
    marker = json.loads((path / "complete.json").read_text())
    if marker["format"] != "salaad_moe.checkpoint.v1" or path.name.endswith(".incomplete"):
        raise ValueError("Incomplete or unsupported checkpoint")
    if not (path / "training.pt").is_file() or any(
        not (path / f"rank_{i:05d}.pt").is_file() for i in range(marker["world_size"])
    ):
        raise ValueError("Checkpoint files are missing")
    return marker


def validate_vanilla_branch(source_config, target_config, step):
    if source_config["salaad"]["enabled"] or not target_config["salaad"]["enabled"]:
        raise ValueError("Branching requires a vanilla source and SALAAD target")
    if step != target_config["salaad"]["state_initialization_step"] or step < 1:
        raise ValueError("Branch exactly at the configured SALAAD initialization step")
    for section in ("seed", "model", "data", "training", "parallel"):
        if fingerprint(source_config[section]) != fingerprint(target_config[section]):
            raise ValueError(f"Vanilla-prefix branch must preserve {section}")


def load_checkpoint(trainer, path, branch_from_vanilla=False):
    path = Path(path)
    error, payload, shards, local = None, None, [], None
    try:
        meta = checkpoint_metadata(path)
        if meta["world_size"] != world_size():
            raise ValueError(
                "Exact training resume requires the original DP size; auxiliary groups can be repartitioned separately"
            )
        if not branch_from_vanilla and meta["config_hash"] != fingerprint(trainer.config):
            raise ValueError(
                "Resume configuration differs; use --stop-after without changing the full schedule"
            )
        payload = load_torch(path / "training.pt")
        if payload["format"] != "salaad_moe.training.v1":
            raise ValueError(
                "This is an evaluation-only conversion; resume it through its original Megatron save root"
            )
        if branch_from_vanilla:
            validate_vanilla_branch(payload["config"], trainer.config, payload["step"])
        if (
            payload["config_hash"] != meta["config_hash"]
            or fingerprint(payload["config"]) != meta["config_hash"]
            or payload["step"] != meta["step"]
        ):
            raise ValueError("Inconsistent checkpoint metadata")
        for i in range(meta["world_size"]):
            state = load_torch(path / f"rank_{i:05d}.pt")
            if i == rank():
                local = state
            if trainer.manager and not branch_from_vanilla:
                if state["salaad"] is None:
                    raise ValueError("SALAAD state is missing")
                shards.append(state["salaad"])
        trainer.model.load_state_dict(payload["model"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.reader.load_state_dict(payload["reader"])
        trainer.step = int(payload["step"])
        if (
            trainer.reader.cursor
            != trainer.step * trainer.config["training"]["global_batch_sequences"]
        ):
            raise ValueError("Reader cursor and optimizer steps disagree")
    except Exception as exc:
        error = exc
    agree_or_raise(error, trainer.device, "Checkpoint load")
    if branch_from_vanilla:
        trainer.manager.initialize(trainer.step)
    elif trainer.manager:
        trainer.manager.load_shards(shards)
    restore_rng(local["rng"])


def prune_checkpoints(root, config):
    """Only remove this trainer's completed checkpoints, after a new save."""
    root = Path(root)
    entries = []
    for path in root.glob("step_*"):
        if path.is_symlink() or not (path / "complete.json").is_file():
            continue
        meta = checkpoint_metadata(path)
        if meta["config_hash"] == fingerprint(config):
            entries.append((path, meta))
    entries.sort(key=lambda pair: pair[1]["step"])
    latest = config["training"]["keep_latest_checkpoints"]
    keep = {p for p, _ in entries[-latest:]}
    for path, _ in entries:
        if path not in keep:
            shutil.rmtree(path)
