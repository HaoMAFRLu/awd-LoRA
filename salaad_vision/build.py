"""Config-driven builders for downstream vision experiments."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import Tensor, nn

from salaad_vision.data import build_imagenet_dataloader, build_voc2012_dataloader
from salaad_vision.models import (
    DinoViTBase8,
    apply_salaad,
    apply_salaad_all_masked_int3,
)
from salaad_vision.models.dino import DINO_VITB8_CHECKPOINT_SHA256
from salaad_vision.tasks import ClassificationTask, SegmentationTask

_ROOT = Path(__file__).resolve().parents[1]
_ENVIRONMENT_VARIABLE = re.compile(
    r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)"
)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config requires a {name!r} mapping")
    return section


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path")
    expanded = os.path.expandvars(str(value))
    unresolved = _ENVIRONMENT_VARIABLE.findall(expanded)
    if unresolved:
        raise ValueError(
            f"{name} contains unset environment variables: {', '.join(unresolved)}"
        )
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else _ROOT / path


def _state(path: Path) -> Mapping[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint must contain a state dict: {path}")
    if not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"checkpoint must map string keys to tensors: {path}")
    return state


def build_model(config: Mapping[str, Any]) -> DinoViTBase8:
    """Build and freeze the backbone selected by config."""
    model_config = _section(config, "model")
    name = model_config.get("name")
    if name != "dino_vitb8":
        raise ValueError(f"unsupported vision model: {name!r}")
    variant = model_config.get("variant")
    valid_variants = {
        "teacher",
        "vanilla",
        "derived",
        "salaad_all",
        "salaad_all_masked_int3",
        "salaad_qkv",
    }
    if variant not in valid_variants:
        raise ValueError(
            f"model.variant must be one of {sorted(valid_variants)}, got {variant!r}"
        )

    checkpoint = _path(model_config.get("checkpoint"), "model.checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {checkpoint}")

    attention_backend = model_config.get("attention_backend", "explicit")
    if attention_backend not in {"explicit", "sdpa"}:
        raise ValueError(
            "model.attention_backend must be 'explicit' or 'sdpa', "
            f"got {attention_backend!r}"
        )
    model = DinoViTBase8(attention_backend=attention_backend)
    checkpoint_kind = model_config.get("checkpoint_kind")
    if checkpoint_kind == "teacher_backbone":
        expected_sha256 = model_config.get(
            "sha256",
            DINO_VITB8_CHECKPOINT_SHA256,
        )
        model.load_checkpoint(
            checkpoint,
            expected_sha256=expected_sha256,
        )
    elif checkpoint_kind == "student_model":
        model.load_state_dict(_state(checkpoint), strict=True)
    elif checkpoint_kind == "derived_backbone":
        model.load_checkpoint(
            checkpoint,
            expected_sha256=model_config.get("sha256"),
        )
    else:
        raise ValueError(
            "model.checkpoint_kind must be 'teacher_backbone', "
            f"'student_model', or 'derived_backbone', got {checkpoint_kind!r}"
        )

    if variant in {"salaad_all", "salaad_all_masked_int3", "salaad_qkv"}:
        if checkpoint_kind != "student_model":
            raise ValueError(f"{variant} requires checkpoint_kind='student_model'")
        matrix_dir = _path(model_config.get("matrix_dir"), "model.matrix_dir")
        if matrix_dir.resolve() != checkpoint.parent.resolve():
            raise ValueError(
                f"{variant} checkpoint and matrix files must be in the same directory"
            )
        if variant == "salaad_all_masked_int3":
            apply_salaad_all_masked_int3(
                model,
                matrix_dir,
                relative_sigma_threshold=model_config.get(
                    "relative_sigma_threshold",
                    1e-2,
                ),
                sparse_zero_threshold=model_config.get(
                    "sparse_zero_threshold",
                    1e-5,
                ),
            )
        else:
            apply_salaad(model, matrix_dir, variant)
    elif "matrix_dir" in model_config:
        raise ValueError(f"model.matrix_dir is not used by variant={variant!r}")

    freeze = model_config.get("freeze", True)
    if not isinstance(freeze, bool):
        raise TypeError("model.freeze must be true or false")
    model.requires_grad_(not freeze)
    model.train(not freeze)
    return model


def build_task(config: Mapping[str, Any]) -> nn.Module:
    """Build the downstream task selected by config."""
    task_config = _section(config, "task")
    name = task_config.get("name")
    num_classes = task_config.get("num_classes")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("task.num_classes must be a positive integer")

    head = task_config.get("head", "linear")
    if name == "classification":
        if head != "linear":
            raise ValueError("classification currently supports only head='linear'")
        return ClassificationTask(num_classes)
    if name == "semantic_segmentation":
        if head not in {"linear", "linear_upsample"}:
            raise ValueError(
                "semantic_segmentation supports head='linear' or "
                "head='linear_upsample'"
            )
        data_config = config.get("data", {})
        if not isinstance(data_config, Mapping):
            raise ValueError("config 'data' must be a mapping")
        output_size = task_config.get(
            "output_size",
            data_config.get("image_size", 224),
        )
        return SegmentationTask(
            num_classes,
            output_size=output_size,
            ignore_index=task_config.get("ignore_index", 255),
            boundary_tolerance=task_config.get("boundary_tolerance", 1),
        )
    raise ValueError(f"unsupported vision task: {name!r}")


def build_data(
    config: Mapping[str, Any],
    phase: str,
    *,
    rank: int = 0,
    world_size: int = 1,
):
    """Build one train or validation DataLoader from the shared data config."""
    if phase not in {"train", "validation"}:
        raise ValueError(f"phase must be 'train' or 'validation', got {phase!r}")

    data_config = _section(config, "data")
    data_name = data_config.get("name")
    if data_name not in {"imagenet", "voc2012"}:
        raise ValueError(f"unsupported vision data: {data_name!r}")
    phase_config = data_config.get(phase)
    if not isinstance(phase_config, Mapping):
        raise ValueError(f"data requires a {phase!r} mapping")

    training_config = _section(config, "training")
    batch_size = phase_config.get(
        "batch_size",
        training_config.get("batch_size"),
    )
    num_workers = phase_config.get(
        "num_workers",
        training_config.get("num_workers", 0),
    )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"data.{phase}.batch_size must be a positive integer")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"data.{phase}.num_workers must be a non-negative integer")

    ignored = {"name", "train", "validation"}
    loader_data: Dict[str, Any] = {
        key: value for key, value in data_config.items() if key not in ignored
    }
    loader_data.update(phase_config)
    for path_name in ("root", "cache_dir"):
        if path_name in loader_data:
            loader_data[path_name] = str(
                _path(loader_data[path_name], f"data.{path_name}")
            )

    loader_config = {
        "seed": config.get("seed", 0),
        "seed_for_shuffle": config.get(
            "seed_for_shuffle",
            config.get("seed", 0),
        ),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data": loader_data,
    }
    if data_name == "imagenet":
        return build_imagenet_dataloader(
            loader_config,
            rank=rank,
            world_size=world_size,
        )
    return build_voc2012_dataloader(
        loader_config,
        rank=rank,
        world_size=world_size,
    )
