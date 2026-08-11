"""Export head-only state dicts from frozen-backbone vision checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor


_EXPECTED_KEYS = {
    "head.linear.weight",
    "head.linear.bias",
}
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Name and source checkpoint path; repeat for multiple heads.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Tuple[Dict[str, Tensor], Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {path}")

    task = checkpoint.get("task")
    if not isinstance(task, Mapping):
        raise TypeError(f"checkpoint has no task state dict: {path}")
    if set(task) != _EXPECTED_KEYS:
        raise ValueError(
            f"unexpected task keys in {path}: {sorted(task)}; "
            f"expected {sorted(_EXPECTED_KEYS)}"
        )
    if not all(isinstance(value, Tensor) for value in task.values()):
        raise TypeError(f"classification head values must be tensors: {path}")

    weight = task["head.linear.weight"]
    bias = task["head.linear.bias"]
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
        raise ValueError(
            f"invalid classification head shapes in {path}: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )

    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in task.items()
    }
    return state_dict, checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_validation(checkpoint: Mapping[str, Any]) -> Dict[str, float]:
    history = checkpoint.get("history")
    if not isinstance(history, Sequence) or not history:
        return {}
    final = history[-1]
    if not isinstance(final, Mapping):
        return {}
    validation = final.get("validation")
    if not isinstance(validation, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in validation.items()
        if isinstance(value, (int, float))
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError(f"checkpoint names must be unique: {names}")

    manifest: Dict[str, Any] = {
        "format": "salaad_vision_classification_head_v1",
        "state_dict_keys": sorted(_EXPECTED_KEYS),
        "heads": {},
    }
    for name, checkpoint_value in args.checkpoint:
        if _SAFE_NAME.fullmatch(name) is None:
            raise ValueError(f"unsafe checkpoint name: {name!r}")

        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        state_dict, checkpoint = _load_checkpoint(checkpoint_path)
        output_path = output_dir / f"{name}.pth"
        torch.save(state_dict, output_path)

        manifest["heads"][name] = {
            "file": output_path.name,
            "source_checkpoint": str(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", 0)),
            "final_validation": _final_validation(checkpoint),
            "shapes": {
                key: list(value.shape)
                for key, value in state_dict.items()
            },
            "dtypes": {
                key: str(value.dtype)
                for key, value in state_dict.items()
            },
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
        print(f"Exported {name}: {output_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
