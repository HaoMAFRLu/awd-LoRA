"""Export trained shared/L/S values without truncation or precision conversion."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpoint import atomic_save, checkpoint_metadata, load_torch
from .model import MoELanguageModel
from .solver import GroupState


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def encode_csr(matrix):
    """Store every nonzero entry at its original precision, in row-major order."""
    m, n = matrix.shape
    flat = matrix.reshape(-1)
    positions = torch.nonzero(flat != 0, as_tuple=False).flatten()
    values = flat[positions]
    rows = torch.div(positions, n, rounding_mode="floor")
    crow = torch.cat(
        (
            torch.zeros(1, device=matrix.device, dtype=torch.int64),
            torch.bincount(rows, minlength=m).cumsum(0),
        )
    )
    return {
        "shape": (m, n),
        "crow_indices": crow.to(device="cpu", dtype=torch.int32),
        "col_indices": (positions % n).to(device="cpu", dtype=torch.int32),
        "values": values.detach().cpu().clone(),
    }


def decode_csr(state, device="cpu"):
    m, n = state["shape"]
    crow = state["crow_indices"].to(device=device, dtype=torch.int64)
    columns = state["col_indices"].to(device=device, dtype=torch.int64)
    values = state["values"].to(device=device)
    if (
        len(crow) != m + 1
        or crow[0] != 0
        or crow[-1] != len(values)
        or len(columns) != len(values)
        or (crow[1:] < crow[:-1]).any()
        or (columns < 0).any()
        or (columns >= n).any()
    ):
        raise ValueError("Invalid CSR artifact")
    rows = torch.repeat_interleave(torch.arange(m, device=device), crow[1:] - crow[:-1])
    matrix = torch.zeros((m, n), device=device, dtype=values.dtype)
    matrix[rows, columns] = values
    return matrix


@torch.no_grad()
def export_group(state):
    # L is already a trained auxiliary matrix. Save it directly instead of
    # running another decomposition that could alter its small components.
    return {
        "shared": state.shared.detach().cpu().clone(),
        "experts": [
            {
                "low_rank": low.detach().cpu().clone(),
                "sparse": encode_csr(sparse),
            }
            for low, sparse in zip(state.low_rank, state.sparse)
        ],
    }


@torch.no_grad()
def materialize_group(group, device="cpu"):
    shared = group["shared"].to(device=device)
    # Use the same addition order as GroupState.reconstruction().
    return torch.stack([
        shared + expert["low_rank"].to(device=device) + decode_csr(expert["sparse"], device)
        for expert in group["experts"]
    ])


def checkpoint_states(path, device="cpu"):
    path = Path(path)
    meta = checkpoint_metadata(path)
    payload = load_torch(path / "training.pt")
    states = {}
    for i in range(meta["world_size"]):
        shard = load_torch(path / f"rank_{i:05d}.pt")["salaad"]
        if shard is None or not shard["initialized"]:
            raise ValueError("This checkpoint has no initialized SALAAD decomposition")
        if set(states) & set(shard["states"]):
            raise ValueError("Duplicate auxiliary group")
        states.update(
            {
                name: GroupState.from_state_dict(value, device)
                for name, value in shard["states"].items()
            }
        )
    return payload, states


def export_checkpoint(checkpoint, output, device="cpu"):
    payload, states = checkpoint_states(checkpoint, device)
    config = payload["config"]
    artifact = {
        "format": "salaad_moe.export.v2",
        "config": config,
        "source_step": payload["step"],
        "source_config_hash": payload["config_hash"],
        "groups": {name: export_group(state) for name, state in states.items()},
        "untouched": {
            name: tensor.detach().cpu().clone()
            for name, tensor in payload["model"].items()
            if name not in states
        },
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Export exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(artifact, output)
    raw_bytes = tensor_bytes(payload["model"])
    report = {
        "format": artifact["format"],
        "source_step": payload["step"],
        "dense_model_tensor_bytes": raw_bytes,
        "export_tensor_bytes": tensor_bytes(artifact),
        "serialized_file_bytes": output.stat().st_size,
        "compression_ratio_tensor_bytes": raw_bytes / max(tensor_bytes(artifact), 1),
        "groups": {
            name: {
                "low_rank_shapes": [list(e["low_rank"].shape) for e in group["experts"]],
                "sparse_nnz": [len(e["sparse"]["values"]) for e in group["experts"]],
            }
            for name, group in artifact["groups"].items()
        },
        "inference_mode": "materialize saved shared + L + S at checkpoint precision",
        "native_sparse_kernel_benchmarked": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_evaluation_model(path, mode="raw", device="cpu"):
    if mode == "exported":
        artifact = load_torch(path)
        if artifact["format"] != "salaad_moe.export.v2":
            raise ValueError("Unsupported export")
        config = artifact["config"]
        weights = dict(artifact["untouched"])
        weights.update(
            {name: materialize_group(group) for name, group in artifact["groups"].items()}
        )
    elif mode in ("raw", "reconstructed"):
        if mode == "raw":
            checkpoint_metadata(path)
            payload = load_torch(Path(path) / "training.pt")
            states = {}
        else:
            payload, states = checkpoint_states(path)
        config = payload["config"]
        weights = dict(payload["model"])
        weights.update({name: state.reconstruction() for name, state in states.items()})
        # Megatron has a persistent BF16 task model; the native reference uses
        # FP32 parameters under autocast. Preserve each backend's raw semantics.
        if (
            payload["format"] == "salaad_moe.megatron_evaluation.v1"
            and config["training"]["task_precision"] == "bfloat16"
        ):
            weights = {name: value.bfloat16() for name, value in weights.items()}
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    model = MoELanguageModel(config, initialize=False)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), config
