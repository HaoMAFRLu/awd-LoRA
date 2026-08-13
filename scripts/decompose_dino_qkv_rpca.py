"""Post-hoc RPCA decomposition of the joint DINO QKV weights.

Each ``[3 * embed_dim, embed_dim]`` QKV weight is decomposed as ``X = L + S``
with the repository's IALM implementation.  The joint matrix is decomposed
before its output rows are split into Q, K, and V, matching the way SALAAD
acts on DINO's combined QKV projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salad.ialm import fit_torch  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT / "data/salaad_vision/pretrained/dino_vitbase8_pretrain.pth"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/salaad_vision/posthoc_rpca_qkv/teacher"
QKV_WEIGHT = re.compile(r"^(?:backbone\.)?blocks\.(\d+)\.attn\.qkv\.weight$")
EXPECTED_BLOCKS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json_save(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as metadata_file:
        json.dump(value, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    os.replace(temporary, destination)


def load_state_dict(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"checkpoint is not a tensor state dict: {path}")
    return dict(state)


def qkv_weight_keys(state: Mapping[str, Tensor]) -> list[str]:
    keyed_blocks = {
        int(match.group(1)): key
        for key in state
        if (match := QKV_WEIGHT.fullmatch(key)) is not None
    }
    expected = set(range(EXPECTED_BLOCKS))
    if set(keyed_blocks) != expected:
        raise ValueError(
            "unexpected DINO ViT-B/8 QKV layout: "
            f"found blocks {sorted(keyed_blocks)}, expected {sorted(expected)}"
        )
    keys = [keyed_blocks[block] for block in range(EXPECTED_BLOCKS)]
    for key in keys:
        if state[key].shape != (2304, 768):
            raise ValueError(f"unexpected QKV shape for {key}: {tuple(state[key].shape)}")
    return keys


def component_path(layer_directory: Path, state_key: str) -> Path:
    return layer_directory / f"{state_key.replace('.', '__')}.pth"


def validate_component(
    payload: object,
    state_key: str,
    reference: Tensor,
    source_digest: str,
) -> tuple[Tensor, Tensor]:
    if not isinstance(payload, Mapping) or payload.get("state_key") != state_key:
        raise ValueError(f"invalid saved RPCA component for {state_key}")
    if payload.get("source_sha256") != source_digest:
        raise ValueError(f"saved RPCA component came from another checkpoint: {state_key}")
    low_rank = payload.get("L")
    sparse = payload.get("S")
    if not isinstance(low_rank, Tensor) or not isinstance(sparse, Tensor):
        raise TypeError(f"saved RPCA component lacks tensor L/S for {state_key}")
    if low_rank.shape != reference.shape or sparse.shape != reference.shape:
        raise ValueError(f"saved RPCA component shape mismatch for {state_key}")
    if low_rank.dtype != torch.float32 or sparse.dtype != torch.float32:
        raise TypeError(f"saved RPCA component must be float32 for {state_key}")
    if not torch.isfinite(low_rank).all() or not torch.isfinite(sparse).all():
        raise ValueError(f"saved RPCA component contains non-finite values: {state_key}")
    return low_rank, sparse


def matrix_statistics(reference: Tensor, low_rank: Tensor, sparse: Tensor) -> dict:
    reconstruction = low_rank + sparse
    squared_reference = float(reference.square().sum())
    squared_error = float((reference - reconstruction).square().sum())
    statistics: dict[str, object] = {
        "shape": list(reference.shape),
        "lambda": 1.0 / math.sqrt(max(reference.shape)),
        "relative_reconstruction_error": math.sqrt(
            squared_error / max(squared_reference, 1e-30)
        ),
        "relative_l_norm": float(low_rank.norm() / reference.norm().clamp_min(1e-12)),
        "relative_s_norm": float(sparse.norm() / reference.norm().clamp_min(1e-12)),
        "sparse_nonzero": int(torch.count_nonzero(sparse)),
        "sparse_density": float(torch.count_nonzero(sparse) / sparse.numel()),
    }
    for name, reference_part, sparse_part in zip(
        ("q", "k", "v"),
        reference.chunk(3, dim=0),
        sparse.chunk(3, dim=0),
    ):
        nonzero = int(torch.count_nonzero(sparse_part))
        statistics[f"{name}_sparse_nonzero"] = nonzero
        statistics[f"{name}_sparse_density"] = nonzero / sparse_part.numel()
        statistics[f"{name}_relative_s_norm"] = float(
            sparse_part.norm() / reference_part.norm().clamp_min(1e-12)
        )
    return statistics


def run(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    layer_directory = output_directory / "layers"
    layer_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / "metadata.json"
    device = choose_device(args.device)
    state = load_state_dict(checkpoint)
    keys = qkv_weight_keys(state)
    source_digest = sha256(checkpoint)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "experiment": "joint DINO QKV post-hoc RPCA",
        "source_checkpoint": str(checkpoint),
        "source_sha256": source_digest,
        "algorithm": "salad.ialm.fit_torch",
        "algorithm_parameters": {
            "lambda": "1 / sqrt(max(rows, columns))",
            "epsilon1": 0.01,
            "epsilon2": 0.01,
            "rho": 1.6,
            "max_iter": 1000,
            "dtype": "torch.float32",
            "approx_svd": False,
        },
        "selection": {
            "rpca": ["attn.qkv.weight"],
            "number_of_rpca_matrices": len(keys),
            "decomposition_scope": "joint QKV before splitting output rows",
            "unchanged": ["attn.qkv.bias", "all non-QKV parameters"],
        },
        "layers": {},
    }

    total_reference_squared = 0.0
    total_error_squared = 0.0
    print(
        f"Teacher QKV RPCA | matrices={len(keys)} | device={device} | "
        f"output={output_directory}",
        flush=True,
    )
    for index, state_key in enumerate(keys, start=1):
        reference = state[state_key].detach().float()
        destination = component_path(layer_directory, state_key)
        started = time.perf_counter()
        if destination.is_file():
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            low_rank, sparse = validate_component(
                payload,
                state_key,
                reference,
                source_digest,
            )
            action = "resumed"
        else:
            matrix = reference.to(device)
            low_rank, sparse = fit_torch(
                matrix,
                lambda_=1.0 / math.sqrt(max(matrix.shape)),
                epsilon1=0.01,
                epsilon2=0.01,
                rho=1.6,
                max_iter=1000,
                verbose=False,
                device=device,
                dtype=torch.float32,
                approx_svd=False,
            )
            low_rank = low_rank.cpu()
            sparse = sparse.cpu()
            atomic_torch_save(
                {
                    "state_key": state_key,
                    "source_sha256": source_digest,
                    "L": low_rank,
                    "S": sparse,
                },
                destination,
            )
            action = "computed"
        statistics = matrix_statistics(reference, low_rank, sparse)
        statistics["elapsed_seconds"] = time.perf_counter() - started
        metadata["layers"][state_key] = statistics
        total_reference_squared += float(reference.square().sum())
        total_error_squared += float((reference - low_rank - sparse).square().sum())
        atomic_json_save(metadata, metadata_path)
        print(
            f"  [{index:02d}/{len(keys)}] {state_key} {action} | "
            f"error={statistics['relative_reconstruction_error']:.6g} | "
            f"S density={statistics['sparse_density']:.4f} | "
            f"elapsed={statistics['elapsed_seconds']:.2f}s",
            flush=True,
        )

    metadata["aggregate_relative_reconstruction_error"] = math.sqrt(
        total_error_squared / max(total_reference_squared, 1e-30)
    )
    metadata["completed"] = True
    atomic_json_save(metadata, metadata_path)
    print(
        "Complete | aggregate relative reconstruction error="
        f"{metadata['aggregate_relative_reconstruction_error']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
