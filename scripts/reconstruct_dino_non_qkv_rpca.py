"""Post-hoc RPCA reconstruction of the non-QKV DINO linear weights.

The script applies the repository's ``salad.ialm.fit_torch`` implementation to
``attn.proj``, ``mlp.fc1``, and ``mlp.fc2`` in every transformer block.  QKV
weights and all non-matrix parameters are copied without modification.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from salad.ialm import fit_torch  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "salaad_vision" / "posthoc_rpca_non_qkv"
)
MODEL_SPECS = {
    "teacher": (
        REPO_ROOT
        / "data"
        / "salaad_vision"
        / "pretrained"
        / "dino_vitbase8_pretrain.pth"
    ),
    "vanilla": (
        REPO_ROOT
        / "data"
        / "salaad_vision"
        / "vit_b8_vanilla"
        / "20260803_101747"
        / "model.pth"
    ),
    "salaad_x": (
        REPO_ROOT
        / "data"
        / "salaad_vision"
        / "vit_b8"
        / "20260803_114805"
        / "model.pth"
    ),
}

TARGET_WEIGHT = re.compile(
    r"^(?:backbone\.)?blocks\.(\d+)\."
    r"(attn\.proj|mlp\.fc1|mlp\.fc2)\.weight$"
)
QKV_WEIGHT = re.compile(r"^(?:backbone\.)?blocks\.(\d+)\.attn\.qkv\.weight$")
EXPECTED_BLOCKS = 12
EXPECTED_TARGETS_PER_BLOCK = 3
OUTPUT_VARIANTS = ("l_plus_s", "l_only", "s_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=None,
        help="Models to reconstruct (default: all three).",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit student checkpoint to reconstruct instead of --models.",
    )
    parser.add_argument(
        "--label",
        help="Output label for --checkpoint (required with --checkpoint).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.checkpoint is not None and not args.label:
        parser.error("--label is required with --checkpoint")
    if args.checkpoint is None and args.label:
        parser.error("--label requires --checkpoint")
    if args.models is None and args.checkpoint is None:
        args.models = list(MODEL_SPECS)
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json_save(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, destination)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"checkpoint is not a tensor state dict: {path}")
    return dict(state)


def target_weight_keys(state: Mapping[str, torch.Tensor]) -> list[str]:
    targets = sorted(key for key in state if TARGET_WEIGHT.fullmatch(key))
    qkv = sorted(key for key in state if QKV_WEIGHT.fullmatch(key))
    expected_targets = EXPECTED_BLOCKS * EXPECTED_TARGETS_PER_BLOCK
    if len(targets) != expected_targets or len(qkv) != EXPECTED_BLOCKS:
        raise ValueError(
            "unexpected DINO ViT-B/8 matrix layout: "
            f"found {len(targets)} non-QKV targets and {len(qkv)} QKV weights; "
            f"expected {expected_targets} and {EXPECTED_BLOCKS}"
        )
    return targets


def component_path(layer_directory: Path, state_key: str) -> Path:
    return layer_directory / f"{state_key.replace('.', '__')}.pth"


def validate_component(
    payload: object,
    state_key: str,
    reference: torch.Tensor,
    source_digest: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(payload, Mapping) or payload.get("state_key") != state_key:
        raise ValueError(f"invalid saved RPCA component for {state_key}")
    if payload.get("source_sha256") != source_digest:
        raise ValueError(f"saved RPCA component came from another checkpoint: {state_key}")
    low_rank = payload.get("L")
    sparse = payload.get("S")
    if not isinstance(low_rank, torch.Tensor) or not isinstance(sparse, torch.Tensor):
        raise TypeError(f"saved RPCA component lacks tensor L/S for {state_key}")
    if low_rank.shape != reference.shape or sparse.shape != reference.shape:
        raise ValueError(f"saved RPCA component shape mismatch for {state_key}")
    if low_rank.dtype != torch.float32 or sparse.dtype != torch.float32:
        raise TypeError(f"saved RPCA component must be float32 for {state_key}")
    if not torch.isfinite(low_rank).all() or not torch.isfinite(sparse).all():
        raise ValueError(f"saved RPCA component contains non-finite values for {state_key}")
    return low_rank, sparse


def layer_statistics(
    reference: torch.Tensor,
    low_rank: torch.Tensor,
    sparse: torch.Tensor,
    elapsed_seconds: float,
) -> dict[str, object]:
    reference = reference.float()
    reconstruction = low_rank + sparse
    squared_reference = torch.sum(reference.square()).item()
    squared_error = torch.sum((reference - reconstruction).square()).item()
    nonzero = int(torch.count_nonzero(sparse).item())
    return {
        "shape": list(reference.shape),
        "lambda": 1.0 / math.sqrt(max(reference.shape)),
        "relative_reconstruction_error": math.sqrt(
            squared_error / max(squared_reference, 1e-30)
        ),
        "sparse_nonzero": nonzero,
        "sparse_density": nonzero / sparse.numel(),
        "reference_squared_frobenius_norm": squared_reference,
        "reconstruction_squared_error": squared_error,
        "elapsed_seconds": elapsed_seconds,
    }


def reconstruct_model(
    label: str,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
) -> None:
    output_directory = output_root / label
    layer_directory = output_directory / "layers"
    layer_directory.mkdir(parents=True, exist_ok=True)
    output_checkpoints = {
        variant: output_directory / f"model_{variant}.pth"
        for variant in OUTPUT_VARIANTS
    }
    metadata_path = output_directory / "metadata.json"

    state = load_state_dict(checkpoint)
    targets = target_weight_keys(state)
    source_digest = sha256(checkpoint)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "model": label,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": source_digest,
        "output_checkpoints": {
            variant: str(path.resolve())
            for variant, path in output_checkpoints.items()
        },
        "algorithm": "salad.ialm.fit_torch",
        "algorithm_parameters": {
            "lambda": "1 / sqrt(max(rows, columns))",
            "epsilon1": 1e-2,
            "epsilon2": 1e-2,
            "rho": 1.6,
            "max_iter": 1000,
            "dtype": "torch.float32",
            "approx_svd": False,
        },
        "selection": {
            "rpca": ["attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"],
            "unchanged": ["attn.qkv.weight", "biases", "normalization", "embeddings"],
            "number_of_rpca_matrices": len(targets),
        },
        "layers": {},
    }

    print(f"[{label}] source={checkpoint}", flush=True)
    for index, state_key in enumerate(targets, start=1):
        reference = state[state_key].detach().float()
        saved_path = component_path(layer_directory, state_key)
        started = time.perf_counter()
        if saved_path.is_file():
            payload = torch.load(saved_path, map_location="cpu", weights_only=True)
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
                epsilon1=1e-2,
                epsilon2=1e-2,
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
                saved_path,
            )
            action = "computed"
        elapsed = time.perf_counter() - started
        statistics = layer_statistics(reference, low_rank, sparse, elapsed)
        metadata["layers"][state_key] = statistics
        atomic_json_save(metadata, metadata_path)
        print(
            f"[{label} {index:02d}/{len(targets)}] {state_key} {action}; "
            f"relative_error={statistics['relative_reconstruction_error']:.6g}; "
            f"S_density={statistics['sparse_density']:.6g}; "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    aggregate_differences: dict[str, float] = {}
    output_digests: dict[str, str] = {}
    for variant, output_checkpoint in output_checkpoints.items():
        reconstructed = dict(state)
        total_reference_squared_norm = 0.0
        total_squared_error = 0.0
        for state_key in targets:
            reference = state[state_key]
            payload = torch.load(
                component_path(layer_directory, state_key),
                map_location="cpu",
                weights_only=True,
            )
            low_rank, sparse = validate_component(
                payload,
                state_key,
                reference,
                source_digest,
            )
            if variant == "l_plus_s":
                replacement = low_rank + sparse
            elif variant == "l_only":
                replacement = low_rank
            elif variant == "s_only":
                replacement = sparse
            else:
                raise ValueError(f"unsupported output variant: {variant}")
            replacement = replacement.to(dtype=reference.dtype)
            reconstructed[state_key] = replacement
            total_reference_squared_norm += torch.sum(reference.float().square()).item()
            total_squared_error += torch.sum(
                (reference.float() - replacement.float()).square()
            ).item()

        for state_key, reference in state.items():
            if state_key not in targets and not torch.equal(
                reconstructed[state_key],
                reference,
            ):
                raise RuntimeError(
                    f"non-target parameter changed unexpectedly: {state_key}"
                )

        atomic_torch_save(reconstructed, output_checkpoint)
        aggregate_differences[variant] = math.sqrt(
            total_squared_error / max(total_reference_squared_norm, 1e-30)
        )
        output_digests[variant] = sha256(output_checkpoint)
        print(
            f"[{label}] materialized {variant}; relative_difference_from_x="
            f"{aggregate_differences[variant]:.6g}; checkpoint={output_checkpoint}",
            flush=True,
        )

    metadata["aggregate_relative_difference_from_x"] = aggregate_differences
    metadata["aggregate_relative_reconstruction_error"] = aggregate_differences[
        "l_plus_s"
    ]
    metadata["output_sha256"] = output_digests
    metadata["completed"] = True
    atomic_json_save(metadata, metadata_path)
    print(
        f"[{label}] complete; aggregate_relative_error="
        f"{metadata['aggregate_relative_reconstruction_error']:.6g}; "
        f"outputs={output_directory}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    device = choose_device(args.device)
    print(f"device={device}; output_root={output_root}", flush=True)
    model_specs = (
        [(args.label, args.checkpoint.expanduser().resolve())]
        if args.checkpoint is not None
        else [(label, MODEL_SPECS[label]) for label in args.models]
    )
    for label, checkpoint in model_specs:
        reconstruct_model(
            label=label,
            checkpoint=checkpoint,
            output_root=output_root,
            device=device,
        )


if __name__ == "__main__":
    main()
