"""Export one prebuilt DINO backbone for the SALAAD-qkv S50 intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salaad_vision.models import DinoViTBase8, apply_salaad_qkv_s50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(path: Path) -> Mapping[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint must contain a state dict: {path}")
    if not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise TypeError(f"checkpoint must map string keys to tensors: {path}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "data/salaad_vision/vit_b8_qkv/20260804_111549/model.pth"
        ),
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=ROOT / "data/salaad_vision/vit_b8_qkv/20260804_111549",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "data/salaad_vision/pretrained"
            / "salaad_qkv_s50_alpha1p5_backbone.pth"
        ),
    )
    parser.add_argument("--sparse-keep-fraction", type=float, default=0.5)
    parser.add_argument("--selected-energy-fraction", type=float, default=0.5)
    parser.add_argument("--reference-rank", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = output.with_suffix(".json")
    temporary = output.with_name(f"{output.name}.tmp")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {checkpoint}")
    if not matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {matrix_dir}")
    for path in (output, manifest, temporary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")

    model = DinoViTBase8()
    model.load_state_dict(_state(checkpoint), strict=True)
    replaced = apply_salaad_qkv_s50(
        model,
        matrix_dir,
        sparse_keep_fraction=args.sparse_keep_fraction,
        selected_energy_fraction=args.selected_energy_fraction,
        reference_rank=args.reference_rank,
        alpha=args.alpha,
    )
    if len(replaced) != 12:
        raise RuntimeError(f"expected 12 replaced qkv layers, got {len(replaced)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    backbone_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.backbone.state_dict().items()
    }
    torch.save(backbone_state, temporary)
    restored = DinoViTBase8()
    restored.load_checkpoint(temporary)
    for name, expected in backbone_state.items():
        actual = restored.backbone.state_dict()[name]
        if not torch.equal(actual, expected):
            raise RuntimeError(f"export verification failed for {name}")
    temporary.replace(output)

    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"))
    metadata = {
        "format": "dino_vitb8_backbone_state_dict",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "matrix_files": {
            path.name: _sha256(path)
            for path in matrix_files
        },
        "intervention": {
            "target": "all 12 qkv layers",
            "qk_sparse_keep_fraction": args.sparse_keep_fraction,
            "selection": "cross-QK output/output low-similarity SVD components",
            "selected_energy_fraction": args.selected_energy_fraction,
            "reference_rank": args.reference_rank,
            "alpha": args.alpha,
            "v": "full L+S",
            "non_qkv": "dense X",
        },
        "replaced_layers": sorted(replaced),
        "checkpoint_sha256": _sha256(output),
    }
    manifest.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint={output}")
    print(f"sha256={metadata['checkpoint_sha256']}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
