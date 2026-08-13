"""Visualize final-block DINO CLS-to-patch self-attention for one or more images."""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from salaad_vision.models.dino import (  # noqa: E402
    DINO_VITB8_CHECKPOINT_SHA256,
    DinoViTBase8,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COMPARISON_VARIANTS = (
    ("teacher", "Teacher"),
    ("vanilla", "Vanilla"),
    ("salaad_x", "SALAAD X"),
    ("salaad_l_plus_s", "SALAAD L+S"),
    ("salaad_l_only", "SALAAD L only"),
    ("salaad_s_only", "SALAAD S only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSONL manifest used to name sample directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "data"
            / "salaad_vision"
            / "pretrained"
            / "dino_vitbase8_pretrain.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "figures" / "salaad_vision",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("teacher_backbone", "derived_backbone", "student_model"),
        default="teacher_backbone",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        help="Directory containing matrix_rank<N>.pkl SALAAD decomposition files.",
    )
    parser.add_argument(
        "--matrix-component",
        choices=("l_plus_s", "l_only", "s_only", "zero"),
        help="Replace every decomposed linear weight with the selected component.",
    )
    parser.add_argument(
        "--qkv-components",
        nargs=3,
        choices=("x", "l", "s", "l_plus_s", "zero"),
        metavar=("Q", "K", "V"),
        help=(
            "Build selected qkv weights from independently chosen Q, K, and V "
            "components. Requires --matrix-layer-group attention_qkv."
        ),
    )
    parser.add_argument(
        "--matrix-layer-group",
        choices=(
            "all",
            "mlp_fc",
            "attention_all",
            "attention_qkv",
            "attention_proj",
            "non_qkv",
        ),
        default="all",
        help="Subset of decomposed linear weights to replace; all others retain X.",
    )
    parser.add_argument(
        "--matrix-block",
        type=int,
        action="append",
        default=[],
        help=(
            "Optionally restrict matrix replacement to these Transformer block "
            "indices; may be provided more than once."
        ),
    )
    parser.add_argument(
        "--low-rank-energy",
        type=float,
        help=(
            "Optionally truncate each selected L matrix to the smallest rank "
            "whose leading squared singular values retain this energy fraction."
        ),
    )
    parser.add_argument(
        "--sparse-retain-fraction",
        type=float,
        help=(
            "Optionally retain only this fraction of each selected S matrix's "
            "existing nonzero entries, chosen by largest absolute magnitude."
        ),
    )
    parser.add_argument("--model-label", default="teacher")
    parser.add_argument(
        "--variant-subdir",
        type=Path,
        help="Optional relative output directory below each sample directory.",
    )
    parser.add_argument(
        "--title-label",
        help="Human-readable model label for the figure title.",
    )
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--skip-transformer-block",
        type=int,
        action="append",
        default=[],
        help="Transformer block index to bypass during attention inference.",
    )
    parser.add_argument(
        "--sweep-single-skipped-blocks",
        action="store_true",
        help="Run one variant per transformer block, skipping exactly one each time.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_model_checkpoint(
    model: DinoViTBase8,
    checkpoint: Path,
    checkpoint_kind: str,
) -> None:
    if checkpoint_kind == "teacher_backbone":
        model.load_checkpoint(
            checkpoint,
            expected_sha256=DINO_VITB8_CHECKPOINT_SHA256,
        )
        return

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(
            "student checkpoint must contain a state-dict mapping, "
            f"got {type(state).__name__}"
        )
    if checkpoint_kind == "derived_backbone":
        model.backbone.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)


def load_matrix_file(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load one trusted rank-local SALAAD pickle onto CPU."""
    original_loader = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda value: torch.load(
            io.BytesIO(value),
            map_location="cpu",
            weights_only=False,
        )
        with path.open("rb") as matrix_file:
            payload = pickle.load(matrix_file)
    finally:
        torch.storage._load_from_bytes = original_loader

    if not isinstance(payload, dict):
        raise TypeError(f"matrix file must contain a dictionary: {path}")
    low_rank = payload.get("LL")
    sparse = payload.get("SS")
    if not isinstance(low_rank, dict) or not isinstance(sparse, dict):
        raise TypeError(f"matrix file must contain LL and SS dictionaries: {path}")
    if set(low_rank) != set(sparse):
        raise ValueError(f"LL and SS layer names differ: {path}")
    return low_rank, sparse


def matrix_rank(path: Path) -> int:
    match = re.fullmatch(r"matrix_rank(\d+)\.pkl", path.name)
    if match is None:
        raise ValueError(f"unexpected matrix filename: {path.name}")
    return int(match.group(1))


@torch.no_grad()
def apply_matrix_component(
    model: DinoViTBase8,
    matrix_dir: Path,
    component: Optional[str],
    layer_group: str = "all",
    low_rank_energy: Optional[float] = None,
    sparse_retain_fraction: Optional[float] = None,
    blocks: Optional[set[int]] = None,
    qkv_components: Optional[tuple[str, str, str]] = None,
) -> dict[str, object]:
    """Replace selected linear weights with saved or energy-truncated L/S."""
    if not matrix_dir.is_dir():
        raise NotADirectoryError(f"matrix directory does not exist: {matrix_dir}")
    matrix_files = sorted(matrix_dir.glob("matrix_rank*.pkl"), key=matrix_rank)
    if not matrix_files:
        raise FileNotFoundError(f"no matrix_rank<N>.pkl files found in {matrix_dir}")
    if (component is None) == (qkv_components is None):
        raise ValueError("select exactly one of component or qkv_components")
    if qkv_components is not None and layer_group != "attention_qkv":
        raise ValueError("mixed Q/K/V components require layer_group=attention_qkv")

    expected_layers = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    layer_selectors = {
        "all": lambda name: True,
        "mlp_fc": lambda name: ".mlp.fc" in name,
        "attention_all": lambda name: ".attn." in name,
        "attention_qkv": lambda name: name.endswith(".attn.qkv"),
        "attention_proj": lambda name: name.endswith(".attn.proj"),
        "non_qkv": lambda name: not name.endswith(".attn.qkv"),
    }
    try:
        select_layer = layer_selectors[layer_group]
    except KeyError as error:
        raise ValueError(f"unsupported matrix layer group: {layer_group}") from error
    selected_layers = {name for name in expected_layers if select_layer(name)}
    if blocks is not None:
        block_count = len(model.backbone.blocks)
        invalid_blocks = sorted(
            block_index
            for block_index in blocks
            if block_index < 0 or block_index >= block_count
        )
        if invalid_blocks:
            raise ValueError(
                f"matrix block indices must be in [0, {block_count - 1}]: "
                f"{invalid_blocks}"
            )
        selected_layers = {
            name
            for name in selected_layers
            if any(
                name.startswith(f"backbone.blocks.{block_index}.")
                for block_index in blocks
            )
        }
    if not selected_layers:
        raise ValueError(
            f"matrix selection contains no layers: group={layer_group}, "
            f"blocks={sorted(blocks) if blocks is not None else None}"
        )

    seen_layers: set[str] = set()
    replaced_layers: set[str] = set()
    reference_squared_norm = 0.0
    component_squared_norm = 0.0
    difference_squared_norm = 0.0
    sparse_nonzero = 0
    original_sparse_nonzero = 0
    sparse_elements = 0
    low_rank_compression: dict[str, dict[str, object]] = {}
    sparse_pruning: dict[str, dict[str, object]] = {}

    for matrix_file in matrix_files:
        low_rank, sparse = load_matrix_file(matrix_file)
        for layer_name in sorted(low_rank):
            if layer_name in seen_layers:
                raise ValueError(f"duplicate decomposed layer: {layer_name}")
            if layer_name not in expected_layers:
                raise ValueError(f"decomposition names a non-linear layer: {layer_name}")

            layer = model.get_submodule(layer_name)
            reference = layer.weight.detach().float()
            low_rank_weight = low_rank[layer_name]
            sparse_weight = sparse[layer_name]
            if not isinstance(low_rank_weight, torch.Tensor) or not isinstance(
                sparse_weight, torch.Tensor
            ):
                raise TypeError(f"L and S must be tensors for {layer_name}")
            if low_rank_weight.shape != reference.shape or sparse_weight.shape != reference.shape:
                raise ValueError(
                    f"matrix shape mismatch for {layer_name}: "
                    f"X={tuple(reference.shape)}, L={tuple(low_rank_weight.shape)}, "
                    f"S={tuple(sparse_weight.shape)}"
                )
            if not torch.isfinite(low_rank_weight).all() or not torch.isfinite(
                sparse_weight
            ).all():
                raise ValueError(f"non-finite decomposition values for {layer_name}")
            seen_layers.add(layer_name)
            if layer_name not in selected_layers:
                continue

            low_rank_weight = low_rank_weight.float()
            sparse_weight = sparse_weight.float()
            if low_rank_energy is not None:
                u, singular_values, vh = torch.linalg.svd(
                    low_rank_weight,
                    full_matrices=False,
                )
                squared = singular_values.square()
                total_energy = squared.sum()
                if total_energy.item() == 0.0:
                    retained_rank = 0
                    achieved_energy = 1.0
                    low_rank_weight = torch.zeros_like(low_rank_weight)
                    numerical_rank = 0
                else:
                    cumulative_energy = squared.cumsum(dim=0)
                    retained_rank = int(
                        torch.searchsorted(
                            cumulative_energy,
                            total_energy * low_rank_energy,
                        ).item()
                    ) + 1
                    low_rank_weight = (
                        u[:, :retained_rank] * singular_values[:retained_rank]
                    ) @ vh[:retained_rank, :]
                    achieved_energy = (
                        cumulative_energy[retained_rank - 1] / total_energy
                    ).item()
                    rank_tolerance = (
                        singular_values[0]
                        * max(low_rank_weight.shape)
                        * torch.finfo(singular_values.dtype).eps
                    )
                    numerical_rank = int(
                        torch.count_nonzero(singular_values > rank_tolerance).item()
                    )
                low_rank_compression[layer_name] = {
                    "original_numerical_rank": numerical_rank,
                    "retained_rank": retained_rank,
                    "retained_rank_ratio": retained_rank / max(numerical_rank, 1),
                    "target_energy": low_rank_energy,
                    "achieved_energy": achieved_energy,
                }
            layer_original_nonzero = int(torch.count_nonzero(sparse_weight).item())
            original_sparse_nonzero += layer_original_nonzero
            if sparse_retain_fraction is not None:
                retained_nonzero = (
                    min(
                        layer_original_nonzero,
                        math.ceil(layer_original_nonzero * sparse_retain_fraction),
                    )
                    if layer_original_nonzero > 0
                    else 0
                )
                if retained_nonzero < layer_original_nonzero:
                    flat_sparse = sparse_weight.flatten()
                    nonzero_indices = torch.nonzero(
                        flat_sparse,
                        as_tuple=False,
                    ).flatten()
                    retained_positions = torch.topk(
                        flat_sparse[nonzero_indices].abs(),
                        k=retained_nonzero,
                        largest=True,
                        sorted=False,
                    ).indices
                    retained_indices = nonzero_indices[retained_positions]
                    pruned_sparse = torch.zeros_like(flat_sparse)
                    pruned_sparse[retained_indices] = flat_sparse[retained_indices]
                    sparse_weight = pruned_sparse.reshape_as(sparse_weight)
                magnitude_threshold = (
                    float(sparse_weight[sparse_weight != 0].abs().min().item())
                    if retained_nonzero > 0
                    else None
                )
                sparse_pruning[layer_name] = {
                    "original_nonzero": layer_original_nonzero,
                    "retained_nonzero": retained_nonzero,
                    "retained_fraction": (
                        retained_nonzero / max(layer_original_nonzero, 1)
                    ),
                    "target_retained_fraction": sparse_retain_fraction,
                    "minimum_retained_magnitude": magnitude_threshold,
                }
            if qkv_components is not None:
                reference_chunks = reference.chunk(3, dim=0)
                low_rank_chunks = low_rank_weight.chunk(3, dim=0)
                sparse_chunks = sparse_weight.chunk(3, dim=0)
                replacement_chunks = []
                for selected, x_chunk, l_chunk, s_chunk in zip(
                    qkv_components,
                    reference_chunks,
                    low_rank_chunks,
                    sparse_chunks,
                ):
                    if selected == "x":
                        replacement_chunks.append(x_chunk)
                    elif selected == "l":
                        replacement_chunks.append(l_chunk)
                    elif selected == "s":
                        replacement_chunks.append(s_chunk)
                    elif selected == "l_plus_s":
                        replacement_chunks.append(l_chunk + s_chunk)
                    elif selected == "zero":
                        replacement_chunks.append(torch.zeros_like(x_chunk))
                    else:
                        raise ValueError(f"unsupported qkv component: {selected}")
                replacement = torch.cat(replacement_chunks, dim=0)
            elif component == "l_plus_s":
                replacement = low_rank_weight + sparse_weight
            elif component == "l_only":
                replacement = low_rank_weight
            elif component == "s_only":
                replacement = sparse_weight
            elif component == "zero":
                replacement = torch.zeros_like(reference)
            else:
                raise ValueError(f"unsupported matrix component: {component}")

            reference_squared_norm += torch.sum(reference.square()).item()
            component_squared_norm += torch.sum(replacement.square()).item()
            difference_squared_norm += torch.sum((replacement - reference).square()).item()
            sparse_nonzero += torch.count_nonzero(sparse_weight).item()
            sparse_elements += sparse_weight.numel()
            layer.weight.copy_(replacement.to(dtype=layer.weight.dtype))
            replaced_layers.add(layer_name)

    missing_selected_layers = selected_layers - seen_layers
    extra_layers = seen_layers - expected_layers
    if missing_selected_layers or extra_layers:
        raise ValueError(
            "decomposition does not cover the selected model linear layers: "
            f"missing={sorted(missing_selected_layers)}, "
            f"extra={sorted(extra_layers)}"
        )

    relative_difference = (
        difference_squared_norm / max(reference_squared_norm, 1e-30)
    ) ** 0.5
    relative_component_norm = (
        component_squared_norm / max(reference_squared_norm, 1e-30)
    ) ** 0.5
    return {
        "matrix_dir": str(matrix_dir.resolve()),
        "matrix_files": [str(path.resolve()) for path in matrix_files],
        "component": component or "mixed_qkv",
        "qkv_components": list(qkv_components) if qkv_components is not None else None,
        "layer_group": layer_group,
        "blocks": sorted(blocks) if blocks is not None else None,
        "low_rank_energy": low_rank_energy,
        "low_rank_compression": low_rank_compression or None,
        "sparse_retain_fraction": sparse_retain_fraction,
        "sparse_pruning": sparse_pruning or None,
        "decomposed_layers": len(seen_layers),
        "replaced_layers": len(replaced_layers),
        "retained_x_linear_layers": len(expected_layers - replaced_layers),
        "sparse_nonzero": sparse_nonzero,
        "original_sparse_nonzero": original_sparse_nonzero,
        "sparse_elements": sparse_elements,
        "sparse_density": sparse_nonzero / max(sparse_elements, 1),
        "relative_frobenius_difference_from_x": relative_difference,
        "relative_frobenius_norm_vs_x": relative_component_norm,
    }


def prepare_image(path: Path) -> tuple[Image.Image, Image.Image, torch.Tensor]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    crop = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )(rgb)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )(crop)
    return rgb, crop, tensor.unsqueeze(0)


def get_last_selfattention(
    model: DinoViTBase8,
    images: torch.Tensor,
    skipped_blocks: set[int],
) -> tuple[torch.Tensor, int]:
    """Return attention from the last executed block after bypassing blocks."""
    backbone = model.backbone
    executed_blocks = [
        index
        for index in range(len(backbone.blocks))
        if index not in skipped_blocks
    ]
    if not executed_blocks:
        raise ValueError("at least one transformer block must be executed")
    attention_block_index = executed_blocks[-1]

    tokens = backbone.prepare_tokens(images)
    for block_index, block in enumerate(backbone.blocks):
        if block_index in skipped_blocks:
            continue
        if block_index == attention_block_index:
            return block(tokens, return_attention=True), attention_block_index
        tokens = block(tokens)
    raise RuntimeError("failed to reach the final transformer block")


def top_attention_mask(attention: torch.Tensor, mass: float) -> torch.Tensor:
    if not 0.0 < mass <= 1.0:
        raise ValueError("attention mass must be in (0, 1]")
    values, indices = torch.sort(attention.flatten(), descending=True)
    cumulative = torch.cumsum(values, dim=0) / values.sum().clamp_min(1e-12)
    count = int(torch.searchsorted(cumulative, mass).item()) + 1
    selected = torch.zeros_like(values, dtype=torch.bool)
    selected[:count] = True
    mask = torch.zeros_like(selected)
    mask.scatter_(0, indices, selected)
    return mask.reshape(attention.shape)


def color_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    normalized = heatmap / max(float(heatmap.max()), 1e-12)
    heatmap_rgb = plt.get_cmap("turbo")(normalized)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap_rgb, 0.0, 1.0)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "sample"


def load_sample_records(
    image_paths: list[Path],
    manifest_path: Optional[Path],
) -> list[dict[str, object]]:
    manifest_records: dict[Path, dict[str, object]] = {}
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            entry = json.loads(line)
            try:
                relative_path = Path(entry["relative_path"])
                index = int(entry["index"])
                label = int(entry["label"])
                class_name = str(entry["class_name"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid manifest entry at line {line_number}: {manifest_path}"
                ) from error
            resolved_path = (manifest_path.parent / relative_path).resolve()
            manifest_records[resolved_path] = {
                "index": index,
                "label": label,
                "class_name": class_name,
                "sample_id": f"{index:02d}_{slugify(class_name.split(',')[0])}",
            }

    records: list[dict[str, object]] = []
    used_sample_ids: set[str] = set()
    for image_path in image_paths:
        resolved_path = image_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        if manifest_path is None:
            record: dict[str, object] = {
                "index": None,
                "label": None,
                "class_name": None,
                "sample_id": slugify(image_path.stem),
            }
        else:
            if resolved_path not in manifest_records:
                raise ValueError(f"image is not listed in manifest: {image_path}")
            record = dict(manifest_records[resolved_path])
        sample_id = str(record["sample_id"])
        if sample_id in used_sample_ids:
            raise ValueError(f"duplicate sample directory name: {sample_id}")
        used_sample_ids.add(sample_id)
        record["path"] = resolved_path
        records.append(record)
    return records


def write_comparison(sample_dir: Path, class_name: object) -> Optional[Path]:
    overlays = [
        (title, sample_dir / variant / "mean_overlay.png")
        for variant, title in COMPARISON_VARIANTS
    ]
    if not all(path.is_file() for _, path in overlays):
        return None

    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, (title, path) in zip(axes.flatten(), overlays):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    suffix = f" — {class_name}" if class_name else ""
    figure.suptitle(
        f"Final-block mean CLS-to-patch attention{suffix}",
        fontsize=16,
    )
    comparison_path = sample_dir / "comparison.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return comparison_path


def write_sample_index(
    output_dir: Path,
    records: list[dict[str, object]],
) -> None:
    columns = 3
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, record in zip(axes.flatten(), records):
        with Image.open(record["path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(str(record["class_name"] or record["sample_id"]), fontsize=12)
        axis.axis("off")
    for axis in axes.flatten()[len(records) :]:
        axis.axis("off")
    figure.suptitle("Self-attention comparison samples", fontsize=16)
    figure.savefig(output_dir / "samples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    index = {
        "variants": [variant for variant, _ in COMPARISON_VARIANTS],
        "samples": [
            {
                "sample_id": record["sample_id"],
                "index": record["index"],
                "label": record["label"],
                "class_name": record["class_name"],
                "source_image": str(Path(record["path"]).resolve()),
                "directory": str((output_dir / str(record["sample_id"])).resolve()),
            }
            for record in records
        ],
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_skipped_block_sweep(
    sample_dir: Path,
    block_count: int,
    attention_source_blocks: dict[int, int],
    class_name: object,
) -> Path:
    group_dir = sample_dir / "ablations" / "l_plus_s_skip_one_block"
    baseline_path = sample_dir / "salaad_l_plus_s" / "mean_overlay.png"
    panels = [("L+S baseline", baseline_path)]
    panels.extend(
        (
            (
                f"Skip block {block_index}"
                if attention_source_blocks[block_index] == block_count - 1
                else (
                    f"Skip block {block_index}\n"
                    f"attention from block {attention_source_blocks[block_index]}"
                )
            ),
            group_dir / f"block_{block_index:02d}" / "mean_overlay.png",
        )
        for block_index in range(block_count)
    )
    missing = [str(path) for _, path in panels if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing skipped-block comparison images: {missing}")

    figure, axes = plt.subplots(4, 4, figsize=(13, 13), constrained_layout=True)
    for axis, (title, path) in zip(axes.flatten(), panels):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    for axis in axes.flatten()[len(panels) :]:
        axis.axis("off")
    figure.suptitle(
        f"{class_name or sample_dir.name} — SALAAD L+S single-block bypass sweep",
        fontsize=17,
    )
    comparison_path = group_dir / "comparison.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "definition": "Each variant completely bypasses exactly one transformer block.",
        "baseline": str(baseline_path.resolve()),
        "variants": [
            {
                "skipped_block": block_index,
                "attention_source_block": attention_source_blocks[block_index],
                "directory": str(
                    (group_dir / f"block_{block_index:02d}").resolve()
                ),
            }
            for block_index in range(block_count)
        ],
        "comparison": str(comparison_path.resolve()),
    }
    (group_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return comparison_path


def render_attention(
    args: argparse.Namespace,
    record: dict[str, object],
    original: Image.Image,
    crop: Image.Image,
    attention: torch.Tensor,
    output_label: str,
    decomposition: Optional[dict[str, object]],
    device: torch.device,
    attention_source_block: int,
    update_standard_comparison: bool = True,
) -> None:
    # [batch, heads, query tokens, key tokens] -> CLS query to patch keys.
    cls_to_patch = attention[0, :, 0, 1:].float().cpu().clone()
    cls_to_patch = cls_to_patch / cls_to_patch.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    num_heads, num_patches = cls_to_patch.shape
    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"patch count is not square: {num_patches}")

    patch_maps = cls_to_patch.reshape(num_heads, grid_size, grid_size)
    mean_patch_map = patch_maps.mean(dim=0)
    mean_mask = top_attention_mask(mean_patch_map, args.attention_mass)
    head_maps = F.interpolate(
        patch_maps.unsqueeze(0),
        size=(224, 224),
        mode="nearest",
    )[0]
    mean_map = F.interpolate(
        mean_patch_map[None, None],
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    mean_mask_image = F.interpolate(
        mean_mask[None, None].float(),
        size=(224, 224),
        mode="nearest",
    )[0, 0]

    original_array = np.asarray(original, dtype=np.float32) / 255.0
    image_array = np.asarray(crop, dtype=np.float32) / 255.0
    overlay = color_overlay(image_array, mean_map.numpy(), alpha=0.55)
    mask_overlay = image_array.copy()
    selected = mean_mask_image.numpy() > 0.5
    mask_overlay[selected] = (
        0.45 * mask_overlay[selected]
        + 0.55 * np.array([1.0, 0.15, 0.0], dtype=np.float32)
    )

    image_path = Path(record["path"])
    sample_dir = args.output_dir / str(record["sample_id"])
    variant_dir = sample_dir / output_label
    variant_dir.mkdir(parents=True, exist_ok=True)
    source_path = sample_dir / f"source{image_path.suffix.lower()}"
    sample_metadata_path = sample_dir / "sample.json"
    overview_path = variant_dir / "overview.png"
    overlay_path = variant_dir / "mean_overlay.png"
    metadata_path = variant_dir / "metadata.json"

    shutil.copy2(image_path, source_path)
    sample_metadata = {
        "sample_id": record["sample_id"],
        "index": record["index"],
        "label": record["label"],
        "class_name": record["class_name"],
        "source_image": str(image_path.resolve()),
        "source_copy": str(source_path.resolve()),
    }
    sample_metadata_path.write_text(
        json.dumps(sample_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plt.imsave(overlay_path, overlay)

    figure, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    flat_axes = axes.flatten()
    flat_axes[0].imshow(original_array)
    flat_axes[0].set_title("Original image")
    flat_axes[1].imshow(image_array)
    flat_axes[1].set_title("Validation center crop")
    flat_axes[2].imshow(overlay)
    flat_axes[2].set_title("Mean over 12 heads")
    flat_axes[3].imshow(mask_overlay)
    flat_axes[3].set_title(f"Top {args.attention_mass:.0%} attention mass")
    for head_index, (axis, head_map) in enumerate(
        zip(flat_axes[4:], head_maps.numpy())
    ):
        normalized = head_map / max(float(head_map.max()), 1e-12)
        axis.imshow(image_array)
        axis.imshow(normalized, cmap="turbo", alpha=0.58, vmin=0.0, vmax=1.0)
        axis.set_title(f"Head {head_index}")
    for axis in flat_axes:
        axis.axis("off")
    figure.suptitle(
        f"DINO ViT-B/8 {args.title_label or args.model_label}: "
        f"block {attention_source_block} CLS-to-patch self-attention",
        fontsize=16,
    )
    figure.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "sample_id": record["sample_id"],
        "class_name": record["class_name"],
        "source_image": str(image_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_sha256": (
            DINO_VITB8_CHECKPOINT_SHA256
            if args.checkpoint_kind == "teacher_backbone"
            else None
        ),
        "model_label": args.model_label,
        "title_label": args.title_label or args.model_label,
        "decomposition": decomposition,
        "device": str(device),
        "transform": "Resize(256, bicubic) -> CenterCrop(224) -> ImageNet normalization",
        "attention_shape": list(attention.shape),
        "patch_grid": [grid_size, grid_size],
        "attention_heads": num_heads,
        "attention_mass_threshold": args.attention_mass,
        "skipped_transformer_blocks": sorted(set(args.skip_transformer_block)),
        "attention_source_block": attention_source_block,
        "overview": str(overview_path.resolve()),
        "overlay": str(overlay_path.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    comparison_path = (
        write_comparison(sample_dir, record["class_name"])
        if update_standard_comparison
        else None
    )

    print(f"sample={record['sample_id']}")
    print(f"  attention_shape={tuple(attention.shape)}")
    print(f"  overview={overview_path}")
    print(f"  overlay={overlay_path}")
    print(f"  metadata={metadata_path}")
    if comparison_path is not None:
        print(f"  comparison={comparison_path}")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not isinstance(args.batch_size, int) or args.batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    has_matrix_selection = (
        args.matrix_component is not None or args.qkv_components is not None
    )
    if (args.matrix_dir is None) != (not has_matrix_selection):
        raise ValueError(
            "--matrix-dir requires either --matrix-component or --qkv-components"
        )
    if args.matrix_component is not None and args.qkv_components is not None:
        raise ValueError(
            "--matrix-component and --qkv-components cannot be combined"
        )
    if args.matrix_dir is not None and args.checkpoint_kind != "student_model":
        raise ValueError("SALAAD matrices require --checkpoint-kind student_model")
    if args.matrix_dir is None and args.matrix_layer_group != "all":
        raise ValueError("--matrix-layer-group requires --matrix-dir")
    if args.matrix_dir is None and args.matrix_block:
        raise ValueError("--matrix-block requires --matrix-dir")
    if args.qkv_components is not None and args.matrix_layer_group != "attention_qkv":
        raise ValueError(
            "--qkv-components requires --matrix-layer-group attention_qkv"
        )
    if args.low_rank_energy is not None:
        if args.matrix_dir is None:
            raise ValueError("--low-rank-energy requires --matrix-dir")
        if not 0.0 < args.low_rank_energy <= 1.0:
            raise ValueError("--low-rank-energy must be in the interval (0, 1]")
        if args.matrix_component not in ("l_plus_s", "l_only"):
            raise ValueError(
                "--low-rank-energy only applies to --matrix-component "
                "l_plus_s or l_only"
            )
    if args.sparse_retain_fraction is not None:
        if args.matrix_dir is None:
            raise ValueError("--sparse-retain-fraction requires --matrix-dir")
        if not 0.0 < args.sparse_retain_fraction <= 1.0:
            raise ValueError(
                "--sparse-retain-fraction must be in the interval (0, 1]"
            )
        if args.matrix_component not in ("l_plus_s", "s_only"):
            raise ValueError(
                "--sparse-retain-fraction only applies to --matrix-component "
                "l_plus_s or s_only"
            )
    if args.sweep_single_skipped_blocks and args.skip_transformer_block:
        raise ValueError(
            "--sweep-single-skipped-blocks cannot be combined with "
            "--skip-transformer-block"
        )
    if args.sweep_single_skipped_blocks and args.matrix_component != "l_plus_s":
        raise ValueError("the skipped-block sweep requires --matrix-component l_plus_s")
    if args.sweep_single_skipped_blocks and args.variant_subdir is not None:
        raise ValueError(
            "--variant-subdir cannot be combined with "
            "--sweep-single-skipped-blocks"
        )
    if args.variant_subdir is not None and (
        args.variant_subdir.is_absolute() or ".." in args.variant_subdir.parts
    ):
        raise ValueError("--variant-subdir must be a safe relative path")
    base_output_label = args.model_label.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", base_output_label):
        raise ValueError("model label may only contain letters, digits, '_' and '-'")
    base_model_label = args.model_label
    base_title_label = args.title_label or args.model_label

    records = load_sample_records(args.image, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_index(args.output_dir, records)

    device = choose_device(args.device)
    model = DinoViTBase8()
    load_model_checkpoint(model, args.checkpoint, args.checkpoint_kind)
    decomposition = None
    if args.matrix_dir is not None:
        decomposition = apply_matrix_component(
            model=model,
            matrix_dir=args.matrix_dir,
            component=args.matrix_component,
            layer_group=args.matrix_layer_group,
            blocks=set(args.matrix_block) if args.matrix_block else None,
            low_rank_energy=args.low_rank_energy,
            sparse_retain_fraction=args.sparse_retain_fraction,
            qkv_components=(
                tuple(args.qkv_components)
                if args.qkv_components is not None
                else None
            ),
        )
    model.eval().to(device)
    block_count = len(model.backbone.blocks)
    if args.sweep_single_skipped_blocks:
        variants = [
            (
                {block_index},
                (
                    "ablations/l_plus_s_skip_one_block/"
                    f"block_{block_index:02d}"
                ),
                f"{base_model_label}_skip_block{block_index}",
                f"{base_title_label} (skip block {block_index})",
            )
            for block_index in range(block_count)
        ]
    else:
        variants = [
            (
                set(args.skip_transformer_block),
                (
                    args.variant_subdir.as_posix()
                    if args.variant_subdir is not None
                    else base_output_label
                ),
                base_model_label,
                base_title_label,
            )
        ]

    prepared = [prepare_image(Path(record["path"])) for record in records]
    print(f"device={device}")
    if decomposition is not None:
        print(
            "decomposition="
            f"{decomposition['component']} "
            f"qkv_components={decomposition['qkv_components']} "
            f"layers={decomposition['replaced_layers']}/"
            f"{decomposition['decomposed_layers']} "
            f"relative_difference_from_x="
            f"{decomposition['relative_frobenius_difference_from_x']:.8f} "
            f"sparse_density={decomposition['sparse_density']:.8f}"
        )

    sweep_attention_source_blocks: dict[int, int] = {}
    for skipped_blocks, output_label, model_label, title_label in variants:
        invalid_skipped_blocks = sorted(
            block_index
            for block_index in skipped_blocks
            if block_index < 0 or block_index >= block_count
        )
        if invalid_skipped_blocks:
            raise ValueError(
                f"transformer block indices out of range: {invalid_skipped_blocks}"
            )
        args.skip_transformer_block = sorted(skipped_blocks)
        args.model_label = model_label
        args.title_label = title_label

        attention_source_block = -1
        for start in range(0, len(records), args.batch_size):
            stop = min(start + args.batch_size, len(records))
            image_batch = torch.cat(
                [prepared[index][2] for index in range(start, stop)],
                dim=0,
            ).to(device)
            with torch.inference_mode():
                attention_batch, attention_source_block = get_last_selfattention(
                    model,
                    image_batch,
                    skipped_blocks,
                )
            for offset, index in enumerate(range(start, stop)):
                original, crop, _ = prepared[index]
                render_attention(
                    args,
                    records[index],
                    original,
                    crop,
                    attention_batch[offset : offset + 1],
                    output_label,
                    decomposition,
                    device,
                    attention_source_block,
                    update_standard_comparison=(
                        not args.sweep_single_skipped_blocks
                    ),
                )
        if args.sweep_single_skipped_blocks:
            skipped_block = next(iter(skipped_blocks))
            sweep_attention_source_blocks[skipped_block] = attention_source_block

    if args.sweep_single_skipped_blocks:
        for record in records:
            sample_dir = args.output_dir / str(record["sample_id"])
            comparison_path = write_skipped_block_sweep(
                sample_dir,
                block_count,
                sweep_attention_source_blocks,
                record["class_name"],
            )
            print(f"sweep_comparison={comparison_path}")


if __name__ == "__main__":
    main()
