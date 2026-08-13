"""Generate additional O3 heatmaps for the LL/SL/LS/SS Q--K paths."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_o3_qk_paths import (  # noqa: E402
    CONDITION_NAMES,
    PATH_NAMES,
    _path_logits,
    _plot_representatives,
)
from analyze_o3_shallow_ll import (  # noqa: E402
    PATCH_GRID,
    QUERY_NAMES,
    O3Dataset,
    _project_attention,
    choose_device,
    load_decompositions,
)
from salaad_vision.models.dino import DinoViTBase8  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/salaad_vision/o3/raw",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/salaad_vision/vit_b8_qkv/20260804_111549/model.pth",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=ROOT / "data/salaad_vision/vit_b8_qkv/20260804_111549",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "data/figures/salaad_vision/o3_qk_paths/additional_samples"
        ),
    )
    parser.add_argument(
        "--existing-metadata",
        type=Path,
        default=ROOT / "data/figures/salaad_vision/o3_qk_paths/metadata.json",
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _existing_samples(metadata_path: Path) -> set[str]:
    path = metadata_path.expanduser().resolve()
    if not path.is_file():
        return set()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    names = metadata.get("representative_images", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TypeError(f"invalid representative_images in {path}")
    return set(names)


def _choose_additional_samples(
    dataset: O3Dataset,
    count: int,
    seed: int,
    excluded_names: set[str],
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    excluded_categories = {
        record.target_type
        for record in dataset.records
        if record.image_name in excluded_names
    }
    used_categories = set(excluded_categories)
    candidates = list(range(len(dataset)))
    random.Random(seed).shuffle(candidates)
    selected: list[int] = []
    for index in candidates:
        record = dataset.records[index]
        if record.image_name in excluded_names or record.target_type in used_categories:
            continue
        if not 3 <= record.num_distractors <= 30:
            continue
        masks = dataset.masks(index)
        areas = masks.flatten(1).mean(dim=1)
        if (
            masks[0].max() < 0.35
            or masks[1].max() < 0.35
            or areas[0] < 0.002
            or areas[1] < 0.02
            or areas[2] < 0.15
        ):
            continue
        selected.append(index)
        used_categories.add(record.target_type)
        if len(selected) == count:
            return sorted(selected)
    raise RuntimeError(f"only found {len(selected)} eligible additional O3 samples")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    dataset = O3Dataset(args.data_root)
    excluded_names = _existing_samples(args.existing_metadata)
    selected_indices = _choose_additional_samples(
        dataset,
        args.count,
        args.seed,
        excluded_names,
    )
    selected_records = [dataset.records[index] for index in selected_indices]

    checkpoint = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    model = DinoViTBase8(attention_backend="sdpa")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    blocks = tuple(range(len(model.backbone.blocks)))
    decomposition = {
        block: (low.to(device), sparse.to(device))
        for block, (low, sparse) in load_decompositions(
            args.matrix_dir,
            blocks,
        ).items()
    }
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    query_indices = np.full((len(dataset), len(QUERY_NAMES)), -1, dtype=np.int64)
    patch_masks = np.zeros(
        (len(dataset), 3, PATCH_GRID, PATCH_GRID),
        dtype=np.float32,
    )
    maps: dict[int, dict[int, dict[str, np.ndarray]]] = {
        index: {} for index in selected_indices
    }
    head_count = model.backbone.blocks[0].attn.num_heads
    print(
        "Additional O3 samples: "
        + ", ".join(record.image_name for record in selected_records),
        flush=True,
    )
    for batch in loader:
        indices = batch["index"].numpy()
        images = batch["image"].to(device, non_blocking=True)
        batch_masks = batch["masks"].to(device, non_blocking=True)
        mask_flat = batch_masks.flatten(2)
        batch_queries = torch.stack(
            (mask_flat[:, 0].argmax(dim=-1), mask_flat[:, 1].argmax(dim=-1)),
            dim=1,
        )
        query_indices[indices] = batch_queries.cpu().numpy()
        patch_masks[indices] = batch_masks.cpu().numpy()

        tokens = model.backbone.prepare_tokens(images)
        for block_index in blocks:
            block = model.backbone.blocks[block_index]
            normalized = block.norm1(tokens)
            patches = normalized[:, 1:]
            low_rank, sparse = decomposition[block_index]
            path_logits = _path_logits(
                patches,
                batch_queries,
                low_rank,
                sparse,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            joint_logits = sum(path_logits.values())
            _, dense_attention = _project_attention(
                patches,
                batch_queries,
                block.attn.qkv.weight,
                block.attn.qkv.bias,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            _, full_attention = _project_attention(
                patches,
                batch_queries,
                low_rank + sparse,
                block.attn.qkv.bias,
                num_heads=head_count,
                scale=block.attn.scale,
            )
            conditions = {
                "dense_x": dense_attention,
                "full_l_plus_s": full_attention,
                "joint_no_bias": joint_logits.softmax(dim=-1),
                **{path: path_logits[path].softmax(dim=-1) for path in PATH_NAMES},
            }
            for local_offset, global_index in enumerate(indices.tolist()):
                maps[global_index][block_index] = {
                    condition: conditions[condition][local_offset]
                    .reshape(
                        len(QUERY_NAMES),
                        head_count,
                        PATCH_GRID,
                        PATCH_GRID,
                    )
                    .cpu()
                    .numpy()
                    for condition in CONDITION_NAMES
                }
            tokens = block(tokens)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_representatives(
        dataset,
        selected_indices,
        blocks,
        maps,
        query_indices,
        patch_masks,
        output_dir,
    )
    manifest = {
        "seed": args.seed,
        "excluded_images": sorted(excluded_names),
        "samples": [
            {
                "index": index,
                "image_name": record.image_name,
                "target_type": record.target_type,
                "attributes": list(record.attributes),
                "num_distractors": record.num_distractors,
            }
            for index, record in zip(selected_indices, selected_records)
        ],
        "blocks": list(blocks),
        "conditions": list(CONDITION_NAMES),
        "queries": list(QUERY_NAMES),
    }
    (output_dir / "samples.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# Additional O3 Q--K path heatmaps",
        "",
        "Six deterministic samples not used by the original visualization. "
        "Each figure compares Dense X, Full L+S, the joint bias-free interaction, "
        "and the isolated LL/SL/LS/SS paths. Heatmaps are split into three "
        "four-block panels for each query type.",
        "",
        "| Sample | Category | Attributes | Distractors | Target query | Distractor query |",
        "|:---|:---|:---|---:|:---|:---|",
    ]
    for record in selected_records:
        stem = Path(record.image_name).stem
        target_links = ", ".join(
            f"[{start:02d}--{start + 3:02d}]"
            f"(sample_{stem}_target_blocks{start:02d}-{start + 3:02d}.png)"
            for start in (0, 4, 8)
        )
        distractor_links = ", ".join(
            f"[{start:02d}--{start + 3:02d}]"
            f"(sample_{stem}_distractor_blocks{start:02d}-{start + 3:02d}.png)"
            for start in (0, 4, 8)
        )
        readme_lines.append(
            f"| `{record.image_name}` | {record.target_type} | "
            f"{', '.join(record.attributes)} | {record.num_distractors} | "
            f"{target_links} | {distractor_links} |"
        )
    readme_lines.extend(
        [
            "",
            "The first column in every figure overlays the O3 masks and marks "
            "the selected query patch. Each heatmap panel is independently "
            "normalized; compare spatial locations rather than absolute strength.",
        ]
    )
    (output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Heatmaps written to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
