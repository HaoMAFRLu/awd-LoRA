"""Test whether shallow SALAAD Q/K low-rank paths perform object grouping.

The experiment follows the patch-to-patch Odd-One-Out (O3) protocol from
Pan et al., "Dissecting Query-Key Interaction in Vision Transformers": one
target patch and one distractor patch are selected from downsampled masks,
and their attention mass on target, distractor, and background regions is
measured.  Blocks 0--2 are evaluated by default.

All conditions at a block receive the same normalized token embeddings.  The
token stream itself is advanced with the trained dense-X qkv weights, so the
comparison changes only the current block's Q/K interaction:

* dense_x: trained dense checkpoint Q/K, including qkv bias;
* full_l_plus_s: reconstructed Q/K, including qkv bias;
* l_only: L-only Q/K, including qkv bias (the usual model ablation);
* ll_bilinear: pure bias-free L_Q^T L_K interaction used in the paper's math;
* shuffled_ll: the same bias-free LL interaction after a common input-feature
  permutation of L_Q and L_K.  This preserves rank, singular values, Frobenius
  norm, and left/right singular-vector cosine while breaking alignment with
  the learned token coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salaad_vision.models.dino import DinoViTBase8  # noqa: E402
from salaad_vision.models.salaad import _files, _load  # noqa: E402


IMAGE_SIZE = 224
PATCH_SIZE = 8
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = PATCH_GRID * PATCH_GRID
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

QUERY_NAMES = ("target", "distractor")
REGION_NAMES = ("target", "distractor", "background")
GROUP_NAMES = ("same", "different", "background")
CONDITION_NAMES = (
    "dense_x",
    "full_l_plus_s",
    "l_only",
    "ll_bilinear",
    "shuffled_ll",
)
CONDITION_LABELS = {
    "dense_x": "Dense X",
    "full_l_plus_s": "Full L+S",
    "l_only": "L-only\n(+ bias)",
    "ll_bilinear": "LL\n(no bias)",
    "shuffled_ll": "Shuffled LL\n(no bias)",
}
GROUP_COLORS = {
    "same": "#2a9d8f",
    "different": "#e76f51",
    "background": "#6c757d",
}


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
        "--output-dir",
        type=Path,
        default=ROOT / "data/figures/salaad_vision/o3_shallow_ll",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--representative-count", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


@dataclass(frozen=True)
class O3Record:
    image_name: str
    num_distractors: int
    target_type: str
    attributes: tuple[str, ...]


def _read_records(data_root: Path) -> list[O3Record]:
    properties_path = data_root / "image_properties.csv"
    if not properties_path.is_file():
        raise FileNotFoundError(f"O3 metadata does not exist: {properties_path}")

    records: list[O3Record] = []
    with properties_path.open(newline="", encoding="utf-8") as properties_file:
        reader = csv.DictReader(properties_file)
        required = {"image_name", "num_distractors", "target_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"O3 metadata is missing columns {sorted(required)}: {properties_path}"
            )
        attribute_columns = tuple(
            name
            for name in (
                "orientation",
                "color",
                "focus",
                "shape",
                "size",
                "location",
                "pattern",
            )
            if name in reader.fieldnames
        )
        for row in reader:
            records.append(
                O3Record(
                    image_name=row["image_name"],
                    num_distractors=int(row["num_distractors"]),
                    target_type=row["target_type"].strip(),
                    attributes=tuple(
                        name for name in attribute_columns if row[name].strip() == "1"
                    ),
                )
            )
    records.sort(key=lambda record: record.image_name)
    return records


def _mask_to_patch_grid(path: Path) -> Tensor:
    with Image.open(path) as mask_image:
        mask = torch.from_numpy(
            (np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128).copy()
        ).float()
    return F.interpolate(
        mask[None, None],
        size=(PATCH_GRID, PATCH_GRID),
        mode="area",
    )[0, 0]


class O3Dataset(Dataset[dict[str, object]]):
    """O3 images and aligned fractional patch masks."""

    def __init__(self, data_root: Path, max_images: int | None = None) -> None:
        self.root = data_root.expanduser().resolve()
        for directory_name in ("images", "targ_labels", "dist_labels"):
            directory = self.root / directory_name
            if not directory.is_dir():
                raise NotADirectoryError(f"O3 directory does not exist: {directory}")
        self.records = _read_records(self.root)
        if max_images is not None:
            if max_images <= 0:
                raise ValueError("max_images must be positive")
            self.records = self.records[:max_images]
        if not self.records:
            raise RuntimeError("O3 dataset contains no images")
        for record in self.records:
            for directory_name in ("images", "targ_labels", "dist_labels"):
                path = self.root / directory_name / record.image_name
                if not path.is_file():
                    raise FileNotFoundError(f"O3 sample file does not exist: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def masks(self, index: int) -> Tensor:
        name = self.records[index].image_name
        target = _mask_to_patch_grid(self.root / "targ_labels" / name)
        distractor = _mask_to_patch_grid(self.root / "dist_labels" / name)
        if torch.any(target + distractor > 1.00001):
            raise ValueError(f"target and distractor masks overlap: {name}")
        background = (1.0 - target - distractor).clamp_(0.0, 1.0)
        masks = torch.stack((target, distractor, background))
        if target.max() <= 0 or distractor.max() <= 0:
            raise ValueError(f"O3 sample has an empty target or distractor mask: {name}")
        return masks

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(self.root / "images" / record.image_name) as source:
            rgb = source.convert("RGB")
            resized = TF.resize(
                rgb,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        image = TF.normalize(
            TF.to_tensor(resized),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        return {
            "index": index,
            "image_name": record.image_name,
            "image": image,
            "masks": self.masks(index),
        }


def choose_representatives(
    dataset: O3Dataset,
    count: int,
    seed: int,
) -> list[int]:
    """Choose non-cherry-picked, well-resolved samples with distinct categories."""
    if count <= 0:
        return []
    candidates = list(range(len(dataset)))
    random.Random(seed).shuffle(candidates)
    selected: list[int] = []
    used_categories: set[str] = set()
    for index in candidates:
        record = dataset.records[index]
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
        category = record.target_type or "unknown"
        if category in used_categories:
            continue
        selected.append(index)
        used_categories.add(category)
        if len(selected) == count:
            return sorted(selected)
    if len(selected) < count:
        raise RuntimeError(
            f"only found {len(selected)} suitable representative O3 samples"
        )
    return sorted(selected)


def load_decompositions(
    matrix_dir: Path,
    blocks: Sequence[int],
) -> dict[int, tuple[Tensor, Tensor]]:
    """Load QKV L/S from SALAAD rank files or post-hoc RPCA layer files."""
    targets = {
        f"backbone.blocks.{block}.attn.qkv": block for block in blocks
    }
    decomposition: dict[int, tuple[Tensor, Tensor]] = {}
    rank_files = sorted(matrix_dir.glob("matrix_rank*.pkl"))
    if rank_files:
        for matrix_file in _files(matrix_dir):
            low_rank, sparse = _load(matrix_file)
            for name, block in targets.items():
                if name not in low_rank:
                    continue
                if block in decomposition:
                    raise ValueError(f"duplicate decomposition for {name}")
                l_weight = low_rank[name]
                s_weight = sparse[name]
                if not isinstance(l_weight, Tensor) or not isinstance(s_weight, Tensor):
                    raise TypeError(f"L/S entries must be tensors: {name}")
                if l_weight.shape != s_weight.shape or l_weight.shape[0] % 3 != 0:
                    raise ValueError(f"invalid qkv L/S shapes for {name}")
                decomposition[block] = (l_weight.float(), s_weight.float())
    else:
        layer_directory = matrix_dir / "layers"
        if not layer_directory.is_dir():
            raise FileNotFoundError(
                "no SALAAD rank files or post-hoc RPCA layers found in "
                f"{matrix_dir}"
            )
        pattern = re.compile(
            r"^(?:backbone\.)?blocks\.(\d+)\.attn\.qkv\.weight$"
        )
        for component_file in sorted(layer_directory.glob("*.pth")):
            payload = torch.load(component_file, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise TypeError(f"invalid RPCA component payload: {component_file}")
            state_key = payload.get("state_key")
            match = pattern.fullmatch(state_key) if isinstance(state_key, str) else None
            if match is None:
                continue
            block = int(match.group(1))
            if block not in blocks:
                continue
            if block in decomposition:
                raise ValueError(f"duplicate RPCA decomposition for block {block}")
            l_weight = payload.get("L")
            s_weight = payload.get("S")
            if not isinstance(l_weight, Tensor) or not isinstance(s_weight, Tensor):
                raise TypeError(f"RPCA component lacks tensor L/S: {component_file}")
            if l_weight.shape != s_weight.shape or l_weight.shape[0] % 3 != 0:
                raise ValueError(f"invalid qkv L/S shapes in {component_file}")
            decomposition[block] = (l_weight.float(), s_weight.float())
    missing = set(blocks) - set(decomposition)
    if missing:
        raise ValueError(f"missing qkv decomposition for blocks {sorted(missing)}")
    return decomposition


def _common_feature_shuffle(weight: Tensor, permutation: Tensor) -> Tensor:
    if permutation.ndim != 1 or permutation.numel() != weight.shape[1]:
        raise ValueError("feature permutation does not match qkv input width")
    return weight[:, permutation]


def _interaction_invariants(
    low_rank: Tensor,
    shuffled: Tensor,
    permutation: Tensor,
    num_heads: int,
) -> dict[str, float]:
    """Verify the common feature permutation preserves every LL spectrum."""
    q_low, k_low, _ = low_rank.chunk(3, dim=0)
    q_shuffled, k_shuffled, _ = shuffled.chunk(3, dim=0)
    head_dim = q_low.shape[0] // num_heads
    maximum_frobenius_relative_error = 0.0
    maximum_conjugation_relative_error = 0.0
    maximum_rank_difference = 0
    for head in range(num_heads):
        row_slice = slice(head * head_dim, (head + 1) * head_dim)
        original = q_low[row_slice].T @ k_low[row_slice]
        permuted = q_shuffled[row_slice].T @ k_shuffled[row_slice]
        expected_permuted = original[permutation][:, permutation]
        conjugation_error = float(
            (permuted - expected_permuted).norm()
            / original.norm().clamp_min(1e-12)
        )
        maximum_conjugation_relative_error = max(
            maximum_conjugation_relative_error,
            conjugation_error,
        )
        relative_error = abs(float(permuted.norm() / original.norm()) - 1.0)
        maximum_frobenius_relative_error = max(
            maximum_frobenius_relative_error,
            relative_error,
        )
        original_rank = int(torch.linalg.matrix_rank(q_low[row_slice]).item())
        shuffled_rank = int(torch.linalg.matrix_rank(q_shuffled[row_slice]).item())
        maximum_rank_difference = max(
            maximum_rank_difference,
            abs(original_rank - shuffled_rank),
        )
    if (
        maximum_frobenius_relative_error > 2e-5
        or maximum_conjugation_relative_error > 2e-5
        or maximum_rank_difference != 0
    ):
        raise AssertionError("shuffled LL failed its rank/Frobenius invariant")
    return {
        "maximum_head_interaction_frobenius_relative_error": (
            maximum_frobenius_relative_error
        ),
        "maximum_head_interaction_conjugation_relative_error": (
            maximum_conjugation_relative_error
        ),
        "maximum_q_rank_difference": maximum_rank_difference,
    }


def _project_attention(
    patches: Tensor,
    query_indices: Tensor,
    qkv_weight: Tensor,
    qkv_bias: Tensor | None,
    *,
    num_heads: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Return patch-only logits and softmax maps for two query patches."""
    batch, num_patches, width = patches.shape
    if num_patches != NUM_PATCHES:
        raise ValueError(f"expected {NUM_PATCHES} patches, got {num_patches}")
    if qkv_weight.shape != (3 * width, width):
        raise ValueError(f"unexpected qkv weight shape: {tuple(qkv_weight.shape)}")
    batch_indices = torch.arange(batch, device=patches.device)[:, None]
    query_features = patches[batch_indices, query_indices]
    q_weight, k_weight, _ = qkv_weight.chunk(3, dim=0)
    q_bias: Tensor | None = None
    k_bias: Tensor | None = None
    if qkv_bias is not None:
        q_bias, k_bias, _ = qkv_bias.chunk(3, dim=0)
    head_dim = width // num_heads
    queries = F.linear(query_features, q_weight, q_bias).reshape(
        batch,
        len(QUERY_NAMES),
        num_heads,
        head_dim,
    )
    keys = F.linear(patches, k_weight, k_bias).reshape(
        batch,
        num_patches,
        num_heads,
        head_dim,
    )
    logits = torch.einsum("bqhd,bnhd->bqhn", queries, keys) * scale
    return logits, logits.softmax(dim=-1)


def _condition_weights(
    x_weight: Tensor,
    low_rank: Tensor,
    sparse: Tensor,
    permutation: Tensor,
    device: torch.device,
) -> dict[str, tuple[Tensor, bool]]:
    full = low_rank + sparse
    shuffled = _common_feature_shuffle(low_rank, permutation)
    return {
        "dense_x": (x_weight.to(device), True),
        "full_l_plus_s": (full.to(device), True),
        "l_only": (low_rank.to(device), True),
        "ll_bilinear": (low_rank.to(device), False),
        "shuffled_ll": (shuffled.to(device), False),
    }


def _group_values(values: np.ndarray) -> np.ndarray:
    """Convert [N,B,C,Q,H,R] directions to same/different/background."""
    same = 0.5 * (values[..., 0, :, 0] + values[..., 1, :, 1])
    different = 0.5 * (values[..., 0, :, 1] + values[..., 1, :, 0])
    background = 0.5 * (values[..., 0, :, 2] + values[..., 1, :, 2])
    return np.stack((same, different, background), axis=-1)


def _bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if samples <= 0 or values.size == 1:
        return mean, math.nan, math.nan
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 200
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return mean, float(low), float(high)


def _summary_rows(
    blocks: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for measure_name, values in arrays.items():
        grouped = _group_values(values).mean(axis=3)  # mean over heads
        for block_offset, block in enumerate(blocks):
            for condition_offset, condition in enumerate(CONDITION_NAMES):
                for group_offset, group in enumerate(GROUP_NAMES):
                    mean, low, high = _bootstrap_mean_ci(
                        grouped[:, block_offset, condition_offset, group_offset],
                        bootstrap_samples,
                        rng,
                    )
                    rows.append(
                        {
                            "block": block,
                            "condition": condition,
                            "measure": measure_name,
                            "group": group,
                            "mean": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
    return rows


def _per_image_rows(
    records: Sequence[O3Record],
    blocks: Sequence[int],
    masses: np.ndarray,
    enrichments: np.ndarray,
    nonself_enrichments: np.ndarray,
    logit_means: np.ndarray,
) -> list[dict[str, object]]:
    grouped_arrays = {
        "mass": _group_values(masses).mean(axis=3),
        "enrichment": _group_values(enrichments).mean(axis=3),
        "nonself_enrichment": _group_values(nonself_enrichments).mean(axis=3),
        "logit_mean": _group_values(logit_means).mean(axis=3),
    }
    rows: list[dict[str, object]] = []
    directional_arrays = {
        "mass": masses.mean(axis=4),
        "enrichment": enrichments.mean(axis=4),
        "nonself_enrichment": nonself_enrichments.mean(axis=4),
        "logit_mean": logit_means.mean(axis=4),
    }
    for image_index, record in enumerate(records):
        for block_offset, block in enumerate(blocks):
            for condition_offset, condition in enumerate(CONDITION_NAMES):
                row: dict[str, object] = {
                    "image_name": record.image_name,
                    "num_distractors": record.num_distractors,
                    "target_type": record.target_type,
                    "attributes": ",".join(record.attributes),
                    "block": block,
                    "condition": condition,
                }
                for measure_name, values in grouped_arrays.items():
                    same, different, background = values[
                        image_index,
                        block_offset,
                        condition_offset,
                    ]
                    row[f"{measure_name}_same"] = float(same)
                    row[f"{measure_name}_different"] = float(different)
                    row[f"{measure_name}_background"] = float(background)
                    row[f"{measure_name}_grouping_margin"] = float(
                        same - different
                    )
                for measure_name, values in directional_arrays.items():
                    directional = values[
                        image_index,
                        block_offset,
                        condition_offset,
                    ]
                    for query_offset, query in enumerate(QUERY_NAMES):
                        for region_offset, region in enumerate(REGION_NAMES):
                            row[f"{measure_name}_{query}_to_{region}"] = float(
                                directional[query_offset, region_offset]
                            )
                    row[f"{measure_name}_distractor_grouping_margin"] = float(
                        directional[1, 1] - directional[1, 0]
                    )
                rows.append(row)
    return rows


def _plot_group_bars(
    summary: pd.DataFrame,
    blocks: Sequence[int],
    measure: str,
    ylabel: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(blocks), figsize=(5.2 * len(blocks), 4.3))
    axes = np.atleast_1d(axes)
    x = np.arange(len(CONDITION_NAMES), dtype=float)
    width = 0.24
    for axis, block in zip(axes, blocks):
        for group_offset, group in enumerate(GROUP_NAMES):
            subset = summary[
                (summary["block"] == block)
                & (summary["measure"] == measure)
                & (summary["group"] == group)
            ].set_index("condition").loc[list(CONDITION_NAMES)]
            means = subset["mean"].to_numpy()
            low_values = subset["ci95_low"].to_numpy()
            high_values = subset["ci95_high"].to_numpy()
            lower = np.where(np.isfinite(low_values), means - low_values, 0.0)
            upper = np.where(np.isfinite(high_values), high_values - means, 0.0)
            axis.bar(
                x + (group_offset - 1) * width,
                means,
                width,
                label=group.capitalize(),
                color=GROUP_COLORS[group],
                yerr=np.vstack((lower, upper)),
                capsize=2,
                linewidth=0,
            )
        axis.set_title(f"Block {block}")
        axis.set_xticks(x, [CONDITION_LABELS[name] for name in CONDITION_NAMES])
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, fontsize=9)
    figure.suptitle("O3 patch-to-patch attention (mean over images and heads)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_grouping_margin(
    per_image: pd.DataFrame,
    blocks: Sequence[int],
    bootstrap_samples: int,
    seed: int,
    output_path: Path,
    *,
    column: str = "enrichment_grouping_margin",
    ylabel: str = "Grouping margin: same − different enrichment",
    title: str = "Does shallow LL prefer same-category object regions?",
) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    rng = np.random.default_rng(seed + 101)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(CONDITION_NAMES)))
    for color, condition in zip(colors, CONDITION_NAMES):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        subset = per_image[per_image["condition"] == condition]
        for block in blocks:
            values = subset[subset["block"] == block][
                column
            ].to_numpy()
            mean, low, high = _bootstrap_mean_ci(values, bootstrap_samples, rng)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        values_array = np.asarray(means)
        low_array = np.asarray(lows)
        high_array = np.asarray(highs)
        lower_errors = np.where(
            np.isfinite(low_array),
            values_array - low_array,
            0.0,
        )
        upper_errors = np.where(
            np.isfinite(high_array),
            high_array - values_array,
            0.0,
        )
        axis.errorbar(
            blocks,
            values_array,
            yerr=np.vstack((lower_errors, upper_errors)),
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=color,
            label=CONDITION_LABELS[condition].replace("\n", " "),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(blocks)
    axis.set_xlabel("Transformer block")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_headwise(
    enrichments: np.ndarray,
    blocks: Sequence[int],
    output_path: Path,
) -> None:
    grouped = _group_values(enrichments).mean(axis=0)  # [B,C,H,G]
    margin = grouped[..., 0] - grouped[..., 1]
    matrix = margin.transpose(1, 0, 2).reshape(len(CONDITION_NAMES), -1)
    limit = max(float(np.nanpercentile(np.abs(matrix), 98)), 1e-6)
    figure, axis = plt.subplots(figsize=(13.5, 3.8))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_yticks(
        np.arange(len(CONDITION_NAMES)),
        [CONDITION_LABELS[name].replace("\n", " ") for name in CONDITION_NAMES],
    )
    head_count = margin.shape[-1]
    axis.set_xticks(
        np.arange(len(blocks) * head_count),
        [f"{head + 1}" for _ in blocks for head in range(head_count)],
        fontsize=7,
    )
    for block_offset in range(1, len(blocks)):
        axis.axvline(block_offset * head_count - 0.5, color="black", linewidth=1.2)
    for block_offset, block in enumerate(blocks):
        axis.text(
            block_offset * head_count + (head_count - 1) / 2,
            -0.85,
            f"Block {block}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xlabel("Attention head (one-based)")
    axis.set_title("Head-wise grouping margin: same − different enrichment")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Grouping margin")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_paired_ll_control(
    per_image: pd.DataFrame,
    blocks: Sequence[int],
    bootstrap_samples: int,
    seed: int,
    output_path: Path,
    *,
    margin_column: str,
    measure_label: str,
) -> list[dict[str, object]]:
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    rng = np.random.default_rng(seed + 211)
    rows: list[dict[str, object]] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for block in blocks:
        block_rows = per_image[per_image["block"] == block]
        ll = block_rows[block_rows["condition"] == "ll_bilinear"].set_index(
            "image_name"
        )[margin_column]
        shuffled = block_rows[
            block_rows["condition"] == "shuffled_ll"
        ].set_index("image_name")[margin_column]
        common = ll.index.intersection(shuffled.index)
        delta = (ll.loc[common] - shuffled.loc[common]).to_numpy()
        mean, low, high = _bootstrap_mean_ci(delta, bootstrap_samples, rng)
        try:
            test = stats.wilcoxon(delta, zero_method="wilcox", alternative="greater")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        except ValueError:
            statistic = math.nan
            p_value = math.nan
        means.append(mean)
        lows.append(low)
        highs.append(high)
        rows.append(
            {
                "block": block,
                "measure": measure_label,
                "paired_images": len(delta),
                "mean_ll_minus_shuffled_grouping_margin": mean,
                "ci95_low": low,
                "ci95_high": high,
                "fraction_images_ll_greater_than_shuffled": float((delta > 0).mean()),
                "wilcoxon_greater_statistic": statistic,
                "wilcoxon_greater_p_value": p_value,
            }
        )
    means_array = np.asarray(means)
    lower_errors = np.where(
        np.isfinite(lows),
        means_array - np.asarray(lows),
        0.0,
    )
    upper_errors = np.where(
        np.isfinite(highs),
        np.asarray(highs) - means_array,
        0.0,
    )
    axis.bar(
        np.arange(len(blocks)),
        means_array,
        color="#457b9d",
        yerr=np.vstack((lower_errors, upper_errors)),
        capsize=4,
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(blocks)), [f"Block {block}" for block in blocks])
    axis.set_ylabel(f"LL − shuffled LL: {measure_label}")
    axis.set_title("Paired semantic-alignment control (95% bootstrap CI)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return rows


def _plot_decomposition_stats(
    decomposition_stats: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    blocks = [int(row["block"]) for row in decomposition_stats]
    density = [float(row["sparse_density"]) for row in decomposition_stats]
    error = [float(row["relative_l_plus_s_minus_x"]) for row in decomposition_stats]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    axes[0].bar(blocks, density, color="#8ecae6")
    axes[0].set_ylabel("S non-zero fraction")
    axes[0].set_title("Sparse component density")
    axes[1].bar(blocks, error, color="#ffb703")
    axes[1].set_ylabel("Relative Frobenius error")
    axes[1].set_title("‖L+S−X‖ / ‖X‖")
    for axis in axes:
        axis.set_xticks(blocks)
        axis.set_xlabel("Transformer block")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _annotated_image(
    image: np.ndarray,
    masks: np.ndarray,
    query_index: int,
    query_name: str,
) -> np.ndarray:
    canvas = image.copy()
    target = np.kron(masks[0], np.ones((PATCH_SIZE, PATCH_SIZE), dtype=float))
    distractor = np.kron(masks[1], np.ones((PATCH_SIZE, PATCH_SIZE), dtype=float))
    overlay = np.zeros_like(canvas)
    overlay[..., 0] = target
    overlay[..., 1] = distractor
    overlay[..., 2] = distractor
    strength = np.maximum(target, distractor)[..., None] * 0.38
    canvas = canvas * (1.0 - strength) + overlay * strength
    row, column = divmod(query_index, PATCH_GRID)
    y0, x0 = row * PATCH_SIZE, column * PATCH_SIZE
    color = np.array((1.0, 0.1, 0.1)) if query_name == "target" else np.array((0.1, 1.0, 1.0))
    canvas[y0 : y0 + 2, x0 : x0 + PATCH_SIZE] = color
    canvas[y0 + PATCH_SIZE - 2 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = color
    canvas[y0 : y0 + PATCH_SIZE, x0 : x0 + 2] = color
    canvas[y0 : y0 + PATCH_SIZE, x0 + PATCH_SIZE - 2 : x0 + PATCH_SIZE] = color
    return np.clip(canvas, 0.0, 1.0)


def _attention_overlay(image: np.ndarray, attention: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(
        Image.fromarray(attention.astype(np.float32), mode="F").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
    )
    upper = max(float(np.percentile(heatmap, 99.5)), 1e-12)
    normalized = np.clip(heatmap / upper, 0.0, 1.0)
    colored = plt.get_cmap("turbo")(normalized)[..., :3]
    strength = (0.18 + 0.58 * normalized)[..., None]
    return np.clip(image * (1.0 - strength) + colored * strength, 0.0, 1.0)


def _load_display_image(dataset: O3Dataset, index: int) -> np.ndarray:
    path = dataset.root / "images" / dataset.records[index].image_name
    with Image.open(path) as source:
        resized = source.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _plot_representative(
    dataset: O3Dataset,
    index: int,
    blocks: Sequence[int],
    maps: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    query_indices: np.ndarray,
    masks: np.ndarray,
    output_dir: Path,
) -> None:
    record = dataset.records[index]
    image = _load_display_image(dataset, index)
    for query_offset, query_name in enumerate(QUERY_NAMES):
        figure, axes = plt.subplots(
            len(blocks),
            len(CONDITION_NAMES) + 1,
            figsize=(3.0 * (len(CONDITION_NAMES) + 1), 3.0 * len(blocks)),
            squeeze=False,
        )
        query_index = int(query_indices[index, query_offset])
        for block_offset, block in enumerate(blocks):
            axes[block_offset, 0].imshow(
                _annotated_image(
                    image,
                    masks[index],
                    query_index,
                    query_name,
                )
            )
            axes[block_offset, 0].set_ylabel(f"Block {block}")
            for condition_offset, condition in enumerate(CONDITION_NAMES, start=1):
                attention = maps[index][block][condition][query_offset].mean(axis=0)
                axes[block_offset, condition_offset].imshow(
                    _attention_overlay(image, attention)
                )
            for axis in axes[block_offset]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes[0, 0].set_title("Target=red\nDistractors=cyan")
        for condition_offset, condition in enumerate(CONDITION_NAMES, start=1):
            axes[0, condition_offset].set_title(
                CONDITION_LABELS[condition].replace("\n", " ")
            )
        attributes = ", ".join(record.attributes) or "unspecified"
        figure.suptitle(
            f"{record.image_name} | {record.target_type} | {attributes} | "
            f"{query_name} query | head-mean attention",
            fontsize=12,
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"sample_{Path(record.image_name).stem}_{query_name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)


def _plot_representative_heads(
    dataset: O3Dataset,
    index: int,
    blocks: Sequence[int],
    maps: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    image = _load_display_image(dataset, index)
    record = dataset.records[index]
    query_offset = QUERY_NAMES.index("distractor")
    for block in blocks:
        figure, axes = plt.subplots(2, 12, figsize=(24, 4.3), squeeze=False)
        for row, condition in enumerate(("ll_bilinear", "shuffled_ll")):
            head_maps = maps[index][block][condition][query_offset]
            for head, axis in enumerate(axes[row]):
                axis.imshow(_attention_overlay(image, head_maps[head]))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"Head {head + 1}", fontsize=9)
            axes[row, 0].set_ylabel(
                CONDITION_LABELS[condition].replace("\n", " "),
                fontsize=9,
            )
        figure.suptitle(
            f"Block {block}, distractor query: LL versus spectrum-preserving "
            f"shuffle ({record.image_name})"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir
            / f"sample_{Path(record.image_name).stem}_block{block}_ll_heads.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def _write_readme(
    output_dir: Path,
    image_count: int,
    blocks: Sequence[int],
    paired_rows: Sequence[Mapping[str, object]],
    representative_records: Sequence[O3Record],
) -> None:
    lines = [
        "# O3 shallow LL grouping experiment",
        "",
        f"Images: {image_count}; blocks: {', '.join(map(str, blocks))}.",
        "",
        "The official O3 patch-to-patch protocol is reported as raw attention "
        "mass. Area-normalized enrichment is also reported so that large "
        "distractor/background masks do not win only because they contain more "
        "patches. Error bars are paired image-level 95% bootstrap confidence "
        "intervals after averaging the 12 heads.",
        "A non-self control removes the selected query patch from every "
        "measurement region before computing enrichment. It therefore rules "
        "out a trivial diagonal-attention explanation, although the O3 "
        "distractor mask still aggregates all distractor instances.",
        "",
        "The main causal control compares bias-free LL with shuffled LL. The "
        "shuffle applies one common input-feature permutation to L_Q and L_K "
        "within a block, preserving rank and the complete singular spectrum of "
        "each head's L_Q^T L_K interaction while breaking its alignment to the "
        "learned token coordinates.",
        "",
        "## Paired LL control",
        "",
        "| Measure | Block | Mean LL − shuffled margin | 95% CI | Fraction > 0 | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            "| {measure} | {block} | {mean:.5f} | [{low:.5f}, {high:.5f}] | "
            "{fraction:.3f} | {p:.3g} |".format(
                measure=row["measure"],
                block=row["block"],
                mean=row["mean_ll_minus_shuffled_grouping_margin"],
                low=row["ci95_low"],
                high=row["ci95_high"],
                fraction=row["fraction_images_ll_greater_than_shuffled"],
                p=row["wilcoxon_greater_p_value"],
            )
        )
    lines.extend(
        [
            "",
            "A positive difference means learned LL coordinates group same-category "
            "O3 regions more strongly than an interaction with identical spectral "
            "capacity but randomized feature alignment.",
            "",
            "## Deterministically selected representative samples",
            "",
        ]
    )
    for record in representative_records:
        lines.append(
            f"- `{record.image_name}`: {record.target_type}; "
            f"attributes={','.join(record.attributes) or 'unspecified'}; "
            f"distractors={record.num_distractors}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `attention_mass_official.png`: paper-compatible raw region mass.",
            "- `area_normalized_enrichment.png`: region mass divided by area fraction.",
            "- `grouping_margin.png`: same-minus-different enrichment.",
            "- `distractor_grouping_margin.png`: distractor-to-distractor minus "
            "distractor-to-target enrichment under the official region protocol.",
            "- `nonself_distractor_grouping_margin.png`: the same comparison after "
            "removing the query patch from all regions.",
            "- `ll_vs_shuffled_paired.png`: paired semantic-alignment test.",
            "- `ll_vs_shuffled_distractor_paired.png`: paired official "
            "distractor-region test.",
            "- `ll_vs_shuffled_nonself_distractor_paired.png`: paired non-self "
            "distractor-region control.",
            "- `headwise_grouping_margin.png`: all 12 heads in Blocks 0--2.",
            "- `decomposition_stats.png`: S density and L+S reconstruction gap.",
            "- `sample_*.png`: head-mean and head-wise attention overlays.",
            "- `summary.csv`, `per_image.csv.gz`, and `raw_metrics.npz`: numerical results.",
            "- `metadata.json`: exact paths, preprocessing, and invariant checks.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    blocks = tuple(sorted(set(args.blocks)))
    if not blocks or blocks[0] < 0 or blocks[-1] >= 12:
        raise ValueError("blocks must be non-empty and lie in [0, 11]")
    if blocks != tuple(range(blocks[-1] + 1)):
        raise ValueError(
            "this controlled-forward experiment requires consecutive blocks from 0"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    dataset = O3Dataset(args.data_root, args.max_images)
    representative_indices = choose_representatives(
        dataset,
        min(args.representative_count, len(dataset)),
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    model = DinoViTBase8(attention_backend="sdpa")
    model.load_state_dict(state, strict=True)
    decomposition = load_decompositions(matrix_dir, blocks)

    x_weights: dict[int, Tensor] = {}
    decomposition_stats: list[dict[str, object]] = []
    permutations: dict[int, Tensor] = {}
    invariant_stats: dict[int, dict[str, float]] = {}
    for block in blocks:
        name = f"backbone.blocks.{block}.attn.qkv.weight"
        x_weight = state[name].float()
        low_rank, sparse = decomposition[block]
        if x_weight.shape != low_rank.shape or sparse.shape != low_rank.shape:
            raise ValueError(f"checkpoint and L/S shapes differ for {name}")
        generator = torch.Generator().manual_seed(args.seed + 1009 * (block + 1))
        permutation = torch.randperm(low_rank.shape[1], generator=generator)
        shuffled = _common_feature_shuffle(low_rank, permutation)
        invariant_stats[block] = _interaction_invariants(
            low_rank,
            shuffled,
            permutation,
            model.backbone.blocks[block].attn.num_heads,
        )
        permutations[block] = permutation
        x_weights[block] = x_weight
        decomposition_stats.append(
            {
                "block": block,
                "sparse_nonzero": int(torch.count_nonzero(sparse).item()),
                "sparse_elements": sparse.numel(),
                "sparse_density": float(torch.count_nonzero(sparse) / sparse.numel()),
                "relative_l_plus_s_minus_x": float(
                    (low_rank + sparse - x_weight).norm()
                    / x_weight.norm().clamp_min(1e-12)
                ),
                "relative_s_norm": float(
                    sparse.norm() / x_weight.norm().clamp_min(1e-12)
                ),
            }
        )

    model.eval().to(device)

    image_count = len(dataset)
    block_count = len(blocks)
    condition_count = len(CONDITION_NAMES)
    head_count = model.backbone.blocks[0].attn.num_heads
    result_shape = (
        image_count,
        block_count,
        condition_count,
        len(QUERY_NAMES),
        head_count,
        len(REGION_NAMES),
    )
    masses = np.full(result_shape, np.nan, dtype=np.float32)
    nonself_masses = np.full(result_shape, np.nan, dtype=np.float32)
    logit_means = np.full(result_shape, np.nan, dtype=np.float32)
    areas = np.full((image_count, len(REGION_NAMES)), np.nan, dtype=np.float32)
    nonself_areas = np.full(
        (image_count, len(QUERY_NAMES), len(REGION_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    query_indices = np.full((image_count, len(QUERY_NAMES)), -1, dtype=np.int64)
    patch_masks = np.full(
        (image_count, len(REGION_NAMES), PATCH_GRID, PATCH_GRID),
        np.nan,
        dtype=np.float32,
    )
    representative_maps: dict[int, dict[int, dict[str, np.ndarray]]] = {
        index: {} for index in representative_indices
    }

    full_attention_checked_blocks: set[int] = set()
    print(
        f"O3 shallow LL | images={image_count} | blocks={blocks} | "
        f"batch={args.batch_size} | device={device}",
        flush=True,
    )
    for batch_number, batch in enumerate(loader, start=1):
        indices = batch["index"].numpy()
        images = batch["image"].to(device, non_blocking=True)
        batch_masks = batch["masks"].to(device, non_blocking=True)
        mask_flat = batch_masks.flatten(2)
        batch_queries = torch.stack(
            (mask_flat[:, 0].argmax(dim=-1), mask_flat[:, 1].argmax(dim=-1)),
            dim=1,
        )
        batch_area = mask_flat.mean(dim=-1)
        query_masks = mask_flat[:, None].expand(
            -1,
            len(QUERY_NAMES),
            -1,
            -1,
        ).clone()
        batch_indices = torch.arange(images.shape[0], device=device)[:, None]
        query_offsets = torch.arange(len(QUERY_NAMES), device=device)[None, :]
        query_masks[
            batch_indices,
            query_offsets,
            :,
            batch_queries,
        ] = 0.0
        batch_nonself_area = query_masks.mean(dim=-1)

        areas[indices] = batch_area.cpu().numpy()
        nonself_areas[indices] = batch_nonself_area.cpu().numpy()
        query_indices[indices] = batch_queries.cpu().numpy()
        patch_masks[indices] = batch_masks.cpu().numpy()

        tokens = model.backbone.prepare_tokens(images)
        for block_offset, block_index in enumerate(blocks):
            block = model.backbone.blocks[block_index]
            normalized = block.norm1(tokens)
            patches = normalized[:, 1:]
            low_rank, sparse = decomposition[block_index]
            conditions = _condition_weights(
                x_weights[block_index],
                low_rank,
                sparse,
                permutations[block_index],
                device,
            )
            qkv_bias = block.attn.qkv.bias
            condition_attention: dict[str, Tensor] = {}
            for condition_offset, condition in enumerate(CONDITION_NAMES):
                weight, include_bias = conditions[condition]
                logits, attention = _project_attention(
                    patches,
                    batch_queries,
                    weight,
                    qkv_bias if include_bias else None,
                    num_heads=head_count,
                    scale=block.attn.scale,
                )
                condition_attention[condition] = attention
                mass = torch.einsum("bqhn,brn->bqhr", attention, mask_flat)
                nonself_mass = torch.einsum(
                    "bqhn,bqrn->bqhr",
                    attention,
                    query_masks,
                )
                logit_mean = torch.einsum(
                    "bqhn,brn->bqhr",
                    logits,
                    mask_flat,
                ) / mask_flat.sum(dim=-1)[:, None, None, :].clamp_min(1e-12)
                masses[indices, block_offset, condition_offset] = mass.cpu().numpy()
                nonself_masses[
                    indices,
                    block_offset,
                    condition_offset,
                ] = nonself_mass.cpu().numpy()
                logit_means[
                    indices,
                    block_offset,
                    condition_offset,
                ] = logit_mean.cpu().numpy()

            if block_index not in full_attention_checked_blocks:
                _, actual_attention = block.attn(normalized, return_attention=True)
                patch_attention = actual_attention[:, :, 1:, 1:]
                patch_attention = patch_attention / patch_attention.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                gathered = torch.gather(
                    patch_attention,
                    dim=2,
                    index=batch_queries[:, None, :, None].expand(
                        -1,
                        head_count,
                        -1,
                        NUM_PATCHES,
                    ),
                ).permute(0, 2, 1, 3)
                maximum_difference = float(
                    (
                        gathered
                        - condition_attention["dense_x"]
                    ).abs().max()
                )
                if maximum_difference > 3e-5:
                    raise AssertionError(
                        "offline dense-X patch attention does not match model "
                        f"attention (maximum difference={maximum_difference:.6g})"
                    )
                full_attention_checked_blocks.add(block_index)

            for local_offset, global_index in enumerate(indices.tolist()):
                if global_index not in representative_maps:
                    continue
                representative_maps[global_index][block_index] = {
                    condition: condition_attention[condition][local_offset]
                    .reshape(len(QUERY_NAMES), head_count, PATCH_GRID, PATCH_GRID)
                    .cpu()
                    .numpy()
                    for condition in CONDITION_NAMES
                }
            tokens = block(tokens)

        if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(loader):
            completed = min(batch_number * args.batch_size, image_count)
            print(
                f"  [{completed:4d}/{image_count}] "
                f"{100.0 * completed / image_count:5.1f}%",
                flush=True,
            )

    if np.isnan(masses).any() or np.isnan(nonself_masses).any():
        raise AssertionError("not every O3 sample received attention metrics")
    if not np.allclose(masses.sum(axis=-1), 1.0, atol=2e-4):
        raise AssertionError("target/distractor/background attention mass does not sum to one")

    enrichments = masses / np.maximum(
        areas[:, None, None, None, None, :],
        1e-12,
    )
    nonself_enrichments = nonself_masses / np.maximum(
        nonself_areas[:, None, None, :, None, :],
        1e-12,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "raw_metrics.npz",
        masses=masses,
        nonself_masses=nonself_masses,
        enrichments=enrichments,
        nonself_enrichments=nonself_enrichments,
        logit_means=logit_means,
        areas=areas,
        nonself_areas=nonself_areas,
        query_indices=query_indices,
        image_names=np.asarray([record.image_name for record in dataset.records]),
        blocks=np.asarray(blocks),
        conditions=np.asarray(CONDITION_NAMES),
        queries=np.asarray(QUERY_NAMES),
        regions=np.asarray(REGION_NAMES),
    )

    summary_rows = _summary_rows(
        blocks,
        {
            "mass": masses,
            "enrichment": enrichments,
            "nonself_enrichment": nonself_enrichments,
            "logit_mean": logit_means,
        },
        args.bootstrap_samples,
        args.seed,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    per_image = pd.DataFrame(
        _per_image_rows(
            dataset.records,
            blocks,
            masses,
            enrichments,
            nonself_enrichments,
            logit_means,
        )
    )
    per_image.to_csv(
        output_dir / "per_image.csv.gz",
        index=False,
        compression="gzip",
    )

    _plot_group_bars(
        summary,
        blocks,
        "mass",
        "Attention mass",
        output_dir / "attention_mass_official.png",
    )
    _plot_group_bars(
        summary,
        blocks,
        "enrichment",
        "Attention enrichment (1 = uniform)",
        output_dir / "area_normalized_enrichment.png",
    )
    _plot_grouping_margin(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed,
        output_dir / "grouping_margin.png",
    )
    _plot_grouping_margin(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed + 1,
        output_dir / "distractor_grouping_margin.png",
        column="enrichment_distractor_grouping_margin",
        ylabel=(
            "Distractor grouping margin: distractor→distractor − "
            "distractor→target enrichment"
        ),
        title="O3 distractor-region grouping from a distractor query",
    )
    _plot_grouping_margin(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed + 2,
        output_dir / "nonself_distractor_grouping_margin.png",
        column="nonself_enrichment_distractor_grouping_margin",
        ylabel=(
            "Non-self distractor margin: distractor→distractor − "
            "distractor→target enrichment"
        ),
        title="Distractor-region grouping after removing the query patch",
    )
    _plot_headwise(
        enrichments,
        blocks,
        output_dir / "headwise_grouping_margin.png",
    )
    paired_rows = _plot_paired_ll_control(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed,
        output_dir / "ll_vs_shuffled_paired.png",
        margin_column="enrichment_grouping_margin",
        measure_label="combined same−different enrichment",
    )
    distractor_paired_rows = _plot_paired_ll_control(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed + 1,
        output_dir / "ll_vs_shuffled_distractor_paired.png",
        margin_column="enrichment_distractor_grouping_margin",
        measure_label="official distractor-region margin",
    )
    nonself_distractor_paired_rows = _plot_paired_ll_control(
        per_image,
        blocks,
        args.bootstrap_samples,
        args.seed + 2,
        output_dir / "ll_vs_shuffled_nonself_distractor_paired.png",
        margin_column="nonself_enrichment_distractor_grouping_margin",
        measure_label="non-self distractor-region margin",
    )
    all_paired_rows = (
        paired_rows
        + distractor_paired_rows
        + nonself_distractor_paired_rows
    )
    _plot_decomposition_stats(
        decomposition_stats,
        output_dir / "decomposition_stats.png",
    )
    for index in representative_indices:
        _plot_representative(
            dataset,
            index,
            blocks,
            representative_maps,
            query_indices,
            patch_masks,
            output_dir,
        )
    if representative_indices:
        _plot_representative_heads(
            dataset,
            representative_indices[0],
            blocks,
            representative_maps,
            output_dir,
        )

    metadata = {
        "experiment": "O3 shallow SALAAD LL grouping",
        "image_count": image_count,
        "blocks": list(blocks),
        "conditions": list(CONDITION_NAMES),
        "queries": list(QUERY_NAMES),
        "regions": list(REGION_NAMES),
        "device": str(device),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "checkpoint": str(checkpoint),
        "matrix_dir": str(matrix_dir),
        "data_root": str(dataset.root),
        "preprocessing": {
            "image": "direct bicubic resize to 224x224, ImageNet normalization",
            "mask": (
                "threshold JPEG mask at 128, area-resize directly to 28x28; "
                "background=1-target-distractor"
            ),
            "attention": (
                "patch queries to patch keys only; CLS key excluded; softmax "
                "renormalized over 784 patch keys"
            ),
            "nonself_control": (
                "remove the selected query patch from target, distractor, and "
                "background region masks before computing enrichment"
            ),
            "upstream_state": (
                "fixed trained dense-X forward path; only current-block Q/K "
                "interaction differs across conditions"
            ),
        },
        "representative_indices": representative_indices,
        "representative_images": [
            dataset.records[index].image_name for index in representative_indices
        ],
        "decomposition": decomposition_stats,
        "shuffle_invariants": {
            str(block): values for block, values in invariant_stats.items()
        },
        "dense_x_attention_consistency_checked_blocks": sorted(
            full_attention_checked_blocks
        ),
        "paired_ll_control": all_paired_rows,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir,
        image_count,
        blocks,
        all_paired_rows,
        [dataset.records[index] for index in representative_indices],
    )

    print("\nPaired LL semantic-alignment control", flush=True)
    for row in all_paired_rows:
        print(
            "  {measure}, Block {block}: Δ={mean:.5f}, "
            "95% CI=[{low:.5f}, {high:.5f}], "
            "P(Δ_image>0)={fraction:.3f}, Wilcoxon p={p:.3g}".format(
                measure=row["measure"],
                block=row["block"],
                mean=row["mean_ll_minus_shuffled_grouping_margin"],
                low=row["ci95_low"],
                high=row["ci95_high"],
                fraction=row["fraction_images_ll_greater_than_shuffled"],
                p=row["wilcoxon_greater_p_value"],
            ),
            flush=True,
        )
    print(f"\nResults written to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
