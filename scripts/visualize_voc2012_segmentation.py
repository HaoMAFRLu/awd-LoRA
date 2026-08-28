"""Compare four frozen vision backbones on Pascal VOC 2012 segmentation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from salaad_vision.build import build_data, build_model, build_task


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    title: str
    semantics: str
    config: Path


SPECS = (
    ModelSpec(
        "teacher",
        "Teacher",
        "official DINO dense backbone",
        ROOT / "configs/vision_voc2012_teacher_segmentation.yaml",
    ),
    ModelSpec(
        "vanilla",
        "Vanilla",
        "student dense backbone",
        ROOT / "configs/vision_voc2012_vanilla_segmentation.yaml",
    ),
    ModelSpec(
        "salaad_all",
        "SALAAD-all L+S",
        "training-time L+S for all 48 decomposed block matrices",
        ROOT / "configs/vision_voc2012_salaad_all_segmentation.yaml",
    ),
    ModelSpec(
        "salaad_qkv",
        "SALAAD-qkv L+S",
        "training-time L+S for all 12 qkv matrices; other matrices use dense X",
        ROOT / "configs/vision_voc2012_salaad_qkv_segmentation.yaml",
    ),
)

VOC_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
NUM_CLASSES = len(VOC_CLASSES)
IGNORE_INDEX = 255
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-samples", type=int, default=6)
    parser.add_argument("--diagnostic-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    return config


def _resolve_repo_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _checkpoint_path(config: Mapping[str, Any]) -> Path:
    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("config requires an output mapping")
    return _resolve_repo_path(output.get("dir"), "output.dir") / "checkpoint.pth"


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _voc_palette() -> np.ndarray:
    palette = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for class_index in range(NUM_CLASSES):
        value = class_index
        bit = 0
        while value:
            palette[class_index, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_index, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_index, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    return palette


PALETTE = _voc_palette()


def _confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE_INDEX
    encoded = (
        target[valid].astype(np.int64) * NUM_CLASSES
        + prediction[valid].astype(np.int64)
    )
    return np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def _summarize_confusion(confusion: np.ndarray) -> dict[str, Any]:
    target_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    union = target_count + predicted_count - true_positive
    class_iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    present = union > 0
    class_iou[present] = 100.0 * true_positive[present] / union[present]
    has_target = target_count > 0
    class_accuracy = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    class_accuracy[has_target] = (
        100.0 * true_positive[has_target] / target_count[has_target]
    )
    total = target_count.sum()
    foreground_iou = class_iou[1:]
    foreground_miou = (
        float(np.nanmean(foreground_iou))
        if np.isfinite(foreground_iou).any()
        else float("nan")
    )
    return {
        "miou": float(np.nanmean(class_iou)),
        "foreground_miou": foreground_miou,
        "pixel_accuracy": float(100.0 * true_positive.sum() / total),
        "mean_accuracy": float(np.nanmean(class_accuracy)),
        "class_iou": {
            name: None if np.isnan(value) else float(value)
            for name, value in zip(VOC_CLASSES, class_iou)
        },
    }


def _per_image_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict[str, float]]:
    result = []
    for prediction, target in zip(predictions.numpy(), labels.numpy()):
        confusion = _confusion(prediction, target)
        summary = _summarize_confusion(confusion)
        result.append(
            {
                "miou": summary["miou"],
                "foreground_miou": summary["foreground_miou"],
                "pixel_accuracy": summary["pixel_accuracy"],
            }
        )
    return result


def _overall_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for prediction, target in zip(predictions.numpy(), labels.numpy()):
        total += _confusion(prediction, target)
    return _summarize_confusion(total)


def _pairwise_disagreement(
    left: torch.Tensor,
    right: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, np.ndarray]:
    per_image = []
    unequal_total = 0
    valid_total = 0
    for left_mask, right_mask, target in zip(
        left.numpy(),
        right.numpy(),
        labels.numpy(),
    ):
        valid = target != IGNORE_INDEX
        unequal = int(np.count_nonzero((left_mask != right_mask) & valid))
        count = int(np.count_nonzero(valid))
        per_image.append(100.0 * unequal / count)
        unequal_total += unequal
        valid_total += count
    return 100.0 * unequal_total / valid_total, np.asarray(per_image)


def _load_predictions(
    spec: ModelSpec,
    config: dict[str, Any],
    loader,
    device: torch.device,
    *,
    collect_labels: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    model = build_model(config).to(device).eval()
    task = build_task(config).to(device).eval()
    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"segmentation checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"invalid task checkpoint: {checkpoint_path}")
    epoch = checkpoint.get("epoch")
    task_state = checkpoint.get("task")
    if not isinstance(epoch, int) or not isinstance(task_state, Mapping):
        raise TypeError(f"task checkpoint lacks epoch/task: {checkpoint_path}")
    task.load_state_dict(task_state, strict=True)

    prediction_batches = []
    label_batches = []
    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = task(model(images))
            prediction_batches.append(logits.argmax(dim=1).to(torch.uint8).cpu())
            if collect_labels:
                label_batches.append(batch["labels"].to(torch.uint8).cpu())

    predictions = torch.cat(prediction_batches)
    labels = torch.cat(label_batches) if collect_labels else None
    print(
        f"{spec.slug}: epoch={epoch}, samples={len(predictions)}, "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    del model, task
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, labels, epoch


def _unnormalize(image: torch.Tensor) -> np.ndarray:
    pixels = image.permute(1, 2, 0).numpy()
    pixels = pixels * IMAGENET_STD + IMAGENET_MEAN
    return np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5)


def _overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    colors = np.zeros_like(image)
    valid_classes = mask < NUM_CLASSES
    colors[valid_classes] = PALETTE[mask[valid_classes]]
    active = valid_classes & (mask != 0)
    blended = image.copy()
    blended[active] = np.uint8(
        0.35 * image[active].astype(np.float32)
        + 0.65 * colors[active].astype(np.float32)
        + 0.5
    )
    ignored = mask == IGNORE_INDEX
    blended[ignored] = np.uint8(
        0.35 * image[ignored].astype(np.float32) + 0.65 * 180.0 + 0.5
    )
    return Image.fromarray(blended, mode="RGB")


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _cell(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    width, height = image.size
    title_height = 54
    cell = Image.new("RGB", (width, height + title_height), "white")
    cell.paste(image, (0, title_height))
    draw = ImageDraw.Draw(cell)
    draw.text((6, 4), title, fill="black", font=_font(15, bold=True))
    if subtitle:
        draw.text((6, 27), subtitle, fill=(55, 55, 55), font=_font(12))
    return cell


def _sample_panel(
    index: int,
    identifier: str,
    dataset,
    labels: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    per_image: Mapping[str, list[dict[str, float]]],
    reason: str,
) -> Image.Image:
    sample = dataset[index]
    image_array = _unnormalize(sample["pixel_values"])
    target = labels[index].numpy()
    image = Image.fromarray(image_array, mode="RGB")
    target_classes = sorted(
        int(value)
        for value in np.unique(target)
        if 0 < value < NUM_CLASSES
    )
    class_text = ", ".join(VOC_CLASSES[value] for value in target_classes)
    cells = [
        _cell(image, f"{identifier} | Input", reason),
        _cell(_overlay(image_array, target), "Ground truth", class_text),
    ]
    for spec in SPECS:
        metrics = per_image[spec.slug][index]
        subtitle = (
            f"fg mIoU {metrics['foreground_miou']:.1f} | "
            f"pixel acc {metrics['pixel_accuracy']:.1f}"
        )
        cells.append(
            _cell(
                _overlay(image_array, predictions[spec.slug][index].numpy()),
                spec.title,
                subtitle,
            )
        )

    panel = Image.new(
        "RGB",
        (sum(cell.width for cell in cells), max(cell.height for cell in cells)),
        (225, 225, 225),
    )
    left = 0
    for cell in cells:
        panel.paste(cell, (left, 0))
        left += cell.width
    return panel


def _save_panels(
    name: str,
    selections: list[dict[str, Any]],
    output_dir: Path,
    identifiers: list[str],
    dataset,
    labels: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    per_image: Mapping[str, list[dict[str, float]]],
) -> None:
    sample_dir = output_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for selection in selections:
        index = selection["index"]
        identifier = identifiers[index]
        panel = _sample_panel(
            index,
            identifier,
            dataset,
            labels,
            predictions,
            per_image,
            selection["reason"],
        )
        panel.save(sample_dir / f"{identifier}.png")
        panels.append(panel)
    if panels:
        sheet = Image.new(
            "RGB",
            (max(panel.width for panel in panels), sum(panel.height for panel in panels)),
            "white",
        )
        top = 0
        for panel in panels:
            sheet.paste(panel, (0, top))
            top += panel.height
        sheet.save(output_dir / f"{name}_samples.png")


def _diagnostic_indices(
    count: int,
    random_indices: set[int],
    per_image: Mapping[str, list[dict[str, float]]],
    disagreement: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    scores = {
        slug: np.asarray([row["foreground_miou"] for row in rows])
        for slug, rows in per_image.items()
    }
    pairwise_mean = np.mean(np.stack(list(disagreement.values())), axis=0)
    criteria = (
        ("largest Teacher - SALAAD-all gap", scores["teacher"] - scores["salaad_all"]),
        ("largest SALAAD-qkv - SALAAD-all gap", scores["salaad_qkv"] - scores["salaad_all"]),
        ("largest Teacher - Vanilla gap", scores["teacher"] - scores["vanilla"]),
        ("largest Vanilla - Teacher gap", scores["vanilla"] - scores["teacher"]),
        ("largest SALAAD-qkv - Vanilla gap", scores["salaad_qkv"] - scores["vanilla"]),
        ("largest mean prediction disagreement", pairwise_mean),
    )
    selected = set(random_indices)
    result = []
    for reason, values in criteria:
        if len(result) >= count:
            break
        order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1]
        index = next((int(item) for item in order if int(item) not in selected), None)
        if index is None:
            continue
        selected.add(index)
        result.append({"index": index, "reason": reason})
    if len(result) < count:
        for index in np.argsort(pairwise_mean)[::-1]:
            index = int(index)
            if index in selected:
                continue
            selected.add(index)
            result.append({"index": index, "reason": "high prediction disagreement"})
            if len(result) == count:
                break
    return result


def _save_legend(output_dir: Path) -> None:
    columns = 3
    rows = (NUM_CLASSES + columns - 1) // columns
    width = 690
    height = 36 * rows + 20
    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    font = _font(15)
    for index, name in enumerate(VOC_CLASSES):
        column = index // rows
        row = index % rows
        x = 10 + column * 225
        y = 10 + row * 36
        color = tuple(int(value) for value in PALETTE[index])
        draw.rectangle((x, y, x + 25, y + 25), fill=color, outline="black")
        draw.text((x + 34, y + 4), f"{index}: {name}", fill="black", font=font)
    legend.save(output_dir / "voc_legend.png")


def main() -> None:
    args = parse_args()
    for value, name in (
        (args.batch_size, "batch-size"),
        (args.random_samples, "random-samples"),
        (args.diagnostic_samples, "diagnostic-samples"),
    ):
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _choose_device(args.device)
    configs = {spec.slug: _read_config(spec.config) for spec in SPECS}
    data_config = copy.deepcopy(configs["teacher"])
    data_config["data"]["root"] = str(args.voc_root.expanduser())
    data_config["data"]["validation"]["batch_size"] = args.batch_size
    data_config["data"]["validation"]["num_workers"] = args.num_workers
    loader = build_data(data_config, "validation", rank=0, world_size=1)
    dataset = loader.dataset
    identifiers = [image_path.stem for image_path, _ in dataset.samples]

    predictions: dict[str, torch.Tensor] = {}
    labels = None
    epochs = {}
    for spec in SPECS:
        config = configs[spec.slug]
        predicted, loaded_labels, epoch = _load_predictions(
            spec,
            config,
            loader,
            device,
            collect_labels=labels is None,
        )
        predictions[spec.slug] = predicted
        if loaded_labels is not None:
            labels = loaded_labels
        epochs[spec.slug] = epoch
    if labels is None:
        raise RuntimeError("validation labels were not collected")
    if any(len(values) != len(labels) for values in predictions.values()):
        raise RuntimeError("prediction and label counts differ")

    overall = {
        slug: _overall_metrics(values, labels)
        for slug, values in predictions.items()
    }
    per_image = {
        slug: _per_image_metrics(values, labels)
        for slug, values in predictions.items()
    }

    disagreement = {}
    disagreement_per_image = {}
    for left, right in combinations((spec.slug for spec in SPECS), 2):
        key = f"{left}__{right}"
        aggregate, per_sample = _pairwise_disagreement(
            predictions[left],
            predictions[right],
            labels,
        )
        disagreement[key] = aggregate
        disagreement_per_image[key] = per_sample

    rng = random.Random(args.seed)
    random_indices = sorted(rng.sample(range(len(labels)), args.random_samples))
    random_selections = [
        {"index": index, "reason": f"fixed random sample (seed={args.seed})"}
        for index in random_indices
    ]
    diagnostic_selections = _diagnostic_indices(
        args.diagnostic_samples,
        set(random_indices),
        per_image,
        disagreement_per_image,
    )

    for selections in (random_selections, diagnostic_selections):
        for selection in selections:
            selection["identifier"] = identifiers[selection["index"]]

    _save_panels(
        "random",
        random_selections,
        output_dir,
        identifiers,
        dataset,
        labels,
        predictions,
        per_image,
    )
    _save_panels(
        "diagnostic",
        diagnostic_selections,
        output_dir,
        identifiers,
        dataset,
        labels,
        predictions,
        per_image,
    )
    _save_legend(output_dir)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("backbone", "epoch", "miou", "foreground_miou", "pixel_accuracy", "mean_accuracy")
        )
        for spec in SPECS:
            metrics = overall[spec.slug]
            writer.writerow(
                (
                    spec.slug,
                    epochs[spec.slug],
                    metrics["miou"],
                    metrics["foreground_miou"],
                    metrics["pixel_accuracy"],
                    metrics["mean_accuracy"],
                )
            )

    with (output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = ["index", "identifier"]
        for spec in SPECS:
            fieldnames.extend(
                (
                    f"{spec.slug}_miou",
                    f"{spec.slug}_foreground_miou",
                    f"{spec.slug}_pixel_accuracy",
                )
            )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            row: dict[str, Any] = {"index": index, "identifier": identifier}
            for spec in SPECS:
                for metric, value in per_image[spec.slug][index].items():
                    row[f"{spec.slug}_{metric}"] = value
            writer.writerow(row)

    metadata = {
        "dataset": "Pascal VOC 2012 val",
        "samples": len(labels),
        "image_size": list(labels.shape[1:]),
        "seed": args.seed,
        "models": {
            spec.slug: {
                "title": spec.title,
                "semantics": spec.semantics,
                "config": str(spec.config),
                "checkpoint": str(_checkpoint_path(configs[spec.slug])),
                "epoch": epochs[spec.slug],
            }
            for spec in SPECS
        },
        "metrics": overall,
        "pairwise_prediction_disagreement_percent": disagreement,
        "random_samples": random_selections,
        "diagnostic_samples": diagnostic_selections,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"metrics": overall, "disagreement": disagreement}, indent=2))
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
