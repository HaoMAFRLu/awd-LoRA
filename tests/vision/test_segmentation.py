"""Tests for the VOC semantic-segmentation probe."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from salaad_vision.build import build_data, build_task
from salaad_vision.models import DinoFeatures, DinoSegmentationHead
from salaad_vision.tasks import SegmentationTask
from salaad_vision.trainer import VisionTrainer

ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_CONFIGS = sorted(
    (ROOT / "configs").glob("vision_voc2012_*_segmentation.yaml")
)


class _TinyPatchBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> DinoFeatures:
        values = images.mean(dim=(1, 2, 3)) * self.scale
        cls = values[:, None].expand(-1, 768)
        patches = values[:, None, None].expand(-1, 784, 768)
        return DinoFeatures(cls=cls, patches=patches)


def _segmentation_loader() -> DataLoader:
    first = torch.zeros((8, 8), dtype=torch.int64)
    first[:, 4:] = 1
    second = torch.ones((8, 8), dtype=torch.int64)
    second[4:, :] = 2
    samples = [
        {
            "pixel_values": torch.full((3, 8, 8), 0.25),
            "labels": first,
        },
        {
            "pixel_values": torch.full((3, 8, 8), 0.75),
            "labels": second,
        },
    ]
    return DataLoader(samples, batch_size=2)


def _write_voc_fixture(root: Path) -> Path:
    voc_root = root / "VOCdevkit" / "VOC2012"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "SegmentationClass").mkdir()
    split_root = voc_root / "ImageSets" / "Segmentation"
    split_root.mkdir(parents=True)

    identifier = "2007_000001"
    image = Image.new("RGB", (320, 240), color=(32, 96, 160))
    image.save(voc_root / "JPEGImages" / f"{identifier}.jpg")
    mask = Image.new("L", (320, 240), color=0)
    mask.paste(1, (106, 0, 212, 240))
    mask.paste(255, (212, 0, 320, 240))
    mask.save(voc_root / "SegmentationClass" / f"{identifier}.png")
    (split_root / "train.txt").write_text(f"{identifier}\n", encoding="utf-8")
    (split_root / "val.txt").write_text(f"{identifier}\n", encoding="utf-8")
    (split_root / "trainval.txt").write_text(f"{identifier}\n", encoding="utf-8")
    return voc_root


class SegmentationProbeTest(unittest.TestCase):
    def test_head_restores_patch_grid_and_ignores_cls_token(self) -> None:
        torch.manual_seed(3)
        head = DinoSegmentationHead(num_classes=21, output_size=(32, 40))
        patches = torch.randn(2, 784, 768)
        first = head(
            DinoFeatures(
                cls=torch.zeros(2, 768),
                patches=patches,
            )
        )
        second = head(
            DinoFeatures(
                cls=torch.ones(2, 768),
                patches=patches,
            )
        )

        self.assertEqual(tuple(first.shape), (2, 21, 32, 40))
        self.assertTrue(torch.equal(first, second))

    def test_head_rejects_non_vitb8_patch_layout(self) -> None:
        head = DinoSegmentationHead(num_classes=21)
        features = DinoFeatures(
            cls=torch.zeros(1, 768),
            patches=torch.zeros(1, 783, 768),
        )

        with self.assertRaisesRegex(ValueError, r"\[B, 784, 768\]"):
            head(features)

    def test_metrics_exclude_ignore_pixels(self) -> None:
        task = SegmentationTask(
            num_classes=3,
            output_size=2,
            ignore_index=255,
            boundary_tolerance=0,
        )
        labels = torch.tensor([[[0, 0], [1, 255]]], dtype=torch.int64)
        predictions = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64)
        logits = torch.full((1, 3, 2, 2), -10.0)
        logits.scatter_(1, predictions[:, None], 10.0)

        stats = task.batch_stats(logits, labels)
        metrics = task.summarize(stats)

        self.assertEqual(task.batch_weight(labels), 3)
        self.assertEqual(stats["confusion"].sum().item(), 3)
        self.assertAlmostEqual(metrics["miou"], 50.0)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["mean_accuracy"], 75.0)
        self.assertAlmostEqual(metrics["boundary_f1"], 250.0 / 3.0)

    def test_loss_rejects_an_all_ignore_batch(self) -> None:
        task = SegmentationTask(num_classes=3, output_size=2)
        labels = torch.full((1, 2, 2), 255, dtype=torch.int64)
        logits = torch.zeros(1, 3, 2, 2)

        with self.assertRaisesRegex(ValueError, "only ignored pixels"):
            task.loss(logits, labels)

    def test_config_builds_segmentation_task_and_voc_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_voc_fixture(root)
            config = {
                "seed": 7,
                "task": {
                    "name": "semantic_segmentation",
                    "head": "linear_upsample",
                    "num_classes": 21,
                    "ignore_index": 255,
                    "boundary_tolerance": 1,
                },
                "data": {
                    "name": "voc2012",
                    "root": "${TEST_VOC2012_ROOT}",
                    "image_size": 32,
                    "resize_size": 40,
                    "train": {"split": "train", "shuffle": True},
                    "validation": {"split": "val", "shuffle": False},
                },
                "training": {
                    "batch_size": 1,
                    "num_workers": 0,
                },
            }

            with patch.dict(os.environ, {"TEST_VOC2012_ROOT": str(root)}):
                task = build_task(config)
                loader = build_data(config, "validation")
                batch = next(iter(loader))

        self.assertIsInstance(task, SegmentationTask)
        self.assertEqual(task.num_classes, 21)
        self.assertEqual(task.head.output_size, (32, 32))
        self.assertEqual(set(batch), {"pixel_values", "labels"})
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(batch["labels"].shape), (1, 32, 32))
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertEqual(set(torch.unique(batch["labels"]).tolist()), {0, 1, 255})

    def test_voc_distributed_loader_uses_distributed_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_voc_fixture(root)
            config = {
                "seed": 7,
                "task": {
                    "name": "semantic_segmentation",
                    "num_classes": 21,
                },
                "data": {
                    "name": "voc2012",
                    "root": str(root),
                    "image_size": 32,
                    "train": {"split": "train", "shuffle": True},
                    "validation": {"split": "val", "shuffle": False},
                },
                "training": {"batch_size": 1, "num_workers": 0},
            }

            loader = build_data(config, "train", rank=1, world_size=2)

        self.assertIsInstance(loader.sampler, DistributedSampler)
        self.assertEqual(loader.sampler.rank, 1)
        self.assertEqual(loader.sampler.num_replicas, 2)

    def test_all_backbone_configs_share_the_probe_protocol(self) -> None:
        expected = {
            "teacher",
            "vanilla",
            "salaad_all",
            "salaad_qkv",
            "salaad_qkv_s50_alpha1",
            "salaad_qkv_s50_alpha1p5",
            "salaad_qkv_s50_alpha3",
        }
        observed = set()

        self.assertEqual(len(SEGMENTATION_CONFIGS), len(expected))
        for path in SEGMENTATION_CONFIGS:
            with self.subTest(config=path.name), path.open(
                "r", encoding="utf-8"
            ) as config_file:
                config = yaml.safe_load(config_file)
                task = config["task"]
                data = config["data"]
                model = config["model"]

                label = model.get("label", model["variant"])
                observed.add(label)
                self.assertEqual(task["name"], "semantic_segmentation")
                self.assertEqual(task["head"], "linear_upsample")
                self.assertEqual(task["num_classes"], 21)
                self.assertTrue(model["freeze"])
                self.assertEqual(data["name"], "voc2012")
                self.assertEqual(data["root"], "${VOC2012_ROOT}")
                self.assertEqual(data["train"]["split"], "train")
                self.assertEqual(data["validation"]["split"], "val")
                checkpoint = ROOT / model["checkpoint"]
                self.assertTrue(checkpoint.is_file(), checkpoint)

        self.assertEqual(observed, expected)

    def test_trainer_updates_only_the_segmentation_head(self) -> None:
        model = _TinyPatchBackbone().requires_grad_(False)
        task = SegmentationTask(num_classes=3, output_size=8)
        model_before = model.scale.detach().clone()
        head_before = task.head.projection.weight.detach().clone()
        config = {
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "precision": "float32",
                "max_steps_per_epoch": 1,
            },
            "validation": {"max_steps": 1},
            "optimizer": {
                "name": "sgd",
                "lr": 0.1,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "scheduler": {"name": "none"},
            "output": {"save": False},
        }
        trainer = VisionTrainer(
            model,
            task,
            _segmentation_loader(),
            _segmentation_loader(),
            config,
            torch.device("cpu"),
        )

        history = trainer.fit()

        self.assertTrue(torch.equal(model.scale, model_before))
        self.assertIsNone(model.scale.grad)
        self.assertFalse(torch.equal(task.head.projection.weight, head_before))
        self.assertIn("miou", history[0]["validation"])
        self.assertIn("pixel_accuracy", history[0]["validation"])
        self.assertIn("boundary_f1", history[0]["validation"])


if __name__ == "__main__":
    unittest.main()
