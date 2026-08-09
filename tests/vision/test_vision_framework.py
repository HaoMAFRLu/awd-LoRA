"""Tests for the config-driven downstream vision framework."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch import nn
from torch.utils.data import DataLoader

from salaad_vision.build import build_data, build_task
from salaad_vision.models import DinoFeatures
from salaad_vision.tasks import ClassificationTask
from salaad_vision.trainer import VisionTrainer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "vision_imagenet_smoke.yaml"


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> DinoFeatures:
        values = images.mean(dim=(1, 2, 3)) * self.scale
        cls = values[:, None].expand(-1, 768)
        patches = values[:, None, None].expand(-1, 2, 768)
        return DinoFeatures(cls=cls, patches=patches)


def _loader() -> DataLoader:
    samples = [
        {
            "pixel_values": torch.full((3, 8, 8), 0.25),
            "labels": torch.tensor(0, dtype=torch.int64),
        },
        {
            "pixel_values": torch.full((3, 8, 8), 0.75),
            "labels": torch.tensor(1, dtype=torch.int64),
        },
    ]
    return DataLoader(samples, batch_size=2)


def _ddp_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_file,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100 + rank)
        model = _TinyBackbone()
        model.requires_grad_(False)
        task = ClassificationTask(num_classes=10)
        sample = {
            "pixel_values": torch.full((3, 8, 8), 0.25 + 0.5 * rank),
            "labels": torch.tensor(rank, dtype=torch.int64),
        }
        loader = DataLoader([sample], batch_size=1)
        config = {
            "training": {
                "epochs": 1,
                "batch_size": 1,
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
            "output": {
                "save": True,
                "dir": str(Path(result_dir) / "output"),
            },
        }
        trainer = VisionTrainer(
            model,
            task,
            loader,
            loader,
            config,
            torch.device("cpu"),
            rank=rank,
            world_size=world_size,
        )
        history = trainer.fit()
        torch.save(
            {
                "history": history,
                "task": task.state_dict(),
                "model_grad": model.scale.grad,
            },
            Path(result_dir) / f"rank{rank}.pth",
        )
    finally:
        dist.destroy_process_group()


class VisionFrameworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG.open("r", encoding="utf-8") as config_file:
            cls.smoke_config = yaml.safe_load(config_file)

    def test_config_selects_classification_task(self) -> None:
        task = build_task(self.smoke_config)

        self.assertIsInstance(task, ClassificationTask)
        self.assertEqual(task.num_classes, 1000)

    def test_config_builds_labeled_imagenet_batch(self) -> None:
        loader = build_data(self.smoke_config, "validation")
        batch = next(iter(loader))
        batch_size = self.smoke_config["training"]["batch_size"]

        self.assertEqual(set(batch), {"pixel_values", "labels"})
        self.assertEqual(
            tuple(batch["pixel_values"].shape),
            (batch_size, 3, 224, 224),
        )
        self.assertEqual(tuple(batch["labels"].shape), (batch_size,))
        self.assertEqual(batch["labels"].dtype, torch.int64)

    def test_frozen_backbone_stays_fixed_while_head_updates(self) -> None:
        model = _TinyBackbone()
        model.requires_grad_(False)
        task = ClassificationTask(num_classes=10)
        model_before = model.scale.detach().clone()
        head_before = task.head.linear.weight.detach().clone()
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
            _loader(),
            _loader(),
            config,
            torch.device("cpu"),
        )

        history = trainer.fit()

        self.assertTrue(torch.equal(model.scale, model_before))
        self.assertIsNone(model.scale.grad)
        self.assertFalse(torch.equal(task.head.linear.weight, head_before))
        self.assertEqual(len(history), 1)
        self.assertIn("top1", history[0]["validation"])
        self.assertIn("top5", history[0]["validation"])

    def test_checkpoint_and_readable_epoch_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
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
                "output": {"save": True, "dir": temporary_root},
            }
            model = _TinyBackbone().requires_grad_(False)
            task = ClassificationTask(num_classes=10)
            trainer = VisionTrainer(
                model,
                task,
                _loader(),
                _loader(),
                config,
                torch.device("cpu"),
            )

            output = io.StringIO()
            with redirect_stdout(output):
                trainer.fit()

            checkpoint_path = Path(temporary_root) / "checkpoint.pth"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

        text = output.getvalue()
        self.assertIn("Epoch 1/1", text)
        self.assertIn("Train", text)
        self.assertIn("Validation", text)
        self.assertIn("Checkpoint", text)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("config", checkpoint)
        self.assertIn("task", checkpoint)
        self.assertNotIn("model", checkpoint)

    def test_wandb_uploads_epoch_metrics_and_finishes(self) -> None:
        run = MagicMock()
        run.name = "unit-test"
        run.id = "run-id"
        fake_wandb = SimpleNamespace(
            login=MagicMock(),
            init=MagicMock(return_value=run),
        )
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
            "wandb": {
                "enabled": True,
                "project": "SALAAD_VISION_DOWNSTREAM",
                "entity": "hao-ma-eth-z-rich",
                "group": "unit-test",
            },
            "output": {"save": False},
        }

        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.dict(
            os.environ,
            {"WANDB_API_KEY": "test-key"},
        ), redirect_stdout(io.StringIO()):
            trainer = VisionTrainer(
                _TinyBackbone().requires_grad_(False),
                ClassificationTask(num_classes=10),
                _loader(),
                _loader(),
                config,
                torch.device("cpu"),
            )
            trainer.fit()

        fake_wandb.login.assert_called_once_with(
            key="test-key",
            relogin=False,
        )
        fake_wandb.init.assert_called_once()
        self.assertRegex(
            fake_wandb.init.call_args.kwargs["name"],
            r"^\d{8}_\d{6}$",
        )
        payload = run.log.call_args.args[0]
        self.assertEqual(run.log.call_args.kwargs["step"], 1)
        self.assertEqual(payload["epoch"], 1)
        self.assertIn("train/loss", payload)
        self.assertIn("validation/loss", payload)
        self.assertIn("validation/top1", payload)
        self.assertIn("validation/top5", payload)
        run.finish.assert_called_once_with()

    def test_ddp_synchronizes_head_and_global_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            init_file = (root / "process_group").as_uri()
            mp.spawn(
                _ddp_worker,
                args=(2, init_file, temporary_root),
                nprocs=2,
                join=True,
            )
            results = [
                torch.load(
                    root / f"rank{rank}.pth",
                    map_location="cpu",
                    weights_only=True,
                )
                for rank in range(2)
            ]
            checkpoint = torch.load(
                root / "output" / "checkpoint.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertFalse((root / "output" / "checkpoint.tmp").exists())

        self.assertEqual(results[0]["history"], results[1]["history"])
        self.assertIsNone(results[0]["model_grad"])
        self.assertIsNone(results[1]["model_grad"])
        for name, value in results[0]["task"].items():
            self.assertTrue(torch.equal(value, results[1]["task"][name]), name)
        self.assertEqual(set(checkpoint["task"]), set(results[0]["task"]))


if __name__ == "__main__":
    unittest.main()
