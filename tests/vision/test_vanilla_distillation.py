"""CPU contract for the vanilla DINO feature-distillation baseline."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, call, patch

import torch
from torch import nn

from salad.simple_timer import SimpleTimer
from salad.trainer_salad import SALADTrainer
from salad.utils import print_epoch
from salaad_vision.models.dino import DinoFeatures


class _TinyFeatureModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, images: torch.Tensor) -> DinoFeatures:
        features = images * self.scale
        return DinoFeatures(
            cls=features.mean(dim=(2, 3)),
            patches=features.flatten(2).transpose(1, 2),
        )


class VanillaDistillationTest(unittest.TestCase):
    def test_vanilla_logging_omits_empty_salaad_table(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            print_epoch(
                epoch=1,
                total_epochs=40,
                num_freq=5,
                lr=1e-5,
                num_images=5,
                losses={
                    "avg_loss": 1.0,
                    "avg_diff": 0.0,
                    "avg_loss_penalty": 0.0,
                },
                layer_stats=[],
            )

        logged_text = output.getvalue()
        self.assertIn("Epoch 1/40", logged_text)
        self.assertIn("Loss: 1.000000", logged_text)
        self.assertNotIn("non-zero", logged_text)
        self.assertNotIn("rho", logged_text)
        self.assertTrue(logged_text.rstrip().endswith("-" * 120))

    def test_vanilla_step_updates_student_without_admm_state(self) -> None:
        teacher = _TinyFeatureModel(scale=2.0)
        teacher.requires_grad_(False)
        teacher.eval()
        student = _TinyFeatureModel(scale=1.0)

        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.teacher_model = teacher
        trainer.ddp_model = student
        trainer.optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer,
            lr_lambda=lambda _: 1.0,
        )
        trainer.config = {
            "distillation": {
                "global_weight": 1.0,
                "patch_weight": 1.0,
            }
        }
        trainer.training_mode = "vanilla"
        trainer.is_clip = 0.0
        trainer.get_global_loss = lambda loss: loss.item()

        images = torch.ones(2, 3, 2, 2)
        student_scale_before = student.scale.detach().clone()
        teacher_scale_before = teacher.scale.detach().clone()

        loss, penalty, layer_diff = trainer.single_step_train(images)

        self.assertGreater(loss, 0.0)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(layer_diff, 0.0)
        self.assertFalse(torch.equal(student.scale.detach(), student_scale_before))
        self.assertTrue(torch.equal(teacher.scale.detach(), teacher_scale_before))
        self.assertIsNone(teacher.scale.grad)

    def test_finite_dataloader_repeats_until_total_iterations(self) -> None:
        dataset = Mock()

        class FiniteDataLoader:
            def __init__(self):
                self.dataset = dataset

            def __iter__(self):
                return iter(
                    [
                        {"pixel_values": torch.tensor([[0.0]])},
                        {"pixel_values": torch.tensor([[1.0]])},
                    ]
                )

        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.ddp_model = nn.Linear(1, 1)
        trainer.dataloader = FiniteDataLoader()
        trainer.num_total_iters = 5
        trainer.num_freq = 10
        trainer.gradient = "coupled"
        trainer.training_mode = "vanilla"
        trainer.world_size = 1
        trainer.rank = 0
        trainer.save_interval = 1
        trainer.is_wandb = False
        trainer.timers = {"train": SimpleTimer("train", sync_cuda=False)}
        trainer.layer_info = {
            "avg_loss": [],
            "avg_loss_penalty": [],
            "avg_diff": [],
            "num_images": [],
        }
        trainer.prepare_batch = lambda batch: batch["pixel_values"]

        seen_batches = []

        def single_step(images, gradient):
            seen_batches.append(int(images.item()))
            return float(images.item()), 0.0, 0.0

        trainer.single_step_train = single_step

        with patch("salad.trainer_salad.dist.destroy_process_group"):
            trainer.train()

        self.assertEqual(seen_batches, [0, 1, 0, 1, 0])
        self.assertEqual(dataset.set_epoch.call_args_list, [call(1), call(2)])
        self.assertEqual(len(trainer.layer_info["avg_loss"]), 5)
        self.assertEqual(trainer.layer_info["num_images"], [1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
