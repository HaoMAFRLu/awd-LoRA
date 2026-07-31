"""CPU contract for the vanilla DINO feature-distillation baseline."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from salad.trainer_salad import SALADTrainer
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


if __name__ == "__main__":
    unittest.main()
