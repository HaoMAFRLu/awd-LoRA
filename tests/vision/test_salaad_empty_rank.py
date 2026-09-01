"""Regression tests for SALAAD runs with fewer layers than DDP ranks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class SalaadEmptyRankTest(unittest.TestCase):
    @staticmethod
    def make_empty_rank_trainer() -> SALADTrainer:
        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.rank = 3
        trainer.world_size = 4
        trainer.device = torch.device("cpu")
        trainer.name2idx = {
            "blocks.0.attn.qkv": 0,
            "blocks.1.attn.qkv": 1,
            "blocks.2.attn.qkv": 2,
        }
        trainer.ADMM_solvers = [
            SimpleNamespace(layer_gpu_map=-1) for _ in trainer.name2idx
        ]
        return trainer

    def test_round_robin_assignment_allows_an_empty_rank(self) -> None:
        layers = [
            {"name": f"blocks.{index}.attn.qkv"}
            for index in range(3)
        ]

        assigned, owner_map = SALADTrainer.assign_layers(layers, rank=3, world_size=4)

        self.assertEqual(assigned, [])
        self.assertEqual(
            owner_map,
            {
                "blocks.0.attn.qkv": 0,
                "blocks.1.attn.qkv": 1,
                "blocks.2.attn.qkv": 2,
            },
        )

    def test_empty_rank_returns_collective_safe_tensors(self) -> None:
        trainer = self.make_empty_rank_trainer()

        diff = trainer.get_diff_per_rank()
        penalty = trainer.get_penalty_loss()
        layer_results = trainer.get_local_results()

        self.assertEqual(diff.shape, torch.Size([]))
        self.assertEqual(penalty.shape, torch.Size([]))
        self.assertEqual(layer_results.shape, (3, 12))
        self.assertEqual(diff.dtype, torch.float32)
        self.assertEqual(penalty.dtype, torch.float32)
        self.assertEqual(layer_results.dtype, torch.float32)
        self.assertEqual(diff.item(), 0.0)
        self.assertEqual(penalty.item(), 0.0)
        self.assertEqual(torch.count_nonzero(layer_results).item(), 0)

    def test_empty_rank_can_enter_layer_statistics_all_reduce(self) -> None:
        trainer = self.make_empty_rank_trainer()

        with patch("salad.trainer_salad.dist.all_reduce") as all_reduce:
            trainer.sync_layer_info()

        reduced = all_reduce.call_args.args[0]
        self.assertEqual(reduced.shape, (3, 12))
        self.assertEqual(torch.count_nonzero(reduced).item(), 0)

    def test_empty_rank_can_complete_a_coupled_training_step(self) -> None:
        trainer = self.make_empty_rank_trainer()
        teacher = _TinyFeatureModel(scale=2.0)
        teacher.requires_grad_(False)
        teacher.eval()
        student = _TinyFeatureModel(scale=1.0)
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
        trainer.cfg_layers = list(trainer.name2idx)
        trainer.training_mode = "salad"
        trainer.is_clip = 0.0
        trainer.use_bfloat16_autocast = False
        images = torch.ones(2, 3, 2, 2)

        with patch("salad.trainer_salad.dist.all_reduce") as all_reduce:
            losses = trainer.single_step_train(images, gradient="coupled")

        self.assertEqual(len(losses), 5)
        self.assertEqual(losses[-1], 0.0)
        self.assertNotEqual(student.scale.item(), 1.0)
        self.assertEqual(all_reduce.call_count, 3)
        self.assertTrue(
            all(
                isinstance(reduce_call.args[0], torch.Tensor)
                for reduce_call in all_reduce.call_args_list
            )
        )

    def test_parameter_broadcast_skips_zero_sized_owner(self) -> None:
        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.rank = 0
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer.per_owner_names = {0: ["0"], 1: []}
        trainer.owner_sizes = {0: 4, 1: 0}
        model = nn.Sequential(nn.Linear(2, 2, bias=False))

        with patch("salad.trainer_salad.dist.broadcast") as broadcast:
            trainer.broadcast_params(model)

        broadcast.assert_called_once()
        buffer = broadcast.call_args.args[0]
        self.assertEqual(buffer.numel(), 4)
        self.assertEqual(broadcast.call_args.kwargs["src"], 0)


if __name__ == "__main__":
    unittest.main()
