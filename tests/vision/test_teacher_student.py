"""Contracts for independently constructed DINO Teacher and Student models."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from salad.operators import opt_copy
from salad.register import get_model
from salad.utils import freeze_model, read_cfg
from salaad_vision.distillation.teacher_student import dino_feature_mse
from salaad_vision.models.dino import (
    DINO_VITB8_CHECKPOINT_SHA256,
    DinoFeatures,
    DinoViTBase8,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "vit_b8_model.json"
TRAIN_CONFIG = REPOSITORY_ROOT / "configs" / "vit_b8.yaml"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "pretrained"
    / "dino_vitbase8_pretrain.pth"
)


class DinoTeacherStudentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        checkpoint = DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise unittest.SkipTest(f"local DINO checkpoint is absent: {checkpoint}")

        train_config = read_cfg(TRAIN_CONFIG)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.teacher = get_model(MODEL_CONFIG)
        cls.student = get_model(MODEL_CONFIG)
        cls.teacher.load_checkpoint(
            checkpoint,
            expected_sha256=DINO_VITB8_CHECKPOINT_SHA256,
        )
        if train_config["distillation"]["initialization"] == "teacher_init":
            opt_copy(cls.teacher, cls.student)
        freeze_model(cls.teacher)
        cls.teacher.to(cls.device)
        cls.student.to(cls.device)
        cls.images = torch.zeros(1, 3, 224, 224, device=cls.device)

    def test_get_model_returns_one_bare_backbone(self) -> None:
        model = get_model(MODEL_CONFIG)
        self.assertIsInstance(model, DinoViTBase8)
        self.assertFalse(hasattr(model, "teacher"))
        self.assertFalse(hasattr(model, "student"))

    def test_main_constructs_independent_teacher_and_student(self) -> None:
        self.assertIsInstance(self.teacher, DinoViTBase8)
        self.assertIsInstance(self.student, DinoViTBase8)
        self.assertIsNot(self.teacher, self.student)

        modules = dict(self.student.named_modules())
        expected_suffixes = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
        expected_layers = [
            f"backbone.blocks.{block}.{suffix}"
            for block in range(12)
            for suffix in expected_suffixes
        ]
        self.assertEqual(len(expected_layers), 48)
        for layer_name in expected_layers:
            self.assertIsInstance(modules.get(layer_name), torch.nn.Linear)

    def test_teacher_stays_frozen_in_training_mode(self) -> None:
        self.student.train()
        self.assertFalse(self.teacher.training)
        self.assertTrue(self.student.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.teacher.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in self.student.parameters())
        )

    def test_identical_features_have_zero_feature_loss(self) -> None:
        with torch.no_grad():
            teacher_features = self.teacher(self.images)
            loss = dino_feature_mse(teacher_features, teacher_features)

        self.assertLessEqual(loss.total.item(), 1e-12)
        self.assertLessEqual(loss.cls.item(), 1e-12)
        self.assertLessEqual(loss.patches.item(), 1e-12)
        self.assertFalse(teacher_features.cls.requires_grad)
        self.assertFalse(teacher_features.patches.requires_grad)

    def test_bfloat16_features_use_float32_mse(self) -> None:
        student_cls = torch.ones(
            2,
            4,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        student_patches = torch.ones(
            2,
            3,
            4,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        student = DinoFeatures(student_cls, student_patches)
        teacher = DinoFeatures(
            torch.zeros_like(student_cls),
            torch.zeros_like(student_patches),
        )

        loss = dino_feature_mse(student, teacher)

        self.assertEqual(loss.total.dtype, torch.float32)
        self.assertEqual(loss.cls.dtype, torch.float32)
        self.assertEqual(loss.patches.dtype, torch.float32)
        loss.total.backward()
        self.assertIsNotNone(student_cls.grad)
        self.assertIsNotNone(student_patches.grad)

    def test_perturbed_student_receives_gradient(self) -> None:
        self.student.train()
        student_bias = self.student.backbone.norm.bias
        original_bias = student_bias.detach().clone()
        self.student.zero_grad(set_to_none=True)

        try:
            with torch.no_grad():
                student_bias.add_(1e-3)
                teacher_features = self.teacher(self.images)
            student_features = self.student(self.images)
            loss = dino_feature_mse(student_features, teacher_features)
            self.assertGreater(loss.total.item(), 0.0)
            loss.total.backward()

            self.assertTrue(
                all(parameter.grad is None for parameter in self.teacher.parameters())
            )
            self.assertIsNotNone(student_bias.grad)
            self.assertGreater(student_bias.grad.abs().sum().item(), 0.0)
        finally:
            with torch.no_grad():
                student_bias.copy_(original_bias)
            self.student.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
