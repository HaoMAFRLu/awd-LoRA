"""Feature-distillation building blocks for SALAAD vision experiments."""

from .teacher_student import (
    DinoDistillationLoss,
    dino_feature_mse,
)

__all__ = [
    "DinoDistillationLoss",
    "dino_feature_mse",
]
