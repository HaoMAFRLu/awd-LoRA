"""Loss contracts for independently managed DINO Teacher and Student models."""

from __future__ import annotations

from typing import NamedTuple

import torch.nn.functional as F
from torch import Tensor

from salaad_vision.models.dino import DinoFeatures


class DinoDistillationLoss(NamedTuple):
    """Scalar losses used by the feature-distillation baseline."""

    total: Tensor
    cls: Tensor
    patches: Tensor


def dino_feature_mse(
    student: DinoFeatures,
    teacher: DinoFeatures,
) -> DinoDistillationLoss:
    """Compute equally weighted MSE over final CLS and patch features."""
    if student.cls.shape != teacher.cls.shape:
        raise ValueError(
            "Student and teacher CLS shapes differ: "
            f"{tuple(student.cls.shape)} != {tuple(teacher.cls.shape)}"
        )
    if student.patches.shape != teacher.patches.shape:
        raise ValueError(
            "Student and teacher patch shapes differ: "
            f"{tuple(student.patches.shape)} != {tuple(teacher.patches.shape)}"
        )

    cls_loss = F.mse_loss(student.cls, teacher.cls)
    patch_loss = F.mse_loss(student.patches, teacher.patches)
    return DinoDistillationLoss(
        total=cls_loss + patch_loss,
        cls=cls_loss,
        patches=patch_loss,
    )
