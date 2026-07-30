"""Vision experiments for applying SALAAD to pretrained backbones."""

from .distillation.teacher_student import (
    DinoDistillationLoss,
    dino_feature_mse,
)
from .models.dino import (
    DINO_VITB8_CHECKPOINT_SHA256,
    DINO_VITB8_EMBED_DIM,
    DINO_VITB8_IMAGE_SIZE,
    DINO_VITB8_NUM_PATCHES,
    DINO_VITB8_PATCH_SIZE,
    DinoFeatures,
    DinoViTBase8,
)

__all__ = [
    "DINO_VITB8_CHECKPOINT_SHA256",
    "DINO_VITB8_EMBED_DIM",
    "DINO_VITB8_IMAGE_SIZE",
    "DINO_VITB8_NUM_PATCHES",
    "DINO_VITB8_PATCH_SIZE",
    "DinoDistillationLoss",
    "DinoFeatures",
    "DinoViTBase8",
    "dino_feature_mse",
]
