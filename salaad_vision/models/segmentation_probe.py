"""Lightweight semantic-segmentation head for DINO patch tokens."""

from __future__ import annotations

from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor, nn

from .dino import (
    DINO_VITB8_EMBED_DIM,
    DINO_VITB8_IMAGE_SIZE,
    DINO_VITB8_NUM_PATCHES,
    DINO_VITB8_PATCH_SIZE,
    DinoFeatures,
)

DINO_VITB8_PATCH_GRID_SIZE = DINO_VITB8_IMAGE_SIZE // DINO_VITB8_PATCH_SIZE


class DinoSegmentationHead(nn.Module):
    """Project the 28 x 28 DINO patch grid and bilinearly upsample it."""

    def __init__(
        self,
        num_classes: int,
        output_size: Union[int, Tuple[int, int]] = DINO_VITB8_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError("num_classes must be an integer")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if isinstance(output_size, int) and not isinstance(output_size, bool):
            normalized_output_size = (output_size, output_size)
        elif (
            isinstance(output_size, (tuple, list))
            and len(output_size) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in output_size
            )
        ):
            normalized_output_size = (output_size[0], output_size[1])
        else:
            raise TypeError("output_size must be an integer or a pair of integers")
        if min(normalized_output_size) <= 0:
            raise ValueError("output_size values must be positive")

        self.num_classes = num_classes
        self.output_size = normalized_output_size
        # A 1 x 1 convolution is exactly a shared Linear over patch tokens.
        self.projection = nn.Conv2d(
            DINO_VITB8_EMBED_DIM,
            num_classes,
            kernel_size=1,
        )
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _feature_grid(features: DinoFeatures) -> Tensor:
        cls, patches = features
        if cls.ndim != 2 or cls.shape[1] != DINO_VITB8_EMBED_DIM:
            raise ValueError(
                f"CLS features must have shape [B, 768], got {tuple(cls.shape)}"
            )
        if patches.ndim != 3 or patches.shape[1:] != (
            DINO_VITB8_NUM_PATCHES,
            DINO_VITB8_EMBED_DIM,
        ):
            raise ValueError(
                "patch features must have shape [B, 784, 768], "
                f"got {tuple(patches.shape)}"
            )
        if cls.shape[0] != patches.shape[0]:
            raise ValueError(
                "CLS and patch batch sizes differ: "
                f"{cls.shape[0]} != {patches.shape[0]}"
            )

        return patches.transpose(1, 2).reshape(
            patches.shape[0],
            DINO_VITB8_EMBED_DIM,
            DINO_VITB8_PATCH_GRID_SIZE,
            DINO_VITB8_PATCH_GRID_SIZE,
        )

    def forward(self, features: DinoFeatures) -> Tensor:
        logits = self.projection(self._feature_grid(features))
        return F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
