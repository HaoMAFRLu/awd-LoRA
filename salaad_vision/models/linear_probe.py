"""Official DINO ViT-B/8 linear classification head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dino import DINO_VITB8_EMBED_DIM, DinoFeatures

DINO_LINEAR_INPUT_DIM = 2 * DINO_VITB8_EMBED_DIM


class DinoLinearHead(nn.Module):
    """Classify the final CLS token and mean patch token with one linear layer."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.linear = nn.Linear(DINO_LINEAR_INPUT_DIM, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    @staticmethod
    def _join_features(features: DinoFeatures) -> Tensor:
        cls, patches = features
        if cls.ndim != 2 or cls.shape[1] != DINO_VITB8_EMBED_DIM:
            raise ValueError(
                "CLS features must have shape [B, 768], "
                f"got {tuple(cls.shape)}"
            )
        if patches.ndim != 3 or patches.shape[2] != DINO_VITB8_EMBED_DIM:
            raise ValueError(
                "patch features must have shape [B, N, 768], "
                f"got {tuple(patches.shape)}"
            )
        if cls.shape[0] != patches.shape[0]:
            raise ValueError(
                "CLS and patch batch sizes differ: "
                f"{cls.shape[0]} != {patches.shape[0]}"
            )
        if patches.shape[1] == 0:
            raise ValueError("patch features must contain at least one token")

        # Match DINO's official avgpool_patchtokens layout exactly:
        # CLS_0, mean_0, CLS_1, mean_1, ..., CLS_767, mean_767.
        return torch.stack((cls, patches.mean(dim=1)), dim=-1).flatten(1)

    def forward(self, features: DinoFeatures) -> Tensor:
        return self.linear(self._join_features(features))
