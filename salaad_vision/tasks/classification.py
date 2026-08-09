"""Image classification task for frozen DINO features."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from salaad_vision.models import DinoFeatures, DinoLinearHead


class ClassificationTask(nn.Module):
    """Linear classification loss and metrics on top of DINO features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = DinoLinearHead(num_classes)

    def forward(self, features: DinoFeatures) -> Tensor:
        return self.head(features)

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        self._validate(logits, labels)
        return F.cross_entropy(logits.float(), labels)

    def batch_weight(self, labels: Tensor) -> int:
        return int(labels.shape[0])

    @torch.no_grad()
    def batch_stats(self, logits: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        self._validate(logits, labels)
        top_k = min(5, self.num_classes)
        predictions = logits.topk(top_k, dim=1).indices
        correct = predictions.eq(labels.unsqueeze(1))
        return {
            "correct1": correct[:, :1].sum().to(torch.float64),
            "correct5": correct.sum().to(torch.float64),
            "samples": torch.tensor(
                labels.shape[0],
                dtype=torch.float64,
                device=labels.device,
            ),
        }

    @staticmethod
    def summarize(stats: Dict[str, Tensor]) -> Dict[str, float]:
        samples = stats["samples"].item()
        if samples <= 0:
            raise ValueError("classification metrics received no samples")
        return {
            "top1": 100.0 * stats["correct1"].item() / samples,
            "top5": 100.0 * stats["correct5"].item() / samples,
        }

    def _validate(self, logits: Tensor, labels: Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must have shape [B, {self.num_classes}], "
                f"got {tuple(logits.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError(
                "labels must have shape [B] with the same batch size as logits, "
                f"got {tuple(labels.shape)}"
            )
        if labels.dtype != torch.int64:
            raise TypeError(f"classification labels must be int64, got {labels.dtype}")
