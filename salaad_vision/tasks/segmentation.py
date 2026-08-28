"""Semantic segmentation task for frozen DINO patch features."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from salaad_vision.models import DinoFeatures, DinoSegmentationHead


def _semantic_boundary(values: Tensor, valid: Tensor) -> Tensor:
    """Return pixels touching a valid four-connected semantic transition."""
    boundary = torch.zeros_like(valid)

    vertical_valid = valid[:, :-1, :] & valid[:, 1:, :]
    vertical = vertical_valid & (values[:, :-1, :] != values[:, 1:, :])
    boundary[:, :-1, :] |= vertical
    boundary[:, 1:, :] |= vertical

    horizontal_valid = valid[:, :, :-1] & valid[:, :, 1:]
    horizontal = horizontal_valid & (values[:, :, :-1] != values[:, :, 1:])
    boundary[:, :, :-1] |= horizontal
    boundary[:, :, 1:] |= horizontal
    return boundary


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius == 0:
        return mask
    kernel_size = 2 * radius + 1
    return (
        F.max_pool2d(
            mask[:, None].to(torch.float32),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )[:, 0]
        > 0
    )


class SegmentationTask(nn.Module):
    """Linear patch probe with cross-entropy and additive dense metrics."""

    def __init__(
        self,
        num_classes: int,
        *,
        output_size: Union[int, Tuple[int, int]] = 224,
        ignore_index: int = 255,
        boundary_tolerance: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise TypeError("ignore_index must be an integer")
        if not isinstance(boundary_tolerance, int) or isinstance(
            boundary_tolerance, bool
        ):
            raise TypeError("boundary_tolerance must be an integer")
        if boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_tolerance = boundary_tolerance
        self.head = DinoSegmentationHead(num_classes, output_size)

    def forward(self, features: DinoFeatures) -> Tensor:
        return self.head(features)

    def _valid_pixels(self, labels: Tensor) -> Tensor:
        valid = labels != self.ignore_index
        invalid_class = valid & ((labels < 0) | (labels >= self.num_classes))
        if invalid_class.any().item():
            invalid_values = torch.unique(labels[invalid_class]).tolist()
            raise ValueError(
                "segmentation labels must be in "
                f"[0, {self.num_classes}) or ignore_index={self.ignore_index}; "
                f"got {invalid_values}"
            )
        return valid

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        self._validate(logits, labels)
        valid = self._valid_pixels(labels)
        if not valid.any().item():
            raise ValueError("segmentation batch contains only ignored pixels")
        return F.cross_entropy(
            logits.float(),
            labels,
            ignore_index=self.ignore_index,
        )

    def batch_weight(self, labels: Tensor) -> int:
        self._validate_labels(labels)
        return int(self._valid_pixels(labels).sum().item())

    @torch.no_grad()
    def batch_stats(self, logits: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        self._validate(logits, labels)
        valid = self._valid_pixels(labels)
        predictions = logits.argmax(dim=1)

        encoded = labels[valid].to(torch.int64) * self.num_classes + predictions[
            valid
        ].to(torch.int64)
        confusion = torch.bincount(
            encoded,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

        precision_matches = []
        predicted_boundaries = []
        recall_matches = []
        target_boundaries = []
        for class_index in range(self.num_classes):
            target_class = (labels == class_index) & valid
            predicted_class = (predictions == class_index) & valid
            target_boundary = _semantic_boundary(target_class, valid) & target_class
            predicted_boundary = (
                _semantic_boundary(predicted_class, valid) & predicted_class
            )
            near_target = _dilate(target_boundary, self.boundary_tolerance)
            near_prediction = _dilate(
                predicted_boundary,
                self.boundary_tolerance,
            )
            precision_matches.append((predicted_boundary & near_target).sum())
            predicted_boundaries.append(predicted_boundary.sum())
            recall_matches.append((target_boundary & near_prediction).sum())
            target_boundaries.append(target_boundary.sum())

        return {
            "confusion": confusion.to(torch.float64),
            "boundary_precision_matches": torch.stack(precision_matches).to(
                torch.float64
            ),
            "boundary_predictions": torch.stack(predicted_boundaries).to(torch.float64),
            "boundary_recall_matches": torch.stack(recall_matches).to(torch.float64),
            "boundary_targets": torch.stack(target_boundaries).to(torch.float64),
        }

    @staticmethod
    def summarize(stats: Dict[str, Tensor]) -> Dict[str, float]:
        confusion = stats["confusion"].to(torch.float64)
        if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
            raise ValueError("segmentation confusion matrix must be square")

        target_count = confusion.sum(dim=1)
        predicted_count = confusion.sum(dim=0)
        true_positive = confusion.diagonal()
        total = target_count.sum()
        if total.item() <= 0:
            raise ValueError("segmentation metrics received no valid pixels")

        union = target_count + predicted_count - true_positive
        classes_with_union = union > 0
        classes_with_targets = target_count > 0
        mean_iou = (
            true_positive[classes_with_union] / union[classes_with_union]
        ).mean()
        mean_accuracy = (
            true_positive[classes_with_targets] / target_count[classes_with_targets]
        ).mean()
        pixel_accuracy = true_positive.sum() / total

        predicted_boundaries = stats["boundary_predictions"].to(torch.float64)
        target_boundaries = stats["boundary_targets"].to(torch.float64)
        if (
            predicted_boundaries.ndim != 1
            or target_boundaries.shape != predicted_boundaries.shape
        ):
            raise ValueError("segmentation boundary statistics must be vectors")
        classes_with_boundaries = (predicted_boundaries > 0) | (target_boundaries > 0)
        if classes_with_boundaries.any().item():
            precision = torch.zeros_like(predicted_boundaries)
            recall = torch.zeros_like(target_boundaries)
            has_predictions = predicted_boundaries > 0
            has_targets = target_boundaries > 0
            precision[has_predictions] = (
                stats["boundary_precision_matches"][has_predictions]
                / predicted_boundaries[has_predictions]
            )
            recall[has_targets] = (
                stats["boundary_recall_matches"][has_targets]
                / target_boundaries[has_targets]
            )
            denominator = precision + recall
            class_f1 = torch.zeros_like(denominator)
            nonzero = denominator > 0
            class_f1[nonzero] = (
                2.0 * precision[nonzero] * recall[nonzero] / denominator[nonzero]
            )
            boundary_f1 = class_f1[classes_with_boundaries].mean().item()
        else:
            boundary_f1 = 1.0

        return {
            "miou": 100.0 * mean_iou.item(),
            "pixel_accuracy": 100.0 * pixel_accuracy.item(),
            "mean_accuracy": 100.0 * mean_accuracy.item(),
            "boundary_f1": 100.0 * boundary_f1,
        }

    def _validate_labels(self, labels: Tensor) -> None:
        if labels.ndim != 3:
            raise ValueError(
                f"segmentation labels must have shape [B, H, W], got {tuple(labels.shape)}"
            )
        if labels.dtype != torch.int64:
            raise TypeError(f"segmentation labels must be int64, got {labels.dtype}")

    def _validate(self, logits: Tensor, labels: Tensor) -> None:
        if logits.ndim != 4 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must have shape [B, {self.num_classes}, H, W], "
                f"got {tuple(logits.shape)}"
            )
        self._validate_labels(labels)
        if logits.shape[0] != labels.shape[0] or logits.shape[2:] != labels.shape[1:]:
            raise ValueError(
                "labels must have the same batch and spatial dimensions as logits, "
                f"got logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )
