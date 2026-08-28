"""Downstream vision tasks used by the shared trainer."""

from .classification import ClassificationTask
from .segmentation import SegmentationTask

__all__ = ["ClassificationTask", "SegmentationTask"]
