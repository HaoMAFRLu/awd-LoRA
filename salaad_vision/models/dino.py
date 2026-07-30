"""A small, explicit feature interface around the official DINO ViT-B/8."""

from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Union

import torch
from torch import Tensor, nn

from salaad_vision.vendor.dino.vision_transformer import vit_base

DINO_VITB8_IMAGE_SIZE = 224
DINO_VITB8_PATCH_SIZE = 8
DINO_VITB8_EMBED_DIM = 768
DINO_VITB8_NUM_PATCHES = (DINO_VITB8_IMAGE_SIZE // DINO_VITB8_PATCH_SIZE) ** 2
DINO_VITB8_CHECKPOINT_SHA256 = (
    "575f0efc4838938314afa09897a62ed2b87919928b5edcd133abb907328995eb"
)

PathLikeValue = Union[str, PathLike]


class DinoFeatures(NamedTuple):
    """Final normalized DINO tokens, separated by their roles."""

    cls: Tensor
    patches: Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DinoViTBase8(nn.Module):
    """Official DINO ViT-B/8 backbone exposing CLS and patch tokens.

    This wrapper intentionally contains no task head and no SALAAD logic. It
    defines the feature contract that later distillation code can depend on.
    """

    def __init__(
        self,
        checkpoint_path: Optional[PathLikeValue] = None,
        *,
        expected_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = vit_base(
            patch_size=DINO_VITB8_PATCH_SIZE,
            num_classes=0,
        )
        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path,
                expected_sha256=expected_sha256,
            )

    def load_checkpoint(
        self,
        checkpoint_path: PathLikeValue,
        *,
        expected_sha256: Optional[str] = None,
    ) -> None:
        """Load a backbone-only state dict and require an exact key match."""
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DINO checkpoint does not exist: {path}")

        if expected_sha256 is not None:
            expected = expected_sha256.lower()
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    "DINO checkpoint SHA-256 mismatch: "
                    f"expected {expected}, got {actual} ({path})"
                )

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Expected a backbone state-dict mapping, "
                f"got {type(state_dict).__name__}"
            )
        if not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state_dict.items()
        ):
            raise TypeError("DINO checkpoint must map string keys directly to tensors")

        self.backbone.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if not isinstance(images, Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, 224, 224], got {tuple(images.shape)}")
        if images.shape[1:] != (
            3,
            DINO_VITB8_IMAGE_SIZE,
            DINO_VITB8_IMAGE_SIZE,
        ):
            raise ValueError(f"images must have shape [B, 3, 224, 224], got {tuple(images.shape)}")
        if not torch.is_floating_point(images):
            raise TypeError(f"images must use a floating-point dtype, got {images.dtype}")

    def extract_features(self, images: Tensor) -> DinoFeatures:
        """Return final normalized CLS and per-patch representations."""
        self._validate_images(images)
        tokens = self.backbone.prepare_tokens(images)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)

        features = DinoFeatures(cls=tokens[:, 0], patches=tokens[:, 1:])
        if features.patches.shape[1:] != (
            DINO_VITB8_NUM_PATCHES,
            DINO_VITB8_EMBED_DIM,
        ):
            raise RuntimeError(
                "DINO ViT-B/8 feature contract changed unexpectedly: "
                f"patch shape is {tuple(features.patches.shape)}"
            )
        return features

    def forward(self, images: Tensor) -> DinoFeatures:
        return self.extract_features(images)
