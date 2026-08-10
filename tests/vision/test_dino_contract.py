"""Contract tests for the first DINO ViT-B/8 integration milestone."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from salaad_vision.data.imagenet import build_imagenet_dataloader
from salaad_vision.models.dino import (
    DINO_VITB8_CHECKPOINT_SHA256,
    DINO_VITB8_EMBED_DIM,
    DINO_VITB8_NUM_PATCHES,
    DinoViTBase8,
)
from salaad_vision.vendor.dino.vision_transformer import Attention

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "pretrained"
    / "dino_vitbase8_pretrain.pth"
)
LOCAL_SMOKE_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "salaad_vision"
    / "smoke"
    / "imagenet_val64_parquet"
)
class DinoViTBase8ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = DinoViTBase8()

    def test_architecture_matches_vit_base8(self) -> None:
        backbone = self.model.backbone
        self.assertEqual(len(backbone.blocks), 12)
        self.assertEqual(backbone.embed_dim, DINO_VITB8_EMBED_DIM)
        self.assertEqual(
            tuple(backbone.patch_embed.proj.weight.shape),
            (DINO_VITB8_EMBED_DIM, 3, 8, 8),
        )
        self.assertEqual(
            tuple(backbone.pos_embed.shape),
            (1, DINO_VITB8_NUM_PATCHES + 1, DINO_VITB8_EMBED_DIM),
        )

    def test_rejects_input_outside_fixed_feature_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "224"):
            self.model(torch.zeros(1, 3, 112, 112))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            self.model(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))

    def test_sdpa_matches_explicit_attention_and_preserves_map_output(self) -> None:
        torch.manual_seed(7)
        explicit = Attention(
            dim=32,
            num_heads=4,
            qkv_bias=True,
            attention_backend="explicit",
        ).eval()
        sdpa = Attention(
            dim=32,
            num_heads=4,
            qkv_bias=True,
            attention_backend="sdpa",
        ).eval()
        sdpa.load_state_dict(explicit.state_dict(), strict=True)
        tokens = torch.randn(2, 11, 32)

        explicit_output, explicit_attention = explicit(tokens)
        sdpa_output, materialized_attention = sdpa(tokens)
        _, sdpa_attention = sdpa(tokens, return_attention=True)

        self.assertIsNone(materialized_attention)
        self.assertTrue(
            torch.allclose(sdpa_output, explicit_output, atol=1e-6, rtol=1e-5)
        )
        self.assertTrue(torch.equal(sdpa_attention, explicit_attention))

    def test_invalid_attention_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention_backend"):
            Attention(dim=32, num_heads=4, attention_backend="unknown")

    def test_official_checkpoint_on_local_imagenet_sample(self) -> None:
        checkpoint = DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            self.skipTest(f"local DINO checkpoint is absent: {checkpoint}")
        if not LOCAL_SMOKE_ROOT.is_dir():
            self.skipTest(f"local ImageNet smoke set is absent: {LOCAL_SMOKE_ROOT}")

        loader = build_imagenet_dataloader(
            {
                "batch_size": 1,
                "num_workers": 0,
                "data": {
                    "dataset": "ILSVRC/imagenet-1k",
                    "location": "local_smoke",
                    "root": str(LOCAL_SMOKE_ROOT),
                    "cache_dir": str(LOCAL_SMOKE_ROOT / "cache"),
                    "split": "validation",
                    "streaming": True,
                    "shuffle": False,
                },
            },
            rank=0,
            world_size=1,
        )
        batch = next(iter(loader))

        self.model.load_checkpoint(
            checkpoint,
            expected_sha256=DINO_VITB8_CHECKPOINT_SHA256,
        )
        self.model.eval()
        with torch.inference_mode():
            features = self.model(batch["pixel_values"])

        self.assertEqual(
            tuple(features.cls.shape),
            (1, DINO_VITB8_EMBED_DIM),
        )
        self.assertEqual(
            tuple(features.patches.shape),
            (1, DINO_VITB8_NUM_PATCHES, DINO_VITB8_EMBED_DIM),
        )
        self.assertTrue(torch.isfinite(features.cls).all().item())
        self.assertTrue(torch.isfinite(features.patches).all().item())


if __name__ == "__main__":
    unittest.main()
