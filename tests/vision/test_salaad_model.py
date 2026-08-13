"""Tests for restoring downstream SALAAD model variants."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from salaad_vision.build import build_model
from salaad_vision.models import (
    SplitQKAttention,
    apply_salaad,
    apply_salaad_qkv_s50,
    split_qk_attention,
)
from salaad_vision.vendor.dino.vision_transformer import Attention

_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(2, 6, bias=False)
        self.proj = nn.Linear(2, 2, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.mlp = _Mlp()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.fill_(7.0)

    def load_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if expected_sha256 is not None:
            raise AssertionError("the synthetic derived checkpoint has no hash")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)


def _layers(variant: str) -> list[str]:
    suffixes = (
        ("attn.qkv",)
        if variant in {"salaad_qkv", "salaad_qkv_s50"}
        else _SUFFIXES
    )
    return [
        f"backbone.blocks.{block}.{suffix}"
        for block in range(12)
        for suffix in suffixes
    ]


def _write_matrices(root: Path, model: nn.Module, names: list[str]) -> None:
    for rank in range(4):
        rank_names = names[rank::4]
        low_rank = {
            name: torch.ones_like(model.get_submodule(name).weight)
            for name in rank_names
        }
        sparse = {
            name: torch.full_like(model.get_submodule(name).weight, 2.0)
            for name in rank_names
        }
        with (root / f"matrix_rank{rank}.pkl").open("wb") as matrix_file:
            pickle.dump({"LL": low_rank, "SS": sparse}, matrix_file)


class SalaadModelTest(unittest.TestCase):
    def test_split_attention_exactly_expands_qk_logits(self) -> None:
        torch.manual_seed(17)
        attention = Attention(
            dim=12,
            num_heads=3,
            qkv_bias=True,
            attention_backend="explicit",
        ).eval()
        sparse = torch.randn_like(attention.qkv.weight) * 0.01
        low_rank = attention.qkv.weight.detach() - sparse
        tokens = torch.randn(2, 7, 12)
        expected_output, expected_attention = attention(
            tokens,
            return_attention=True,
        )
        original_keys = set(attention.state_dict())

        split = SplitQKAttention(attention, low_rank, sparse).eval()
        output, actual_attention, components = split(
            tokens,
            return_attention=True,
            return_logit_components=True,
        )
        _, implicit_attention = split(tokens)

        self.assertEqual(tuple(components), ("LL", "SL", "LS", "SS"))
        self.assertEqual(split.logit_scales, {name: 1.0 for name in components})
        self.assertEqual(set(split.state_dict()), original_keys)
        self.assertIsNotNone(implicit_attention)
        self.assertTrue(
            torch.allclose(actual_attention, expected_attention, atol=1e-6, rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
        )

        qkv = attention.qkv(tokens).reshape(2, 7, 3, 3, 4).permute(2, 0, 3, 1, 4)
        direct_logits = (qkv[0] @ qkv[1].transpose(-2, -1)) * attention.scale
        expanded_logits = sum(components.values())
        self.assertTrue(
            torch.allclose(expanded_logits, direct_logits, atol=1e-6, rtol=1e-5)
        )

    def test_split_attention_scales_paths_independently(self) -> None:
        torch.manual_seed(23)
        attention = Attention(dim=8, num_heads=2, qkv_bias=True).eval()
        sparse = torch.randn_like(attention.qkv.weight) * 0.01
        low_rank = attention.qkv.weight.detach() - sparse
        split = SplitQKAttention(attention, low_rank, sparse).eval()
        tokens = torch.randn(1, 5, 8)
        split.set_logit_scales(SL=0.0, LS=2.0, SS=-1.0)

        _, actual_attention, components = split(
            tokens,
            return_attention=True,
            return_logit_components=True,
        )
        expected_logits = components["LL"] + 2.0 * components["LS"] - components["SS"]
        self.assertTrue(
            torch.allclose(actual_attention, expected_logits.softmax(dim=-1))
        )

        split.reset_logit_scales()
        replaced_components = dict(components)
        q_low, q_sparse, k_low, _ = split.project_qk(tokens)
        projected_zero_sparse = split.project_operand(
            tokens,
            torch.zeros_like(split.q_sparse_weight),
        )
        replaced_components["SL"] = split.build_logit_component(
            projected_zero_sparse,
            k_low,
        )
        _, replaced_attention = split.forward_from_logit_components(
            tokens,
            replaced_components,
            return_attention=True,
        )
        expected_replaced_logits = (
            components["LL"] + components["LS"] + components["SS"]
        )
        self.assertTrue(
            torch.allclose(
                replaced_attention,
                expected_replaced_logits.softmax(dim=-1),
            )
        )

        split.reset_logit_scales()
        self.assertEqual(
            split.logit_scales,
            {"LL": 1.0, "SL": 1.0, "LS": 1.0, "SS": 1.0},
        )

    def test_split_helper_changes_only_requested_one_based_layer(self) -> None:
        model = _Model()
        for block in model.backbone.blocks:
            block.attn = Attention(dim=2, num_heads=1, qkv_bias=False)
        untouched_weight = model.backbone.blocks[5].attn.qkv.weight.detach().clone()
        names = _layers("salaad_qkv")
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_matrices(root, model, names)
            split = split_qk_attention(model, root, layer=7, restore=True)

        self.assertIs(model.backbone.blocks[6].attn, split)
        self.assertIsInstance(model.backbone.blocks[6].attn, SplitQKAttention)
        self.assertTrue(
            torch.equal(
                model.backbone.blocks[6].attn.qkv.weight,
                torch.full_like(model.backbone.blocks[6].attn.qkv.weight, 3.0),
            )
        )
        self.assertTrue(
            all(
                not isinstance(block.attn, SplitQKAttention)
                for index, block in enumerate(model.backbone.blocks)
                if index != 6
            )
        )
        self.assertTrue(
            torch.equal(
                model.backbone.blocks[5].attn.qkv.weight,
                untouched_weight,
            )
        )

    def test_builder_applies_config_selected_qkv_variant(self) -> None:
        source = _Model()
        qkv_names = _layers("salaad_qkv")
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            checkpoint = root / "model.pth"
            torch.save(source.state_dict(), checkpoint)
            _write_matrices(root, source, qkv_names)
            config = {
                "model": {
                    "name": "dino_vitb8",
                    "variant": "salaad_qkv",
                    "checkpoint": str(checkpoint),
                    "checkpoint_kind": "student_model",
                    "matrix_dir": str(root),
                    "freeze": True,
                }
            }

            with patch("salaad_vision.build.DinoViTBase8", return_value=_Model()):
                model = build_model(config)

        self.assertTrue(
            torch.equal(
                model.backbone.blocks[0].attn.qkv.weight,
                torch.full((6, 2), 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                model.backbone.blocks[0].attn.proj.weight,
                torch.full((2, 2), 7.0),
            )
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )

    def test_all_replaces_all_48_linear_weights(self) -> None:
        model = _Model()
        names = _layers("salaad_all")
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_matrices(root, model, names)

            replaced = apply_salaad(model, root, "salaad_all")

        self.assertEqual(replaced, set(names))
        self.assertEqual(len(replaced), 48)
        for name in names:
            self.assertTrue(
                torch.equal(
                    model.get_submodule(name).weight,
                    torch.full_like(model.get_submodule(name).weight, 3.0),
                ),
                name,
            )

    def test_qkv_replaces_12_qkv_weights_and_keeps_other_x(self) -> None:
        model = _Model()
        qkv_names = _layers("salaad_qkv")
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_matrices(root, model, qkv_names)

            replaced = apply_salaad(model, root, "salaad_qkv")

        self.assertEqual(replaced, set(qkv_names))
        self.assertEqual(len(replaced), 12)
        for block in range(12):
            for suffix in _SUFFIXES:
                name = f"backbone.blocks.{block}.{suffix}"
                expected = 3.0 if suffix == "attn.qkv" else 7.0
                self.assertTrue(
                    torch.equal(
                        model.get_submodule(name).weight,
                        torch.full_like(
                            model.get_submodule(name).weight,
                            expected,
                        ),
                    ),
                    name,
                )

    def test_qkv_s50_alpha_preserves_v_and_other_x(self) -> None:
        source = _Model()
        qkv_names = _layers("salaad_qkv_s50")
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_matrices(root, source, qkv_names)
            enhanced = _Model()
            enhanced.load_state_dict(source.state_dict())
            apply_salaad_qkv_s50(
                enhanced,
                root,
                sparse_keep_fraction=0.5,
                selected_energy_fraction=0.5,
                reference_rank=1,
                alpha=1.5,
            )
            baseline = _Model()
            baseline.load_state_dict(source.state_dict())
            apply_salaad_qkv_s50(
                baseline,
                root,
                sparse_keep_fraction=0.5,
                selected_energy_fraction=0.5,
                reference_rank=1,
                alpha=1.0,
            )

        enhanced_q, enhanced_k, enhanced_v = (
            enhanced.backbone.blocks[0].attn.qkv.weight.chunk(3, dim=0)
        )
        baseline_q, baseline_k, baseline_v = (
            baseline.backbone.blocks[0].attn.qkv.weight.chunk(3, dim=0)
        )
        self.assertFalse(torch.equal(enhanced_q, baseline_q))
        self.assertFalse(torch.equal(enhanced_k, baseline_k))
        self.assertTrue(torch.equal(enhanced_v, baseline_v))
        self.assertTrue(torch.equal(enhanced_v, torch.full((2, 2), 3.0)))
        self.assertTrue(
            torch.equal(
                enhanced.backbone.blocks[0].attn.proj.weight,
                torch.full((2, 2), 7.0),
            )
        )

    def test_builder_loads_a_prebuilt_derived_backbone(self) -> None:
        source = _Model()
        with tempfile.TemporaryDirectory() as temporary_root:
            checkpoint = Path(temporary_root) / "backbone.pth"
            torch.save(source.backbone.state_dict(), checkpoint)
            config = {
                "model": {
                    "name": "dino_vitb8",
                    "variant": "derived",
                    "checkpoint": str(checkpoint),
                    "checkpoint_kind": "derived_backbone",
                    "freeze": True,
                }
            }

            with patch("salaad_vision.build.DinoViTBase8", return_value=_Model()):
                restored = build_model(config)

        self.assertTrue(
            all(
                torch.equal(restored.state_dict()[name], value)
                for name, value in source.state_dict().items()
            )
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in restored.parameters())
        )

    def test_missing_target_layer_is_rejected(self) -> None:
        model = _Model()
        names = _layers("salaad_qkv")[:-1]
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_matrices(root, model, names)

            with self.assertRaisesRegex(ValueError, "incomplete"):
                apply_salaad(model, root, "salaad_qkv")


if __name__ == "__main__":
    unittest.main()
