"""Tests for the official DINO ViT-B/8 linear classification head."""

from __future__ import annotations

import unittest

import torch

from salaad_vision.models import DinoFeatures, DinoLinearHead


class DinoLinearHeadTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = DinoLinearHead(num_classes=1000)
        features = DinoFeatures(
            cls=torch.randn(2, 768),
            patches=torch.randn(2, 784, 768),
        )

        logits = head(features)

        self.assertEqual(tuple(logits.shape), (2, 1000))

    def test_interleaves_cls_and_mean_patch_features(self) -> None:
        head = DinoLinearHead(num_classes=1)
        with torch.no_grad():
            head.linear.weight.zero_()
            head.linear.bias.zero_()
            head.linear.weight[0, 0] = 2.0
            head.linear.weight[0, 1] = 3.0

        cls = torch.zeros(1, 768)
        cls[0, 0] = 5.0
        patches = torch.zeros(1, 2, 768)
        patches[0, :, 0] = torch.tensor([7.0, 9.0])

        logits = head(DinoFeatures(cls=cls, patches=patches))

        self.assertEqual(logits.item(), 2.0 * 5.0 + 3.0 * 8.0)

    def test_initialization_matches_official_linear_probe(self) -> None:
        torch.manual_seed(0)
        head = DinoLinearHead(num_classes=1000)

        self.assertTrue(torch.count_nonzero(head.linear.bias).item() == 0)
        self.assertAlmostEqual(head.linear.weight.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(head.linear.weight.std().item(), 0.01, places=4)

    def test_rejects_mismatched_batch_sizes(self) -> None:
        head = DinoLinearHead()
        features = DinoFeatures(
            cls=torch.randn(2, 768),
            patches=torch.randn(3, 784, 768),
        )

        with self.assertRaisesRegex(ValueError, "batch sizes differ"):
            head(features)


if __name__ == "__main__":
    unittest.main()
