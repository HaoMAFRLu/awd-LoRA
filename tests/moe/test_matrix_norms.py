"""Check matrix norms against explicit hard/soft maps and real saved checkpoints."""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from salaad_moe.checkpoint import save_checkpoint
from salaad_moe.config import load_config
from salaad_moe.data import make_synthetic_corpus, TokenCorpus
from salaad_moe.solver import DenseResidualState
from salaad_moe.trainer import Trainer
from scripts.analyze_moe_matrix_norms import analyze_checkpoint, analyze_group, permutation_rows

ROOT = Path(__file__).resolve().parents[2]


class MatrixNormTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_hard_map_both_axes_matches_explicit_matrix(self):
        indices = torch.tensor([[2, 0, 1], [0, 1, 2]])
        permutation = torch.zeros(2, 3, 3)
        permutation.scatter_(1, indices[:, None, :], 1)
        for axis, projection in ((0, "gate"), (1, "down")):
            with self.subTest(projection=projection):
                shared = torch.arange(1, 7).float().reshape(3, 2)
                if axis:
                    shared = shared.T.contiguous()
                expected_mapped = shared @ permutation if axis else permutation.mT @ shared
                residual = torch.ones_like(expected_mapped) * 0.25
                weight = expected_mapped + residual
                state = DenseResidualState(shared, residual, torch.ones_like(weight), indices, axis)
                group, rows, _ = analyze_group(f"layers.0.moe.experts.{projection}", weight, state, 0.01)
                self.assertAlmostEqual(group["consensus_fro"], math.sqrt(91))
                for expert, row in enumerate(rows):
                    self.assertAlmostEqual(row["mapped_consensus_fro"], math.sqrt(91))
                    self.assertAlmostEqual(row["expert_specific_fro"], math.sqrt(6) * 0.25)
                    self.assertAlmostEqual(row["full_expert_fro"], float(weight[expert].double().norm()))
                    self.assertEqual(row["reconstruction_fro"], 0)
                    self.assertAlmostEqual(row["multiplier_fro"], math.sqrt(6) * 0.01)
        rows = permutation_rows(0, indices)
        self.assertEqual([row["is_identity"] for row in rows], [False, True])
        self.assertEqual([row["moved_channels"] for row in rows], [3, 0])
        self.assertAlmostEqual(rows[0]["permutation_fro"], math.sqrt(3))
        self.assertAlmostEqual(rows[0]["identity_distance_fro"], math.sqrt(6))
        self.assertAlmostEqual(rows[0]["relative_identity_distance_fro"], math.sqrt(2))
        self.assertEqual(rows[0]["diagonal_mean"], 0)
        self.assertEqual(rows[1]["identity_distance_fro"], 0)
        self.assertEqual(rows[1]["diagonal_mean"], 1)
        self.assertNotAlmostEqual(rows[0]["permutation_fro"], float(indices[0].float().norm()))
        with self.assertRaisesRegex(ValueError, "bijection"):
            permutation_rows(0, torch.tensor([[0, 0, 2]]))

    def test_soft_contraction_and_zero_norm_ratios(self):
        shared = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        permutation = torch.full((2, 2, 2), 0.5)
        residual = torch.stack([torch.ones(2, 2), torch.zeros(2, 2)])
        state = DenseResidualState(shared, residual, torch.zeros_like(residual), permutation, 0)
        group, rows, _ = analyze_group("layers.0.moe.experts.up", residual, state, 1.0)
        self.assertEqual(group["consensus_fro"], 2)
        self.assertEqual(group["mapped_consensus_fro"]["max"], 0)
        self.assertIsNone(rows[0]["specific_over_mapped_consensus"])
        self.assertIsNone(rows[1]["specific_over_full"])
        self.assertEqual(rows[0]["specific_over_full"], 1)
        json.dumps({"group": group, "rows": rows}, allow_nan=False)
        row = permutation_rows(0, permutation)[0]
        self.assertEqual(row["permutation_fro"], 1)
        self.assertEqual(row["identity_distance_fro"], 1)
        self.assertAlmostEqual(row["relative_identity_distance_fro"], 1 / math.sqrt(2))
        self.assertEqual(row["diagonal_mean"], 0.5)

    def test_hungarian_checkpoint_reports_every_expert_and_deduplicates_permutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(ROOT / "configs/smoke_hungarian.yaml")
            config["salaad"]["state_initialization_step"] = 0
            make_synthetic_corpus(config, root / "tokens", 8)
            trainer = Trainer(config, TokenCorpus(root / "tokens/manifest.json", config), "cpu")
            trainer.train_step()
            checkpoint = save_checkpoint(trainer, root / "checkpoints")
            with redirect_stdout(io.StringIO()):
                report = analyze_checkpoint(checkpoint, root / "norms", expected_step=1)
            count = config["model"]["num_layers"] * config["model"]["num_experts"]
            self.assertEqual(report["per_expert_rows"], count * 3)
            self.assertEqual(report["permutation_summary"]["count"], count)
            self.assertEqual(report["permutation_summary"]["identity_count"], count)
            self.assertTrue((root / "norms/per_expert_norms.csv").is_file())
            for row in report["experts"]:
                self.assertAlmostEqual(row["consensus_fro"], row["mapped_consensus_fro"])
            with self.assertRaises(FileExistsError):
                analyze_checkpoint(checkpoint, root / "norms")
            with self.assertRaisesRegex(ValueError, "evaluated step"):
                analyze_checkpoint(checkpoint, root / "wrong", expected_step=2100)
            self.assertFalse((root / "wrong").exists())


if __name__ == "__main__":
    unittest.main()
