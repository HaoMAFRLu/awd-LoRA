"""Check joint channel descriptors, assignment direction, and initialization provenance."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from salaad_moe.config import load_config
from scripts.analyze_moe_initial_alignment import analyze, descriptors, match_similarity


class InitialAlignmentAnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_joint_triplet_recovers_non_self_inverse_channel_permutation(self):
        base = torch.tensor([[1., 0., 0., 1.], [0., 2., 0., 1.], [0., 0., 3., 1.]])
        order = torch.tensor([2, 0, 1])
        gate = torch.stack((base, base[order]))
        up = gate * .75
        down = (gate * .25).mT
        value = descriptors({"gate": gate, "up": up, "down": down})
        torch.testing.assert_close(value[0, :, 8:], down[0].T.double())
        for metric in ("cosine", "squared_l2"):
            with self.subTest(metric=metric):
                indices, rows = match_similarity(value, value[0], metric)
                np.testing.assert_array_equal(indices, [[0, 1, 2], [2, 0, 1]])
                self.assertTrue(rows[0]["is_identity"])
                self.assertEqual(rows[1]["moved_channels"], 3)

    def test_assignment_enforces_one_to_one_matching(self):
        value = torch.tensor([[[.9, .8, 0.], [.8, .1, 0.], [0., 0., 1.]]])
        for metric in ("cosine", "squared_l2"):
            indices, _ = match_similarity(value, torch.eye(3), metric)
            np.testing.assert_array_equal(indices, [[1, 0, 2]])

    def test_tied_solver_result_is_used_without_identity_fallback(self):
        value = torch.ones(1, 3, 4)
        with patch("scripts.analyze_moe_initial_alignment.linear_sum_assignment",
                   return_value=(np.arange(3), np.array([2, 0, 1]))):
            indices, rows = match_similarity(value, value[0], "cosine")
        np.testing.assert_array_equal(indices, [[2, 0, 1]])
        self.assertFalse(rows[0]["is_identity"])

    def test_step_zero_analysis_saves_all_cases_without_training_or_wandb(self):
        config = load_config("configs/smoke_sinkhorn.yaml")
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            with patch("salaad_moe.trainer.Trainer", side_effect=AssertionError("No training")), \
                 patch("salaad_moe.tracking.WandbTracker", side_effect=AssertionError("No W&B")):
                report = analyze(config, temporary)
            count = config["model"]["num_layers"] * config["model"]["num_experts"]
            self.assertEqual(report["step"], 0)
            self.assertEqual(len(report["cases"]), 4)
            self.assertTrue(report["expert_weights_unchanged"])
            arrays = np.load(Path(temporary) / "initial_permutations.npz")
            for case, result in report["cases"].items():
                self.assertEqual(result["summary"]["matrix_count"], count)
                hard = arrays[f"{case}_P_e"]
                self.assertTrue(np.isin(hard, [0, 1]).all())
                self.assertTrue((hard.sum(-1) == 1).all() and (hard.sum(-2) == 1).all())
                np.testing.assert_array_equal(hard.argmax(-2), arrays[f"{case}_native_to_shared"])


if __name__ == "__main__":
    unittest.main()
