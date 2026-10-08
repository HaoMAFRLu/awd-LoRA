"""Exact cosine initialization followed by soft Sinkhorn training and resume."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from salaad_moe.alignment import PROJECTIONS, aligned_mean, initialize_mean_cosine_alignment
from salaad_moe.checkpoint import load_checkpoint, rng_state, save_checkpoint
from salaad_moe.config import COSINE_INITIALIZATION, load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import checkpoint_states, export_checkpoint, load_evaluation_model
from salaad_moe.sinkhorn import log_sinkhorn, validate_soft_state
from salaad_moe.solver import ConsensusManager
from salaad_moe.trainer import Trainer
from scripts.analyze_moe_initial_alignment import descriptors, match_similarity
from test_alignment import assert_nested_equal, groups_from


ROOT = Path(__file__).resolve().parents[2]


def smoke_config():
    config = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
    config["salaad"].update(initialization=COSINE_INITIALIZATION, state_initialization_step=0)
    config["salaad"]["channel_alignment"]["sinkhorn"]["early_stopping"] = False
    return config


class CosineInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.c = smoke_config()

    def test_initial_maps_match_prior_analysis_and_reconstruct_raw_experts(self):
        weights = {"gate": torch.randn(16, 12, 8), "up": torch.randn(16, 12, 8),
                   "down": torch.randn(16, 8, 12) * .25}
        original = copy.deepcopy(weights)
        mean = descriptors({p: w.mean(0, keepdim=True) for p, w in weights.items()})[0]
        indices, _ = match_similarity(descriptors(weights), mean, "cosine")
        self.assertGreater(np.count_nonzero(indices != np.arange(12)), 0)
        manager = ConsensusManager(groups_from(weights), self.c)
        with patch("scipy.optimize.linear_sum_assignment", wraps=linear_sum_assignment) as matching:
            manager.initialize(0)
        self.assertEqual(matching.call_count, len(indices))  # Exactly one pass, including expert zero.
        expected_p = torch.nn.functional.one_hot(torch.from_numpy(indices), 12).float().mT
        for name, state in manager.states.items():
            projection = name.rsplit(".", 1)[-1]
            torch.testing.assert_close(state.permutation, expected_p, rtol=0, atol=0)
            torch.testing.assert_close(
                state.shared, aligned_mean(weights[projection], torch.from_numpy(indices), int(projection == "down")),
                rtol=0, atol=0,
            )
            torch.testing.assert_close(state.reconstruction(), weights[projection])
            self.assertEqual(torch.count_nonzero(state.dual), 0)
            self.assertIsNone(state.alignment_logits)
        assert_nested_equal(self, weights, original)
        restored = ConsensusManager(manager.groups, self.c)
        restored.load_shards([manager.local_state_dict()])
        assert_nested_equal(self, restored.local_state_dict(), manager.local_state_dict())

    def test_tied_matching_keeps_solver_result_and_zero_channels_are_rejected(self):
        weights = {"gate": torch.ones(2, 3, 4), "up": torch.ones(2, 3, 4),
                   "down": torch.ones(2, 4, 3)}
        with patch("scipy.optimize.linear_sum_assignment",
                   return_value=(np.arange(3), np.array([2, 0, 1]))):
            indices, _ = initialize_mean_cosine_alignment(weights)
        torch.testing.assert_close(indices, torch.tensor([[2, 0, 1], [2, 0, 1]]))
        weights = {p: torch.zeros_like(w) for p, w in weights.items()}
        with self.assertRaisesRegex(ValueError, "nonzero"):
            initialize_mean_cosine_alignment(weights)

    def test_fixed_iterations_do_not_stop_early_and_still_enforce_tolerance(self):
        settings = self.c["salaad"]["channel_alignment"]["sinkhorn"]
        for early, calls in ((False, 300), (True, 10)):
            with patch("salaad_moe.sinkhorn.torch.logsumexp", wraps=torch.logsumexp) as balancing:
                p = log_sinkhorn(torch.zeros(2, 3, 3), {**settings, "early_stopping": early}, None)
            self.assertEqual(balancing.call_count, calls)
            torch.testing.assert_close(p, torch.full_like(p, 1 / 3))
        logits = torch.tensor([[[1., 1.], [1e-8, 1.]]]).log()
        with self.assertRaisesRegex(FloatingPointError, "after 150 iterations"):
            log_sinkhorn(logits, settings, None)

    def test_missing_logits_only_allowed_for_exact_initial_permutations(self):
        alignment = self.c["salaad"]["channel_alignment"]
        p = torch.eye(3).roll(1, 0).repeat(2, 1, 1)
        validate_soft_state(p, None, alignment, allow_permutation=True)
        for candidate, allowed in ((p, False), (.8 * p + .2 / 3, True)):
            with self.assertRaisesRegex(ValueError, "Missing Sinkhorn logits"):
                validate_soft_state(candidate, None, alignment, allow_permutation=allowed)
        manager = ConsensusManager(groups_from({
            "gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5), "down": torch.randn(3, 5, 4),
        }), self.c)
        manager.initialize(0)
        invalid = manager.local_state_dict()
        invalid.update(sweeps=1, last_structure_step=2, last_matching_step=2)
        with self.assertRaisesRegex(RuntimeError, "Missing Sinkhorn logits"):
            manager.load_shards([invalid])

    def test_soft_training_and_resume_before_and_after_first_update(self):
        self.c["training"]["task_precision"] = "bfloat16"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            make_synthetic_corpus(self.c, path / "data", 64)
            corpus = TokenCorpus(path / "data/manifest.json", self.c)
            reference = Trainer(self.c, corpus, "cpu")
            checkpoints = {0: save_checkpoint(reference, path / "checkpoints")}
            _, initial = checkpoint_states(checkpoints[0])
            self.assertTrue(all(s.alignment_logits is None for s in initial.values()))
            with patch("scipy.optimize.linear_sum_assignment", side_effect=AssertionError("No training Hungarian")):
                for step in range(1, 9):
                    reference.train_step()
                    if step == 2:
                        checkpoints[step] = save_checkpoint(reference, path / "checkpoints")
            self.assertEqual(reference.manager.sweeps, 4)
            for state in reference.manager.states.values():
                self.assertIsNotNone(state.alignment_logits)
                self.assertTrue(((state.permutation > 0) & (state.permutation < 1)).any())
            expected_rng = rng_state()
            for start, checkpoint in checkpoints.items():
                with patch("scipy.optimize.linear_sum_assignment", side_effect=AssertionError("Restore saved P")):
                    resumed = Trainer(self.c, corpus, "cpu", initialize_auxiliary=False)
                    load_checkpoint(resumed, checkpoint)
                    for _ in range(start, 8):
                        resumed.train_step()
                for expected, actual in (
                    (reference.model.state_dict(), resumed.model.state_dict()),
                    (reference.optimizer.state_dict(), resumed.optimizer.state_dict()),
                    (reference.manager.local_state_dict(), resumed.manager.local_state_dict()),
                    (reference.reader.state_dict(), resumed.reader.state_dict()),
                    (expected_rng, rng_state()),
                ):
                    assert_nested_equal(self, expected, actual)
            # Evaluation constructs models and consumes RNG: do it after the
            # training/resume comparisons, outside both training trajectories.
            export_checkpoint(checkpoints[0], path / "initial.pt")
            reconstructed, _ = load_evaluation_model(checkpoints[0], "reconstructed")
            exported, _ = load_evaluation_model(path / "initial.pt", "exported")
            assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())

    def test_formal_config_preserves_training_settings_and_old_modes(self):
        formal = load_config(ROOT / "configs/ns97m_sinkhorn_cosine_init.yaml")
        baseline = load_config(ROOT / "configs/ns97m_sinkhorn_10k.yaml")
        validate_config(formal, 4)
        expected = copy.deepcopy(baseline)
        expected["experiment"] = formal["experiment"]
        expected["salaad"]["initialization"] = COSINE_INITIALIZATION
        expected["salaad"]["channel_alignment"]["sinkhorn"].update(max_iterations=150, early_stopping=False)
        self.assertEqual(formal, expected)
        for name in ("ns97m_sinkhorn", "ns97m_sinkhorn_10k", "ns97m_sinkhorn_hungarian", "ns97m_hungarian"):
            validate_config(load_config(ROOT / f"configs/{name}.yaml"), 4)
        for field, value in (("early_stopping", 0), ("early_stopping", "false"), ("initial_softening", .1)):
            invalid = copy.deepcopy(formal)
            invalid["salaad"]["channel_alignment"]["sinkhorn"][field] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)
        formal["salaad"]["channel_alignment"]["method"] = "sinkhorn_hungarian"
        with self.assertRaisesRegex(ValueError, "identity/shared-mean"):
            validate_config(formal)


if __name__ == "__main__":
    unittest.main()
