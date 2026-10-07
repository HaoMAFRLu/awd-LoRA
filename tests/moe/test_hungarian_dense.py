"""Consensus-first hard matching, free expert residuals, and persistence."""
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from salaad_moe.alignment import PROJECTIONS, initialize_alignment, match_channels, native_shared, validate_permutation
from salaad_moe.checkpoint import atomic_save, load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import fingerprint, load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.solver import ConsensusManager, DenseResidualState, aligned_structure_sweep, initial_state
from salaad_moe.trainer import Trainer
from test_alignment import assert_nested_equal, groups_from


ROOT = Path(__file__).resolve().parents[2]


class HungarianDenseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(127)
        self.c = load_config(ROOT / "configs/smoke_hungarian.yaml")

    def test_initialization_preserves_weights_rng_and_exact_mean(self):
        weights = {
            "gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5),
            "down": torch.randn(3, 5, 4),
        }
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        before_rng = torch.get_rng_state().clone()
        with (
            patch("salaad_moe.alignment.match_channels", side_effect=AssertionError("no initial matching")),
            patch("salaad_moe.solver.initialize_soft_alignment", side_effect=AssertionError("no soft P")),
            patch("salaad_moe.solver.initial_svd_basis", side_effect=AssertionError("no residual SVD")),
        ):
            manager.initialize(3)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        for group in manager.groups:
            state = manager.states[group.name]
            value = weights[group.name.rsplit(".", 1)[1]]
            self.assertIsInstance(state, DenseResidualState)
            torch.testing.assert_close(group.weight(), value, rtol=0, atol=0)
            torch.testing.assert_close(state.shared, value.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(state.residual, value - value.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(state.permutation, torch.arange(4).expand(3, -1), rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(state.dual), 0)
            self.assertIsNone(state.alignment_logits)
            torch.testing.assert_close(state.anchor(), value, rtol=0, atol=3e-7)
        for triplet in manager.layers.values():
            self.assertIs(manager.states[triplet["gate"].name].permutation,
                          manager.states[triplet["down"].name].permutation)

    def test_sweep_uses_old_p_then_new_shared_and_matches_exhaustive_objective(self):
        template = {
            "gate": torch.tensor([[0., 1.], [10., 2.], [20., 3.]]),
            "up": torch.tensor([[-1., 0.], [2., 15.], [3., 30.]]),
            "down": torch.tensor([[2., -2., 4.], [0., 6., 12.]]),
        }
        old_p = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        target_p = torch.tensor([[1, 0, 2], [1, 2, 0], [2, 0, 1]])
        targets = {p: native_shared(template[p], target_p, int(p == "down")) for p in PROJECTIONS}
        old, weights = {}, {}
        for p in PROJECTIONS:
            # Deliberately different old X makes using X(old) for matching detectable.
            shared = template[p].flip(int(p == "down"))
            state = initial_state(targets[p], self.c, shared=shared,
                                  permutation=old_p, channel_axis=int(p == "down"))
            state.residual = torch.arange(targets[p].numel()).reshape_as(targets[p]).float() / 8
            state.dual = torch.arange(targets[p].numel()).reshape_as(targets[p]).float() / 16
            old[p] = state
            weights[p] = targets[p] + state.residual - state.dual
        before = {p: state.state_dict() for p, state in old.items()}
        before_weights = copy.deepcopy(weights)
        # Independent dense-matrix consensus formula, not aligned_mean/index gathers.
        old_dense = F.one_hot(old_p, 3).float().mT
        expected_shared = {
            p: ((targets[p] @ old_dense.mT) if p == "down" else
                (old_dense @ targets[p])).mean(0)
            for p in PROJECTIONS
        }
        candidates = list(itertools.permutations(range(3)))
        expected_p = []
        for expert in range(3):
            errors = []
            for candidate in candidates:
                matrix = F.one_hot(torch.tensor(candidate), 3).double().mT
                error = 0.0
                for p in PROJECTIONS:
                    shared = expected_shared[p].double()
                    mapped = shared @ matrix if p == "down" else matrix.T @ shared
                    error += (targets[p][expert].double() - mapped).square().sum().item()
                errors.append(error)
            expected_p.append(candidates[min(range(len(errors)), key=errors.__getitem__)])
        expected_p = torch.tensor(expected_p)
        torch.testing.assert_close(expected_p, target_p, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected_p[0], old_p[0]))
        old_shared_p = match_channels(targets, {p: old[p].shared for p in PROJECTIONS},
                                      old_p, self.c["salaad"]["channel_alignment"])
        self.assertFalse(torch.equal(expected_p, old_shared_p))
        with (
            patch("salaad_moe.solver.update_soft_alignment", side_effect=AssertionError("no Sinkhorn")),
            patch("salaad_moe.solver.least_squares_shared", side_effect=AssertionError("no pseudoinverse")),
        ):
            actual = aligned_structure_sweep(weights, old, self.c, rematch=True)
        new_dense = F.one_hot(expected_p, 3).float().mT
        rho = self.c["salaad"]["rho"]
        for p in PROJECTIONS:
            state = actual[p]
            torch.testing.assert_close(state.permutation, expected_p, rtol=0, atol=0)
            torch.testing.assert_close(state.shared, expected_shared[p], rtol=0, atol=0)
            mapped = expected_shared[p] @ new_dense if p == "down" else new_dense.mT @ expected_shared[p]
            expected_residual = weights[p] - mapped + old[p].dual
            torch.testing.assert_close(state.residual, expected_residual, rtol=0, atol=0)
            expected_y = rho * old[p].dual + rho * (weights[p] - mapped - expected_residual)
            torch.testing.assert_close(rho * state.dual, expected_y, rtol=0, atol=1e-7)
            torch.testing.assert_close(state.native_shared().flatten(1).norm(dim=1),
                                       state.shared.norm().expand(3), rtol=1e-6, atol=0)
            averaged_again = ((targets[p] @ new_dense.mT) if p == "down" else
                              (new_dense @ targets[p])).mean(0)
            self.assertFalse(torch.allclose(state.shared, averaged_again))
            self.assertIs(state.permutation, actual["gate"].permutation)
        assert_nested_equal(self, before, {p: state.state_dict() for p, state in old.items()})
        assert_nested_equal(self, before_weights, weights)

    def test_equal_cost_preserves_current_permutations(self):
        targets = {"gate": torch.zeros(3, 4, 5), "up": torch.zeros(3, 4, 5),
                   "down": torch.zeros(3, 5, 4)}
        old_p = torch.tensor([[3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]])
        actual = match_channels(targets, {p: t[0] for p, t in targets.items()},
                                old_p, self.c["salaad"]["channel_alignment"])
        torch.testing.assert_close(actual, old_p, rtol=0, atol=0)

    def test_legacy_reference_initialization_is_preserved_when_later_p_is_free(self):
        # A small reference expert can drift after several matching/mean rounds
        # if the new free-reference update policy leaks into legacy initialization.
        torch.manual_seed(167)
        weights = {
            p: torch.randn(8, 2, 8) if p == "down" else torch.randn(8, 8, 2)
            for p in PROJECTIONS
        }
        for value in weights.values():
            value[0] *= 0.01
        settings = load_config(ROOT / "configs/smoke_aligned.yaml")["salaad"]["channel_alignment"]
        expected = initialize_alignment(weights, settings)
        actual = initialize_alignment(weights, {**settings, "fix_reference": False})
        assert_nested_equal(self, actual, expected)
        torch.testing.assert_close(actual[0][0], torch.arange(8), rtol=0, atol=0)

    def test_every_sweep_matches_including_final_flush_and_failure_is_atomic(self):
        weights = {"gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5),
                   "down": torch.randn(3, 5, 4)}
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        manager.initialize(3)
        before, anchors = manager.local_state_dict(), copy.deepcopy(manager.anchors)
        with patch("salaad_moe.solver.match_channels", side_effect=RuntimeError("injected assignment failure")):
            with self.assertRaisesRegex(RuntimeError, "no state committed"):
                manager.after_step(5)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)
        with patch("salaad_moe.solver.match_channels", wraps=match_channels) as match:
            for step, due in ((4, False), (5, True), (6, False), (7, True), (8, True)):
                self.assertEqual(manager.after_step(step, final=step == 8), due)
                if due:
                    self.assertEqual(manager.last_matching_step, step)
            self.assertFalse(manager.after_step(8, final=True))
            self.assertEqual(match.call_count, 6)  # Two layers, three sweeps.
        self.assertEqual(manager.sweeps, 3)

    def test_configs_are_opt_in_and_preserve_training_hyperparameters(self):
        for name in ("smoke_hungarian", "smoke_hungarian_dp2", "ns97m_hungarian"):
            config = load_config(ROOT / "configs" / (name + ".yaml"))
            before = fingerprint(config)
            validate_config(config)
            self.assertEqual(fingerprint(config), before)
        hard = load_config(ROOT / "configs/ns97m_hungarian.yaml")
        soft = load_config(ROOT / "configs/ns97m_sinkhorn.yaml")
        for key in ("model", "data", "training", "parallel"):
            self.assertEqual(hard[key], soft[key])
        for key in ("rho", "state_initialization_step", "initialization", "guidance_period_optimizer_steps"):
            self.assertEqual(hard["salaad"][key], soft["salaad"][key])
        for key, value in (
            ("structure_order", ["permutation", "shared", "residual", "dual"]),
            ("initialization", "aligned_shared_mean_L_zero_S_residual_dual_zero"),
            ("low_rank_enabled", True), ("sparse_enabled", True), ("controller", {}),
        ):
            bad = copy.deepcopy(self.c)
            bad["salaad"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)
        for key, value in (("fix_reference", True), ("enabled", False),
                           ("match_every_optimizer_steps", 0), ("match_every_optimizer_steps", 4),
                           ("method", "sinkhorn")):
            bad = copy.deepcopy(self.c)
            bad["salaad"]["channel_alignment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)
        for name in ("ns97m", "ns97m_aligned", "ns97m_aligned_fixed", "ns97m_sinkhorn", "ns97m_vanilla"):
            legacy = load_config(ROOT / "configs" / (name + ".yaml"))
            before = fingerprint(legacy)
            validate_config(legacy)
            self.assertEqual(fingerprint(legacy), before)

    def test_training_resume_from_prefix_initialization_and_sweep(self):
        self.c["training"]["task_precision"] = "bfloat16"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            corpus = TokenCorpus(path / "data/manifest.json", self.c)
            trainer = Trainer(self.c, corpus, "cpu")
            checkpoints = {}
            for step in range(1, 9):
                trainer.train_step()
                if step in (2, 3, 5, 7):
                    checkpoints[step] = save_checkpoint(trainer, path / "checkpoints")
            expected_rng = rng_state()
            for start, checkpoint in checkpoints.items():
                resumed = Trainer(self.c, corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(resumed, checkpoint)
                for _ in range(start, 8):
                    resumed.train_step()
                assert_nested_equal(self, trainer.model.state_dict(), resumed.model.state_dict())
                assert_nested_equal(self, trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
                assert_nested_equal(self, trainer.manager.local_state_dict(), resumed.manager.local_state_dict())
                assert_nested_equal(self, trainer.reader.state_dict(), resumed.reader.state_dict())
                assert_nested_equal(self, expected_rng, rng_state())
                self.assertEqual(resumed.manager.last_matching_step, 8)
            wrong_config = copy.deepcopy(self.c)
            wrong_config["salaad"] = load_config(ROOT / "configs/smoke_sinkhorn.yaml")["salaad"]
            wrong_trainer = Trainer(wrong_config, corpus, "cpu", initialize_auxiliary=False)
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                load_checkpoint(wrong_trainer, checkpoints[5])

    def test_checkpoint_export_handles_identity_and_nonidentity_first_expert(self):
        self.c["salaad"]["state_initialization_step"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            trainer = Trainer(self.c, TokenCorpus(path / "data/manifest.json", self.c), "cpu")
            for stage in ("identity", "rematched"):
                if stage == "rematched":
                    trainer.train_step()
                    # Force real nonidentity matches for expert 0 in each layer;
                    # most experts retain the old channels, fixing the new mean.
                    with torch.no_grad():
                        for group in trainer.manager.groups:
                            state = trainer.manager.states[group.name]
                            target_p = state.permutation.clone()
                            target_p[0] = target_p[0].roll(1)
                            target = native_shared(state.shared, target_p, state.channel_axis)
                            group.parameter.copy_(target + state.residual - state.dual)
                    trainer.train_step()
                checkpoint = save_checkpoint(trainer, path / stage / "checkpoints")
                restored = Trainer(self.c, trainer.corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(restored, checkpoint)
                assert_nested_equal(self, trainer.manager.local_state_dict(), restored.manager.local_state_dict())
                artifact_path = path / (stage + ".pt")
                report = export_checkpoint(checkpoint, artifact_path)
                self.assertEqual(report["format"], "salaad_moe.export.v5")
                reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
                exported, _ = load_evaluation_model(artifact_path, "exported")
                assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
                artifact = load_torch(artifact_path)
                for name, state in trainer.manager.states.items():
                    self.assertEqual(set(state.state_dict()), {"shared", "residual", "dual", "permutation", "channel_axis"})
                    validate_permutation(state.permutation, 8, 24)
                    if stage == "rematched":
                        self.assertFalse(torch.equal(state.permutation[0], torch.arange(24)))
                    self.assertEqual(artifact["groups"][name]["permutation_convention"], "native_to_shared")
                    torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
            bad = copy.deepcopy(artifact)
            bad["groups"]["layers.0.moe.experts.up"]["permutation"][0].zero_()
            atomic_save(bad, path / "bad.pt")
            with self.assertRaises(ValueError):
                load_evaluation_model(path / "bad.pt", "exported")


if __name__ == "__main__":
    unittest.main()
