"""Five-step ADMM equations, singular consensus, and dense-state persistence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from salaad_moe.alignment import PROJECTIONS, native_shared
from salaad_moe.checkpoint import atomic_save, load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.sinkhorn import least_squares_shared, validate_soft_state
from salaad_moe.solver import ConsensusManager, DenseResidualState, aligned_structure_sweep, initial_state, structure_sweep
from salaad_moe.tracking import flatten_metrics
from salaad_moe.trainer import Trainer
from test_alignment import assert_nested_equal, groups_from, permuted_experts


ROOT = Path(__file__).resolve().parents[2]


class DenseResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(93)
        self.c = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
        self.alignment = self.c["salaad"]["channel_alignment"]

    def test_identity_initialization_is_exact_and_does_not_fit_or_change_weights(self):
        weights = {
            "gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5),
            "down": torch.randn(3, 5, 4),
        }
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        before_weights = {g.name: g.weight().clone() for g in manager.groups}
        before_rng = torch.get_rng_state().clone()
        with (
            patch("salaad_moe.sinkhorn.initialize_alignment", side_effect=AssertionError("no hard matching")),
            patch("salaad_moe.sinkhorn.log_sinkhorn", side_effect=AssertionError("no initial Sinkhorn")),
            patch("salaad_moe.sinkhorn.least_squares_shared", side_effect=AssertionError("mean needs no solve")),
            patch("salaad_moe.solver.initial_svd_basis", side_effect=AssertionError("no residual SVD")),
        ):
            manager.initialize(3)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        for group in manager.groups:
            state = manager.states[group.name]
            expected_shared = before_weights[group.name].mean(0)
            torch.testing.assert_close(group.weight(), before_weights[group.name], rtol=0, atol=0)
            torch.testing.assert_close(state.permutation, torch.eye(4).repeat(3, 1, 1), rtol=0, atol=0)
            torch.testing.assert_close(state.shared, expected_shared, rtol=0, atol=0)
            torch.testing.assert_close(state.residual, group.weight() - expected_shared, rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(state.dual), 0)
            self.assertIsNone(state.alignment_logits)
            self.assertNotIn("alignment_logits", state.state_dict())
            validate_soft_state(state.permutation, None, self.alignment, allow_identity=True)
            torch.testing.assert_close(state.anchor(), group.weight(), rtol=0, atol=3e-7)
        before = manager.local_state_dict()
        restored = ConsensusManager(manager.groups, self.c)
        restored.load_shards([before])
        assert_nested_equal(self, before, restored.local_state_dict())
        for problem in ("after_sweep", "nonidentity"):
            bad = copy.deepcopy(before)
            if problem == "after_sweep":
                bad["sweeps"] = 1
            else:
                for value in bad["states"].values():
                    value["permutation"] = value["permutation"].roll(1, -1)
            with self.subTest(problem=problem), self.assertRaisesRegex(RuntimeError, "identity initialization"):
                restored.load_shards([bad])
            assert_nested_equal(self, before, restored.local_state_dict())

    def test_identity_checkpoint_exports_before_any_structure_update(self):
        self.c["salaad"]["state_initialization_step"] = 0
        self.c["training"]["task_precision"] = "bfloat16"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            corpus = TokenCorpus(path / "data/manifest.json", self.c)
            trainer = Trainer(self.c, corpus, "cpu")
            baseline_config = copy.deepcopy(self.c)
            baseline_config["salaad"]["enabled"] = False
            baseline_config["salaad"]["channel_alignment"]["enabled"] = False
            baseline = Trainer(baseline_config, corpus, "cpu")
            assert_nested_equal(self, baseline.model.state_dict(), trainer.model.state_dict())
            checkpoint = save_checkpoint(trainer, path / "checkpoints")
            exported_path = path / "identity.pt"
            report = export_checkpoint(checkpoint, exported_path)
            self.assertEqual(report["format"], "salaad_moe.export.v5")
            reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
            exported, _ = load_evaluation_model(exported_path, "exported")
            assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
            artifact = load_torch(exported_path)
            for name, state in trainer.manager.states.items():
                self.assertIsNone(state.alignment_logits)
                group = artifact["groups"][name]
                self.assertEqual(group["permutation_convention"], "shared_to_native_doubly_stochastic")
                torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
                torch.testing.assert_close(group["permutation"], state.permutation, rtol=0, atol=0)

    def test_full_sweep_matches_independent_least_squares_and_unscaled_dual(self):
        shared, _, _ = permuted_experts()
        # The joint least-squares solution has negative entries to exercise clipping,
        # and a dense first expert to detect an unintended P_0=I constraint.
        candidates = 0.5 + torch.rand(3, 4, 4)
        candidates[1, 0, 2] = -0.2
        targets = {
            p: native_shared(shared[p], candidates, int(p == "down"))
            + 0.001 * torch.randn(3, *shared[p].shape)
            for p in PROJECTIONS
        }
        old, weights = {}, {}
        with patch("salaad_moe.solver.initial_svd_basis", side_effect=AssertionError("dense mode has no SVD state")):
            for p in PROJECTIONS:
                old[p] = initial_state(
                    targets[p], self.c, shared=shared[p],
                    permutation=torch.eye(4).repeat(3, 1, 1), channel_axis=int(p == "down"),
                )
                old[p].residual = torch.randn_like(targets[p]) * 0.1
                old[p].dual = torch.randn_like(targets[p]) * 0.07
                weights[p] = targets[p] + old[p].residual - old[p].dual
        before = copy.deepcopy((weights, {p: old[p].state_dict() for p in PROJECTIONS}))
        design = torch.cat((shared["up"].mT, shared["gate"].mT, shared["down"]), 0).double()
        rhs = torch.cat((targets["up"].mT, targets["gate"].mT, targets["down"]), -2).double()
        unconstrained = torch.linalg.lstsq(design.expand(3, -1, -1), rhs, driver="gelsd").solution
        expected_p = unconstrained.clamp_min(1e-8)
        for _ in range(500):
            expected_p = expected_p / expected_p.sum(-1, keepdim=True)
            expected_p = expected_p / expected_p.sum(-2, keepdim=True)
        with patch("salaad_moe.solver.svt", side_effect=AssertionError("dense mode has no SVT")):
            actual = aligned_structure_sweep(weights, old, self.c, rematch=True)
        p = actual["gate"].permutation
        torch.testing.assert_close(p, expected_p.float(), rtol=0, atol=1e-5)
        self.assertFalse(torch.equal(p[0], torch.eye(4)))
        # Solve the consensus directly from the stacked design, without forming
        # its Gram matrix. This independently checks orientation and averaging.
        consensus_design = p.double().mT.reshape(-1, 4)
        for name in PROJECTIONS:
            target = (weights[name] - old[name].residual + old[name].dual).double()
            target_rows = target.mT if name == "down" else target
            fitted = torch.linalg.lstsq(
                consensus_design, target_rows.reshape(-1, 5), driver="gelsd",
            ).solution
            if name == "down":
                fitted = fitted.mT
            state = actual[name]
            self.assertIsInstance(state, DenseResidualState)
            torch.testing.assert_close(state.shared, fitted.float(), rtol=2e-4, atol=2e-5)
            mapped = native_shared(state.shared.double(), p.double(), int(name == "down"))
            rho = self.c["salaad"]["rho"]
            old_y = rho * old[name].dual.double()
            expected_residual = weights[name].double() - mapped + old_y / rho
            torch.testing.assert_close(state.residual, expected_residual.float(), rtol=2e-5, atol=2e-6)
            expected_y = old_y + rho * (weights[name].double() - mapped - state.residual.double())
            torch.testing.assert_close(rho * state.dual, expected_y.float(), rtol=0, atol=2e-7)
            torch.testing.assert_close(state.dual, torch.zeros_like(state.dual), rtol=0, atol=1e-6)
            validate_soft_state(state.permutation, state.alignment_logits, self.alignment)
        assert_nested_equal(self, before, (weights, {p: old[p].state_dict() for p in PROJECTIONS}))

    def test_finite_components_with_overflowing_anchor_are_rejected(self):
        weights = torch.full((3, 4, 5), 3e38)
        old = DenseResidualState(torch.zeros(4, 5), torch.zeros_like(weights), torch.full_like(weights, 1e38))
        # X_new + X_e_new overflows although each stored component is finite.
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
            structure_sweep(weights, old, self.c, shared=torch.full((4, 5), 2e38))

    def test_sinkhorn_150_iteration_failure_preserves_auxiliary_state(self):
        template = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        weights = {
            "gate": template.repeat(3, 1, 1), "up": (2 * template).repeat(3, 1, 1),
            "down": (3 * template.mT).repeat(3, 1, 1),
        }
        manager = ConsensusManager(groups_from(weights), self.c)
        manager.initialize(3)
        before = manager.local_state_dict()
        anchors = copy.deepcopy(manager.anchors)
        candidate = torch.tensor([[1., 1.], [0., 1.]]).repeat(3, 1, 1)
        # This triangular fitting target remains unbalanced at the real limit;
        # exercise the numerical failure itself, rather than mocking an error.
        for group in manager.groups:
            state = manager.states[group.name]
            group.weight().copy_(native_shared(state.shared, candidate, state.channel_axis))
        with patch("salaad_moe.sinkhorn.torch.logsumexp", wraps=torch.logsumexp) as normalization:
            with self.assertRaisesRegex(RuntimeError, "after 150 iterations:.*tolerance"):
                manager.update(5)
        self.assertEqual(normalization.call_count, 2 * 150)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)

    def test_evaluation_and_export_reject_inconsistent_checkpoint_metadata(self):
        self.c["salaad"]["state_initialization_step"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 32)
            trainer = Trainer(self.c, TokenCorpus(path / "data/manifest.json", self.c), "cpu")
            checkpoint = save_checkpoint(trainer, path / "checkpoints")
            payload = load_torch(checkpoint / "training.pt")
            for problem in ("step", "config_hash", "config"):
                bad = copy.deepcopy(payload)
                if problem == "step":
                    bad["step"] += 1
                elif problem == "config_hash":
                    bad["config_hash"] = "invalid"
                else:
                    bad["config"]["salaad"]["rho"] *= 2
                atomic_save(bad, checkpoint / "training.pt")
                for mode in ("raw", "reconstructed"):
                    with self.subTest(problem=problem, mode=mode), self.assertRaisesRegex(ValueError, "metadata"):
                        load_evaluation_model(checkpoint, mode)
                with self.subTest(problem=problem, mode="export"), self.assertRaisesRegex(ValueError, "metadata"):
                    export_checkpoint(checkpoint, path / (problem + ".pt"))
            atomic_save(payload, checkpoint / "training.pt")

    def test_singular_consensus_uses_minimum_norm_without_fixed_reference(self):
        permutation = torch.full((3, 4, 4), 0.25)
        targets = {"gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5), "down": torch.randn(3, 5, 4)}
        with patch("torch.linalg.cholesky", side_effect=AssertionError("singular consensus")):
            result = least_squares_shared(targets, permutation, allow_singular=True)
        for name in PROJECTIONS:
            rows = targets[name].mT if name == "down" else targets[name]
            expected = rows.mean((0, 1)).expand(4, -1)
            actual = result[name].mT if name == "down" else result[name]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_dense_gradient_is_the_augmented_lagrangian_gradient(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights), self.c)
        manager.initialize(0)
        for state in manager.states.values():
            state.residual.add_(0.02)
            state.dual.fill_(0.03)
        manager.refresh_anchors()
        rho = self.c["salaad"]["rho"]
        expected = {}
        for group in manager.groups:
            w = group.parameter
            state = manager.states[group.name]
            task = w.sin().sum()
            residual = w - state.native_shared() - state.residual
            objective = task + (rho * state.dual * residual).sum() + rho / 2 * residual.square().sum()
            expected[group.name] = torch.autograd.grad(objective, group.parameter, retain_graph=True)[0]
            group.parameter.grad = torch.autograd.grad(task, group.parameter)[0]
        manager.inject_gradients(1.0)
        for group in manager.groups:
            torch.testing.assert_close(group.parameter.grad, expected[group.name])

    def test_dense_checkpoint_validation_and_atomic_failure(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        manager.initialize(3)
        before = manager.local_state_dict()
        anchors = copy.deepcopy(manager.anchors)
        with patch("salaad_moe.solver.structure_sweep", side_effect=FloatingPointError("injected residual failure")):
            with self.assertRaisesRegex(RuntimeError, "no state committed"):
                manager.after_step(5)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)
        for problem in ("mode", "shape", "nan", "mixed", "triplet", "anchor_overflow"):
            bad = copy.deepcopy(before)
            value = bad["states"]["layers.0.moe.experts.gate"]
            if problem == "mode":
                bad["residual_mode"] = "low_rank_sparse"
            elif problem == "shape":
                value["residual"] = value["residual"][:1]
            elif problem == "nan":
                value["residual"][0, 0, 0] = float("nan")
            elif problem == "mixed":
                value["sparse"] = value["residual"].clone()
            elif problem == "anchor_overflow":
                value["shared"].fill_(2e38)
                value["residual"].fill_(2e38)
            else:
                value["permutation"][0] = value["permutation"][0].roll(1, 0)
            with self.subTest(problem=problem), self.assertRaises(RuntimeError):
                manager.load_shards([bad])
            assert_nested_equal(self, before, manager.local_state_dict())
        restored = ConsensusManager(manager.groups, self.c)
        restored.load_shards([before])
        assert_nested_equal(self, before, restored.local_state_dict())

    def test_configs_select_documented_updates_and_reject_mixed_modes(self):
        for name in ("smoke_sinkhorn", "smoke_sinkhorn_dp2", "ns97m_sinkhorn"):
            config = load_config(ROOT / "configs" / (name + ".yaml"))
            validate_config(config)
            self.assertEqual(config["salaad"]["structure_order"], ["permutation", "shared", "residual", "dual"])
            self.assertFalse(config["salaad"]["channel_alignment"]["fix_reference"])
            self.assertEqual(config["salaad"]["initialization"], "identity_shared_mean_residual_dual_zero")
            self.assertNotIn("initial_softening", config["salaad"]["channel_alignment"]["sinkhorn"])
            self.assertEqual(config["salaad"]["channel_alignment"]["sinkhorn"]["max_iterations"], 150)
        for key, value in (("residual_mode", "unknown"), ("low_rank_enabled", True),
                           ("sparse_enabled", True), ("controller", {}), ("initialization", "unknown")):
            bad = copy.deepcopy(self.c)
            bad["salaad"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)
        bad = copy.deepcopy(self.c)
        bad["salaad"]["channel_alignment"]["sinkhorn"]["initial_softening"] = 0.1
        with self.assertRaisesRegex(ValueError, "initial_softening"):
            validate_config(bad)

    def test_training_resume_export_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.c["training"]["task_precision"] = "bfloat16"
            make_synthetic_corpus(self.c, path / "data", 64)
            corpus = TokenCorpus(path / "data/manifest.json", self.c)
            trainer = Trainer(self.c, corpus, "cpu")
            checkpoints = {}
            for step in range(1, 9):
                record = trainer.train_step()
                flatten_metrics(record)
                if step in (2, 3, 5):
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
            for state in trainer.manager.states.values():
                self.assertEqual(set(state.state_dict()), {"shared", "residual", "dual", "permutation", "channel_axis", "alignment_logits"})
                torch.testing.assert_close(state.dual, torch.zeros_like(state.dual), rtol=0, atol=1e-7)
                state.dual.fill_(7)  # Inference must exclude the scaled multiplier.
            checkpoint = save_checkpoint(trainer, path / "checkpoints")
            artifact_path = path / "dense.pt"
            report = export_checkpoint(checkpoint, artifact_path)
            self.assertEqual(report["format"], "salaad_moe.export.v5")
            reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
            exported, _ = load_evaluation_model(artifact_path, "exported")
            assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
            for name, state in trainer.manager.states.items():
                torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
            artifact = load_torch(artifact_path)
            for problem in ("mode", "shape", "mixed", "shared_nan", "shared_inf", "reconstruction_overflow"):
                bad = copy.deepcopy(artifact)
                group = bad["groups"]["layers.0.moe.experts.gate"]
                expert = group["experts"][0]
                if problem == "mode":
                    bad["format"] = "salaad_moe.export.v4"
                elif problem == "shape":
                    expert["residual"] = expert["residual"][:1]
                elif problem == "mixed":
                    expert["sparse"] = {}
                elif problem == "shared_nan":
                    group["shared"][0, 0] = float("nan")
                elif problem == "shared_inf":
                    group["shared"][0, 0] = float("inf")
                else:
                    group["shared"].fill_(2e38)
                    expert["residual"].fill_(2e38)
                bad_path = path / (problem + ".pt")
                atomic_save(bad, bad_path)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    load_evaluation_model(bad_path, "exported")


if __name__ == "__main__":
    unittest.main()
