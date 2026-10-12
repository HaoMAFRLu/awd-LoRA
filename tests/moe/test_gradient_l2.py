"""Direct P gradients, dense residual proximal optimality, and persistence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from salaad_moe.alignment import PROJECTIONS, native_shared
from salaad_moe.checkpoint import load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import IDENTITY_INITIALIZATION, load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.sinkhorn import log_sinkhorn, update_soft_alignment, validate_soft_state
from salaad_moe.solver import ConsensusManager, aligned_structure_sweep, initial_state, structure_sweep
from salaad_moe.tracking import flatten_metrics
from salaad_moe.trainer import Trainer
from test_alignment import assert_nested_equal, groups_from


ROOT = Path(__file__).resolve().parents[2]


class GradientL2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(101)
        self.c = load_config(ROOT / "configs/smoke_sinkhorn_gradient_l2.yaml")
        self.alignment = self.c["salaad"]["channel_alignment"]
        self.settings = self.alignment["sinkhorn"]
        self.weights = {
            "gate": 0.1 * torch.randn(3, 4, 5),
            "up": 0.1 * torch.randn(3, 4, 5),
            "down": 0.1 * torch.randn(3, 5, 4),
        }

    def test_initialization_keeps_exact_mean_identity_difference_and_zero_dual(self):
        manager = ConsensusManager(groups_from(self.weights), self.c)
        before = {g.name: g.weight().clone() for g in manager.groups}
        random_state = torch.get_rng_state().clone()
        with (
            patch("salaad_moe.sinkhorn.log_sinkhorn", side_effect=AssertionError("no initial balancing")),
            patch("salaad_moe.solver.structure_sweep", side_effect=AssertionError("no initial shrinkage")),
        ):
            manager.initialize(0)
        torch.testing.assert_close(random_state, torch.get_rng_state(), rtol=0, atol=0)
        for group in manager.groups:
            state = manager.states[group.name]
            torch.testing.assert_close(group.weight(), before[group.name], rtol=0, atol=0)
            torch.testing.assert_close(state.shared, group.weight().mean(0), rtol=0, atol=0)
            torch.testing.assert_close(state.permutation, torch.eye(4).expand(3, -1, -1), rtol=0, atol=0)
            torch.testing.assert_close(state.residual, group.weight() - state.shared, rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(state.dual), 0)
            self.assertIsNone(state.alignment_logits)

    def test_gradient_matches_autograd_for_all_projections_before_soft_balancing(self):
        shared = {p: value.mean(0) for p, value in self.weights.items()}
        previous = torch.eye(4).repeat(3, 1, 1)
        targets = {p: value.roll(1, int(p == "down") + 1) for p, value in self.weights.items()}
        variable = previous.double().requires_grad_()
        objective = sum(
            0.5 * (native_shared(shared[p].double(), variable, int(p == "down")) - targets[p].double()).square().sum()
            for p in PROJECTIONS
        )
        gradient, = torch.autograd.grad(objective, variable)
        rate = self.settings["learning_rate"]
        kernel = (previous.double() - rate * gradient).clamp_min(self.settings["clip_min"])
        expected = kernel.clone()
        for _ in range(10000):
            expected /= expected.sum(-1, keepdim=True)
            expected /= expected.sum(-2, keepdim=True)
        before = copy.deepcopy((shared, targets, previous))
        self.settings["temperature"] = 0.37
        with (
            patch("salaad_moe.sinkhorn.torch.linalg.pinv", side_effect=AssertionError("P needs no pseudoinverse")),
            patch("salaad_moe.sinkhorn.log_sinkhorn", wraps=log_sinkhorn) as balancing,
        ):
            p, logits = update_soft_alignment(targets, shared, self.alignment, previous=previous)
        self.assertEqual(balancing.call_count, 1)
        torch.testing.assert_close((logits / 0.37).exp(), kernel.float(), rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(p, expected.float(), rtol=0, atol=2e-5)
        self.assertTrue((p > 0).all())
        for expert in range(3):
            self.assertGreater((p[expert] - previous[expert]).norm().item(), 1e-4)
        validate_soft_state(p, logits, self.alignment)
        assert_nested_equal(self, before, (shared, targets, previous))

    def test_residual_update_solves_regularized_subproblem_and_dual_is_not_reset(self):
        x = self.weights["gate"]
        old = initial_state(x, self.c)
        old.dual = 0.02 * torch.randn_like(x)
        shared = 0.05 * torch.randn_like(old.shared)
        rho = self.c["salaad"]["rho"]
        for coefficient in (0.0, 1e-4, 0.5):
            self.c["salaad"]["residual_l2_lambda"] = coefficient
            state = structure_sweep(x, old, self.c, shared=shared)
            z = state.residual.double().requires_grad_()
            candidate = x.double() - shared.double() + old.dual.double()
            objective = coefficient / 2 * z.square().sum() + rho / 2 * (candidate - z).square().sum()
            gradient, = torch.autograd.grad(objective, z)
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=3e-9)
            torch.testing.assert_close(rho * state.dual, coefficient * state.residual, rtol=1e-4, atol=3e-9)
            if coefficient:
                self.assertGreater(state.dual.norm().item(), 0)
                self.assertGreater((state.anchor() - x).norm().item(), 0)
            else:
                torch.testing.assert_close(state.residual, x - shared + old.dual, rtol=0, atol=0)

    def test_full_sweep_uses_old_residual_for_P_and_new_P_for_consensus(self):
        manager = ConsensusManager(groups_from(self.weights), self.c)
        manager.initialize(0)
        old = {p: manager.states[f"layers.0.moe.experts.{p}"] for p in PROJECTIONS}
        for state in old.values():
            state.dual = 0.01 * torch.randn_like(state.dual)
        weights = {p: w + 0.02 * torch.randn_like(w) for p, w in self.weights.items()}
        targets = {p: weights[p] - old[p].residual + old[p].dual for p in PROJECTIONS}
        before = copy.deepcopy({p: state.state_dict() for p, state in old.items()})
        with patch("salaad_moe.solver.update_soft_alignment", wraps=update_soft_alignment) as update:
            new = aligned_structure_sweep(weights, old, self.c, True)
        self.assertEqual(update.call_count, 1)
        assert_nested_equal(self, update.call_args.args[0], targets)
        assert_nested_equal(self, update.call_args.kwargs["previous"], old["gate"].permutation)
        p = new["gate"].permutation.double()
        matrix = (p @ p.mT).sum(0)
        rho, coefficient = self.c["salaad"]["rho"], self.c["salaad"]["residual_l2_lambda"]
        for name, state in new.items():
            rows = targets[name].double().mT if name == "down" else targets[name].double()
            solution = state.shared.double().mT if name == "down" else state.shared.double()
            torch.testing.assert_close(matrix @ solution, (p @ rows).sum(0), rtol=2e-5, atol=1e-6)
            expected = rho / (rho + coefficient) * (weights[name] - state.native_shared() + old[name].dual)
            torch.testing.assert_close(state.residual, expected)
            torch.testing.assert_close(rho * state.dual, coefficient * state.residual, rtol=1e-5, atol=2e-8)
            torch.testing.assert_close(state.permutation, new["gate"].permutation, rtol=0, atol=0)
        assert_nested_equal(self, before, {p: state.state_dict() for p, state in old.items()})

    def test_failed_gradient_update_preserves_all_layers_and_anchors(self):
        manager = ConsensusManager(groups_from(self.weights, layers=2), self.c)
        manager.initialize(0)
        before = copy.deepcopy(manager.local_state_dict())
        anchors = copy.deepcopy(manager.anchors)
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("injected gradient failure")
            return update_soft_alignment(*args, **kwargs)
        with patch("salaad_moe.solver.update_soft_alignment", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "no state committed"):
                manager.update(2)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)

    def test_config_coefficients_rules_and_old_configs(self):
        formal = load_config(ROOT / "configs/ns97m_sinkhorn_gradient_l2.yaml")
        validate_config(formal)
        self.assertEqual(formal["salaad"]["rho"], 1e-5)
        self.assertEqual(formal["salaad"]["residual_l2_lambda"], 1e-4)
        self.assertEqual(formal["salaad"]["initialization"], IDENTITY_INITIALIZATION)
        self.assertEqual(formal["salaad"]["state_initialization_step"], 0)
        self.assertEqual(formal["salaad"]["structure_order"], ["permutation", "shared", "residual", "dual"])
        validate_config(self.c)
        validate_config(load_config(ROOT / "configs/smoke_sinkhorn_gradient_l2_dp2.yaml"))
        for field in ("residual_l2_lambda", "learning_rate"):
            for value in (-1, True, "0.1", float("nan"), float("inf")):
                bad = copy.deepcopy(self.c)
                settings = bad["salaad"] if field == "residual_l2_lambda" else bad["salaad"]["channel_alignment"]["sinkhorn"]
                settings[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_config(bad)
        for name in ("ns97m_sinkhorn", "ns97m_sinkhorn_10k", "ns97m_sinkhorn_cosine_init", "ns97m_hungarian"):
            original = load_config(ROOT / "configs" / (name + ".yaml"))
            validate_config(original)
            self.assertNotIn("residual_l2_lambda", original["salaad"])
        bad = load_config(ROOT / "configs/ns97m.yaml")
        bad["salaad"]["residual_l2_lambda"] = 1e-4
        with self.assertRaisesRegex(ValueError, "dense residuals"):
            validate_config(bad)

    def test_gradient_requires_previous_P_and_rejects_nonfinite_targets(self):
        shared = {p: w.mean(0) for p, w in self.weights.items()}
        with self.assertRaisesRegex(ValueError, "previous"):
            update_soft_alignment(self.weights, shared, self.alignment)
        self.weights["gate"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
            update_soft_alignment(self.weights, shared, self.alignment, previous=torch.eye(4).repeat(3, 1, 1))

    def test_bfloat16_training_bitwise_resume_export_and_lambda_logging(self):
        self.c["training"]["task_precision"] = "bfloat16"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            corpus = TokenCorpus(path / "data/manifest.json", self.c)
            trainer = Trainer(self.c, corpus, "cpu")
            checkpoints = {0: save_checkpoint(trainer, path / "checkpoints")}
            for step in range(1, 9):
                record = trainer.train_step()
                if step in (2, 4):
                    checkpoints[step] = save_checkpoint(trainer, path / "checkpoints")
            expected_rng = rng_state()
            for start, checkpoint in checkpoints.items():
                resumed = Trainer(self.c, corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(resumed, checkpoint)
                for _ in range(start, 8):
                    resumed.train_step()
                for first, second in (
                    (trainer.model.state_dict(), resumed.model.state_dict()),
                    (trainer.optimizer.state_dict(), resumed.optimizer.state_dict()),
                    (trainer.manager.local_state_dict(), resumed.manager.local_state_dict()),
                    (trainer.reader.state_dict(), resumed.reader.state_dict()),
                    (expected_rng, rng_state()),
                ):
                    assert_nested_equal(self, first, second)
            self.assertGreater(record["constraint_gradient_norm"], 0)
            flat = flatten_metrics(record)
            self.assertEqual(flat["salaad_hyperparameters/residual_l2_lambda"], 0.5)
            checkpoint = save_checkpoint(trainer, path / "checkpoints")
            artifact_path = path / "gradient_l2.pt"
            export_checkpoint(checkpoint, artifact_path)
            reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
            exported, _ = load_evaluation_model(artifact_path, "exported")
            assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
            self.assertEqual(load_torch(artifact_path)["config"]["salaad"]["residual_l2_lambda"], 0.5)
            for name, state in trainer.manager.states.items():
                self.assertGreater(state.dual.norm().item(), 0)
                torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
                self.assertGreater((exported.state_dict()[name] - trainer.model.state_dict()[name]).norm().item(), 0)
            bad = copy.deepcopy(self.c)
            bad["salaad"]["residual_l2_lambda"] *= 2
            wrong = Trainer(bad, corpus, "cpu", initialize_auxiliary=False)
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                load_checkpoint(wrong, checkpoint)


if __name__ == "__main__":
    unittest.main()
