"""Numerical, persistence and training contracts for soft channel alignment."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from salaad_moe.alignment import PROJECTIONS, aligned_mean, native_shared
from salaad_moe.checkpoint import atomic_save, load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.sinkhorn import (
    initialize_soft_alignment, least_squares_shared, log_sinkhorn,
    update_soft_alignment, validate_transport,
)
from salaad_moe.solver import ConsensusManager
from salaad_moe.trainer import Trainer
from test_alignment import assert_nested_equal, dense_permutations, groups_from, permuted_experts

ROOT = Path(__file__).resolve().parents[2]


class SinkhornNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.c = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
        self.alignment = self.c["salaad"]["channel_alignment"]
        self.settings = self.alignment["sinkhorn"]

    def test_balancing_is_differentiable_and_reference_is_fixed(self):
        logits = (torch.randn(3, 4, 4) * 0.2).requires_grad_()
        p = log_sinkhorn(logits, self.settings, 0)
        validate_transport(p, 3, 4, 0, self.settings["marginal_tolerance"])
        (p * torch.randn_like(p)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(torch.count_nonzero(logits.grad[0]), 0)
        self.assertGreater(logits.grad[1:].norm().item(), 0)
        shifted = logits.detach() + torch.randn(3, 4, 1) + torch.randn(3, 1, 4)
        torch.testing.assert_close(log_sinkhorn(shifted, self.settings, 0), p, rtol=0, atol=2e-5)
        invalid = copy.deepcopy(self.settings)
        invalid.update(max_iterations=1, marginal_tolerance=1e-12)
        with self.assertRaisesRegex(FloatingPointError, "tolerance"):
            log_sinkhorn(torch.randn(3, 4, 4), invalid, 0)

    def test_soft_mapping_and_consensus_solve_recover_known_shared(self):
        shared, indices, _ = permuted_experts()
        p = 0.8 * dense_permutations(indices) + 0.2 / 4
        p[0] = torch.eye(4)
        residuals = {
            name: shared[name] @ p if name == "down" else p.mT @ shared[name]
            for name in PROJECTIONS
        }
        actual = least_squares_shared(residuals, p)
        matrix = (p @ p.mT).sum(0)
        for name in PROJECTIONS:
            torch.testing.assert_close(native_shared(shared[name], p, int(name == "down")), residuals[name])
            torch.testing.assert_close(actual[name], shared[name], rtol=1e-5, atol=2e-6)
            rows = residuals[name].mT if name == "down" else residuals[name]
            solution = actual[name].mT if name == "down" else actual[name]
            torch.testing.assert_close(matrix @ solution, (p @ rows).sum(0), rtol=1e-5, atol=2e-6)
            # Merely mapping and averaging is no longer the minimizer.
            naive = (p @ rows).mean(0)
            self.assertGreater((naive - solution).norm().item(), 0.01)
        hard = dense_permutations(indices)
        fitted = least_squares_shared(residuals, hard)
        for name in PROJECTIONS:
            torch.testing.assert_close(fitted[name], aligned_mean(residuals[name], indices, int(name == "down")))

    def test_eight_steps_match_literal_joint_loss_and_penalty_uses_outer_anchor(self):
        shared, _, residuals = permuted_experts()
        shared = {p: value * 0.2 for p, value in shared.items()}
        residuals = {p: value * 0.2 + 0.1 * torch.randn_like(value) for p, value in residuals.items()}
        logits = torch.randn(3, 4, 4) * 0.2
        logits[0].zero_()
        old_p = log_sinkhorn(logits, self.settings, 0).detach()
        before = (logits.clone(), old_p.clone())
        expected = logits.clone()
        def loss(p):
            reconstruction = sum(
                (residuals[name] - native_shared(shared[name], p, int(name == "down"))).square().sum()
                for name in PROJECTIONS
            ) / 2
            return reconstruction + self.settings["move_penalty_over_rho"] * (p - old_p).square().sum() / 2
        for _ in range(8):
            expected.requires_grad_()
            candidate = log_sinkhorn(expected, self.settings, 0)
            gradient, = torch.autograd.grad(loss(candidate), expected)
            expected = expected.detach() - self.settings["learning_rate"] * gradient
            expected[0].zero_()
        with patch("salaad_moe.sinkhorn.torch.autograd.grad", wraps=torch.autograd.grad) as gradients:
            p, actual = update_soft_alignment(residuals, shared, old_p, logits, self.alignment)
        self.assertEqual(gradients.call_count, 8)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
        self.assertLess(loss(p).item(), loss(old_p).item())
        self.assertGreater((p - old_p).norm().item(), 1e-4)
        without_penalty = copy.deepcopy(self.alignment)
        without_penalty["sinkhorn"]["move_penalty_over_rho"] = 0
        _, unpenalized = update_soft_alignment(residuals, shared, old_p, logits, without_penalty)
        self.assertGreater((actual - unpenalized).norm().item(), 1e-5)
        assert_nested_equal(self, before, (logits, old_p))
        self.assertFalse(actual.requires_grad)
        self.assertFalse(p.requires_grad)

    def test_initialization_schedules_and_failed_update_are_atomic(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        for step in (1, 2):
            self.assertFalse(manager.after_step(step))
        self.assertTrue(manager.after_step(3))
        for group in manager.groups:
            torch.testing.assert_close(manager.anchors[group.name], group.weight())
        before = manager.local_state_dict()
        anchors = {name: value.clone() for name, value in manager.anchors.items()}
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("injected soft update failure")
            return update_soft_alignment(*args)
        with patch("salaad_moe.solver.update_soft_alignment", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "no state committed"):
                manager.after_step(5)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)
        for step in (5, 7, 8):
            self.assertTrue(manager.after_step(step, final=step == 8))
            self.assertEqual(manager.last_matching_step, step)
            self.assertEqual(manager.last_structure_step, step)
        self.assertEqual(manager.sweeps, 3)
        self.assertFalse(manager.after_step(8, final=True))

    def test_checkpoint_rejects_corrupt_soft_states_and_mode_mismatch(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights), self.c)
        manager.initialize(3)
        before = manager.local_state_dict()
        for problem in ("missing", "negative", "rows", "reference", "logits", "triplet", "mode"):
            invalid = copy.deepcopy(before)
            group = invalid["states"]["layers.0.moe.experts.gate"]
            if problem == "missing":
                del group["alignment_logits"]
            elif problem == "negative":
                group["permutation"][1, 0, 0] = -0.1
            elif problem == "rows":
                group["permutation"][1, 0].zero_()
            elif problem == "reference":
                group["permutation"][0] = torch.eye(4).roll(1, 0)
            elif problem == "logits":
                for value in invalid["states"].values():
                    value["alignment_logits"][1, 0, 0] += 1
            elif problem == "triplet":
                group["alignment_logits"][1, 0, 0] += 0.1
            else:
                invalid["alignment_method"] = "hungarian"
            with self.subTest(problem=problem), self.assertRaises(RuntimeError):
                manager.load_shards([invalid])
            assert_nested_equal(self, before, manager.local_state_dict())
        restored = ConsensusManager(manager.groups, self.c)
        restored.load_shards([before])
        assert_nested_equal(self, before, restored.local_state_dict())
        states = list(restored.states.values())
        for state in states[1:]:
            self.assertIs(state.permutation, states[0].permutation)
            self.assertIs(state.alignment_logits, states[0].alignment_logits)

    def test_configs_preserve_task_settings_and_enforce_cadence(self):
        formal = load_config(ROOT / "configs/ns97m_sinkhorn.yaml")
        baseline = load_config(ROOT / "configs/ns97m_aligned_warm100_every200.yaml")
        validate_config(formal, world_size=4)
        expected = copy.deepcopy(baseline)
        expected["experiment"] = formal["experiment"]
        expected["salaad"]["state_initialization_step"] = 0
        expected["salaad"]["initialization"] = formal["salaad"]["initialization"]
        expected["salaad"]["channel_alignment"] = formal["salaad"]["channel_alignment"]
        self.assertEqual(formal, expected)
        self.assertEqual(formal["salaad"]["channel_alignment"]["sinkhorn"]["inner_steps"], 128)
        self.assertEqual(formal["salaad"]["state_initialization_step"], 0)
        for key, value in (
            ("inner_steps", 1), ("inner_steps", True), ("temperature", 0),
            ("learning_rate", float("nan")), ("initial_softening", 1),
            ("move_penalty_over_rho", -1), ("max_iterations", 0), ("marginal_tolerance", 0.1),
        ):
            invalid = copy.deepcopy(self.c)
            invalid["salaad"]["channel_alignment"]["sinkhorn"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)
        for interval in (0, 4):
            invalid = copy.deepcopy(self.c)
            invalid["salaad"]["channel_alignment"]["match_every_optimizer_steps"] = interval
            with self.assertRaisesRegex(ValueError, "every L/S"):
                validate_config(invalid)

    def test_legacy_hard_checkpoint_without_method_metadata_still_loads(self):
        _, _, weights = permuted_experts()
        config = load_config(ROOT / "configs/smoke_aligned.yaml")
        original = ConsensusManager(groups_from(weights), config)
        original.initialize(0)
        legacy = copy.deepcopy(original.local_state_dict())
        del legacy["alignment_method"]
        restored = ConsensusManager(original.groups, config)
        restored.load_shards([legacy])
        assert_nested_equal(self, original.local_state_dict(), restored.local_state_dict())
        assert_nested_equal(self, original.anchors, restored.anchors)


class SinkhornWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.c = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
        make_synthetic_corpus(self.c, self.path / "data", 64)
        self.corpus = TokenCorpus(self.path / "data/manifest.json", self.c)

    def test_vanilla_prefix_and_checkpoint_resume_are_bitwise_equal(self):
        vanilla_config = copy.deepcopy(self.c)
        vanilla_config["salaad"]["enabled"] = False
        vanilla_config["salaad"]["channel_alignment"]["enabled"] = False
        vanilla = Trainer(vanilla_config, self.corpus, "cpu")
        prefix = []
        for _ in range(3):
            vanilla.train_step()
            prefix.append(copy.deepcopy((vanilla.model.state_dict(), vanilla.optimizer.state_dict())))
        reference = Trainer(self.c, self.corpus, "cpu")
        checkpoints = {}
        for step in range(1, 9):
            record = reference.train_step()
            if step <= 3:
                self.assertEqual(record["constraint_gradient_norm"], 0)
                assert_nested_equal(self, prefix[step - 1], (reference.model.state_dict(), reference.optimizer.state_dict()))
            if step in (2, 3, 5, 7):
                checkpoints[step] = save_checkpoint(reference, self.path / "checkpoints")
        expected_rng = rng_state()
        for start, checkpoint in checkpoints.items():
            with patch("salaad_moe.solver.initialize_soft_alignment", side_effect=AssertionError("must restore")):
                resumed = Trainer(self.c, self.corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(resumed, checkpoint)
            for _ in range(start, 8):
                resumed.train_step()
            for actual, expected in (
                (resumed.model.state_dict(), reference.model.state_dict()),
                (resumed.optimizer.state_dict(), reference.optimizer.state_dict()),
                (resumed.manager.local_state_dict(), reference.manager.local_state_dict()),
                (resumed.reader.state_dict(), reference.reader.state_dict()),
                (rng_state(), expected_rng),
            ):
                assert_nested_equal(self, actual, expected)
            self.assertEqual(resumed.manager.last_matching_step, 8)
        initial = load_torch(checkpoints[3] / "rank_00000.pt")["salaad"]["states"]
        self.assertTrue(any(
            not torch.equal(state.permutation, initial[name]["permutation"])
            for name, state in reference.manager.states.items()
        ))

    def test_bfloat16_export_keeps_soft_mapping_and_excludes_dual(self):
        self.c["training"]["task_precision"] = "bfloat16"
        trainer = Trainer(self.c, self.corpus, "cpu")
        for _ in range(8):
            trainer.train_step()
        for state in trainer.manager.states.values():
            self.assertTrue(all(t.dtype == torch.float32 for t in state.tensors()))
            state.dual.fill_(7)
        checkpoint = save_checkpoint(trainer, self.path / "checkpoints")
        path = self.path / "soft.pt"
        report = export_checkpoint(checkpoint, path)
        self.assertEqual(report["format"], "salaad_moe.export.v4")
        reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
        exported, _ = load_evaluation_model(path, "exported")
        assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
        for name, state in trainer.manager.states.items():
            torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
        inputs, _ = self.corpus.batch("validation", [0, 1], "cpu")
        torch.testing.assert_close(reconstructed(inputs).logits, exported(inputs).logits, rtol=0, atol=0)
        artifact = load_torch(path)
        for problem in ("rows", "reference", "triplet", "format", "missing", "marginal_tolerance"):
            invalid = copy.deepcopy(artifact)
            group = invalid["groups"]["layers.0.moe.experts.gate"]
            if problem == "rows":
                group["permutation"][1, 0].zero_()
            elif problem == "reference":
                group["permutation"][0] = group["permutation"][0].roll(1, 0)
            elif problem == "triplet":
                group["permutation"][1] = group["permutation"][1].roll(1, 0)
            elif problem == "format":
                invalid["format"] = "salaad_moe.export.v3"
            elif problem == "marginal_tolerance":
                for value in invalid["groups"].values():
                    value["permutation"][1, 0, 0] += 5e-5
            else:
                del group["permutation"]
            invalid_path = self.path / (problem + ".pt")
            atomic_save(invalid, invalid_path)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                load_evaluation_model(invalid_path, "exported")


if __name__ == "__main__":
    unittest.main()
