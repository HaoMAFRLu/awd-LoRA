"""Numerical and end-to-end contracts for native-coordinate channel alignment."""
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from salaad_moe.alignment import (
    PROJECTIONS, aligned_mean, initialize_alignment, joint_cost, match_channels, native_shared,
)
from salaad_moe.checkpoint import atomic_save, load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.groups import SequentialFusedGroup, StackedGroup
from salaad_moe.solver import (
    ConsensusManager, aligned_structure_sweep, initial_state, initial_svd_basis, linear_mass_rank,
)
from salaad_moe.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


def assert_nested_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(first, second)


def permuted_experts():
    shared = {"gate": torch.randn(4, 5), "up": torch.randn(4, 5), "down": torch.randn(5, 4)}
    permutation = torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1], [1, 3, 0, 2]])
    weights = {p: native_shared(shared[p], permutation, int(p == "down")) for p in PROJECTIONS}
    return shared, permutation, weights


def groups_from(weights, layers=1):
    return [
        StackedGroup(f"layers.{layer}.moe.experts.{p}", torch.nn.Parameter(weights[p].clone()))
        for layer in range(layers) for p in PROJECTIONS
    ]


def dense_permutations(permutation):
    # P[b,a]=1 iff the native channel a maps to shared channel b.
    return F.one_hot(permutation, permutation.shape[-1]).float().mT


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.c = load_config(ROOT / "configs/smoke_aligned.yaml")
        self.settings = self.c["salaad"]["channel_alignment"]

    def test_joint_cost_and_assignment_match_exhaustive_permutations(self):
        shared, _, residuals = permuted_experts()
        residuals = {p: value + 0.1 * torch.randn_like(value) for p, value in residuals.items()}
        cost = joint_cost(residuals, shared)
        for i, a, b in itertools.product(range(3), range(4), range(4)):
            expected = (
                (residuals["gate"][i, a] - shared["gate"][b]).square().sum()
                + (residuals["up"][i, a] - shared["up"][b]).square().sum()
                + (residuals["down"][i, :, a] - shared["down"][:, b]).square().sum()
            )
            torch.testing.assert_close(cost[i, a, b], expected)
        identity = torch.arange(4).expand(3, -1).clone()
        actual = match_channels(residuals, shared, identity, self.settings)
        torch.testing.assert_close(actual[0], identity[0])
        for i in (1, 2):
            best = min(
                itertools.permutations(range(4)),
                key=lambda indices: sum(cost[i, a, b].item() for a, b in enumerate(indices)),
            )
            self.assertEqual(actual[i].tolist(), list(best))

    def test_global_assignment_and_tie_acceptance(self):
        shared, permutation, residuals = permuted_experts()
        identity = torch.arange(2).expand(2, -1).clone()
        # Greedy channel 0 -> position 0 costs 10; optimal one-to-one costs 3.
        cost = torch.tensor([[[0., 1.], [1., 0.]], [[1., 2.], [1., 9.]]])
        with patch("salaad_moe.alignment.joint_cost", return_value=cost):
            matched = match_channels(residuals, shared, identity, self.settings)
        self.assertEqual(matched.tolist(), [[0, 1], [1, 0]])
        # Ties and improvements below epsilon retain the previous pairing.
        for cost in (torch.zeros(3, 4, 4), torch.full((3, 4, 4), 1e-10)):
            with patch("salaad_moe.alignment.joint_cost", return_value=cost):
                matched = match_channels(residuals, shared, permutation, self.settings)
            torch.testing.assert_close(matched, permutation, rtol=0, atol=0)

    def test_recovers_known_permutation_without_changing_ffn(self):
        shared, permutation, weights = permuted_experts()
        actual, center = initialize_alignment(weights, self.settings)
        torch.testing.assert_close(actual, permutation, rtol=0, atol=0)
        for p in PROJECTIONS:
            torch.testing.assert_close(center[p], shared[p])
            torch.testing.assert_close(aligned_mean(weights[p], actual, int(p == "down")), shared[p])
        inputs = torch.randn(7, 5)
        reference = F.linear(F.silu(F.linear(inputs, shared["gate"])) * F.linear(inputs, shared["up"]), shared["down"])
        for i in range(3):
            output = F.linear(
                F.silu(F.linear(inputs, weights["gate"][i])) * F.linear(inputs, weights["up"][i]),
                weights["down"][i],
            )
            torch.testing.assert_close(output, reference)
        groups = groups_from(weights)
        before = [g.weight().clone() for g in groups]
        manager = ConsensusManager(groups, self.c)
        manager.initialize(0)
        for group, expected in zip(groups, before):
            torch.testing.assert_close(group.weight(), expected, rtol=0, atol=0)
            torch.testing.assert_close(manager.anchors[group.name], expected)
            torch.testing.assert_close(manager.states[group.name].permutation, permutation, rtol=0, atol=0)
        states = list(manager.states.values())
        self.assertIs(states[0].permutation, states[1].permutation)
        self.assertIs(states[0].permutation, states[2].permutation)

    def test_sweep_matches_dense_permutation_and_exact_svd_reference(self):
        shared, expected_permutation, residuals = permuted_experts()
        identity = torch.arange(4).expand(3, -1).clone()
        old, weights = {}, {}
        for p in PROJECTIONS:
            state = initial_state(
                residuals[p], self.c, shared=shared[p].clone(),
                permutation=identity, channel_axis=int(p == "down"),
            )
            state.low_rank = torch.randn_like(residuals[p])
            state.sparse = torch.randn_like(residuals[p]) * 0.3
            state.dual = torch.randn_like(residuals[p]) * 0.2
            state.tau_l.fill_(0.15)
            state.tau_s.fill_(0.1)
            old[p] = state
            weights[p] = residuals[p] + state.low_rank + state.sparse - state.dual
        matrices = dense_permutations(expected_permutation)
        expected_shared, expected = {}, {}
        for p in PROJECTIONS:
            aligned = residuals[p] @ matrices.mT if p == "down" else matrices @ residuals[p]
            h = expected_shared[p] = aligned.mean(0)
            mapped = h @ matrices if p == "down" else matrices.mT @ h
            a = weights[p] - mapped - old[p].sparse + old[p].dual
            old[p].svd_basis, _ = initial_svd_basis(a, 3)
            u, sigma, vh = torch.linalg.svd(a, full_matrices=False)
            shrunk = (sigma - 0.15).clamp_min(0)
            low = (u * shrunk[:, None, :]) @ vh
            remainder = weights[p] - mapped - low + old[p].dual
            sparse = remainder.sign() * (remainder.abs() - 0.1).clamp_min(0)
            dual = old[p].dual + weights[p] - mapped - low - sparse
            expected[p] = (mapped, low, sparse, dual, shrunk)
        before = {p: old[p].state_dict() for p in PROJECTIONS}
        new = aligned_structure_sweep(weights, old, self.c, rematch=True)
        for p in PROJECTIONS:
            mapped, low, sparse, dual, spectrum = expected[p]
            torch.testing.assert_close(new[p].permutation, expected_permutation, rtol=0, atol=0)
            torch.testing.assert_close(new[p].shared, expected_shared[p])
            for actual, target in ((new[p].low_rank, low), (new[p].sparse, sparse), (new[p].dual, dual)):
                torch.testing.assert_close(actual, target, rtol=1e-4, atol=3e-6)
            torch.testing.assert_close(new[p].anchor(), mapped + low + sparse - dual, rtol=1e-4, atol=5e-6)
            torch.testing.assert_close(new[p].tau_l, old[p].tau_l + 0.02 * (linear_mass_rank(spectrum) - 0.15))
            torch.testing.assert_close(new[p].tau_s, old[p].tau_s + 0.002 * (sparse.ne(0).float().mean((-1, -2)) - 0.1))
            assert_nested_equal(self, before[p], old[p].state_dict())

    def test_schedules_final_flush_and_initialization_only(self):
        _, _, weights = permuted_experts()
        for interval, expected_step in ((4, 4), (0, 0)):
            self.settings["match_every_optimizer_steps"] = interval
            manager = ConsensusManager(groups_from(weights), self.c)
            manager.initialize(0)
            updates = [s for s in range(1, 8) if manager.after_step(s, final=s == 7)]
            self.assertEqual(updates, [2, 4, 6, 7])
            self.assertEqual(manager.last_matching_step, expected_step)
            self.assertFalse(manager.after_step(7, final=True))
            self.assertEqual(manager.sweeps, 4)

    def test_delayed_matching_uses_initialization_offset_and_fixed_p_stays_fixed(self):
        _, _, weights = permuted_experts()
        self.c["salaad"]["state_initialization_step"] = 3
        for interval, expected_last_match in ((4, 7), (0, 3)):
            self.settings["match_every_optimizer_steps"] = interval
            manager = ConsensusManager(groups_from(weights), self.c)
            for step in (1, 2):
                self.assertFalse(manager.after_step(step))
                self.assertFalse(manager.initialized)
                self.assertEqual(manager.states, {})
                self.assertEqual(manager.anchors, {})
                self.assertEqual(manager.inject_gradients(1), (0.0, None))
            self.assertTrue(manager.after_step(3))
            initial_p = manager.states[manager.groups[0].name].permutation.clone()
            updates, matching_steps = [], []
            with patch("salaad_moe.solver.match_channels", wraps=match_channels) as matching:
                for step in range(4, 9):
                    before = matching.call_count
                    if manager.after_step(step, final=step == 8):
                        updates.append(step)
                    if matching.call_count != before:
                        matching_steps.append(step)
                    if interval == 0:
                        torch.testing.assert_close(
                            manager.states[manager.groups[0].name].permutation,
                            initial_p, rtol=0, atol=0,
                        )
            self.assertEqual(updates, [5, 7, 8])
            self.assertEqual(matching_steps, [7] if interval else [])
            self.assertEqual(manager.last_matching_step, expected_last_match)
            self.assertEqual(manager.sweeps, 3)
            self.assertFalse(manager.after_step(8, final=True))

    def test_layer_owners_and_failed_matching_are_atomic(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        for current_rank in range(4):
            with patch("salaad_moe.solver.world_size", return_value=4), patch("salaad_moe.solver.rank", return_value=current_rank):
                names = [group.name for _, group in manager.owned()]
            expected = [f"layers.{current_rank}.moe.experts.{p}" for p in PROJECTIONS] if current_rank < 2 else []
            self.assertEqual(names, expected)
        manager.initialize(0)
        before = manager.local_state_dict()
        anchors = {name: value.clone() for name, value in manager.anchors.items()}
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("matching failed on second layer")
            return match_channels(*args)
        with patch("salaad_moe.solver.match_channels", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "no state committed.*matching failed"):
                manager.update(4)
        assert_nested_equal(self, before, manager.local_state_dict())
        assert_nested_equal(self, anchors, manager.anchors)

    def test_checkpoint_rejects_missing_invalid_and_inconsistent_permutations(self):
        _, _, weights = permuted_experts()
        manager = ConsensusManager(groups_from(weights), self.c)
        manager.initialize(0)
        before = manager.local_state_dict()
        for problem in ("missing", "float", "duplicate", "reference", "triplet", "axis", "step"):
            invalid = copy.deepcopy(before)
            state = invalid["states"]["layers.0.moe.experts.up"]
            if problem == "missing":
                del state["permutation"]
            elif problem == "float":
                state["permutation"] = state["permutation"].float()
            elif problem == "duplicate":
                state["permutation"][1].zero_()
            elif problem == "reference":
                state["permutation"][0] = state["permutation"][0].roll(1)
            elif problem == "triplet":
                state["permutation"][1] = state["permutation"][1].roll(1)
            elif problem == "axis":
                state["channel_axis"] = 1
            else:
                invalid["last_matching_step"] = 5
            with self.subTest(problem=problem), self.assertRaises(RuntimeError):
                manager.load_shards([invalid])
            assert_nested_equal(self, before, manager.local_state_dict())
        restored = ConsensusManager(manager.groups, self.c)
        restored.load_shards([before])
        assert_nested_equal(self, before, restored.local_state_dict())
        assert_nested_equal(self, manager.anchors, restored.anchors)

    def test_fused_fp32_masters_and_constraint_gradients_stay_native(self):
        _, permutation, weights = permuted_experts()
        fc1, fc2 = [], []
        for i in range(3):
            fused = torch.cat((weights["gate"][i], weights["up"][i]))
            for target, value in ((fc1, fused), (fc2, weights["down"][i])):
                parameter = torch.nn.Parameter(value.bfloat16())
                parameter.main_param = torch.nn.Parameter(value.clone())
                target.append(parameter)
        groups = [
            SequentialFusedGroup(f"layers.0.moe.experts.{p}", fc2 if p == "down" else fc1, p)
            for p in PROJECTIONS
        ]
        manager = ConsensusManager(groups, self.c)
        manager.initialize(0)
        for group in groups:
            torch.testing.assert_close(manager.states[group.name].permutation, permutation)
            manager.states[group.name].dual.fill_(0.2)
        manager.refresh_anchors()
        expected = {
            group.name: self.c["salaad"]["rho"] * (group.weight() - manager.anchors[group.name])
            for group in groups
        }
        manager.inject_gradients(0)
        for i in range(3):
            torch.testing.assert_close(fc1[i].main_param.grad, torch.cat((expected[groups[0].name][i], expected[groups[1].name][i])))
            torch.testing.assert_close(fc2[i].main_param.grad, expected[groups[2].name][i])
            torch.testing.assert_close(fc1[i].main_param, torch.cat((weights["gate"][i], weights["up"][i])), rtol=0, atol=0)

    def test_config_rejects_incompatible_alignment(self):
        validate_config(self.c)
        formal = load_config(ROOT / "configs/ns97m_aligned.yaml")
        validate_config(formal)
        self.assertEqual(formal["salaad"]["channel_alignment"]["match_every_optimizer_steps"], 100)
        for key, value in (
            ("projections", ["down"]), ("shared_mode", "none"),
            ("state_initialization_step", -1), ("state_initialization_step", 1.5),
            ("state_initialization_step", True), ("state_initialization_step", 8),
            ("structure_inner_steps", 2), ("auxiliary_owner", "deterministic_group_id_mod_dp"),
        ):
            invalid = copy.deepcopy(self.c)
            invalid["salaad"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)
        for key, value in (
            ("match_every_optimizer_steps", 3), ("match_every_optimizer_steps", -2),
            ("initialization_max_iterations", 0), ("improvement_tolerance", float("nan")),
            ("reference_expert", 8),
        ):
            invalid = copy.deepcopy(self.c)
            invalid["salaad"]["channel_alignment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_formal_schedule_ablations_preserve_all_other_settings(self):
        baseline = load_config(ROOT / "configs/ns97m_aligned.yaml")
        for name, start, interval in (
            ("ns97m_aligned_fixed", 0, 0),
            ("ns97m_aligned_warm100_fixed", 100, 0),
            ("ns97m_aligned_warm100_every200", 100, 200),
        ):
            actual = load_config(ROOT / "configs" / (name + ".yaml"))
            validate_config(actual, world_size=4)
            expected = copy.deepcopy(baseline)
            expected["experiment"] = "moe_" + name
            expected["salaad"]["state_initialization_step"] = start
            expected["salaad"]["channel_alignment"]["match_every_optimizer_steps"] = interval
            self.assertEqual(actual, expected)


class AlignedWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.c = load_config(ROOT / "configs/smoke_aligned.yaml")
        make_synthetic_corpus(self.c, self.path / "data", 64)
        self.corpus = TokenCorpus(self.path / "data/manifest.json", self.c)

    def test_delayed_prefix_matches_vanilla_and_branch_preserves_training_state(self):
        self.c["salaad"]["state_initialization_step"] = 3
        vanilla_config = copy.deepcopy(self.c)
        vanilla_config["salaad"]["enabled"] = False
        vanilla_config["salaad"]["channel_alignment"]["enabled"] = False
        vanilla = Trainer(vanilla_config, self.corpus, "cpu")
        prefix = []
        for _ in range(3):
            record = vanilla.train_step()
            prefix.append(copy.deepcopy((
                vanilla.model.state_dict(), vanilla.optimizer.state_dict(),
                vanilla.reader.state_dict(), record["lm_nll"],
            )))
        checkpoint = save_checkpoint(vanilla, self.path / "vanilla_prefix")
        reference = Trainer(self.c, self.corpus, "cpu")
        for step, expected in enumerate(prefix, 1):
            record = reference.train_step()
            self.assertEqual(record["constraint_gradient_norm"], 0)
            assert_nested_equal(self, expected, (
                reference.model.state_dict(), reference.optimizer.state_dict(),
                reference.reader.state_dict(), record["lm_nll"],
            ))
            self.assertEqual(reference.manager.initialized, step == 3)
        initial_auxiliary = reference.manager.local_state_dict()
        for _ in range(5):
            reference.train_step()
        expected_rng = rng_state()
        branched = Trainer(self.c, self.corpus, "cpu", initialize_auxiliary=False)
        load_checkpoint(branched, checkpoint, branch_from_vanilla=True)
        assert_nested_equal(self, initial_auxiliary, branched.manager.local_state_dict())
        for _ in range(5):
            branched.train_step()
        for actual, expected in (
            (branched.model.state_dict(), reference.model.state_dict()),
            (branched.optimizer.state_dict(), reference.optimizer.state_dict()),
            (branched.manager.local_state_dict(), reference.manager.local_state_dict()),
            (branched.reader.state_dict(), reference.reader.state_dict()),
            (rng_state(), expected_rng),
        ):
            assert_nested_equal(self, actual, expected)
        self.assertEqual(branched.manager.last_matching_step, 7)

    def test_delayed_resume_before_initialization_and_at_matching_boundary(self):
        self.c["salaad"]["state_initialization_step"] = 3
        reference = Trainer(self.c, self.corpus, "cpu")
        checkpoints = {}
        for step in range(1, 9):
            reference.train_step()
            if step in (2, 3, 7):
                checkpoints[step] = save_checkpoint(reference, self.path / "delayed_resume")
        expected_rng = rng_state()
        for start, checkpoint in checkpoints.items():
            with patch("salaad_moe.solver.initialize_alignment", wraps=initialize_alignment) as initialize:
                resumed = Trainer(self.c, self.corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(resumed, checkpoint)
                self.assertEqual(initialize.call_count, 0)
                for _ in range(start, 8):
                    resumed.train_step()
                self.assertEqual(initialize.call_count, self.c["model"]["num_layers"] if start < 3 else 0)
            for actual, expected in (
                (resumed.model.state_dict(), reference.model.state_dict()),
                (resumed.optimizer.state_dict(), reference.optimizer.state_dict()),
                (resumed.manager.local_state_dict(), reference.manager.local_state_dict()),
                (resumed.reader.state_dict(), reference.reader.state_dict()),
                (rng_state(), expected_rng),
            ):
                assert_nested_equal(self, actual, expected)
            self.assertEqual(resumed.manager.last_matching_step, 7)

    def test_resume_between_updates_and_at_matching_boundary_is_bitwise_identical(self):
        reference = Trainer(self.c, self.corpus, "cpu")
        checkpoints = {}
        for step in range(1, 9):
            reference.train_step()
            if step in (3, 4):
                checkpoints[step] = save_checkpoint(reference, self.path / "checkpoints")
        final_rng = rng_state()
        for start, checkpoint in checkpoints.items():
            with patch("salaad_moe.solver.initialize_alignment", side_effect=AssertionError("must restore P")), patch(
                "salaad_moe.solver.initial_svd_basis", side_effect=AssertionError("must restore basis")
            ):
                resumed = Trainer(self.c, self.corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(resumed, checkpoint)
            for _ in range(start, 8):
                resumed.train_step()
            for first, second in (
                (reference.model.state_dict(), resumed.model.state_dict()),
                (reference.optimizer.state_dict(), resumed.optimizer.state_dict()),
                (reference.manager.local_state_dict(), resumed.manager.local_state_dict()),
                (reference.reader.state_dict(), resumed.reader.state_dict()),
                (final_rng, rng_state()),
            ):
                assert_nested_equal(self, first, second)
            self.assertEqual(resumed.manager.last_matching_step, 8)

    def test_export_preserves_native_reconstruction_and_excludes_dual(self):
        trainer = Trainer(self.c, self.corpus, "cpu")
        for _ in range(4):
            trainer.train_step()
        # Make including U an obvious error in both checkpoint and artifact readers.
        for state in trainer.manager.states.values():
            state.dual.fill_(7)
        checkpoint = save_checkpoint(trainer, self.path / "checkpoints")
        path = self.path / "aligned.pt"
        report = export_checkpoint(checkpoint, path)
        self.assertEqual(report["format"], "salaad_moe.export.v3")
        artifact = load_torch(path)
        for group in artifact["groups"].values():
            self.assertEqual(group["permutation_convention"], "native_to_shared")
            self.assertEqual(group["permutation"].dtype, torch.int64)
        reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
        exported, _ = load_evaluation_model(path, "exported")
        assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
        for name, state in trainer.manager.states.items():
            torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
        raw, _ = load_evaluation_model(checkpoint, "raw")
        assert_nested_equal(self, raw.state_dict(), trainer.model.state_dict())
        inputs, _ = self.corpus.batch("validation", [0, 1], "cpu")
        torch.testing.assert_close(reconstructed(inputs).logits, exported(inputs).logits, rtol=0, atol=0)
        for problem in ("missing", "triplet", "axis"):
            invalid = copy.deepcopy(artifact)
            group = invalid["groups"]["layers.0.moe.experts.up"]
            if problem == "missing":
                del invalid["groups"]["layers.0.moe.experts.up"]
            elif problem == "triplet":
                group["permutation"][1] = group["permutation"][1].roll(1)
            else:
                group["channel_axis"] = 1
            invalid_path = self.path / (problem + ".pt")
            atomic_save(invalid, invalid_path)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                load_evaluation_model(invalid_path, "exported")

    def test_bfloat16_training_retains_float32_aligned_states(self):
        self.c["training"]["task_precision"] = "bfloat16"
        trainer = Trainer(self.c, self.corpus, "cpu")
        for _ in range(4):
            trainer.train_step()
        for state in trainer.manager.states.values():
            for value in state.tensors():
                self.assertEqual(value.dtype, torch.int64 if value is state.permutation else torch.float32)


if __name__ == "__main__":
    unittest.main()
