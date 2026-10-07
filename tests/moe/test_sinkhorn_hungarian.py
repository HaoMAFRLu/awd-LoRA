"""Soft-candidate projection, hard-coordinate updates, and persistence."""
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from salaad_moe.alignment import PROJECTIONS, native_shared, validate_permutation
from salaad_moe.checkpoint import atomic_save, load_checkpoint, load_torch, rng_state, save_checkpoint
from salaad_moe.config import fingerprint, load_config, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import export_checkpoint, load_evaluation_model
from salaad_moe.sinkhorn import log_sinkhorn, project_transport_to_permutation, update_soft_alignment
from salaad_moe.solver import ConsensusManager, aligned_structure_sweep, initial_state
from salaad_moe.tracking import flatten_metrics
from salaad_moe.trainer import Trainer
from test_alignment import assert_nested_equal, groups_from


ROOT = Path(__file__).resolve().parents[2]


class SinkhornHungarianTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(127)
        self.c = load_config(ROOT / "configs/smoke_sinkhorn_hungarian.yaml")

    def test_projection_minimizes_exhaustive_frobenius_distance_and_orientation(self):
        settings = self.c["salaad"]["channel_alignment"]["sinkhorn"]
        soft = log_sinkhorn(torch.randn(8, 4, 4), settings, None)
        cycle = torch.tensor([1, 2, 3, 0])  # Different from its inverse.
        soft[0] = 0.8 * F.one_hot(cycle, 4).float().T + 0.05
        previous = torch.arange(4).expand(8, -1).clone()
        before, before_rng = soft.clone(), torch.get_rng_state().clone()
        actual = project_transport_to_permutation(soft, previous)
        validate_permutation(actual, 8, 4)
        torch.testing.assert_close(actual[0], cycle, rtol=0, atol=0)
        candidates = torch.stack([
            F.one_hot(torch.tensor(p), 4).double().T
            for p in itertools.permutations(range(4))
        ])
        for expert in range(8):
            matrix = F.one_hot(actual[expert], 4).double().T
            error = (matrix - soft[expert].double()).square().sum()
            optimum = (candidates - soft[expert].double()).square().sum((1, 2)).min()
            torch.testing.assert_close(error, optimum, rtol=0, atol=1e-14)
        torch.testing.assert_close(soft, before, rtol=0, atol=0)
        torch.testing.assert_close(previous, torch.arange(4).expand(8, -1), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)

    def test_colliding_row_maxima_exact_ties_and_small_improvements(self):
        soft = torch.tensor([[[.46, .40, .14], [.44, .31, .25], [.10, .29, .61]]])
        actual = project_transport_to_permutation(soft, torch.arange(3)[None])
        torch.testing.assert_close(actual, torch.tensor([[1, 0, 2]]), rtol=0, atol=0)
        previous = torch.tensor([[2, 0, 1]])
        torch.testing.assert_close(
            project_transport_to_permutation(torch.full((1, 3, 3), 1 / 3), previous),
            previous, rtol=0, atol=0,
        )
        soft = torch.tensor([[[.5 - 1e-7, .5 + 1e-7], [.5 + 1e-7, .5 - 1e-7]]])
        torch.testing.assert_close(
            project_transport_to_permutation(soft, torch.arange(2)[None]),
            torch.tensor([[1, 0]]), rtol=0, atol=0,
        )

    def test_projection_rejects_invalid_candidates_and_old_indices(self):
        soft = torch.eye(3)[None]
        for bad in (soft * 2, soft * float("nan"), -soft, soft[0], soft.double()):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                project_transport_to_permutation(bad, torch.arange(3)[None])
        with self.assertRaisesRegex(ValueError, "bijection"):
            project_transport_to_permutation(soft, torch.zeros(1, 3, dtype=torch.long))

    def test_sweep_projects_soft_candidate_before_updating_shared(self):
        template = {
            "gate": torch.eye(3), "up": torch.diag(torch.tensor([2., 3., 4.])),
            "down": torch.tensor([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.]]),
        }
        soft = torch.tensor([
            [[.1, .8, .1], [.1, .1, .8], [.8, .1, .1]],
            [[.46, .40, .14], [.44, .31, .25], [.10, .29, .61]],
        ])
        expected_p = torch.tensor([[2, 0, 1], [1, 0, 2]])
        old_p = torch.arange(3).expand(2, -1).clone()
        old, weights, targets = {}, {}, {}
        for p in PROJECTIONS:
            target = native_shared(template[p], soft, int(p == "down"))
            state = initial_state(target, self.c, shared=template[p],
                                  permutation=old_p, channel_axis=int(p == "down"))
            state.residual = 0.1 * torch.randn_like(target)
            state.dual = 0.05 * torch.randn_like(target)
            old[p] = state
            weights[p] = target + state.residual - state.dual
            targets[p] = weights[p] - state.residual + state.dual
        before = copy.deepcopy({p: state.state_dict() for p, state in old.items()})
        before_weights = copy.deepcopy(weights)
        with (
            patch("salaad_moe.solver.update_soft_alignment", wraps=update_soft_alignment) as update,
            patch("salaad_moe.solver.match_channels", side_effect=AssertionError("no reconstruction matching")),
            patch("salaad_moe.solver.least_squares_shared", side_effect=AssertionError("no soft X solve")),
        ):
            new = aligned_structure_sweep(weights, old, self.c, rematch=True)
        self.assertEqual(update.call_count, 1)
        assert_nested_equal(self, update.call_args.args[0], targets)
        assert_nested_equal(self, update.call_args.args[1], template)
        dense_p = F.one_hot(expected_p, 3).float().mT
        for p, state in new.items():
            shared = ((targets[p] @ dense_p.mT) if p == "down" else
                      (dense_p @ targets[p])).mean(0)
            torch.testing.assert_close(state.shared, shared)
            self.assertFalse(torch.allclose(shared, targets[p].mean(0)))
            torch.testing.assert_close(state.permutation, expected_p, rtol=0, atol=0)
            self.assertIs(state.permutation, new["gate"].permutation)
            self.assertIsNone(state.alignment_logits)
            torch.testing.assert_close(state.native_shared().flatten(1).norm(dim=1),
                                       state.shared.norm().expand(2))
            torch.testing.assert_close(state.residual, weights[p] - state.native_shared() + old[p].dual)
            torch.testing.assert_close(state.dual, torch.zeros_like(state.dual), rtol=0, atol=5e-7)
        assert_nested_equal(self, before, {p: state.state_dict() for p, state in old.items()})
        assert_nested_equal(self, before_weights, weights)

    def test_initialization_and_atomic_failures_at_both_update_stages(self):
        weights = {"gate": torch.randn(3, 4, 5), "up": torch.randn(3, 4, 5),
                   "down": torch.randn(3, 5, 4)}
        manager = ConsensusManager(groups_from(weights, layers=2), self.c)
        before_rng = torch.get_rng_state().clone()
        with (
            patch("salaad_moe.alignment.match_channels", side_effect=AssertionError("no initial matching")),
            patch("salaad_moe.solver.initialize_soft_alignment", side_effect=AssertionError("no soft init")),
            patch("salaad_moe.solver.initial_svd_basis", side_effect=AssertionError("no residual SVD")),
        ):
            manager.initialize(3)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        for group in manager.groups:
            state = manager.states[group.name]
            value = weights[group.name.rsplit(".", 1)[1]]
            torch.testing.assert_close(group.weight(), value, rtol=0, atol=0)
            torch.testing.assert_close(state.shared, value.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(state.residual, value - value.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(state.permutation, torch.arange(4).expand(3, -1), rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(state.dual), 0)
        before, anchors = copy.deepcopy(manager.local_state_dict()), copy.deepcopy(manager.anchors)
        for function in ("update_soft_alignment", "project_transport_to_permutation"):
            with patch("salaad_moe.solver." + function, side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "no state committed"):
                    manager.after_step(5)
            assert_nested_equal(self, before, manager.local_state_dict())
            assert_nested_equal(self, anchors, manager.anchors)
        with patch("salaad_moe.solver.project_transport_to_permutation", wraps=project_transport_to_permutation) as project:
            for step, due in ((4, False), (5, True), (6, False), (7, True), (8, True)):
                self.assertEqual(manager.after_step(step, final=step == 8), due)
            self.assertFalse(manager.after_step(8, final=True))
            self.assertEqual(project.call_count, 6)  # Two layers, three sweeps.
        self.assertEqual(manager.last_matching_step, 8)

    def test_config_is_opt_in_and_preserves_training_parameters(self):
        original = load_config(ROOT / "configs/ns97m_sinkhorn.yaml")
        hard = load_config(ROOT / "configs/ns97m_sinkhorn_hungarian.yaml")
        self.assertEqual(hard["training"]["weight_decay"], 0)
        restored = copy.deepcopy(hard)
        restored["experiment"] = original["experiment"]
        restored["salaad"]["channel_alignment"]["method"] = "sinkhorn"
        self.assertEqual(hard["salaad"]["channel_alignment"]["sinkhorn"]["max_iterations"], 10000)
        restored["salaad"]["channel_alignment"]["sinkhorn"]["max_iterations"] = 150
        self.assertEqual(restored, original)
        for changes in (
            {"initialization": "soft_aligned_shared_least_squares_residual_dual_zero"},
            {"structure_order": ["shared", "permutation", "residual", "dual"]},
            {"channel_alignment": {"fix_reference": True}},
            {"channel_alignment": {"match_every_optimizer_steps": 4}},
            {"channel_alignment": {"sinkhorn": {"max_iterations": 0}}},
        ):
            bad = copy.deepcopy(self.c)
            if "channel_alignment" in changes:
                bad["salaad"]["channel_alignment"].update(changes["channel_alignment"])
            else:
                bad["salaad"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(bad)
        bad = load_config(ROOT / "configs/smoke_aligned.yaml")
        bad["salaad"]["channel_alignment"]["method"] = "sinkhorn_hungarian"
        with self.assertRaisesRegex(ValueError, "dense residuals"):
            validate_config(bad)
        for name in ("ns97m", "ns97m_aligned", "ns97m_aligned_fixed", "ns97m_sinkhorn",
                     "ns97m_hungarian", "ns97m_vanilla", "ns97m_sinkhorn_hungarian"):
            config = load_config(ROOT / "configs" / (name + ".yaml"))
            before = fingerprint(config)
            validate_config(config)
            self.assertEqual(fingerprint(config), before)

    def test_bfloat16_training_and_bitwise_resume_including_final_flush(self):
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
            wrong = copy.deepcopy(self.c)
            wrong["salaad"]["channel_alignment"]["method"] = "sinkhorn"
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                load_checkpoint(Trainer(wrong, corpus, "cpu", initialize_auxiliary=False), checkpoints[5])

    def test_original_iteration_cap_fails_without_committing_partial_states(self):
        self.c["salaad"]["channel_alignment"]["sinkhorn"]["max_iterations"] = 150
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            trainer = Trainer(self.c, TokenCorpus(path / "data/manifest.json", self.c), "cpu")
            for _ in range(6):
                trainer.train_step()
            before = copy.deepcopy(trainer.manager.local_state_dict())
            anchors = copy.deepcopy(trainer.manager.anchors)
            with self.assertRaisesRegex(RuntimeError, "after 150 iterations"):
                trainer.train_step()
            assert_nested_equal(self, before, trainer.manager.local_state_dict())
            assert_nested_equal(self, anchors, trainer.manager.anchors)

    def test_hard_checkpoint_export_and_nonidentity_metrics(self):
        self.c["salaad"]["state_initialization_step"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_synthetic_corpus(self.c, path / "data", 64)
            trainer = Trainer(self.c, TokenCorpus(path / "data/manifest.json", self.c), "cpu")
            for stage in ("identity", "rematched"):
                if stage == "rematched":
                    trainer.train_step()
                    # Exercise a real nonidentity Sinkhorn projection for expert 0.
                    with torch.no_grad():
                        for group in trainer.manager.groups:
                            state = trainer.manager.states[group.name]
                            target_p = state.permutation.clone()
                            target_p[0] = target_p[0].roll(1)
                            group.parameter.copy_(native_shared(state.shared, target_p, state.channel_axis)
                                                  + state.residual - state.dual)
                    trainer.train_step()
                checkpoint = save_checkpoint(trainer, path / stage / "checkpoints")
                restored = Trainer(self.c, trainer.corpus, "cpu", initialize_auxiliary=False)
                load_checkpoint(restored, checkpoint)
                assert_nested_equal(self, trainer.manager.local_state_dict(), restored.manager.local_state_dict())
                artifact_path = path / (stage + ".pt")
                self.assertEqual(export_checkpoint(checkpoint, artifact_path)["format"], "salaad_moe.export.v5")
                reconstructed, _ = load_evaluation_model(checkpoint, "reconstructed")
                exported, _ = load_evaluation_model(artifact_path, "exported")
                assert_nested_equal(self, reconstructed.state_dict(), exported.state_dict())
                artifact = load_torch(artifact_path)
                metrics = trainer.manager.metrics()
                logged = flatten_metrics({"step": trainer.step, "salaad": metrics})
                for name, state in trainer.manager.states.items():
                    self.assertEqual(set(state.state_dict()), {"shared", "residual", "dual", "permutation", "channel_axis"})
                    validate_permutation(state.permutation, 8, 24)
                    expected_count = 24 if stage == "rematched" else 0
                    self.assertEqual((state.permutation[0] != torch.arange(24)).sum().item(), expected_count)
                    self.assertEqual(artifact["groups"][name]["permutation_convention"], "native_to_shared")
                    torch.testing.assert_close(exported.state_dict()[name], state.reconstruction(), rtol=0, atol=0)
                    if name.endswith(".gate"):
                        layer = name.split(".")[1]
                        self.assertEqual(metrics[name + ".expert_0"]["permutation_nonidentity_channels"], expected_count)
                        self.assertEqual(logged[f"salaad_structure/layer_{layer}_gate_expert_0_permutation_nonidentity_fraction"],
                                         expected_count / 24)
            bad = copy.deepcopy(artifact)
            bad["groups"]["layers.0.moe.experts.up"]["permutation"][0].zero_()
            atomic_save(bad, path / "bad.pt")
            with self.assertRaises(ValueError):
                load_evaluation_model(path / "bad.pt", "exported")


if __name__ == "__main__":
    unittest.main()
