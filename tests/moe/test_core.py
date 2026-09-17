"""Numerical contracts for routing, consensus ADMM and expert weight views."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from salaad_moe.config import learning_rate, load_config, parameter_counts, validate_config
from salaad_moe.groups import GroupedFusedGroup, SequentialFusedGroup, StackedGroup, native_groups
from salaad_moe.model import MoELanguageModel, route
from salaad_moe.solver import (
    ConsensusManager,
    initial_state,
    linear_mass_rank,
    soft_threshold,
    streaming_svd_step,
    structure_sweep,
    svt,
)

ROOT = Path(__file__).resolve().parents[2]


def smoke_config():
    return load_config(ROOT / "configs/smoke.yaml")


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.c = smoke_config()

    def test_architecture_counts_match_the_plan(self):
        for name, total, tokens in (
            ("ns97m", 97194240, 4404019200),
            ("ns480m", 479736320, 7600078848),
        ):
            c = load_config(ROOT / f"configs/{name}.yaml")
            validate_config(c)
            self.assertEqual(parameter_counts(c)["total_parameters"], total)
            self.assertEqual(parameter_counts(c)["prediction_tokens"], tokens)
        model = MoELanguageModel(self.c)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()), parameter_counts(self.c)["total_parameters"]
        )
        self.assertEqual(len(native_groups(model)), 6)
        self.assertFalse(any("shared" in n for n, _ in model.named_parameters()))

    def test_reject_incompatible_architectures_and_batch(self):
        for group, key, value in (
            ("model", "num_shared_experts", 1),
            ("model", "router_pre_softmax", True),
            ("parallel", "expert_parallel_size", 2),
            ("training", "global_batch_sequences", 7),
            ("salaad", "rho", 0),
        ):
            c = copy.deepcopy(self.c)
            c[group][key] = value
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_wsd_uses_full_budget_and_reaches_minimum(self):
        c = load_config(ROOT / "configs/ns97m.yaml")
        self.assertAlmostEqual(learning_rate(c, 1), 3e-4 / 21)
        self.assertEqual(learning_rate(c, 21), 3e-4)
        self.assertEqual(learning_rate(c, 500), 3e-4)
        self.assertEqual(learning_rate(c, 1890), 3e-4)
        self.assertAlmostEqual(learning_rate(c, 2100), 3e-5)

    def test_router_selected_softmax_and_full_probability_balance(self):
        logits = torch.randn(11, 8, requires_grad=True)
        w, ids, balance, z, _ = route(logits, 2)
        full = logits.softmax(-1)
        selected = full.gather(-1, ids)
        torch.testing.assert_close(w, selected / selected.sum(-1, keepdim=True))
        torch.testing.assert_close(w.sum(-1), torch.ones(11))
        counts = torch.bincount(ids.flatten(), minlength=8) / 22
        expected = 8 * (counts * full.mean(0)).sum()
        torch.testing.assert_close(balance, expected)
        torch.testing.assert_close(z, logits.logsumexp(-1).square().mean())
        torch.testing.assert_close(
            torch.autograd.grad(balance, logits, retain_graph=True)[0],
            torch.autograd.grad(expected, logits)[0],
        )
        self.assertEqual(route(torch.zeros(10, 8), 2)[2].item(), 1.0)

    def test_attention_is_causal_and_labels_shift_only_once(self):
        model = MoELanguageModel(self.c).eval()
        ids = torch.randint(128, (2, 9))
        changed = ids.clone()
        changed[:, 6:] = torch.randint(128, (2, 3))
        first, second = model(ids), model(changed)
        torch.testing.assert_close(first.logits[:, :6], second.logits[:, :6])
        labels = torch.randint(128, ids.shape)
        output = model(ids, labels)
        torch.testing.assert_close(
            output.lm_loss,
            torch.nn.functional.cross_entropy(output.logits.flatten(0, 1), labels.flatten()),
        )

    def test_sequential_dispatch_matches_explicit_all_expert_formula(self):
        model = MoELanguageModel(self.c)
        moe = model.layers[0].moe
        x = torch.randn(2, 3, 32)
        actual = moe(x)[0]
        flat = x.flatten(0, 1)
        weights, ids, *_ = route(moe.router(flat), 2)
        expected = torch.zeros_like(flat)
        for i in range(8):
            expert = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(flat, moe.experts.gate[i]))
                * torch.nn.functional.linear(flat, moe.experts.up[i]),
                moe.experts.down[i],
            )
            selected_weight = ((ids == i) * weights).sum(-1)
            expected += expert * selected_weight[:, None]
        torch.testing.assert_close(actual, expected.view_as(x))
        grad_a = torch.autograd.grad(actual.square().sum(), moe.experts.gate, retain_graph=True)[0]
        grad_b = torch.autograd.grad(expected.square().sum(), moe.experts.gate)[0]
        torch.testing.assert_close(grad_a, grad_b)

    def test_linear_mass_rank_is_not_squared_energy(self):
        self.assertEqual(linear_mass_rank(torch.tensor([[3.0, 1.0]]), 0.8).item(), 1.0)
        self.assertEqual(linear_mass_rank(torch.tensor([[1.0, 3.0]]), 0.7).item(), 0.5)
        self.assertEqual(linear_mass_rank(torch.zeros(2, 3), 0.999).sum().item(), 0.0)
        self.assertEqual(
            linear_mass_rank(torch.tensor([[3.0, 0, 0]]), 1.0).item(), torch.tensor(1 / 3).item()
        )

    def test_streaming_svd_converges_for_tall_wide_and_rank_deficient_matrices(self):
        for rows, columns in ((7, 4), (4, 7), (4, 4)):
            with self.subTest(shape=(rows, columns)):
                k = min(rows, columns)
                left, _ = torch.linalg.qr(torch.randn(3, rows, k))
                right, _ = torch.linalg.qr(torch.randn(3, columns, k))
                expected = torch.tensor([[5.0, 2.0, 1.0, 0.25], [5.0, 2.0, 0, 0], [0, 0, 0, 0]])
                x = (left * expected[:, None, :]) @ right.mT
                basis = torch.eye(k).repeat(3, 1, 1)
                for _ in range(40):
                    u, sigma, vh, basis = streaming_svd_step(x, basis)
                torch.testing.assert_close(
                    sigma.sort(descending=True).values, expected, rtol=1e-4, atol=1e-5
                )
                torch.testing.assert_close((u * sigma[:, None, :]) @ vh, x, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(basis.mT @ basis, torch.eye(k).repeat(3, 1, 1))
                self.assertTrue(torch.isfinite(u).all() and torch.isfinite(vh).all())

    def test_streaming_svt_preserves_input_basis_and_matches_chunked_updates(self):
        x = torch.randn(5, 4, 7)
        basis = torch.eye(4).repeat(5, 1, 1)
        previous = basis.clone()
        thresholds = torch.tensor([0.0, 0.2, -0.1, 0.5, 0.1])
        chunked = svt(x, thresholds, basis, chunk_size=2)
        single = [svt(x[i:i + 1], thresholds[i:i + 1], basis[i:i + 1]) for i in range(len(x))]
        for index, value in enumerate(chunked):
            torch.testing.assert_close(value, torch.cat([item[index] for item in single]))
        torch.testing.assert_close(basis, previous, rtol=0, atol=0)
        self.assertFalse(torch.equal(chunked[2], previous))
        # A complete orthogonal basis reconstructs X exactly at zero threshold.
        torch.testing.assert_close(chunked[0][0], x[0])

    def test_threshold_operators_with_an_aligned_streaming_basis(self):
        x = torch.stack(
            (torch.diag(torch.tensor([5.0, 2.0, 0.5])), torch.diag(torch.tensor([4.0, 1.0, 0.0])))
        )
        basis = torch.eye(3).repeat(2, 1, 1)
        low, sigma, _ = svt(x, torch.tensor([1.0, 2.0]), basis, chunk_size=1)
        expected = torch.stack(
            (torch.diag(torch.tensor([4.0, 1.0, 0.0])), torch.diag(torch.tensor([2.0, 0.0, 0.0])))
        )
        torch.testing.assert_close(low, expected)
        torch.testing.assert_close(sigma, torch.tensor([[4.0, 1.0, 0.0], [2.0, 0.0, 0.0]]))
        torch.testing.assert_close(
            soft_threshold(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), 1),
            torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0]),
        )
        # At a converged basis, streaming thresholding equals the nuclear proximal operator.
        self.assertLessEqual(torch.linalg.matrix_norm(x[0] - low[0], ord=2).item(), 1.00001)

        # Negative thresholds expand values while the operators keep their clamping rules.
        expanded, spectrum, _ = svt(
            torch.diag(torch.tensor([2.0, 1.0])).unsqueeze(0), torch.tensor([-0.5]),
            torch.eye(2).unsqueeze(0),
        )
        torch.testing.assert_close(expanded[0], torch.diag(torch.tensor([2.5, 1.5])))
        torch.testing.assert_close(spectrum, torch.tensor([[2.5, 1.5]]))
        torch.testing.assert_close(
            soft_threshold(torch.tensor([-0.2, 0.0, 0.2]), -0.1),
            torch.tensor([-0.3, 0.0, 0.3]),
        )

    def test_initialization_is_exact_and_expert_specific(self):
        x = torch.randn(4, 5, 3)
        state = initial_state(x, self.c)
        torch.testing.assert_close(state.reconstruction(), x)
        torch.testing.assert_close(state.shared, x.mean(0))
        self.assertEqual(state.low_rank.count_nonzero(), 0)
        self.assertEqual(state.dual.count_nonzero(), 0)
        self.assertEqual(state.tau_s.shape, (4,))
        self.assertEqual(state.svd_basis.shape, (4, 3, 3))
        torch.testing.assert_close(state.svd_basis.mT @ state.svd_basis, torch.eye(3).repeat(4, 1, 1))

    def test_one_sweep_matches_exact_svt_with_a_converged_basis(self):
        x = torch.randn(4, 5, 3)
        old = initial_state(x, self.c)
        old.low_rank = torch.randn_like(x) * 0.1
        old.sparse = torch.randn_like(x) * 0.1
        old.dual = torch.randn_like(x) * 0.05
        old.tau_l[:] = 0.2
        old.tau_s[:] = 0.1
        shared = (x - old.low_rank - old.sparse + old.dual).mean(0)
        residual = x - shared - old.sparse + old.dual
        _, _, vh = torch.linalg.svd(residual, full_matrices=False)
        old.svd_basis = vh.mT.contiguous()
        new = structure_sweep(x, old, self.c)
        low = []
        for i in range(len(x)):
            u, singular, vh = torch.linalg.svd(
                x[i] - shared - old.sparse[i] + old.dual[i], full_matrices=False
            )
            low.append((u * (singular - 0.2).clamp_min(0)) @ vh)
        low = torch.stack(low)
        sparse = soft_threshold(x - shared - low + old.dual, 0.1)
        torch.testing.assert_close(new.shared, shared)
        torch.testing.assert_close(new.low_rank, low)
        torch.testing.assert_close(new.sparse, sparse)
        torch.testing.assert_close(new.dual, old.dual + x - shared - low - sparse)
        torch.testing.assert_close(new.anchor(), shared + low + sparse - new.dual)

    def test_fixed_targets_allow_signed_overshoot_and_update_through_training_end(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 3, 3))
        manager = ConsensusManager([StackedGroup("g", parameter)], self.c)
        manager.initialize(0)
        state = manager.states["g"]
        state.tau_l.fill_(0.001)
        state.tau_s.fill_(0.00001)
        updates = []
        end = self.c["training"]["total_optimizer_steps"]
        for step in range(1, end + 1):
            before = manager.states["g"]
            if manager.after_step(step, final=step == end):
                updates.append(step)
                new = manager.states["g"]
                # Even the last sweeps must keep adjusting both thresholds.
                self.assertTrue((new.tau_l != before.tau_l).all())
                self.assertTrue((new.tau_s < before.tau_s).all())
                if len(updates) == 1:
                    # Fixed targets apply immediately; both thresholds cross zero.
                    torch.testing.assert_close(new.tau_l, torch.full((2,), -0.002))
                    torch.testing.assert_close(new.tau_s, torch.full((2,), -0.00004))
        self.assertEqual(updates, [2, 4, 6, 8])

    def test_independent_and_fixed_mean_ablation(self):
        x = torch.randn(3, 5, 4)
        self.c["salaad"]["shared_mode"] = "none"
        state = initial_state(x, self.c)
        torch.testing.assert_close(state.sparse, x)
        self.assertEqual(structure_sweep(x + 0.1, state, self.c).shared.count_nonzero(), 0)
        self.c["salaad"]["shared_mode"] = "fixed"
        state = initial_state(x, self.c)
        torch.testing.assert_close(structure_sweep(x + 0.1, state, self.c).shared, state.shared)

    def test_constraint_gradient_is_a_sum_and_includes_unused_experts(self):
        p = torch.nn.Parameter(torch.randn(8, 3, 4))
        manager = ConsensusManager([StackedGroup("g", p)], self.c)
        manager.initialize(1)
        manager.anchors["g"] = torch.randn_like(p)
        task = p[:2].square().mean()  # six unvisited experts
        full = task + 0.5 * self.c["salaad"]["rho"] * (p - manager.anchors["g"]).square().sum()
        expected = torch.autograd.grad(full, p, retain_graph=True)[0]
        task.backward()
        manager.inject_gradients()
        torch.testing.assert_close(p.grad, expected)
        self.assertGreater(p.grad[2:].norm().item(), 0)

    def test_structure_failure_leaves_all_states_and_anchors_unchanged(self):
        p = torch.nn.Parameter(torch.randn(3, 4, 5))
        manager = ConsensusManager([StackedGroup("g", p)], self.c)
        manager.initialize(1)
        before = manager.local_state_dict()
        anchor = manager.anchors["g"].clone()
        with patch("salaad_moe.solver.streaming_svd_step", side_effect=RuntimeError("injected SVD failure")):
            with self.assertRaisesRegex(RuntimeError, "no state committed"):
                manager.update(3)
        torch.testing.assert_close(manager.anchors["g"], anchor, rtol=0, atol=0)
        self.assertEqual(manager.sweeps, 0)
        for key, value in before["states"]["g"].items():
            torch.testing.assert_close(
                manager.local_state_dict()["states"]["g"][key], value, rtol=0, atol=0
            )

    def test_final_flush_and_restore(self):
        # End between scheduled updates to exercise the final structure flush.
        self.c["training"]["total_optimizer_steps"] = 7
        p = torch.nn.Parameter(torch.randn(3, 4, 5))
        manager = ConsensusManager([StackedGroup("g", p)], self.c)
        manager.initialize(0)
        updated_steps = [
            step for step in range(1, 8) if manager.after_step(step, final=step == 7)
        ]
        self.assertEqual(updated_steps, [2, 4, 6, 7])
        self.assertEqual(manager.sweeps, 4)
        fresh = ConsensusManager([StackedGroup("g", p)], self.c)
        fresh.load_shards([manager.local_state_dict()])
        torch.testing.assert_close(fresh.anchors["g"], manager.anchors["g"], rtol=0, atol=0)

    def test_fused_sequential_views_use_fp32_master_not_bf16_copy(self):
        parameters = []
        values = torch.randn(3, 2, 4, 5)
        for i in range(3):
            p = torch.nn.Parameter(torch.zeros(8, 5, dtype=torch.bfloat16))
            p.main_param = values[i].reshape(8, 5).clone()
            parameters.append(p)
        gate = SequentialFusedGroup("gate", parameters, "gate")
        up = SequentialFusedGroup("up", parameters, "up")
        torch.testing.assert_close(gate.weight(), values[:, 0])
        torch.testing.assert_close(up.weight(), values[:, 1])
        gate.add_gradient(torch.ones_like(gate.weight()))
        up.add_gradient(2 * torch.ones_like(up.weight()))
        for p in parameters:
            torch.testing.assert_close(p.main_param.grad[:4], torch.ones(4, 5))
            torch.testing.assert_close(p.main_param.grad[4:], torch.full((4, 5), 2.0))
            self.assertIsNone(p.grad)

    def test_fused_grouped_views_and_transposed_gradients(self):
        k, d, f = 3, 5, 4
        logical = torch.randn(k, d, 2 * f)
        p = torch.nn.Parameter(logical.reshape(d, k * 2 * f).clone())
        gate = GroupedFusedGroup("gate", p, "gate", k, d)
        up = GroupedFusedGroup("up", p, "up", k, d)
        torch.testing.assert_close(gate.weight(), logical[:, :, :f].transpose(1, 2))
        torch.testing.assert_close(up.weight(), logical[:, :, f:].transpose(1, 2))
        delta = torch.randn_like(gate.weight())
        gate.add_gradient(delta)
        torch.testing.assert_close(p.grad.view_as(logical)[:, :, :f], delta.transpose(1, 2))
        self.assertEqual(p.grad.view_as(logical)[:, :, f:].count_nonzero(), 0)
        down_p = torch.nn.Parameter(torch.randn(k * f, d))
        down = GroupedFusedGroup("down", down_p, "down", k, d)
        torch.testing.assert_close(down.weight(), down_p.view(k, f, d).transpose(1, 2))


if __name__ == "__main__":
    unittest.main()
