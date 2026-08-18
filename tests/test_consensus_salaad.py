import unittest

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from models.Llama import LlamaForCausalLM
from models.consensus import ConsensusLinear
from salad.consensus import (
    ConsensusADMM,
    apply_decomposition,
    compose_decomposition,
    singular_value_threshold,
    soft_threshold,
)
from salad.loop import LoopSampler


def tiny_config(num_loop_weights=3):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        # The count-based loop protocol overrides this default automatically.
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
        use_cache=False,
    )
    config.loop = {
        "num_entry_blocks": 1,
        "num_blocks_per_loop": 3,
        "num_exit_blocks": 1,
        "num_loops": 2,
        "max_num_loops": num_loop_weights,
    }
    config.consensus_salaad = {
        "target_modules": ["self_attn.q_proj"],
    }
    return config


def consensus_solver_config(
    *,
    components=None,
    rho=1.0,
    energy=0.999,
    rate_rank=0.15,
    rate_sparsity=0.05,
    alpha_init=0.0,
    beta_init=0.0,
    alpha_rate_decay=0.0,
    beta_rate_decay=0.0,
):
    config = {
        "rho": rho,
        "energy": energy,
        "rate_rank": rate_rank,
        "rate_sparsity": rate_sparsity,
        "alpha_dict": {
            "init": alpha_init,
            "mode": "adaptive",
            "rate_decay": alpha_rate_decay,
            "drate": 0.01,
            "start_epoch": 1500,
        },
        "beta_dict": {
            "init": beta_init,
            "mode": "adaptive",
            "rate_decay": beta_rate_decay,
            "drate": 0.01,
            "start_epoch": 1500,
        },
    }
    if components is not None:
        config["components"] = components
    return config


class ConsensusSALAADTests(unittest.TestCase):
    def test_consensus_linear_selects_the_requested_weight(self):
        dense = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            dense.weight.copy_(torch.tensor([[1.0, 2.0]]))
        linear = ConsensusLinear.from_linear(dense, num_loops=2)
        with torch.no_grad():
            linear.weight[1].copy_(torch.tensor([[3.0, 4.0]]))

        inputs = torch.tensor([[2.0, 1.0]])
        torch.testing.assert_close(linear(inputs, 0), torch.tensor([[4.0]]))
        torch.testing.assert_close(linear(inputs, 1), torch.tensor([[10.0]]))

    def test_proximal_operators_use_the_configured_threshold(self):
        values = torch.tensor([-2.0, -0.5, 0.25, 3.0])
        torch.testing.assert_close(
            soft_threshold(values, 1.0),
            torch.tensor([-1.0, 0.0, 0.0, 2.0]),
        )

        matrix = torch.diag(torch.tensor([3.0, 1.0]))
        torch.testing.assert_close(
            singular_value_threshold(matrix, 1.5),
            torch.diag(torch.tensor([1.5, 0.0])),
        )

    def test_consensus_step_uses_mean_and_reconstructs_effective_weights(self):
        effective = nn.Parameter(
            torch.tensor(
                [
                    [[3.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 3.0]],
                ]
            )
        )
        solver = ConsensusADMM(
            "matrix",
            effective,
            consensus_solver_config(),
        )
        with torch.no_grad():
            solver.low_rank[0].fill_(0.25)
            solver.sparse[1].fill_(0.5)
            solver.scaled_dual[0].fill_(0.75)

        expected_shared = (
            effective.detach().float()
            - solver.low_rank
            - solver.sparse
            + solver.scaled_dual
        ).mean(dim=0)
        solver.step()

        torch.testing.assert_close(solver.shared, expected_shared)

        clean_solver = ConsensusADMM(
            "clean_matrix",
            effective,
            consensus_solver_config(),
        )
        clean_solver.step()
        torch.testing.assert_close(
            clean_solver.reconstruction(),
            effective.detach().float(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            clean_solver.residual(),
            torch.zeros_like(effective),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_consensus_uses_independent_integral_controller_state(self):
        effective = nn.Parameter(
            torch.tensor(
                [
                    [[2.0, 0.0], [0.0, 0.0]],
                    [[-2.0, 0.0], [0.0, 0.0]],
                ]
            )
        )
        config = consensus_solver_config(
            rho=2.0,
            energy=1.0,
            rate_rank=0.25,
            rate_sparsity=0.10,
            alpha_rate_decay=0.4,
            beta_rate_decay=0.2,
        )
        solver = ConsensusADMM("matrix", effective, config)

        self.assertIsNot(solver.alpha_controllers[0], solver.alpha_controllers[1])
        self.assertIsNot(solver.beta_controllers[0], solver.beta_controllers[1])

        solver.step()
        for controller in solver.alpha_controllers:
            self.assertAlmostEqual(controller.value, 0.2)
        for controller in solver.beta_controllers:
            self.assertEqual(controller.value, 0.0)

        solver.step()
        for controller in solver.alpha_controllers:
            self.assertAlmostEqual(controller.value, 0.4)
        for controller in solver.beta_controllers:
            self.assertAlmostEqual(controller.value, 0.06)

        state = solver.state_dict()
        restored = ConsensusADMM("matrix", effective, config)
        restored.load_state_dict(state)
        for expected, actual in zip(
            solver.alpha_controllers, restored.alpha_controllers
        ):
            self.assertEqual(actual.nr_updates, expected.nr_updates)
            self.assertAlmostEqual(actual.value, expected.value)
            self.assertAlmostEqual(actual.pre_rate, expected.pre_rate)
        for expected, actual in zip(
            solver.beta_controllers, restored.beta_controllers
        ):
            self.assertEqual(actual.nr_updates, expected.nr_updates)
            self.assertAlmostEqual(actual.value, expected.value)
            self.assertAlmostEqual(actual.pre_rate, expected.pre_rate)

    def test_integral_controller_never_makes_a_negative_threshold(self):
        effective = nn.Parameter(torch.zeros(2, 2, 2))
        solver = ConsensusADMM(
            "matrix",
            effective,
            consensus_solver_config(
                rho=1.0,
                rate_rank=0.5,
                rate_sparsity=0.5,
                alpha_rate_decay=1.0,
                beta_rate_decay=1.0,
            ),
        )

        solver.step()

        self.assertTrue(
            all(controller.value == 0.0 for controller in solver.alpha_controllers)
        )
        self.assertTrue(
            all(controller.value == 0.0 for controller in solver.beta_controllers)
        )

    def test_augmented_penalty_backpropagates_to_effective_weights(self):
        effective = nn.Parameter(torch.zeros(2, 2, 2))
        solver = ConsensusADMM(
            "matrix",
            effective,
            consensus_solver_config(rho=2.0),
        )
        with torch.no_grad():
            effective[1, 0, 0] = 1.0

        solver.penalty().backward()

        self.assertIsNotNone(effective.grad)
        self.assertEqual(effective.grad[1, 0, 0].item(), 2.0)

    def test_shared_only_penalty_backpropagates_to_each_loop_weight(self):
        effective = nn.Parameter(torch.tensor([[[0.0]], [[1.0]]]))
        solver = ConsensusADMM(
            "matrix",
            effective,
            {"components": ["shared"], "rho": 0.5},
        )

        solver.penalty().backward()

        self.assertIsNotNone(effective.grad)
        torch.testing.assert_close(
            effective.grad,
            torch.tensor([[[-0.25]], [[0.25]]]),
        )

    def test_saved_decomposition_can_be_materialized_for_evaluation(self):
        model = nn.Sequential(ConsensusLinear(2, 1, num_loops=2, bias=False))
        states = {
            "0": {
                "shared": torch.tensor([[1.0, 2.0]]),
                "low_rank": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                "sparse": torch.tensor([[[0.0, 3.0]], [[4.0, 0.0]]]),
            }
        }

        apply_decomposition(model, states)

        torch.testing.assert_close(
            model[0].weight,
            torch.tensor([[[2.0, 5.0]], [[5.0, 3.0]]]),
        )

    def test_selected_decomposition_components_can_be_materialized(self):
        model = nn.Sequential(ConsensusLinear(2, 1, num_loops=2, bias=False))
        states = {
            "0": {
                "shared": torch.tensor([[1.0, 2.0]]),
                "low_rank": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                "sparse": torch.tensor([[[0.0, 3.0]], [[4.0, 0.0]]]),
            }
        }
        cases = (
            (("shared",), torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]]])),
            (
                ("shared", "low_rank"),
                torch.tensor([[[2.0, 2.0]], [[1.0, 3.0]]]),
            ),
            (
                ("shared", "sparse"),
                torch.tensor([[[1.0, 5.0]], [[5.0, 2.0]]]),
            ),
        )

        for components, expected in cases:
            apply_decomposition(model, states, components=components)
            torch.testing.assert_close(model[0].weight, expected)

    def test_llama_routes_recurrent_passes_to_their_weight_slices(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(tiny_config())
        loop_model = model.model

        self.assertEqual(
            loop_model.execution_plan,
            (
                (0, None),
                (1, 0),
                (2, 0),
                (3, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, None),
            ),
        )
        q_proj = loop_model.layers[1].self_attn.q_proj
        self.assertIsInstance(q_proj, ConsensusLinear)
        self.assertEqual(q_proj.weight.shape, (3, 8, 8))

        input_ids = torch.randint(1, 32, (2, 6))
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()

        self.assertIsNotNone(q_proj.weight.grad)
        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[0])), 0)
        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[1])), 0)
        self.assertEqual(int(torch.count_nonzero(q_proj.weight.grad[2])), 0)

        loop_model.set_num_loops(3)
        self.assertEqual(len(loop_model.execution_plan), 11)

    def test_shared_only_keeps_loop_weights_and_uses_consensus_admm(self):
        torch.manual_seed(0)
        config = tiny_config()
        config.consensus_salaad["components"] = ["shared"]
        model = LlamaForCausalLM(config)
        loop_model = model.model
        q_proj = loop_model.layers[1].self_attn.q_proj

        self.assertIsInstance(q_proj, ConsensusLinear)
        self.assertEqual(q_proj.weight.shape, (3, 8, 8))
        self.assertEqual(loop_model.consensus_components, ("shared",))
        self.assertEqual(loop_model.max_num_loops, 3)

        solver = ConsensusADMM(
            "q_proj",
            q_proj.weight,
            {"components": ["shared"], "rho": 0.5},
        )
        self.assertIsNone(solver.low_rank)
        self.assertIsNone(solver.sparse)
        self.assertEqual(solver.alpha_controllers, [])
        self.assertEqual(solver.beta_controllers, [])

        with torch.no_grad():
            q_proj.weight[0].add_(1.0)
            q_proj.weight[1].sub_(0.5)
            solver.scaled_dual[2].add_(0.25)
        expected_shared = (
            q_proj.weight.detach().float() + solver.scaled_dual
        ).mean(dim=0)
        solver.step()
        torch.testing.assert_close(solver.shared, expected_shared)
        torch.testing.assert_close(
            solver.reconstruction(),
            solver.shared.unsqueeze(0).expand_as(q_proj.weight),
        )
        state = solver.state_dict()
        self.assertNotIn("low_rank", state)
        self.assertNotIn("sparse", state)
        self.assertEqual(state["components"], ["shared"])
        torch.testing.assert_close(
            compose_decomposition(state),
            solver.shared.unsqueeze(0).expand_as(q_proj.weight),
        )
        restored = ConsensusADMM(
            "q_proj",
            q_proj.weight,
            {"components": ["shared"], "rho": 0.5},
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.shared, solver.shared)
        torch.testing.assert_close(restored.scaled_dual, solver.scaled_dual)

        input_ids = torch.randint(1, 32, (2, 6))
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[0])), 0)
        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[1])), 0)

        loop_model.set_num_loops(3)
        self.assertEqual(len(loop_model.execution_plan), 11)
        with self.assertRaisesRegex(ValueError, "available loop-specific weights"):
            loop_model.set_num_loops(4)

    def test_consensus_components_reject_partial_residual_modes(self):
        config = tiny_config()
        config.consensus_salaad["components"] = ["shared", "low_rank"]
        with self.assertRaisesRegex(ValueError, "must be either"):
            LlamaForCausalLM(config)

    def test_model_rejects_depth_without_a_loop_specific_weight(self):
        model = LlamaForCausalLM(tiny_config(num_loop_weights=2))
        with self.assertRaisesRegex(ValueError, "available loop-specific weights"):
            model.model.set_num_loops(3)

    def test_standard_transformer_is_the_r1_nm_extreme(self):
        config = tiny_config(num_loop_weights=1)
        config.loop.update({
            "num_blocks_per_loop": 4,
            "num_loops": 1,
        })
        model = LlamaForCausalLM(config)
        loop_model = model.model
        consensus_modules = [
            module for module in model.modules()
            if isinstance(module, ConsensusLinear)
        ]

        self.assertEqual(len(loop_model.layers), 6)
        self.assertEqual(loop_model.logical_num_layers, 6)
        self.assertEqual(len(consensus_modules), 4)
        self.assertTrue(all(module.weight.shape[0] == 1 for module in consensus_modules))

    def test_maximum_shared_center_is_the_rm_n1_extreme(self):
        config = tiny_config(num_loop_weights=4)
        config.loop.update({
            "num_blocks_per_loop": 1,
            "num_loops": 4,
        })
        model = LlamaForCausalLM(config)
        loop_model = model.model
        consensus_modules = [
            module for module in model.modules()
            if isinstance(module, ConsensusLinear)
        ]

        self.assertEqual(len(loop_model.layers), 3)
        self.assertEqual(loop_model.logical_num_layers, 6)
        self.assertEqual(len(consensus_modules), 1)
        self.assertEqual(consensus_modules[0].weight.shape, (4, 8, 8))
        self.assertEqual(
            loop_model.execution_plan,
            ((0, None), (1, 0), (1, 1), (1, 2), (1, 3), (2, None)),
        )

    def test_explicit_layer_index_protocol_remains_supported(self):
        config = tiny_config()
        config.loop = {
            "entry_layers": [0],
            "loop_layers": [1, 2, 3],
            "exit_layers": [4],
            "num_loops": 2,
        }
        config.num_hidden_layers = 5
        config.consensus_salaad["num_loop_weights"] = 3

        model = LlamaForCausalLM(config)

        self.assertEqual(len(model.model.layers), 5)
        self.assertEqual(model.model.logical_num_layers, 8)

    def test_gradient_checkpointing_preserves_each_loop_index(self):
        torch.manual_seed(1)
        model = LlamaForCausalLM(tiny_config())
        model.train()
        model.model.gradient_checkpointing = True
        q_proj = model.model.layers[1].self_attn.q_proj

        input_ids = torch.randint(1, 32, (2, 6))
        model(input_ids=input_ids, labels=input_ids).loss.backward()

        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[0])), 0)
        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad[1])), 0)
        self.assertEqual(int(torch.count_nonzero(q_proj.weight.grad[2])), 0)

    def test_task_update_and_consensus_update_run_in_one_training_step(self):
        torch.manual_seed(2)
        model = LlamaForCausalLM(tiny_config())
        solvers = [
            ConsensusADMM(
                name,
                module.weight,
                consensus_solver_config(
                    rho=0.1,
                    alpha_init=1.0e-4,
                    beta_init=1.0e-5,
                ),
            )
            for name, module in model.named_modules()
            if isinstance(module, ConsensusLinear)
        ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        input_ids = torch.randint(1, 32, (2, 6))

        task_loss = model(input_ids=input_ids, labels=input_ids).loss
        penalty = sum((solver.penalty() for solver in solvers), task_loss.new_zeros(()))
        (task_loss + penalty).backward()
        optimizer.step()
        for solver in solvers:
            solver.step()

        self.assertTrue(all(torch.isfinite(solver.shared).all() for solver in solvers))
        self.assertTrue(
            all(torch.isfinite(solver.reconstruction()).all() for solver in solvers)
        )

    def test_loop_sampler_validates_and_reproduces_its_sequence(self):
        first = LoopSampler([1, 2], [0.25, 0.75], seed=7, expected_value=1.75)
        second = LoopSampler([1, 2], [0.25, 0.75], seed=7, expected_value=1.75)
        self.assertEqual(
            [first.sample() for _ in range(20)],
            [second.sample() for _ in range(20)],
        )


if __name__ == "__main__":
    unittest.main()
