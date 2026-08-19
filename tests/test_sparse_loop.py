import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from models.Llama import LlamaForCausalLM
from models.sparse_loop import SparseLoopLinear
from salad.sparse_loop import SparseLoopADMM, apply_sparse_residuals
from scripts.evaluate_consensus_c4 import (
    _apply_sparse_loop_reconstruction,
    _copy_sparse_loop_effective_weights,
)
from scripts.train_salad import _write_effective_model_config


def tiny_sparse_loop_config(max_num_loops=3):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=8,
        num_attention_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
        use_cache=False,
    )
    config.loop = {
        "num_entry_blocks": 1,
        "num_blocks_per_loop": 2,
        "num_exit_blocks": 1,
        "num_loops": 2,
        "max_num_loops": max_num_loops,
    }
    config.specific_sparsity = {
        "target_modules": ["self_attn.q_proj"],
    }
    return config


def sparse_solver_config(
    *,
    rho=1.0,
    rate_sparsity=0.5,
    beta_init=0.0,
    beta_rate_decay=0.2,
):
    return {
        "rho": rho,
        "rate_sparsity": rate_sparsity,
        "beta_dict": {
            "init": beta_init,
            "mode": "adaptive",
            "rate_decay": beta_rate_decay,
            "drate": 0.01,
            "start_epoch": 1500,
        },
    }


class SparseLoopTests(unittest.TestCase):
    def test_linear_starts_from_shared_function_and_selects_residual(self):
        dense = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            dense.weight.copy_(torch.tensor([[1.0, 2.0]]))
        linear = SparseLoopLinear.from_linear(dense, num_loops=2)
        with torch.no_grad():
            linear.specific_weight[1].copy_(torch.tensor([[3.0, 4.0]]))

        inputs = torch.tensor([[2.0, 1.0]])
        torch.testing.assert_close(linear(inputs, 0), torch.tensor([[4.0]]))
        torch.testing.assert_close(linear(inputs, 1), torch.tensor([[14.0]]))

    def test_llama_adds_specific_residuals_only_inside_recurrent_body(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(tiny_sparse_loop_config())
        decoder = model.model

        self.assertEqual(
            decoder.execution_plan,
            (
                (0, None),
                (1, 0),
                (2, 0),
                (1, 1),
                (2, 1),
                (3, None),
            ),
        )
        self.assertIsInstance(decoder.layers[0].self_attn.q_proj, nn.Linear)
        self.assertIsInstance(decoder.layers[3].self_attn.q_proj, nn.Linear)
        q_proj = decoder.layers[1].self_attn.q_proj
        self.assertIsInstance(q_proj, SparseLoopLinear)
        self.assertEqual(q_proj.weight.shape, (8, 8))
        self.assertEqual(q_proj.specific_weight.shape, (3, 8, 8))

        input_ids = torch.randint(1, 32, (2, 6))
        model(input_ids=input_ids, labels=input_ids).loss.backward()

        self.assertGreater(int(torch.count_nonzero(q_proj.weight.grad)), 0)
        self.assertGreater(
            int(torch.count_nonzero(q_proj.specific_weight.grad[0])), 0
        )
        self.assertGreater(
            int(torch.count_nonzero(q_proj.specific_weight.grad[1])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(q_proj.specific_weight.grad[2])), 0
        )

    def test_model_rejects_consensus_and_specific_sparsity_together(self):
        config = tiny_sparse_loop_config()
        config.consensus_salaad = {
            "target_modules": ["self_attn.q_proj"],
        }
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LlamaForCausalLM(config)

    def test_admm_penalty_does_not_constrain_shared_weight(self):
        linear = SparseLoopLinear(3, 2, num_loops=2, bias=False)
        with torch.no_grad():
            linear.weight.fill_(2.0)
            linear.specific_weight.copy_(torch.tensor([
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]))
        solver = SparseLoopADMM(
            "linear", linear, sparse_solver_config(rho=2.0)
        )

        penalty = solver.penalty()
        expected = linear.specific_weight.float().square().mean()
        torch.testing.assert_close(penalty, expected)
        penalty.backward()

        self.assertIsNone(linear.weight.grad)
        self.assertIsNotNone(linear.specific_weight.grad)

    def test_admm_dynamically_thresholds_each_loop_and_updates_scaled_dual(self):
        linear = SparseLoopLinear(3, 2, num_loops=2, bias=False)
        residual = torch.tensor([
            [[1.0, 6.0, 2.0], [5.0, 3.0, 4.0]],
            [[12.0, 7.0, 11.0], [8.0, 10.0, 9.0]],
        ])
        with torch.no_grad():
            linear.specific_weight.copy_(residual)
        solver = SparseLoopADMM(
            "linear",
            linear,
            sparse_solver_config(beta_init=3.5),
        )

        solver.step()

        self.assertEqual(int(torch.count_nonzero(solver.sparse[0])), 3)
        self.assertEqual(int(torch.count_nonzero(solver.sparse[1])), 6)
        torch.testing.assert_close(
            solver.sparse[0],
            torch.tensor([[0.0, 2.5, 0.0], [1.5, 0.0, 0.5]]),
        )
        torch.testing.assert_close(
            solver.sparse[1],
            torch.tensor([[8.5, 3.5, 7.5], [4.5, 6.5, 5.5]]),
        )
        torch.testing.assert_close(
            solver.scaled_dual, residual - solver.sparse
        )
        self.assertAlmostEqual(solver.beta_controllers[0].value, 3.5)
        self.assertAlmostEqual(solver.beta_controllers[1].value, 3.6)

    def test_saved_sparse_state_can_be_materialized_for_evaluation(self):
        model = nn.Sequential(SparseLoopLinear(2, 1, num_loops=2, bias=False))
        solver = SparseLoopADMM(
            "0", model[0], sparse_solver_config()
        )
        with torch.no_grad():
            solver.sparse.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]]))
        state = solver.state_dict()

        restored = SparseLoopADMM(
            "0",
            model[0],
            sparse_solver_config(
                rho=3.0,
                rate_sparsity=0.25,
                beta_init=1.0,
            ),
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.sparse, solver.sparse)
        torch.testing.assert_close(restored.scaled_dual, solver.scaled_dual)

        apply_sparse_residuals(model, {"0": state})
        torch.testing.assert_close(model[0].specific_weight, solver.sparse)

    def test_evaluator_compares_dense_and_sparse_effective_weights(self):
        model = nn.Sequential(SparseLoopLinear(2, 1, num_loops=2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
            model[0].specific_weight.copy_(
                torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
            )
        states = {
            "0": {
                "sparse": torch.tensor([[[0.0, 1.0]], [[2.0, 0.0]]]),
                "rate_sparsity": 0.25,
            }
        }

        reference = _copy_sparse_loop_effective_weights(model, states)
        relative_error, actual_density, target_density = (
            _apply_sparse_loop_reconstruction(model, states, reference)
        )

        self.assertAlmostEqual(relative_error, (10.0 / 25.0) ** 0.5)
        self.assertEqual(actual_density, 0.5)
        self.assertEqual(target_density, 0.25)
        torch.testing.assert_close(
            model[0].specific_weight,
            states["0"]["sparse"],
        )

        dense_states = {
            "0": {
                "sparse": torch.tensor([[[1.0, 4.0]], [[3.0, 2.0]]]),
                "rate_sparsity": 0.25,
            }
        }
        relative_error, actual_density, target_density = (
            _apply_sparse_loop_reconstruction(
                model,
                dense_states,
                reference,
                sparse_density=0.5,
            )
        )
        self.assertGreater(relative_error, 0.0)
        self.assertEqual(actual_density, 0.5)
        self.assertEqual(target_density, 0.5)
        torch.testing.assert_close(
            model[0].specific_weight,
            torch.tensor([[[0.0, 4.0]], [[3.0, 0.0]]]),
        )

    def test_training_yaml_builds_effective_sparse_loop_model_config(self):
        source = {
            "model_type": "llama",
            "num_hidden_layers": 8,
            "use_cache": True,
            "consensus_salaad": {"target_modules": ["self_attn.k_proj"]},
        }
        training = {
            "training_mode": "sparse_loop",
            "loop": {
                "num_entry_blocks": 1,
                "num_blocks_per_loop": 2,
                "num_exit_blocks": 1,
                "num_loops": 2,
            },
            "specific_sparsity": {
                "rho": 1.0,
                "rate_sparsity": 0.1,
                "beta_dict": {
                    "init": 0.0,
                    "mode": "adaptive",
                    "rate_decay": 0.002,
                    "drate": 0.01,
                    "start_epoch": 1500,
                },
                "target_modules": ["self_attn.q_proj"],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "effective.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            _write_effective_model_config(
                str(source_path), str(output_path), training
            )
            effective = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(effective["num_hidden_layers"], 4)
        self.assertEqual(effective["loop"]["num_loops"], 2)
        self.assertEqual(effective["loop"]["max_num_loops"], 2)
        self.assertEqual(
            effective["specific_sparsity"],
            {"target_modules": ["self_attn.q_proj"]},
        )
        self.assertFalse(effective["use_cache"])
        self.assertNotIn("consensus_salaad", effective)


if __name__ == "__main__":
    unittest.main()
