import unittest

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from models.Llama import LlamaForCausalLM
from salad.loop import (
    DEFAULT_TIED_PARAMETER_NAMES,
    LoopSampler,
    LoopStabilitySampler,
    block_distance,
    block_parameter_errors,
    get_block_reference_norms,
    monotonic_stability_loss,
)


def _tiny_looped_model(num_loops=2):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        max_position_embeddings=32,
        use_cache=False,
        loop={
            "entry_layers": [0],
            "loop_layers": [1, 2, 3],
            "exit_layers": [4],
            "num_loops": num_loops,
        },
    )
    return LlamaForCausalLM(config)


class LoopedLlamaTest(unittest.TestCase):
    def test_loop_distribution_has_expected_value_two(self):
        sampler = LoopSampler(
            values=[1, 2, 3, 4],
            probabilities=[0.4, 0.3, 0.2, 0.1],
            seed=42,
            expected_value=2.0,
        )
        self.assertAlmostEqual(sampler.expected_value, 2.0)
        self.assertTrue(all(sampler.sample() in {1, 2, 3, 4} for _ in range(100)))

    def test_num_loops_is_configurable(self):
        model = _tiny_looped_model()
        for num_loops in (1, 2, 3, 4):
            model.model.set_num_loops(num_loops)
            self.assertEqual(
                model.model.layer_order,
                (0,) + (1, 2, 3) * num_loops + (4,),
            )

    def test_stability_sampler_returns_configured_positive_deltas(self):
        sampler = LoopStabilitySampler(
            probability=1.0,
            deltas=[1, 2, 4],
            seed=42,
        )
        self.assertTrue(all(sampler.sample() in {1, 2, 4} for _ in range(100)))

        disabled_sampler = LoopStabilitySampler(
            probability=0.0,
            deltas=[1],
            seed=42,
        )
        self.assertIsNone(disabled_sampler.sample())

    def test_monotonic_loss_only_updates_a_worse_long_path(self):
        short_loss = torch.tensor(1.0, requires_grad=True)
        long_loss = torch.tensor(1.5, requires_grad=True)
        stability_loss = monotonic_stability_loss(short_loss, long_loss)
        stability_loss.backward()

        self.assertEqual(stability_loss.item(), 0.5)
        self.assertIsNone(short_loss.grad)
        self.assertEqual(long_loss.grad.item(), 1.0)

        better_long_loss = torch.tensor(0.5, requires_grad=True)
        zero_loss = monotonic_stability_loss(short_loss, better_long_loss)
        zero_loss.backward()
        self.assertEqual(zero_loss.item(), 0.0)
        self.assertEqual(better_long_loss.grad.item(), 0.0)

    def test_middle_blocks_are_reused(self):
        model = _tiny_looped_model(num_loops=2)
        counts = [0] * len(model.model.layers)
        handles = []

        for index, layer in enumerate(model.model.layers):
            def count_call(_module, _inputs, _outputs, index=index):
                counts[index] += 1

            handles.append(layer.register_forward_hook(count_call))

        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
        model(input_ids=input_ids, labels=input_ids)
        for handle in handles:
            handle.remove()

        self.assertEqual(model.model.layer_order, (0, 1, 2, 3, 1, 2, 3, 4))
        self.assertEqual(counts, [1, 2, 2, 2, 1])

    def test_soft_tie_is_name_aligned_and_differentiable(self):
        model = _tiny_looped_model()
        reference_norms = get_block_reference_norms(
            model, "layers.0", DEFAULT_TIED_PARAMETER_NAMES
        )
        distance = block_distance(
            model,
            "layers.0",
            "layers.3",
            DEFAULT_TIED_PARAMETER_NAMES,
            reference_norms=reference_norms,
        )
        distance.backward()

        source = model.model.layers[0].self_attn.q_proj.weight
        target = model.model.layers[3].self_attn.q_proj.weight
        self.assertTrue(torch.isfinite(distance))
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        self.assertGreater(source.grad.norm().item(), 0.0)
        self.assertGreater(target.grad.norm().item(), 0.0)

    def test_parameter_errors_are_reported_separately(self):
        model = _tiny_looped_model()
        reference_norms = get_block_reference_norms(
            model, "layers.0", DEFAULT_TIED_PARAMETER_NAMES
        )
        errors = block_parameter_errors(
            model,
            "layers.0",
            "layers.3",
            DEFAULT_TIED_PARAMETER_NAMES,
            reference_norms=reference_norms,
        )

        self.assertEqual(tuple(errors), DEFAULT_TIED_PARAMETER_NAMES)
        self.assertEqual(len(errors), 7)
        self.assertTrue(all(error.ndim == 0 for error in errors.values()))
        mean_error = torch.stack(tuple(errors.values())).mean()
        self.assertTrue(
            torch.allclose(
                mean_error,
                block_distance(
                    model,
                    "layers.0",
                    "layers.3",
                    DEFAULT_TIED_PARAMETER_NAMES,
                    reference_norms=reference_norms,
                ),
            )
        )

        parameter_name = "self_attn.q_proj.weight"
        source = model.model.layers[0].self_attn.q_proj.weight
        target = model.model.layers[3].self_attn.q_proj.weight
        expected = (
            (source.float() - target.float()).square().sum()
            / reference_norms[0]
        )
        self.assertTrue(torch.allclose(errors[parameter_name], expected))

    def test_identical_blocks_have_zero_distance(self):
        model = _tiny_looped_model()
        model.model.layers[3].load_state_dict(model.model.layers[0].state_dict())
        distance = block_distance(
            model,
            "layers.0",
            "layers.3",
            DEFAULT_TIED_PARAMETER_NAMES,
        )
        self.assertEqual(distance.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
