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
    contraction_losses,
    get_block_reference_norms,
    hidden_distance,
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

    def test_contraction_loop_distribution_has_expected_value_four(self):
        sampler = LoopSampler(
            values=[3, 4, 5, 6],
            probabilities=[0.4, 0.3, 0.2, 0.1],
            seed=42,
            expected_value=4.0,
        )

        self.assertAlmostEqual(sampler.expected_value, 4.0)
        self.assertTrue(all(sampler.sample() in {3, 4, 5, 6} for _ in range(100)))

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

    def test_loop_states_are_captured_after_complete_loops(self):
        model = _tiny_looped_model(num_loops=3)
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

        with torch.no_grad():
            ordinary = model(input_ids=input_ids, return_dict=True)

        tn_minus_one_outputs = []

        def capture_tn_minus_one(_module, _inputs, outputs):
            tn_minus_one_outputs.append(outputs[0].detach().clone())

        handle = model.model.layers[3].register_forward_hook(
            capture_tn_minus_one
        )
        with torch.no_grad():
            captured = model(
                input_ids=input_ids,
                output_loop_states=True,
                return_dict=True,
            )
        handle.remove()

        self.assertEqual(len(captured.loop_states), 3)
        self.assertEqual(len(tn_minus_one_outputs), 3)
        for loop_state, layer_output in zip(
            captured.loop_states,
            tn_minus_one_outputs,
        ):
            self.assertTrue(torch.allclose(loop_state, layer_output))
        self.assertTrue(torch.allclose(ordinary.logits, captured.logits))

    def test_random_contraction_depths_capture_only_sampled_loops(self):
        model = _tiny_looped_model()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
        attention_mask = torch.ones_like(input_ids)

        for num_loops in (3, 4, 5, 6):
            model.model.set_num_loops(num_loops)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_loop_states=True,
                return_dict=True,
            )
            result = contraction_losses(
                output.loop_states,
                attention_mask,
                start_loop=1,
                gamma=0.9,
            )

            self.assertEqual(len(output.loop_states), num_loops)
            self.assertEqual(
                tuple(result["distances"].shape),
                (input_ids.shape[0], num_loops - 1),
            )
            self.assertEqual(
                tuple(result["ratios"].shape),
                (input_ids.shape[0], num_loops - 2),
            )
            self.assertEqual(result["fixed_point"].item(), 0.0)

    def test_hidden_distance_ignores_padding(self):
        first = torch.zeros(2, 3, 2)
        second = torch.zeros_like(first)
        second[0, 0] = 2.0
        second[0, 2] = 100.0
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        distance = hidden_distance(first, second, mask)

        self.assertTrue(torch.allclose(distance, torch.tensor([2.0**0.5, 0.0])))

    def test_contraction_loss_uses_adjacent_loop_distances(self):
        # Distances are d1=2, d2=1, d3=0.4. With gamma=0.5, both
        # contraction inequalities hold. The d1 hinge above 0.5 is 1.5.
        states = tuple(
            torch.tensor([[[value]]], requires_grad=True)
            for value in (0.0, 2.0, 3.0, 3.4)
        )
        result = contraction_losses(
            states,
            attention_mask=torch.ones(1, 1),
            start_loop=1,
            gamma=0.5,
            fixed_point_epsilon=0.5,
        )

        self.assertAlmostEqual(result["contraction"].item(), 0.0, places=6)
        self.assertAlmostEqual(result["fixed_point"].item(), 2.25, places=6)
        self.assertAlmostEqual(result["violation_rate"].item(), 0.0, places=6)

        result["fixed_point"].backward()
        self.assertIsNotNone(states[0].grad)
        self.assertIsNotNone(states[1].grad)
        self.assertEqual(states[-1].grad.abs().item(), 0.0)

    def test_fixed_point_loss_is_zero_when_disabled(self):
        states = tuple(
            torch.tensor([[[value]]], requires_grad=True)
            for value in (0.0, 2.0, 3.0, 3.4)
        )
        result = contraction_losses(
            states,
            attention_mask=None,
            start_loop=1,
            gamma=0.5,
        )

        self.assertEqual(result["fixed_point"].item(), 0.0)
        self.assertEqual(tuple(result["distances"].shape), (1, 3))

    def test_fixed_point_hinge_is_zero_below_epsilon(self):
        states = tuple(
            torch.tensor([[[value]]], requires_grad=True)
            for value in (0.0, 0.4, 0.6)
        )
        result = contraction_losses(
            states,
            attention_mask=None,
            start_loop=1,
            gamma=0.9,
            fixed_point_epsilon=0.5,
        )

        self.assertEqual(result["fixed_point"].item(), 0.0)

    def test_fixed_point_anchor_backpropagates_through_recurrent_blocks(self):
        model = _tiny_looped_model(num_loops=3)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_loop_states=True,
            return_dict=True,
        )
        result = contraction_losses(
            output.loop_states,
            attention_mask=attention_mask,
            start_loop=1,
            gamma=0.9,
            fixed_point_epsilon=0.0,
        )

        result["fixed_point"].backward()

        recurrent_gradients = [
            parameter.grad
            for layer_index in (1, 2, 3)
            for parameter in model.model.layers[layer_index].parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(recurrent_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in recurrent_gradients)
        )
        self.assertGreater(
            sum(gradient.abs().sum().item() for gradient in recurrent_gradients),
            0.0,
        )

    def test_contraction_reference_distance_is_detached(self):
        states = tuple(
            torch.tensor([[[value]]], requires_grad=True)
            for value in (0.0, 1.0, 3.0)
        )
        result = contraction_losses(
            states,
            attention_mask=None,
            start_loop=1,
            gamma=0.5,
        )

        result["contraction"].backward()

        self.assertEqual(states[0].grad.abs().item(), 0.0)
        self.assertIsNotNone(states[-1].grad)
        self.assertGreater(states[-1].grad.abs().item(), 0.0)

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
