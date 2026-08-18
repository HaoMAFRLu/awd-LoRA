import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from models.Llama import LlamaForCausalLM
from models.consensus import ConsensusLinear
from scripts.train_salad import _write_effective_model_config


def tiny_plain_loop_config(num_loops=3, num_blocks=2):
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
        "num_blocks_per_loop": num_blocks,
        "num_exit_blocks": 1,
        "num_loops": num_loops,
    }
    return config


class PlainLoopTests(unittest.TestCase):
    def test_plain_loop_reuses_standard_linear_layers(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(tiny_plain_loop_config())
        decoder = model.model

        self.assertEqual(len(decoder.layers), 4)
        self.assertEqual(decoder.logical_num_layers, 8)
        self.assertEqual(
            decoder.execution_plan,
            (
                (0, None),
                (1, 0),
                (2, 0),
                (1, 1),
                (2, 1),
                (1, 2),
                (2, 2),
                (3, None),
            ),
        )
        self.assertIsInstance(decoder.layers[1].self_attn.q_proj, nn.Linear)
        self.assertFalse(
            any(isinstance(module, ConsensusLinear) for module in model.modules())
        )

        input_ids = torch.randint(1, 32, (2, 6))
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        gradient = decoder.layers[1].self_attn.q_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(int(torch.count_nonzero(gradient)), 0)

    def test_plain_loop_depth_can_be_changed_without_extra_weights(self):
        model = LlamaForCausalLM(tiny_plain_loop_config(num_loops=2))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        model.model.set_num_loops(5)

        self.assertEqual(model.model.logical_num_layers, 12)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            parameter_count,
        )

    def test_training_yaml_overrides_the_effective_model_structure(self):
        source = {
            "model_type": "llama",
            "num_hidden_layers": 8,
            "use_cache": True,
            "consensus_salaad": {"target_modules": ["self_attn.q_proj"]},
        }
        training = {
            "training_mode": "loop",
            "loop": {
                "num_entry_blocks": 1,
                "num_blocks_per_loop": 3,
                "num_exit_blocks": 1,
                "num_loops": 2,
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

        self.assertEqual(effective["loop"], training["loop"])
        self.assertEqual(effective["num_hidden_layers"], 5)
        self.assertFalse(effective["use_cache"])
        self.assertNotIn("consensus_salaad", effective)


if __name__ == "__main__":
    unittest.main()
