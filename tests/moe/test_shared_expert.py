"""Shared-expert forward/gradient, parameter-budget, and checkpoint contracts."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from salaad_moe.checkpoint import load_checkpoint, save_checkpoint
from salaad_moe.config import load_config, parameter_counts, validate_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import load_evaluation_model
from salaad_moe.megatron import megatron_arguments
from salaad_moe.model import MoELanguageModel, route
from salaad_moe.trainer import Trainer, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def shared_smoke(topk=2):
    config = load_config(ROOT / "configs/smoke.yaml")
    config["model"].update(
        family="llama_style_shared_expert", num_experts=7,
        num_shared_experts=1, router_topk=topk,
    )
    config["salaad"]["enabled"] = False
    config["is_wandb"] = False
    return config


class SharedExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_and_all_gradients_match_explicit_shared_plus_routed_sum(self):
        for topk, shared in ((2, 1), (3, 1), (2, 2)):
            with self.subTest(topk=topk, shared=shared):
                config = shared_smoke(topk)
                config["model"]["num_shared_experts"] = shared
                validate_config(config)
                seed_everything(41)
                moe = MoELanguageModel(config).layers[0].moe
                x = torch.randn(2, 3, 32, requires_grad=True)
                actual, balance, z_loss, stats = moe(x)
                flat = x.flatten(0, 1)
                weights, ids, expected_balance, expected_z, _ = route(moe.router(flat), topk)
                expected = torch.zeros_like(flat)
                for i in range(7):
                    value = F.linear(
                        F.silu(F.linear(flat, moe.experts.gate[i]))
                        * F.linear(flat, moe.experts.up[i]), moe.experts.down[i],
                    )
                    expected = expected + value * ((ids == i) * weights).sum(-1, keepdim=True)
                for i in range(shared):
                    channels = slice(i * 24, (i + 1) * 24)
                    expected = expected + F.linear(
                        F.silu(F.linear(flat, moe.shared_experts.gate[channels]))
                        * F.linear(flat, moe.shared_experts.up[channels]),
                        moe.shared_experts.down[:, channels],
                    )
                torch.testing.assert_close(actual, expected.view_as(x))
                torch.testing.assert_close(balance, expected_balance)
                torch.testing.assert_close(z_loss, expected_z)
                self.assertEqual(stats["counts"].shape, (7,))
                self.assertEqual(stats["counts"].sum().item(), 6 * topk)
                probe = torch.randn_like(actual)
                parameters = (x, *moe.parameters())
                actual_grad = torch.autograd.grad((actual * probe).sum(), parameters, retain_graph=True)
                expected_grad = torch.autograd.grad((expected.view_as(x) * probe).sum(), parameters)
                for a, b in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(a, b)
                    self.assertTrue(torch.isfinite(a).all())
                self.assertGreater(actual_grad[-1].abs().sum().item(), 0)

    def test_parameter_counts_include_shared_and_only_routed_router_parameters(self):
        config = shared_smoke()
        model = MoELanguageModel(config)
        counts = parameter_counts(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), counts["total_parameters"])
        expert_count = sum(p.numel() for n, p in model.named_parameters() if "experts." in n)
        self.assertEqual(expert_count, counts["expert_parameters"])
        vanilla = load_config(ROOT / "configs/smoke.yaml")
        # Move one of eight experts to the shared branch: only two router rows disappear.
        self.assertEqual(parameter_counts(vanilla)["total_parameters"] - counts["total_parameters"], 64)
        self.assertEqual(model.layers[0].moe.router.out_features, 7)

    def test_two_topks_start_from_identical_weights_and_training_recipe(self):
        a, b = shared_smoke(2), shared_smoke(3)
        seed_everything(a["seed"])
        first = MoELanguageModel(a)
        seed_everything(b["seed"])
        second = MoELanguageModel(b)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)

    def test_bfloat16_training_resume_and_raw_evaluation_include_shared_weights(self):
        for topk in (2, 3):
            with self.subTest(topk=topk), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = shared_smoke(topk)
                config["training"]["task_precision"] = "bfloat16"
                make_synthetic_corpus(config, root / "data", 64)
                corpus = TokenCorpus(root / "data/manifest.json", config)
                reference = Trainer(config, corpus, "cpu")
                self.assertIsNone(reference.manager)
                initial = reference.model.layers[0].moe.shared_experts.down.detach().clone()
                for _ in range(2):
                    reference.train_step()
                self.assertFalse(torch.equal(initial, reference.model.layers[0].moe.shared_experts.down))
                for layer in reference.model.layers:
                    for p in layer.moe.shared_experts.parameters():
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertIn(p, reference.optimizer.state)
                checkpoint = save_checkpoint(reference, root / "checkpoints")
                evaluator, restored_config = load_evaluation_model(checkpoint)
                inputs, labels = corpus.batch("validation", [0, 1], "cpu")
                self.assertEqual(restored_config, config)
                torch.testing.assert_close(
                    evaluator(inputs, labels).logits, reference.model(inputs, labels).logits,
                    rtol=0, atol=0,
                )
                for _ in range(2):
                    reference.train_step()
                resumed = Trainer(config, corpus, "cpu")
                load_checkpoint(resumed, checkpoint)
                for _ in range(2):
                    resumed.train_step()
                for name, tensor in reference.model.state_dict().items():
                    torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
                for p, q in zip(reference.model.parameters(), resumed.model.parameters()):
                    for key, value in reference.optimizer.state[p].items():
                        torch.testing.assert_close(value, resumed.optimizer.state[q][key], rtol=0, atol=0)
                self.assertEqual(reference.reader.state_dict(), resumed.reader.state_dict())

    def test_reject_inconsistent_families_and_unsupported_consensus_combination(self):
        for section, key, value in (
            ("model", "num_shared_experts", -1),
            ("model", "num_shared_experts", True),
            ("model", "num_shared_experts", 0),
            ("model", "family", "llama_style_no_shared_expert"),
            ("salaad", "enabled", True),
        ):
            config = shared_smoke()
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        with self.assertRaisesRegex(ValueError, "native trainer"):
            megatron_arguments(shared_smoke(), "/data", "/tokenizer", "/run")

    def test_formal_baselines_match_current_training_and_compute_budgets(self):
        vanilla = load_config(ROOT / "configs/ns97m_vanilla.yaml")
        current = load_config(ROOT / "configs/ns97m_sinkhorn_cosine_init.yaml")
        configs = []
        for topk in (7, 8):
            config = load_config(ROOT / f"configs/ns97m_shared_1plus{topk}.yaml")
            validate_config(config, 4)
            self.assertEqual(config["training"], current["training"])
            self.assertEqual(config["data"], current["data"])
            self.assertEqual(config["parallel"], current["parallel"])
            self.assertEqual(config["evaluation"], current["evaluation"])
            self.assertEqual(config["seed"], current["seed"])
            self.assertTrue(config["is_wandb"])
            self.assertFalse(config["salaad"]["enabled"])
            self.assertEqual(config["model"]["num_shared_experts"], 1)
            self.assertEqual(config["model"]["num_experts"], 63)
            self.assertEqual(config["model"]["router_topk"], topk)
            counts = parameter_counts(config)
            self.assertEqual(counts["expert_parameters"], parameter_counts(vanilla)["expert_parameters"])
            self.assertEqual(counts["total_parameters"], 97192192)
            self.assertEqual(counts["prediction_tokens"], 4404019200)
            with torch.device("meta"):
                model = MoELanguageModel(config, initialize=False)
            self.assertEqual(sum(p.numel() for p in model.parameters()), counts["total_parameters"])
            submit = (ROOT / f"sub/moe_ns97m_shared_1plus{topk}.sub").read_text()
            self.assertIn(f"--cfg_version ns97m_shared_1plus{topk}", submit)
            self.assertIn("WANDB_MODE=online", submit)
            configs.append(config)
        a, b = (copy.deepcopy(c) for c in configs)
        a.pop("experiment"); b.pop("experiment")
        a["model"].pop("router_topk"); b["model"].pop("router_topk")
        self.assertEqual(a, b)
        difference = parameter_counts(configs[1])["active_parameters_convention"] - parameter_counts(configs[0])["active_parameters_convention"]
        self.assertEqual(difference, 8 * 3 * 256 * 176)


if __name__ == "__main__":
    unittest.main()
