"""CPU protocol tests; actual CUDA Megatron execution remains a separate check."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from salaad_moe.config import load_config
from salaad_moe.groups import StackedGroup
from salaad_moe.megatron import (
    MegatronOptimizerHook,
    install_training_integration,
    megatron_arguments,
    native_weights_from_megatron,
)
from salaad_moe.model import MoELanguageModel

ROOT = Path(__file__).resolve().parents[2]


class OptimizerProtocol:
    """prepare -> global clip -> ready step -> copy, as in the pinned optimizer."""

    def __init__(self, parameter, clipping):
        self.parameter = parameter
        self.optimizer = torch.optim.SGD([parameter.main_param], lr=0.001)
        self.param_groups = self.optimizer.param_groups
        self.clip = clipping
        self.calls = []

    def prepare_grads(self):
        self.calls.append("prepare")
        self.parameter.main_param.grad = self.parameter.main_grad.clone()
        return False

    def step(self):
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None
        self.before_clip = self.parameter.main_param.grad.clone()
        norm = torch.nn.utils.clip_grad_norm_([self.parameter.main_param], self.clip)
        self.calls.append("clip")
        self.optimizer.step()
        self.calls.append("step")
        self.parameter.data.copy_(self.parameter.main_param)
        return True, norm.item(), 0


class MegatronAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c = load_config(ROOT / "configs/smoke.yaml")

    def test_native_flags_exclude_shared_experts_and_pre_softmax(self):
        c = load_config(ROOT / "configs/ns97m.yaml")
        args = megatron_arguments(c, "/data", "/tokenizer", "/run")
        for absent in (
            "--moe-router-pre-softmax",
            "--moe-shared-expert-intermediate-size",
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--moe-grouped-gemm",
        ):
            self.assertNotIn(absent, args)
        for flag, expected in (
            ("--moe-layer-freq", "1"),
            ("--moe-router-topk", "8"),
            ("--moe-aux-loss-coeff", "0.1"),
            ("--moe-z-loss-coeff", "0.001"),
            ("--lr-wsd-decay-style", "cosine"),
            ("--train-iters", "2100"),
            ("--eval-iters", "0"),
            ("--eval-interval", "2101"),
        ):
            self.assertEqual(args[args.index(flag) + 1], expected)

    def test_master_gradient_injection_precedes_upstream_global_clip(self):
        p = torch.nn.Parameter(torch.randn(3, 4, 5).bfloat16())
        p.main_param = torch.nn.Parameter(torch.randn(3, 4, 5))
        p.main_grad = torch.randn_like(p.main_param)
        optimizer = OptimizerProtocol(p, 0.2)
        hook = MegatronOptimizerHook(optimizer, [StackedGroup("g", p)], self.c, step=1)
        hook.manager.initialize(1)
        hook.manager.anchors["g"].zero_()
        initial = p.main_param.detach().clone()
        expected_gradient = p.main_grad + self.c["salaad"]["rho"] * initial
        success, norm, _ = optimizer.step()
        self.assertTrue(success)
        self.assertEqual(optimizer.calls, ["prepare", "clip", "step"])
        torch.testing.assert_close(optimizer.before_clip, expected_gradient)
        self.assertAlmostEqual(norm, expected_gradient.norm().item(), places=5)
        clipped = expected_gradient * min(1.0, 0.2 / (expected_gradient.norm().item() + 1e-6))
        torch.testing.assert_close(p.main_param, initial - 0.001 * clipped)
        torch.testing.assert_close(p, p.main_param.bfloat16(), rtol=0, atol=0)
        self.assertEqual(hook.step, 2)
        with self.assertRaises(ValueError):
            MegatronOptimizerHook(optimizer, [StackedGroup("g", p)], self.c)
        hook.uninstall()

    def test_nonfinite_task_gradient_aborts_before_clip_or_step(self):
        p = torch.nn.Parameter(torch.randn(2, 3, 4).bfloat16())
        p.main_param = torch.nn.Parameter(torch.randn(2, 3, 4))
        p.main_grad = torch.full_like(p.main_param, float("nan"))
        before = p.main_param.detach().clone()
        optimizer = OptimizerProtocol(p, 1.0)
        hook = MegatronOptimizerHook(optimizer, [StackedGroup("g", p)], self.c)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            optimizer.step()
        self.assertEqual(optimizer.calls, ["prepare"])
        self.assertEqual(hook.step, 0)
        torch.testing.assert_close(p.main_param, before, rtol=0, atol=0)

    def test_full_mha_qkv_swiglu_and_te_norm_conversion(self):
        source = MoELanguageModel(self.c)
        fake = self.make_megatron_fixture(source)
        weights = native_weights_from_megatron([fake], self.c)
        self.assertEqual(set(weights), set(source.state_dict()))
        for key, tensor in source.state_dict().items():
            torch.testing.assert_close(weights[key], tensor, rtol=0, atol=0)
        restored = MoELanguageModel(self.c)
        restored.load_state_dict(weights)
        tokens = torch.randint(128, (2, 8))
        torch.testing.assert_close(source(tokens).logits, restored(tokens).logits, rtol=0, atol=0)

    def make_megatron_fixture(self, source):
        m = self.c["model"]
        h, d = m["num_attention_heads"], m["hidden_size"]
        layers = []
        for layer in source.layers:
            qkv = (
                layer.attention.qkv.weight.detach()
                .view(3, h, d // h, d)
                .transpose(0, 1)
                .reshape(3 * d, d)
            )
            locals_ = []
            for i in range(m["num_experts"]):
                fc1 = torch.nn.Parameter(
                    torch.cat((layer.moe.experts.gate[i], layer.moe.experts.up[i])).detach().clone()
                )
                fc2 = torch.nn.Parameter(layer.moe.experts.down[i].detach().clone())
                locals_.append(
                    SimpleNamespace(
                        linear_fc1=SimpleNamespace(weight=fc1),
                        linear_fc2=SimpleNamespace(weight=fc2),
                    )
                )
            layers.append(
                SimpleNamespace(
                    self_attention=SimpleNamespace(
                        linear_qkv=SimpleNamespace(
                            weight=torch.nn.Parameter(qkv.clone()),
                            layer_norm_weight=layer.attention_norm.weight,
                        ),
                        linear_proj=layer.attention.out,
                    ),
                    pre_mlp_layernorm=layer.moe_norm,
                    mlp=SimpleNamespace(
                        router=layer.moe.router,
                        experts=SimpleNamespace(local_experts=locals_),
                        shared_experts=None,
                    ),
                )
            )
        return SimpleNamespace(
            embedding=SimpleNamespace(word_embeddings=source.embedding),
            output_layer=source.lm_head,
            decoder=SimpleNamespace(final_layernorm=source.norm, layers=layers),
        )

    def test_checkpoint_sidecars_without_validation(self):
        from salaad_moe.checkpoint import checkpoint_metadata

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = MoELanguageModel(self.c)
            fake = self.make_megatron_fixture(source)
            p = torch.nn.Parameter(torch.randn(3, 4, 5).bfloat16())
            p.main_param = torch.nn.Parameter(p.float().detach())
            optimizer = OptimizerProtocol(p, 1.0)
            native_args = SimpleNamespace(
                expert_model_parallel_size=1,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                use_distributed_optimizer=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                async_save=False,
                moe_shared_expert_intermediate_size=None,
                moe_router_pre_softmax=False,
                moe_expert_capacity_factor=None,
                moe_router_enable_expert_bias=False,
                moe_router_topk=2,
                num_experts=8,
                padded_vocab_size=128,
                ckpt_format="torch",
                iteration=0,
                save=str(root / "run"),
            )
            Path(native_args.save).mkdir()
            training = SimpleNamespace(
                get_args=lambda: native_args,
                setup_model_and_optimizer=lambda: ([fake], optimizer, None),
                train_step=lambda: None,
            )
            training.save_checkpoint = lambda iteration, *a, **k: (
                Path(native_args.save) / f"iter_{iteration:07d}"
            ).mkdir()
            active = install_training_integration(training, self.c, "test-data-identity")
            training.setup_model_and_optimizer()
            self.assertTrue(active["hook"].manager.initialized)
            active["hook"].step = 1
            rng_before = torch.get_rng_state().clone()
            weights_before = native_weights_from_megatron([fake], self.c)
            training.save_checkpoint(1, [fake], optimizer, None)
            torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
            for name, tensor in weights_before.items():
                torch.testing.assert_close(
                    native_weights_from_megatron([fake], self.c)[name], tensor, rtol=0, atol=0
                )
            meta = checkpoint_metadata(Path(native_args.save) / "salaad/iter_0000001")
            self.assertNotIn("validation_nll", meta)
            self.assertEqual(meta["resume_backend"], "megatron_only")


if __name__ == "__main__":
    unittest.main()
