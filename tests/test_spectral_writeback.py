"""Contracts for direct post-optimizer spectral writeback."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from salad.spectral_writeback import (
    SpectralWriteback,
    balance_singular_energies,
)
from salad.trainer_salad import SALADTrainer, normalize_training_mode
from salad.utils import print_epoch, print_wandb
from scripts.vit_config_generator import generate_vit_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "vit_b8_block0_qkv_spectral_writeback_smoke.yaml"
)
FORMAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "vit_b8_block0_qkv_spectral_writeback.yaml"
)
REFERENCE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_shallow_qkv_r35_s35.yaml"
)
SUBMIT_PATH = (
    REPOSITORY_ROOT
    / "sub"
    / "salaad_vision_block0_qkv_spectral_writeback.sub"
)


def _params(balance_rate: float = 0.25) -> dict:
    return {
        "energy": 0.999,
        "energy_balance_rate": balance_rate,
    }


class SpectralWritebackTest(unittest.TestCase):
    def test_balancing_preserves_energy_and_reduces_variance(self) -> None:
        singular_values = torch.tensor([4.0, 2.0, 1.0, 0.5])

        balanced = balance_singular_energies(singular_values, 0.25)

        initial_energy = singular_values.square()
        balanced_energy = balanced.square()
        torch.testing.assert_close(
            balanced_energy.sum(),
            initial_energy.sum(),
        )
        self.assertLess(
            balanced_energy.var(unbiased=False).item(),
            initial_energy.var(unbiased=False).item(),
        )
        self.assertGreater(balanced_energy[-1].item(), initial_energy[-1].item())
        self.assertLess(balanced_energy[0].item(), initial_energy[0].item())

    def test_writeback_preserves_parameter_identity_and_autograd(self) -> None:
        weight = nn.Parameter(torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5])))
        operator = SpectralWriteback(
            "linear",
            _params(),
            weight,
            nr_layers=1,
            is_full=True,
        )
        parameter_id = id(weight)
        initial_norm = torch.linalg.vector_norm(weight.detach())

        operator.project_and_writeback()

        self.assertEqual(id(weight), parameter_id)
        self.assertTrue(weight.is_leaf)
        self.assertTrue(weight.requires_grad)
        self.assertIsNone(weight.grad_fn)
        torch.testing.assert_close(
            torch.linalg.vector_norm(weight.detach()),
            initial_norm,
        )
        self.assertGreater(operator.projection_relative_change, 0.0)
        self.assertLess(
            operator.spectrum_cv_after,
            operator.spectrum_cv_before,
        )
        self.assertFalse(hasattr(operator, "L"))
        self.assertFalse(hasattr(operator, "Y"))
        self.assertFalse(hasattr(operator, "rho"))

        weight.square().sum().backward()
        self.assertIsNotNone(weight.grad)

    def test_invalid_balance_rates_are_rejected(self) -> None:
        weight = nn.Parameter(torch.eye(2))
        for rate in (0.0, 1.0, -0.1, 1.1, True):
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(ValueError, "energy_balance_rate"):
                    SpectralWriteback(
                        "linear",
                        _params(rate),
                        weight,
                        nr_layers=1,
                        is_full=True,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_writeback_allows_the_next_adamw_backward(self) -> None:
        layer = nn.Linear(32, 96, bias=False, device="cuda")
        operator = SpectralWriteback(
            "linear",
            _params(5e-4),
            layer.weight,
            nr_layers=1,
            is_full=True,
        )
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        parameter = layer.weight

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = layer(torch.randn(4, 32, device="cuda")).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            operator.project_and_writeback()

        self.assertIs(layer.weight, parameter)
        self.assertIn(parameter, optimizer.state)
        self.assertTrue(torch.isfinite(layer.weight).all().item())


class SpectralWritebackTrainerTest(unittest.TestCase):
    @staticmethod
    def _make_trainer() -> tuple[SALADTrainer, SpectralWriteback, nn.Module]:
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5])))
        operator = SpectralWriteback(
            "0",
            _params(),
            model[0].weight,
            nr_layers=1,
            is_full=True,
        )
        operator.layer_gpu_map = 0
        operator.layer_idx = 0
        operator.init_T(1)

        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.training_mode = "spectral_writeback"
        trainer.rank = 0
        trainer.world_size = 1
        trainer.device = torch.device("cpu")
        trainer.ddp_model = model
        trainer.ADMM_solvers = [operator]
        trainer.per_owner_names = {0: ["0"]}
        trainer.owner_sizes = {0: 16}
        trainer.name2idx = {"0": 0}
        trainer.layer_info = {
            "0": {
                "pre_rank": [],
                "rank": [],
                "total_rank": [],
                "total_elements": [],
                "energy_balance_rate": [],
                "projection_relative_change": [],
                "spectrum_cv_before": [],
                "spectrum_cv_after": [],
            }
        }
        return trainer, operator, model

    def test_owner_projects_then_broadcasts_the_same_parameter(self) -> None:
        trainer, operator, model = self._make_trainer()
        parameter_id = id(model[0].weight)

        trainer.update_spectral_writeback()
        operator.cal_results()
        with patch("salad.trainer_salad.dist.broadcast") as broadcast:
            trainer.broadcast_params(model)

        self.assertEqual(id(model[0].weight), parameter_id)
        broadcast.assert_called_once()
        self.assertEqual(broadcast.call_args.args[0].numel(), 16)

    def test_projection_statistics_are_gathered_without_admm_fields(self) -> None:
        trainer, operator, _ = self._make_trainer()
        trainer.update_spectral_writeback()
        operator.cal_results()

        trainer.gather_layer_info(operator.T)

        info = trainer.layer_info["0"]
        self.assertEqual(len(info["rank"]), 1)
        self.assertEqual(info["energy_balance_rate"], [0.25])
        self.assertGreater(info["projection_relative_change"][0], 0.0)
        self.assertNotIn("rho", info)
        self.assertNotIn("loss", info)

    def test_mode_name_is_case_insensitive(self) -> None:
        self.assertEqual(
            normalize_training_mode("SPECTRAL_WRITEBACK"),
            "spectral_writeback",
        )


class SpectralWritebackConfigTest(unittest.TestCase):
    def test_generator_emits_only_direct_writeback_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "writeback.yaml"
            config = generate_vit_config(
                name="writeback",
                training_mode="spectral_writeback",
                include_mlp=False,
                vit_layers=1,
                excluded_suffixes=("attn.proj",),
                energy_balance_rate=5e-4,
                output_path=str(output_path),
            )

        self.assertEqual(config["training_mode"], "spectral_writeback")
        self.assertEqual(
            [entry["name"] for entry in config["layers"]],
            ["backbone.blocks.0.attn.qkv"],
        )
        self.assertEqual(
            set(config["layers"][0]["params"]),
            {"energy", "energy_balance_rate"},
        )
        self.assertEqual(
            config["layers"][0]["params"]["energy_balance_rate"],
            5e-4,
        )

    def test_committed_configs_use_the_requested_protocol(self) -> None:
        import yaml

        with FORMAL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            formal = yaml.safe_load(config_file)
        with SMOKE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            smoke = yaml.safe_load(config_file)
        with REFERENCE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            reference = yaml.safe_load(config_file)

        for key in set(reference) - {"name", "training_mode", "layers"}:
            with self.subTest(key=key):
                self.assertEqual(formal[key], reference[key])
        self.assertEqual(formal["training_mode"], "spectral_writeback")
        self.assertEqual(formal["num_freq"], 20)
        self.assertEqual(formal["num_total_iters"], 120_000)
        self.assertEqual(
            formal["layers"],
            [{
                "name": "backbone.blocks.0.attn.qkv",
                "params": {
                    "energy": 0.999,
                    "energy_balance_rate": 5e-4,
                },
            }],
        )

        self.assertEqual(smoke["training_mode"], "spectral_writeback")
        self.assertEqual(smoke["runtime"], "local")
        self.assertEqual(smoke["num_total_iters"], 2)
        self.assertEqual(smoke["num_freq"], 2)
        self.assertFalse(smoke["is_wandb"])

    def test_logging_exposes_projection_diagnostics(self) -> None:
        losses = {
            "avg_loss": 1.0,
            "avg_cls_loss": 0.4,
            "avg_patch_loss": 0.6,
            "avg_loss_penalty": 0.0,
            "avg_diff": 0.0,
        }
        stats = [{
            "name": "backbone.blocks.0.attn.qkv",
            "pre_rank": 3,
            "rank": 4,
            "total_rank": 4,
            "total_elements": 16,
            "energy_balance_rate": 5e-4,
            "projection_relative_change": 1e-3,
            "spectrum_cv_before": 1.0,
            "spectrum_cv_after": 0.9,
        }]

        output = io.StringIO()
        with redirect_stdout(output):
            print_epoch(1, 5, 2, 1e-5, 2, losses, stats)
        self.assertIn("relative writeback", output.getvalue())

        with patch("salad.utils.wandb.log") as log:
            print_wandb(
                None,
                epoch=1,
                total_epochs=5,
                num_freq=2,
                lr=1e-5,
                num_images=2,
                losses=losses,
                layer_stats=stats,
            )
        payload = log.call_args.args[0]
        prefix = "layer/backbone.blocks.0.attn.qkv"
        self.assertEqual(payload[f"{prefix}/pre_rank_ratio"], 0.75)
        self.assertEqual(payload[f"{prefix}/rank_ratio"], 1.0)
        self.assertEqual(
            payload[f"{prefix}/projection_relative_change"],
            1e-3,
        )
        self.assertNotIn(f"{prefix}/rho", payload)
        self.assertNotIn(f"{prefix}/diff", payload)

    def test_submit_file_uses_the_formal_config_and_four_h100s(self) -> None:
        submit = SUBMIT_PATH.read_text(encoding="utf-8")

        self.assertIn("--nproc_per_node=4", submit)
        self.assertIn(
            "--cfg_version vit_b8_block0_qkv_spectral_writeback",
            submit,
        )
        self.assertIn("request_cpus    = 12", submit)
        self.assertIn("request_memory  = 64000", submit)
        self.assertIn("request_gpus    = 4", submit)


if __name__ == "__main__":
    unittest.main()
