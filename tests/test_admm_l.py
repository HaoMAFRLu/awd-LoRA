"""Contracts for the X=L-only ADMM training feature."""

from __future__ import annotations

import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from salad.admm_l_solver import ADMM_L
from salad.trainer_salad import SALADTrainer, normalize_training_mode
from salad.utils import print_epoch, print_wandb
from scripts.vit_config_generator import generate_vit_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_admm_l_smoke.yaml"
)
FORMAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "vit_b8_block0_qkv_admm_l_full_spectrum.yaml"
)
FORMAL_REFERENCE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vit_b8_shallow_qkv_r35_s35.yaml"
)
FORMAL_SUBMIT_PATH = (
    REPOSITORY_ROOT
    / "sub"
    / "salaad_vision_block0_qkv_admm_l_full_spectrum.sub"
)


def _params(
    *,
    energy_balance_rate: float = 0.25,
) -> dict:
    return {
        "energy": 0.999,
        "energy_balance_rate": energy_balance_rate,
        "rho_dict": {
            "rho": 1.0,
            "mode": "fixed",
        },
    }


class ADMMLSolverTest(unittest.TestCase):
    def test_initial_state_is_x_equals_l_without_sparse_state(self) -> None:
        weight = nn.Parameter(torch.tensor([[3.0, 0.0], [0.0, 1.0]]))

        solver = ADMM_L(
            "linear",
            _params(),
            weight,
            nr_layers=1,
            is_full=True,
        )

        torch.testing.assert_close(solver.L, weight.detach())
        torch.testing.assert_close(solver.Y, torch.zeros_like(weight))
        self.assertFalse(hasattr(solver, "S"))
        self.assertFalse(hasattr(solver, "alpha_solver"))
        self.assertFalse(hasattr(solver, "beta_solver"))
        self.assertFalse(hasattr(solver, "target_rank"))
        self.assertEqual(solver.get_diff().item(), 0.0)
        self.assertEqual(solver.get_penalty().item(), 0.0)

    def test_l_update_balances_all_spectral_energy_by_one_step(self) -> None:
        weight = nn.Parameter(torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5])))
        solver = ADMM_L(
            "linear",
            _params(energy_balance_rate=0.25),
            weight,
            nr_layers=1,
            is_full=True,
        )

        solver.update_L()

        torch.testing.assert_close(
            solver.L,
            torch.diag(
                torch.sqrt(
                    torch.tensor([13.328125, 4.328125, 2.078125, 1.515625])
                )
            ),
        )
        balanced_energy = torch.linalg.svdvals(solver.L).square()
        self.assertEqual(solver.nr_rank, 4)
        self.assertAlmostEqual(balanced_energy.sum().item(), 21.25, places=5)
        self.assertLess(balanced_energy[0].item(), 16.0)
        self.assertGreater(balanced_energy[2].item(), 1.0)
        self.assertGreater(balanced_energy[3].item(), 0.25)

    def test_repeated_updates_flatten_energy_gradually(self) -> None:
        weight = nn.Parameter(torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5])))
        solver = ADMM_L(
            "linear",
            _params(energy_balance_rate=0.10),
            weight,
            nr_layers=1,
            is_full=True,
        )

        initial_energy = torch.tensor([16.0, 4.0, 1.0, 0.25])
        initial_variance = initial_energy.var(unbiased=False)
        solver.update_L()
        first_energy = torch.linalg.svdvals(solver.L).square()
        first_variance = first_energy.var(unbiased=False)

        # Mimic the primal variable following the previous ADMM projection.
        with torch.no_grad():
            weight.copy_(solver.L)
        solver.Y.zero_()
        solver.update_L()
        second_energy = torch.linalg.svdvals(solver.L).square()
        second_variance = second_energy.var(unbiased=False)

        self.assertGreater(initial_variance.item(), first_variance.item())
        self.assertGreater(first_variance.item(), second_variance.item())
        self.assertFalse(torch.allclose(first_energy, first_energy.mean()))
        torch.testing.assert_close(first_energy.sum(), initial_energy.sum())
        torch.testing.assert_close(second_energy.sum(), initial_energy.sum())

    def test_gradient_and_dual_updates_use_x_minus_l_only(self) -> None:
        weight = nn.Parameter(torch.eye(2))
        solver = ADMM_L(
            "linear",
            _params(),
            weight,
            nr_layers=1,
            is_full=True,
        )
        solver.L.zero_()

        torch.testing.assert_close(
            solver.get_gradient(),
            torch.eye(2),
        )
        self.assertAlmostEqual(solver.get_penalty().item(), 1.0, places=6)

        solver.update_Y()

        torch.testing.assert_close(solver.Y, torch.eye(2))

    def test_rank_and_balance_rates_are_validated(self) -> None:
        weight = nn.Parameter(torch.eye(2))
        for field, value in (
            ("energy_balance_rate", 0.0),
            ("energy_balance_rate", 1.0),
            ("energy_balance_rate", 1.01),
        ):
            params = _params()
            params[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    ADMM_L(
                        "linear",
                        params,
                        weight,
                        nr_layers=1,
                        is_full=True,
                    )

    def test_statistics_use_padding_without_exposing_sparse_state(self) -> None:
        weight = nn.Parameter(torch.eye(2))
        solver = ADMM_L(
            "linear",
            _params(),
            weight,
            nr_layers=1,
            is_full=True,
        )
        solver.layer_idx = 0
        solver.init_T(1)
        solver.get_diff()

        solver.cal_results()

        self.assertEqual(tuple(solver.T.shape), (1, 12))
        self.assertEqual(solver.T[0, 1].item(), 0.0)
        self.assertEqual(solver.T[0, 3].item(), 0.0)
        self.assertEqual(solver.T[0, 6].item(), 0.0)
        self.assertEqual(solver.T[0, 9].item(), 0.0)


class ADMMTrainerIntegrationTest(unittest.TestCase):
    @staticmethod
    def _trainer_and_solver() -> tuple[SALADTrainer, ADMM_L]:
        model = nn.Linear(2, 2, bias=False)
        solver = ADMM_L(
            "0",
            _params(),
            model.weight,
            nr_layers=1,
            is_full=True,
        )
        solver.layer_gpu_map = 0
        solver.layer_idx = 0
        solver.init_T(1)

        trainer = SALADTrainer.__new__(SALADTrainer)
        trainer.training_mode = "admm_l"
        trainer.rank = 0
        trainer.world_size = 1
        trainer.device = torch.device("cpu")
        trainer.ddp_model = model
        trainer.ADMM_solvers = [solver]
        trainer.name2idx = {"0": 0}
        trainer.layer_info = {
            "0": {
                "loss": [],
                "rank": [],
                "rho": [],
                "total_rank": [],
                "total_elements": [],
                "energy_balance_rate": [],
            }
        }
        return trainer, solver

    def test_mode_name_is_case_insensitive(self) -> None:
        self.assertEqual(normalize_training_mode("ADMM_L"), "admm_l")
        self.assertEqual(normalize_training_mode("admm_l"), "admm_l")
        with self.assertRaisesRegex(ValueError, "ADMM_L"):
            normalize_training_mode("unknown")

    def test_vit_generator_emits_l_only_layer_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "admm_l.yaml"
            config = generate_vit_config(
                name="admm_l",
                training_mode="ADMM_L",
                include_mlp=False,
                vit_layers=1,
                excluded_suffixes=("attn.proj",),
                output_path=str(output_path),
            )

        self.assertEqual(config["training_mode"], "admm_l")
        self.assertEqual(
            [entry["name"] for entry in config["layers"]],
            ["backbone.blocks.0.attn.qkv"],
        )
        params = config["layers"][0]["params"]
        self.assertIn("rho_dict", params)
        self.assertEqual(params["energy_balance_rate"], 0.05)
        self.assertNotIn("rate_rank", params)
        self.assertNotIn("alpha_dict", params)
        self.assertNotIn("rate_sparsity", params)
        self.assertNotIn("beta_dict", params)
        self.assertNotIn("block_p", params)
        self.assertNotIn("block_q", params)

    def test_committed_smoke_config_uses_iterative_balance(self) -> None:
        import yaml

        with SMOKE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(config["training_mode"], "admm_l")
        self.assertEqual(config["runtime"], "local")
        self.assertEqual(config["num_total_iters"], 10)
        self.assertEqual(config["num_freq"], 2)
        self.assertEqual(
            config["scheduler"]["params"]["total_steps"],
            config["num_total_iters"],
        )
        self.assertEqual(len(config["layers"]), 1)
        layer = config["layers"][0]
        self.assertEqual(layer["name"], "backbone.blocks.0.attn.qkv")
        self.assertEqual(layer["params"]["energy_balance_rate"], 0.05)
        self.assertNotIn("rate_rank", layer["params"])
        self.assertNotIn("alpha_dict", layer["params"])
        self.assertNotIn("beta_dict", layer["params"])

    def test_formal_config_matches_previous_training_protocol(self) -> None:
        import yaml

        with FORMAL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        with FORMAL_REFERENCE_CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as reference_file:
            reference = yaml.safe_load(reference_file)

        for key in set(reference) - {"name", "training_mode", "layers"}:
            with self.subTest(key=key):
                self.assertEqual(config[key], reference[key])

        self.assertEqual(
            config["name"],
            "vit_b8_block0_qkv_admm_l_full_spectrum",
        )
        self.assertEqual(config["training_mode"], "admm_l")
        self.assertEqual(len(config["layers"]), 1)
        layer = config["layers"][0]
        reference_layer = reference["layers"][0]
        self.assertEqual(layer["name"], "backbone.blocks.0.attn.qkv")
        self.assertEqual(
            layer["params"]["rho_dict"],
            reference_layer["params"]["rho_dict"],
        )
        self.assertEqual(
            layer["params"]["energy"],
            reference_layer["params"]["energy"],
        )
        self.assertEqual(layer["params"]["energy_balance_rate"], 0.05)
        self.assertEqual(
            set(layer["params"]),
            {"energy", "energy_balance_rate", "rho_dict"},
        )

    def test_formal_submit_matches_previous_resources(self) -> None:
        submit = FORMAL_SUBMIT_PATH.read_text(encoding="utf-8")

        self.assertIn("--nproc_per_node=4", submit)
        self.assertIn(
            "--cfg_version vit_b8_block0_qkv_admm_l_full_spectrum",
            submit,
        )
        self.assertIn("request_cpus    = 12", submit)
        self.assertIn("request_memory  = 64000", submit)
        self.assertIn("request_gpus    = 4", submit)
        self.assertIn(
            'TARGET.Machine != "i205.internal.cluster.is.localnet"',
            submit,
        )
        self.assertIn(
            '!regexp("GPU-b211f480", TARGET.DetectedGPUs)',
            submit,
        )

    def test_trainer_uses_solver_without_reading_s(self) -> None:
        trainer, solver = self._trainer_and_solver()
        with torch.no_grad():
            solver.X_with_grad.add_(1.0)

        diff = trainer.get_diff_per_rank()
        penalty = trainer.get_penalty_loss()
        gradients = trainer.get_gradient_per_layer()

        self.assertGreater(diff.item(), 0.0)
        self.assertGreater(penalty.item(), 0.0)
        self.assertEqual(set(gradients), {"0"})
        self.assertFalse(hasattr(solver, "S"))

    def test_sparse_updates_are_rejected(self) -> None:
        trainer, _ = self._trainer_and_solver()

        with self.assertRaisesRegex(ValueError, "no alpha"):
            trainer.update_ADMM_single_step("alpha")
        with self.assertRaisesRegex(ValueError, "no S"):
            trainer.update_ADMM_single_step("S")
        with self.assertRaisesRegex(ValueError, "no beta"):
            trainer.update_ADMM_single_step("beta")

    def test_gathered_stats_expose_balance_rate(self) -> None:
        trainer, solver = self._trainer_and_solver()
        solver.get_diff()
        solver.cal_results()

        trainer.gather_layer_info(solver.T)

        self.assertEqual(
            trainer.layer_info["0"]["energy_balance_rate"],
            [0.25],
        )

    def test_admm_l_logging_uses_full_spectrum_without_alpha(self) -> None:
        losses = {
            "avg_loss": 1.0,
            "avg_cls_loss": 0.4,
            "avg_patch_loss": 0.6,
            "avg_loss_penalty": 0.1,
            "avg_diff": 0.2,
        }
        stats = [{
            "name": "backbone.blocks.0.attn.qkv",
            "loss": 0.2,
            "rho": 5e-6,
            "rank": 2,
            "total_rank": 4,
            "total_elements": 16,
            "energy_balance_rate": 0.05,
        }]

        output = io.StringIO()
        with redirect_stdout(output):
            print_epoch(1, 5, 2, 1e-5, 2, losses, stats)
        self.assertIn("balance rate", output.getvalue())

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
        self.assertEqual(payload[f"{prefix}/energy_balance_rate"], 0.05)
        self.assertNotIn(f"{prefix}/target_rank_ratio", payload)
        self.assertNotIn(f"{prefix}/alpha", payload)

    def test_saved_matrix_shard_contains_only_l_and_y(self) -> None:
        trainer, _ = self._trainer_and_solver()

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.save_results(temporary_directory)
            matrix_path = Path(temporary_directory) / "matrix_rank0.pkl"
            with matrix_path.open("rb") as matrix_file:
                payload = pickle.load(matrix_file)

        self.assertEqual(set(payload), {"LL", "YY"})
        self.assertEqual(set(payload["LL"]), {"0"})
        self.assertEqual(set(payload["YY"]), {"0"})


if __name__ == "__main__":
    unittest.main()
