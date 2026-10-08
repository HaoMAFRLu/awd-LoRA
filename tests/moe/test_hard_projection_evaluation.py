"""Verify direct assignments, frozen X/X_e, and the complete held-out evaluation path."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from salaad_moe.alignment import native_shared
from salaad_moe.checkpoint import save_checkpoint
from salaad_moe.config import fingerprint, load_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import load_evaluation_model
from salaad_moe.solver import DenseResidualState
from salaad_moe.trainer import Trainer, evaluate_model
from scripts.evaluate_moe_hard_projection import direct_hungarian, main, projected_weights

ROOT = Path(__file__).resolve().parents[2]


class HardProjectionEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_direct_assignment_accepts_solver_choice_on_tie(self):
        soft = torch.full((1, 3, 3), 1 / 3)
        with patch("scripts.evaluate_moe_hard_projection.linear_sum_assignment",
                   return_value=(np.array([0, 1, 2]), np.array([1, 2, 0]))) as solve:
            indices = direct_hungarian(soft)
        self.assertTrue(solve.call_args.kwargs["maximize"])
        np.testing.assert_array_equal(solve.call_args.args[0], soft[0].double().numpy())
        torch.testing.assert_close(indices, torch.tensor([[2, 0, 1]]), rtol=0, atol=0)

    def test_direct_projection_changes_only_mapped_shared_component_on_both_axes(self):
        soft = torch.tensor([[[.05, .90, .05], [.05, .05, .90], [.90, .05, .05]]])
        hard = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        shared = torch.arange(1, 7).float().reshape(3, 2)
        payload = {"config": {"model": {"num_layers": 1}}, "model": {"untouched": torch.tensor([7.])}}
        states, originals = {}, {}
        for projection in ("gate", "up", "down"):
            x = shared.T.contiguous() if projection == "down" else shared.clone()
            residual = torch.full((1, *x.shape), .25)
            name = f"layers.0.moe.experts.{projection}"
            state = DenseResidualState(x, residual, torch.zeros_like(residual), soft.clone(), int(projection == "down"))
            states[name] = state
            payload["model"][name] = state.native_shared() + residual
            originals[name] = (x.clone(), residual.clone(), state.permutation.clone())
        weights, indices = projected_weights(payload, states)
        torch.testing.assert_close(indices, torch.tensor([[[2, 0, 1]]]), rtol=0, atol=0)
        self.assertIs(weights["untouched"], payload["model"]["untouched"])
        for name, state in states.items():
            expected = state.shared @ hard if name.endswith("down") else hard.T @ state.shared
            torch.testing.assert_close(weights[name], expected[None] + state.residual, rtol=0, atol=0)
            self.assertGreater(float((weights[name] - payload["model"][name]).norm()), 0)
            for actual, before in zip((state.shared, state.residual, state.permutation), originals[name]):
                torch.testing.assert_close(actual, before, rtol=0, atol=0)

    def test_checkpoint_to_bf16_evaluation_and_binary_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
            config["salaad"]["state_initialization_step"] = 0
            config["salaad"]["channel_alignment"]["sinkhorn"]["max_iterations"] = 10000
            config["training"]["task_precision"] = "bfloat16"
            make_synthetic_corpus(config, root / "tokens", 8)
            manifest = root / "tokens/manifest.json"
            corpus = TokenCorpus(manifest, config)
            trainer = Trainer(config, corpus, "cpu")
            trainer.train_step()
            trainer.train_step()
            checkpoint = save_checkpoint(trainer, root / "checkpoints")
            with redirect_stdout(io.StringIO()):
                report = main([
                    "--checkpoint", str(checkpoint), "--data-manifest", str(manifest),
                    "--output", str(root / "results"), "--device", "cpu", "--cpu-threads", "2",
                    "--sequences", "3", "--batch-size", "2", "--expected-step", "2",
                    "--expected-config-hash", fingerprint(config),
                    "--expected-corpus-identity", corpus.identity,
                ])
            self.assertEqual(report["status"], "complete")
            self.assertEqual([r["name"] for r in report["results"]], ["hungarian_projected", "original_raw"])
            self.assertTrue(all(r["prediction_tokens"] == 3 * 32 for r in report["results"]))
            raw, _ = load_evaluation_model(checkpoint, "raw", "cpu")
            expected = evaluate_model(raw, corpus, "validation", 3, 2, config, torch.device("cpu"), distributed=False)
            self.assertEqual(report["results"][1]["nll"], expected["nll"])
            arrays = np.load(root / "results/hard_permutations.npz")
            hard = arrays["P_e"]
            self.assertEqual(hard.dtype, np.uint8)
            self.assertTrue(np.isin(hard, [0, 1]).all())
            self.assertTrue((hard.sum(-1) == 1).all() and (hard.sum(-2) == 1).all())
            np.testing.assert_array_equal(hard.argmax(-2), arrays["native_to_shared"])
            np.testing.assert_array_equal(hard.argmax(-1), arrays["shared_to_native"])
            saved = json.loads((root / "results/summary.json").read_text())
            self.assertEqual(saved["results"], report["results"])
            self.assertTrue((root / "results/summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", str(checkpoint), "--data-manifest", str(manifest),
                      "--output", str(root / "results"), "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
