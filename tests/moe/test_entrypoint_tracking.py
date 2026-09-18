"""Config-name entrypoint and W&B integration, without contacting the service."""
import copy
import bdb
from contextlib import redirect_stdout
import dis
import io
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from salaad_moe.checkpoint import rng_state
from salaad_moe.config import load_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.tracking import tracking_run
from salaad_moe.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


class FakeRun:
    def __init__(self, kwargs):
        self.kwargs, self.summary, self.logs = kwargs, {}, []
        self.step, self.finished, self.fail_log = 0, None, False

    def define_metric(self, *args, **kwargs):
        pass

    def get_url(self):
        return f"https://wandb.ai/test/{self.kwargs['project']}/runs/{self.kwargs['id']}"

    def log(self, payload, step):
        random.random()
        np.random.rand()
        torch.rand(2)
        if self.fail_log:
            raise RuntimeError("logger write failed")
        self.logs.append((copy.deepcopy(payload), step))
        self.step = step + 1

    def finish(self, exit_code=0):
        random.random()
        np.random.rand()
        torch.rand(2)
        self.finished = exit_code


class EntrypointTests(unittest.TestCase):
    def test_both_commands_hit_a_breakpoint_inside_shared_main(self):
        from scripts import train_salad

        filename = str(ROOT / "scripts/train_salad.py")
        line = next(
            line for _, line in dis.findlinestarts(train_salad.main.__code__)
            if line > train_salad.main.__code__.co_firstlineno
        )

        class MainBreakpoint(bdb.Bdb):
            def user_line(self, frame):
                if self.canonic(frame.f_code.co_filename) == filename and frame.f_code.co_name == "main":
                    hits.append(frame.f_lineno)
                self.set_continue()

        for entry in ("train_salad.py", "train_moe.py"):
            with self.subTest(entry=entry):
                hits = []
                debugger = MainBreakpoint()
                self.assertIsNone(debugger.set_break(filename, line))
                command = [str(ROOT / "scripts" / entry), "--cfg_version", "smoke_bf16", "--dry-run"]
                try:
                    with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()):
                        debugger.runcall(runpy.run_path, command[0], run_name="__main__")
                finally:
                    debugger.clear_all_breaks()
                self.assertEqual(hits, [line])

    def test_programmatic_main_uses_its_arguments(self):
        from scripts.train_salad import main

        stdout = io.StringIO()
        with patch.object(sys, "argv", ["train_salad.py", "--cfg_version", "ns480m"]), redirect_stdout(stdout):
            main(path_cfg=ROOT / "configs/smoke.yaml", rho=0.02, dry_run=True)
        config = json.loads(stdout.getvalue())["config"]
        expected = load_config(ROOT / "configs/smoke.yaml")
        expected["salaad"]["rho"] = 0.02
        self.assertEqual(config, expected)

    def test_shared_entry_preserves_pause_resume_and_full_schedule(self):
        def initialize_tracking(**kwargs):
            self.assertTrue(dist.is_initialized())
            self.assertEqual(dist.get_rank(), 0)
            self.assertEqual(dist.get_world_size(), 1)
            return FakeRun(kwargs)

        common = ["--cfg_version", "smoke", "--device", "cpu", "--cpu-threads", "2"]
        with tempfile.TemporaryDirectory() as directory, patch(
            "wandb.init", side_effect=initialize_tracking
        ), patch.dict(os.environ, {"WORLD_SIZE": "1"}), patch.object(
            dist, "broadcast", wraps=dist.broadcast
        ) as broadcast, patch.object(
            dist, "all_reduce", wraps=dist.all_reduce
        ) as all_reduce, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            entry = str(ROOT / "scripts/train_salad.py")
            with patch.object(sys, "argv", [
                entry, *common, "--output", str(output), "--num_total_iters", "2"
            ]):
                runpy.run_path(entry, run_name="__main__")
            self.assertFalse(dist.is_initialized())
            broadcast.assert_called()
            all_reduce.assert_called()
            checkpoint = output / "checkpoints/step_00000002"
            self.assertTrue(checkpoint.is_dir())
            alias = str(ROOT / "scripts/train_moe.py")
            with patch.object(sys, "argv", [
                alias, *common, "--resume", str(checkpoint), "--stop-after", "4"
            ]):
                runpy.run_path(alias, run_name="__main__")
            self.assertFalse(dist.is_initialized())
            self.assertTrue((output / "checkpoints/step_00000004").is_dir())
            config = json.loads((output / "config.resolved.json").read_text())
            self.assertEqual(config["training"]["total_optimizer_steps"], 8)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["resume_from"], str(checkpoint))

    def test_legacy_llm_and_vision_configs_are_rejected(self):
        from scripts.train_salad import main

        for name in ("llama_debug", "vit_b8_admm_l_smoke"):
            with self.subTest(config=name), self.assertRaisesRegex(ValueError, "MoE configuration"):
                main(cfg_version=name, dry_run=True)

    def test_entry_runs_without_legacy_framework_imports(self):
        # Block legacy imports in a fresh process to verify MoE startup is independent.
        code = '''
import importlib.abc
import runpy
import sys
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"salad", "models", "dataloaders", "salaad_vision"}:
            raise ImportError("Legacy training dependency: " + fullname)
sys.meta_path.insert(0, RejectLegacy())
sys.argv = ["scripts/train_salad.py", "--cfg_version", "smoke_bf16", "--dry-run"]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=40, check=True,
        )
        self.assertEqual(json.loads(result.stdout)["config"]["experiment"], "moe_salaad_bf16_smoke")

    def test_existing_config_name_entry_resolves_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/train_salad.py"),
                    "--cfg_version",
                    "smoke_bf16",
                    "--dry-run",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=40,
                check=True,
            )
        config = json.loads(result.stdout)["config"]
        self.assertEqual(config, load_config(ROOT / "configs/smoke_bf16.yaml"))
        self.assertTrue(config["is_wandb"])
        self.assertEqual(config["wandb_project"], "SALAAD-MoE")

    def test_preserved_user_default_and_direct_entry_match(self):
        configs = []
        for entry in ("train_salad.py", "train_moe.py"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / entry), "--dry-run"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=40,
                check=True,
            )
            configs.append(json.loads(result.stdout)["config"])
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0]["experiment"], "moe_salaad_dclm_bf16_smoke")


class TrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.config = load_config(ROOT / "configs/smoke.yaml")
        self.runs = []

        def initialize(**kwargs):
            random.random()
            np.random.rand()
            torch.rand(2)
            run = FakeRun(kwargs)
            self.runs.append(run)
            return run

        sdk = patch("wandb.init", side_effect=initialize)
        sdk.start()
        self.addCleanup(sdk.stop)
        mode = patch.dict(os.environ, {"WANDB_MODE": "online"})
        mode.start()
        self.addCleanup(mode.stop)

    def corpus(self):
        make_synthetic_corpus(self.config, self.path / "data", 64)
        return TokenCorpus(self.path / "data/manifest.json", self.config)

    def test_actual_trainer_logs_training_metrics_without_validation(self):
        self.config["training"]["checkpoint_interval_steps"] = 2
        trainer = Trainer(self.config, self.corpus(), "cpu")
        with patch(
            "salaad_moe.trainer.evaluate_model",
            side_effect=AssertionError("Training must not run validation"),
        ) as evaluate:
            trainer.run(self.path / "run")
        evaluate.assert_not_called()
        self.assertEqual(len(self.runs), 1)
        run = self.runs[0]
        self.assertEqual(run.kwargs["project"], "SALAAD-MoE")
        self.assertEqual(run.kwargs["config"], self.config)
        self.assertEqual([step for _, step in run.logs], list(range(1, 9)))
        first, last = run.logs[0][0], run.logs[-1][0]
        self.assertEqual(
            {key for key in first if key.startswith("train/")},
            {"train/lm_nll", "train/load_balancing_loss", "train/router_z_loss", "train/learning_rate"},
        )
        # The first optimizer update makes the structural gradient nonzero at step 2.
        second = run.logs[1][0]
        self.assertEqual(
            {key for key in second if key.startswith("train_stats/")},
            {
                "train_stats/consumed_sequences",
                "train_stats/prediction_tokens",
                "train_stats/task_gradient_norm",
                "train_stats/constraint_gradient_norm",
                "train_stats/constraint_task_gradient_ratio",
                "train_stats/constraint_task_gradient_cosine",
                "train_stats/combined_gradient_norm_before_clip",
                "train_stats/step_seconds",
            },
        )
        self.assertIn("router/mean_token_entropy/0", first)
        for payload, _ in run.logs:
            self.assertFalse(any(key.startswith("validation_") for key in payload))
            self.assertNotIn("train/reconstruction_nll_gap", payload)
            self.assertFalse(any("effective_experts_from_entropy" in key for key in payload))
            cosine_key = "train_stats/constraint_task_gradient_cosine"
            if payload["train_stats/task_gradient_norm"] and payload["train_stats/constraint_gradient_norm"]:
                self.assertGreaterEqual(payload[cosine_key], -1.0)
                self.assertLessEqual(payload[cosine_key], 1.0)
            else:
                self.assertNotIn(cosine_key, payload)
        checkpoints = self.path / "run/checkpoints"
        self.assertEqual(
            {path.name for path in checkpoints.iterdir()}, {"step_00000008"}
        )
        for checkpoint in checkpoints.iterdir():
            metadata = json.loads((checkpoint / "complete.json").read_text())
            self.assertNotIn("validation_nll", metadata)
        # With step-zero initialization, the first structural update is at step 2.
        first_structure_update = run.logs[1][0]
        self.assertIn(
            "salaad_structure/layer_0_gate_expert_0_effective_rank_ratio", first_structure_update
        )
        # Keep each expert's five values and log the global rho only once.
        expected_keys = {"salaad_hyperparameters/rho"}
        rho = self.config["salaad"]["rho"]
        self.assertEqual(last["salaad_hyperparameters/rho"], rho)
        for group in trainer.manager.groups:
            layer, projection = group.name.split(".")[1], group.name.split(".")[-1]
            state = trainer.manager.states[group.name]
            expected = {
                "diff": (group.weight() - state.reconstruction()).flatten(1).norm(dim=1),
                "density": state.sparse.ne(0).float().mean((-1, -2)),
                "effective_rank_ratio": state.rank_ratio,
                "alpha": rho * state.tau_l,
                "beta": rho * state.tau_s,
            }
            for expert in range(self.config["model"]["num_experts"]):
                for metric, values in expected.items():
                    section = "hyperparameters" if metric in ("alpha", "beta") else "structure"
                    key = f"salaad_{section}/layer_{layer}_{projection}_expert_{expert}_{metric}"
                    self.assertEqual(last[key], values[expert].item())
                    expected_keys.add(key)
        salaad_keys = {key for key in last if key.startswith("salaad_")}
        self.assertEqual(salaad_keys, expected_keys)
        self.assertEqual(
            {key.split("/")[0] for key in salaad_keys},
            {"salaad_structure", "salaad_hyperparameters"},
        )
        self.assertTrue(all(key.count("/") == 1 for key in expected_keys))
        self.assertFalse(any(key.startswith("layer/") for key in last))
        self.assertFalse(any(key.startswith("salaad/") for key in last))
        self.assertEqual(run.finished, 0)
        self.assertEqual(
            json.loads((self.path / "run/wandb_run.json").read_text())["last_logged_step"], 8
        )

    def test_vanilla_does_not_log_an_undefined_gradient_cosine(self):
        self.config["salaad"]["enabled"] = False
        trainer = Trainer(self.config, self.corpus(), "cpu")
        trainer.run(self.path / "vanilla", stop_after=2)
        for payload, _ in self.runs[0].logs:
            self.assertEqual(payload["train_stats/constraint_gradient_norm"], 0.0)
            self.assertNotIn("train_stats/constraint_task_gradient_cosine", payload)

    def test_tracking_does_not_change_weights_optimizer_or_rng(self):
        corpus = self.corpus()
        disabled = copy.deepcopy(self.config)
        disabled["is_wandb"] = False
        baseline = Trainer(disabled, corpus, "cpu")
        baseline.run(self.path / "without_logging")
        before = rng_state()
        logged = Trainer(self.config, corpus, "cpu")
        logged.run(self.path / "with_logging")
        after = rng_state()
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"], after["numpy"])
        torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)
        for first, second in zip(before["cuda"], after["cuda"]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for first, second in zip(baseline.model.parameters(), logged.model.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            for key, value in baseline.optimizer.state[first].items():
                torch.testing.assert_close(
                    value, logged.optimizer.state[second][key], rtol=0, atol=0
                )

    def test_resume_keeps_run_id_and_checkpoint_step(self):
        corpus = self.corpus()
        trainer = Trainer(self.config, corpus, "cpu")
        checkpoint = trainer.run(self.path / "run", stop_after=4)
        resumed = Trainer(self.config, corpus, "cpu")
        resumed.run(self.path / "run", resume=checkpoint)
        first, second = self.runs
        self.assertEqual(first.kwargs["id"], second.kwargs["id"])
        self.assertEqual(second.kwargs["resume"], "allow")
        self.assertEqual([step for _, step in second.logs], [5, 6, 7, 8])

    def test_rewind_and_new_output_use_separate_histories(self):
        with tracking_run(self.config, self.path, "cpu") as tracker:
            tracker.log({"step": 4, "lm_nll": 1.0})
        with tracking_run(self.config, self.path, "cpu", resume=True, step=2):
            pass
        new_output = self.path / "branch"
        new_output.mkdir()
        with tracking_run(self.config, new_output, "cpu"):
            pass
        self.assertEqual(len({run.kwargs["id"] for run in self.runs}), 3)
        self.assertEqual(self.runs[1].summary["parent_run_id"], self.runs[0].kwargs["id"])

    def test_nonprimary_rank_and_disabled_config_do_not_initialize_wandb(self):
        with patch("salaad_moe.tracking.rank", return_value=1):
            with tracking_run(self.config, self.path, "cpu") as tracker:
                self.assertIsNone(tracker)
        self.config["is_wandb"] = False
        with tracking_run(self.config, self.path, "cpu") as tracker:
            self.assertIsNone(tracker)
        self.assertEqual(self.runs, [])

    def test_initialization_failure_is_reported_collectively(self):
        with patch("salaad_moe.tracking.WandbTracker", side_effect=RuntimeError("init failed")):
            with self.assertRaisesRegex(RuntimeError, "W&B initialization.*init failed"):
                with tracking_run(self.config, self.path, "cpu"):
                    self.fail("Failed initialization must not enter training")

    def test_log_failure_finishes_run_as_failed(self):
        with self.assertRaisesRegex(RuntimeError, "logger write failed"):
            with tracking_run(self.config, self.path, "cpu") as tracker:
                self.runs[0].fail_log = True
                tracker.log({"step": 1, "lm_nll": 1.0})
        self.assertEqual(self.runs[0].finished, 1)


if __name__ == "__main__":
    unittest.main()
