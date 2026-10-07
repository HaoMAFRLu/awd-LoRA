"""Numerical evaluation, fair-comparison guards, and recoverable result artifacts."""
import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
import yaml

from salaad_moe.checkpoint import save_checkpoint
from salaad_moe.config import fingerprint, load_config
from salaad_moe.data import TokenCorpus, make_synthetic_corpus
from salaad_moe.export import load_evaluation_model
from salaad_moe.trainer import Trainer
from scripts.evaluate_moe_comparison import main

ROOT = Path(__file__).resolve().parents[2]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
        make_synthetic_corpus(config, self.root / "tokens", 8)
        self.corpus = TokenCorpus(self.root / "tokens/manifest.json", config)
        self.spec = {
            "format": "salaad_moe.comparison.v1", "checkpoint_step": 1,
            "data_manifest": "tokens/manifest.json", "corpus_identity": self.corpus.identity,
            "runs": [],
        }
        for enabled in (False, True):
            for wd in (0.0, 0.1):
                c = copy.deepcopy(config)
                if not enabled:
                    c["salaad"] = load_config(ROOT / "configs/smoke.yaml")["salaad"]
                c["salaad"]["enabled"] = enabled
                c["salaad"]["state_initialization_step"] = 0
                c["training"]["weight_decay"] = wd
                name = ("sinkhorn" if enabled else "vanilla") + ("_wd01" if wd else "_wd0")
                c["experiment"] = name
                trainer = Trainer(c, self.corpus, "cpu")
                trainer.train_step()
                run = self.root / name
                run.mkdir()
                (run / "config.resolved.json").write_text(json.dumps(c))
                (run / "run_metadata.json").write_text(json.dumps({"corpus_identity": self.corpus.identity}))
                checkpoint = save_checkpoint(trainer, run / "checkpoints")
                self.spec["runs"].append({
                    "name": name, "checkpoint": str(checkpoint.relative_to(self.root)),
                    "config_hash": fingerprint(c), "weight_decay": wd, "salaad_enabled": enabled,
                })
        self.config_path = self.root / "comparison.yaml"
        self.write_spec()
        self.output = self.root / "results"

    def write_spec(self):
        self.config_path.write_text(yaml.safe_dump(self.spec))

    def run_evaluation(self, *extra):
        with redirect_stdout(io.StringIO()):
            return main([
                "--config", str(self.config_path), "--repo-root", str(self.root),
                "--output", str(self.output), "--device", "cpu", "--cpu-threads", "2",
                "--batch-size", "2", *extra,
            ])

    def test_four_runs_match_manual_token_loss_with_partial_final_batch(self):
        result = self.run_evaluation("--sequences", "3")
        self.assertTrue(result["debug"])
        self.assertEqual(len(result["results"]), 4)
        for entry, record in zip(self.spec["runs"], result["results"]):
            model, _ = load_evaluation_model(self.root / entry["checkpoint"])
            inputs, labels = self.corpus.batch("validation", [0, 1, 2], "cpu")
            with torch.no_grad():
                logits = model(inputs).logits.float()
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()
            self.assertAlmostEqual(record["nll_metrics"]["nll"], nll, places=5)
            self.assertAlmostEqual(record["nll_metrics"]["perplexity"], torch.tensor(nll).exp().item(), places=4)
            self.assertEqual(record["nll_metrics"]["prediction_tokens"], 3 * 32)
            self.assertEqual(record["status"], "complete")
        with (self.output / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["debug"] == "True" and row["sequences"] == "3" for row in rows))
        before = (self.output / "summary.json").read_text()
        with self.assertRaises(FileExistsError):
            self.run_evaluation()
        self.assertEqual(before, (self.output / "summary.json").read_text())

    def test_full_test_split_and_run_subset(self):
        result = self.run_evaluation("--split", "test", "--runs", "sinkhorn_wd01")
        self.assertFalse(result["debug"])
        self.assertEqual(len(result["results"]), 1)
        record = result["results"][0]
        self.assertEqual(record["split"], "test")
        self.assertEqual(record["nll_metrics"]["sequences"], 8)
        self.assertEqual(record["nll_metrics"]["prediction_tokens"], 8 * 32)

    def test_dry_run_needs_neither_model_loading_nor_cuda(self):
        with patch("scripts.evaluate_moe_comparison.load_evaluation_model") as loader, patch(
            "scripts.evaluate_moe_comparison.TokenCorpus"
        ) as corpus, patch("torch.cuda.is_available") as cuda:
            result = self.run_evaluation("--dry-run", "--device", "cuda")
        loader.assert_not_called()
        corpus.assert_not_called()
        cuda.assert_not_called()
        self.assertEqual(result["evaluated_sequences"], 8)
        self.assertFalse(self.output.exists())

    def test_preflight_rejects_wrong_step_label_and_corpus(self):
        for key, value, error in (
            ("checkpoint_step", 2100, "wrong checkpoint step"),
            ("corpus_identity", "wrong", "Corpus identity"),
        ):
            with self.subTest(key=key):
                original = self.spec[key]
                self.spec[key] = value
                self.write_spec()
                with self.assertRaisesRegex(ValueError, error):
                    self.run_evaluation("--dry-run")
                self.spec[key] = original
        self.spec["runs"][0]["weight_decay"] = 0.1
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "weight decay differs"):
            self.run_evaluation("--dry-run")
        self.assertFalse(self.output.exists())

    def test_preflight_rejects_extra_training_difference(self):
        entry = self.spec["runs"][-1]
        checkpoint = self.root / entry["checkpoint"]
        config_path = checkpoint.parent.parent / "config.resolved.json"
        config = json.loads(config_path.read_text())
        config["training"]["learning_rate"] *= 2
        config_path.write_text(json.dumps(config))
        entry["config_hash"] = fingerprint(config)
        marker_path = checkpoint / "complete.json"
        marker = json.loads(marker_path.read_text())
        marker["config_hash"] = entry["config_hash"]
        marker_path.write_text(json.dumps(marker))
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "incomparable training"):
            self.run_evaluation("--dry-run")

    def test_task_failure_keeps_completed_nll(self):
        with patch("scripts.evaluate_moe_tasks.evaluate_tasks", side_effect=RuntimeError("dataset unavailable")):
            with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
                self.run_evaluation("--suite", "all", "--sequences", "3")
        result = json.loads((self.output / "summary.json").read_text())["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["nll_metrics"]["nll"], 0)
        self.assertIn("dataset unavailable", result["error"])

    def test_norm_inspection_happens_after_evaluation_and_keeps_nll_on_failure(self):
        def fail_norms(*args, **kwargs):
            saved = json.loads((self.output / "summary.json").read_text())
            self.assertTrue(all(record["status"] == "complete" for record in saved["results"]))
            self.assertTrue(all("nll_metrics" in record for record in saved["results"]))
            raise RuntimeError("norm analysis interrupted")

        with patch("scripts.analyze_moe_matrix_norms.analyze_checkpoint", side_effect=fail_norms):
            with self.assertRaisesRegex(RuntimeError, "norm analysis interrupted"):
                self.run_evaluation("--matrix-norms", "--sequences", "1")
        saved = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(len(saved["results"]), 4)
        self.assertTrue(saved["matrix_norms_requested"])

    def test_evaluation_with_norms_writes_real_checkpoint_report(self):
        result = self.run_evaluation("--matrix-norms", "--sequences", "1", "--runs", "sinkhorn_wd0")
        metadata = result["results"][0]["matrix_norms"]
        report = json.loads(Path(metadata["report"]).read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(report["config_hash"], result["results"][0]["config_hash"])
        self.assertEqual(report["checkpoint_step"], 1)

    def test_tasks_use_shared_adapter_and_save_all_six_metrics(self):
        config = load_config(ROOT / "configs/smoke_sinkhorn.yaml")
        tasks = config["evaluation"]["tasks"]
        fake = {"results": {
            task: {"acc,none": (i + 1) / 10, "acc_norm,none": (i + 2) / 10}
            for i, task in enumerate(tasks)
        }, "samples": {}}
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = config["data"]["tokenizer_length"]
        tokenizer.eos_token_id = config["data"]["eod_id"]
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
            "lm_eval.evaluator.simple_evaluate", return_value=fake
        ) as evaluate:
            result = self.run_evaluation(
                "--suite", "all", "--sequences", "3", "--task-limit", "2", "--runs", "vanilla_wd0"
            )
        kwargs = evaluate.call_args.kwargs
        self.assertEqual(kwargs["num_fewshot"], 0)
        self.assertEqual(kwargs["limit"], 2)
        self.assertEqual(kwargs["random_seed"], 42)
        self.assertEqual(kwargs["tasks"], tasks)
        self.assertEqual(kwargs["model"].batch_size, 1)
        self.assertAlmostEqual(result["results"][0]["task_metrics"]["six_task_mean_acc"], 0.35)
        self.assertTrue((self.output / "vanilla_wd0.tasks.json").is_file())
        with (self.output / "summary.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertAlmostEqual(float(row["social_iqa/acc_norm"]), 0.7)

    def test_real_harness_scoring_with_in_memory_tasks(self):
        # Exercise the real request builder, adapter and metric aggregation without
        # downloading benchmarks. Identical questions with opposite gold answers
        # must produce 50% accuracy regardless of the randomly initialized model.
        from datasets import Dataset, DatasetDict
        from lm_eval.api.task import ConfigurableTask

        class TinyTokenizer:
            eos_token_id = 0

            def __len__(self):
                return 128

            def encode(self, text, add_special_tokens=False):
                return [1 + ord(char) % 127 for char in text]

        dataset = DatasetDict({"validation": Dataset.from_list([
            {"question": "Q:", "choices": ["A", "B"], "gold": gold} for gold in (0, 1)
        ])})
        names = load_config(ROOT / "configs/smoke_sinkhorn.yaml")["evaluation"]["tasks"]
        tasks = {name: ConfigurableTask(config={
            "task": name, "dataset_path": "in_memory", "custom_dataset": lambda **_: dataset,
            "validation_split": "validation", "num_fewshot": 0,
            "output_type": "multiple_choice", "doc_to_text": "question",
            "doc_to_choice": "choices", "doc_to_target": "gold",
            "metric_list": [
                {"metric": metric, "aggregation": "mean", "higher_is_better": True}
                for metric in ("acc", "acc_norm")
            ],
        }) for name in names}
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), patch(
            "lm_eval.evaluator.get_task_dict", return_value=tasks
        ), patch("scripts.evaluate_moe_comparison.TokenCorpus") as corpus:
            result = self.run_evaluation("--suite", "tasks", "--runs", "vanilla_wd0", "--task-limit", "2")
        corpus.assert_not_called()
        metrics = result["results"][0]["task_metrics"]
        self.assertEqual(metrics["six_task_mean_acc"], 0.5)
        self.assertTrue(all(values["acc,none"] == 0.5 for values in metrics["results"].values()))
        saved = json.loads((self.output / "vanilla_wd0.tasks.json").read_text())
        self.assertEqual(sum(len(samples) for samples in saved["samples"].values()), 12)


if __name__ == "__main__":
    unittest.main()
