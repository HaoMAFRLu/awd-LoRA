"""Check the conditions and duplicate-submission protection for the cluster follow-up."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import submit_moe_salaad_when_ready as followup


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        (self.run / "config.resolved.json").write_text(json.dumps({
            "experiment": "moe_ns97m_vanilla", "salaad": {"enabled": False},
        }))

    def write_training(self, steps=100, checkpoint=True):
        records = [{
            "step": step, "lm_nll": 10.0, "load_balancing_loss": 1.0,
            "router_z_loss": 0.1, "task_gradient_norm": 0.5,
            "combined_gradient_norm_before_clip": 0.5,
        } for step in range(1, steps + 1)]
        if checkpoint:
            path = self.run / "checkpoints" / f"step_{steps:08d}"
            path.mkdir(parents=True)
            (path / "complete.json").write_text(json.dumps({
                "format": "salaad_moe.checkpoint.v1", "step": steps, "world_size": 4,
            }))
            for name in ["training.pt"] + [f"rank_{rank:05d}.pt" for rank in range(4)]:
                (path / name).touch()
            records[-1].update(checkpoint=str(path), validation_raw={"nll": 9.8})
        (self.run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        return records

    def test_waits_for_100_steps_and_complete_checkpoint(self):
        self.write_training(99, checkpoint=False)
        self.assertEqual(followup.check_training(self.run, 100)["stage"], "training")
        self.write_training(100, checkpoint=False)
        self.assertEqual(followup.check_training(self.run, 100)["stage"], "waiting_for_validation_and_checkpoint")
        self.write_training()
        self.assertEqual(followup.check_training(self.run, 100)["stage"], "ready")
        (self.run / "checkpoints/step_00000100/rank_00003.pt").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            followup.check_training(self.run, 100)

    def test_rejects_nonfinite_metrics(self):
        records = self.write_training()
        records[40]["task_gradient_norm"] = float("nan")
        (self.run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            followup.check_training(self.run, 100)

    def test_ignores_a_partially_written_line(self):
        self.write_training(99, checkpoint=False)
        with (self.run / "metrics.jsonl").open("a") as handle:
            handle.write('{"step":100,')
        self.assertEqual(followup.check_training(self.run, 100)["step"], 99)

    def test_reuses_an_existing_submission(self):
        state = self.run / "state.json"
        with patch.object(followup, "query_jobs", return_value=[{"ClusterId": 123, "ProcId": 0}]), \
             patch.object(followup.subprocess, "run") as submit:
            result = followup.submit_once(42, 100, state, {"stage": "ready"})
        self.assertEqual(result["salaad_job"], "123.0")
        submit.assert_not_called()

    def test_uncertain_submission_is_not_repeated(self):
        state = self.run / "state.json"
        with patch.object(followup, "query_jobs", return_value=[]), \
             patch.object(followup.subprocess, "run", side_effect=subprocess.TimeoutExpired("submit", 60)) as submit:
            with self.assertRaises(subprocess.TimeoutExpired):
                followup.submit_once(42, 100, state, {"stage": "ready"})
            self.assertEqual(json.loads(state.read_text())["stage"], "submission_uncertain")
            with self.assertRaisesRegex(RuntimeError, "refusing to submit twice"):
                followup.submit_once(42, 100, state, {"stage": "ready"})
        self.assertEqual(submit.call_count, 1)

    def test_records_success_with_the_authorized_bid(self):
        state = self.run / "state.json"
        result = subprocess.CompletedProcess([], 0, "Submitting job(s).\n1 job(s) submitted to cluster 456.\n", "")
        with patch.object(followup, "query_jobs", return_value=[]), \
             patch.object(followup.subprocess, "run", return_value=result) as submit:
            submitted = followup.submit_once(42, 100, state, {"stage": "ready"})
        self.assertEqual(submitted["salaad_job"], "456.0")
        self.assertEqual(json.loads(state.read_text())["stage"], "submitted")
        self.assertEqual(submit.call_args.args[0][:2], ["/usr/local/bin/condor_submit_bid", "100"])
        self.assertEqual(submit.call_args.kwargs["env"]["DEFAULT_JOB_BID"], "100")


if __name__ == "__main__":
    unittest.main()
