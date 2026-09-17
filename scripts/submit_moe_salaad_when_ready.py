#!/usr/bin/env python3
"""Submit SALAAD once the vanilla run has passed its first training checks.

This monitor uses only the standard library and reads logs/checkpoint metadata.
All model training still starts through scripts/train_salad.py:main().
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = "ClusterId,ProcId,JobStatus,DAGNodeName,ExitCode"


def query_jobs(constraint, history=True):
    """Check live jobs first, then recently completed jobs."""
    commands = ("condor_q", "condor_history") if history else ("condor_q",)
    for command in commands:
        args = [command, "-constraint", constraint, "-json", "-attributes", ATTRIBUTES]
        if command == "condor_history":
            args.extend(["-limit", "10", "-scanlimit", "10000"])
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=45)
        jobs = json.loads(result.stdout)
        if jobs:
            return jobs
    return []


def check_training(run, minimum_step):
    """Require finite training metrics, validation, and a complete checkpoint."""
    metrics = run / "metrics.jsonl"
    if not metrics.exists():
        return {"stage": "waiting_for_training_logs"}
    # Ignore the last line while the training process is still writing it.
    lines = metrics.read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.endswith("\n")]
    if not records:
        return {"stage": "waiting_for_training_logs"}
    step = records[-1]["step"]
    if step < minimum_step:
        return {"stage": "training", "step": step}

    config = json.loads((run / "config.resolved.json").read_text())
    if config["salaad"]["enabled"] or config["experiment"] != "moe_ns97m_vanilla":
        raise ValueError("The monitored run is not the expected vanilla baseline")
    steps = {record["step"] for record in records}
    if not set(range(1, minimum_step + 1)).issubset(steps):
        raise ValueError("The initial vanilla training metrics are incomplete")
    for record in records:
        for key in (
            "lm_nll", "load_balancing_loss", "router_z_loss",
            "task_gradient_norm", "combined_gradient_norm_before_clip",
        ):
            if not math.isfinite(record[key]):
                raise ValueError(f"Nonfinite {key} at vanilla step {record['step']}")

    for record in reversed(records):
        if record["step"] < minimum_step:
            break
        if "checkpoint" not in record or "validation_raw" not in record:
            continue
        validation_nll = record["validation_raw"]["nll"]
        if not math.isfinite(validation_nll):
            raise ValueError("Nonfinite vanilla validation NLL")
        checkpoint = Path(record["checkpoint"])
        marker_path = checkpoint / "complete.json"
        if not marker_path.exists():
            continue  # Older checkpoints may already have been pruned.
        marker = json.loads(marker_path.read_text())
        files = [checkpoint / "training.pt"] + [
            checkpoint / f"rank_{rank:05d}.pt" for rank in range(marker["world_size"])
        ]
        if (
            marker["format"] != "salaad_moe.checkpoint.v1"
            or marker["step"] != record["step"]
            or marker["world_size"] != 4
            or not all(path.is_file() for path in files)
        ):
            raise ValueError("The vanilla checkpoint is incomplete or inconsistent")
        return {
            "stage": "ready", "step": step, "checkpoint": str(checkpoint),
            "validation_nll": validation_nll,
        }
    return {"stage": "waiting_for_validation_and_checkpoint", "step": step}


def save_state(path, state):
    state = dict(state, updated_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print(json.dumps(state, allow_nan=False), flush=True)


def submit_once(baseline_dag, bid, state_path, ready):
    """Reconcile prior submissions before issuing one irreversible queue request."""
    constraint = f'MoeBaselineDag == {baseline_dag} && MoeRole == "salaad_followup"'
    existing = query_jobs(constraint)
    if existing:
        job = existing[0]
        result = dict(ready, stage="submitted", salaad_job=f"{job['ClusterId']}.{job['ProcId']}")
        save_state(state_path, result)
        return result
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get("stage") in ("submitting", "submission_uncertain"):
        raise RuntimeError("A previous submission needs reconciliation; refusing to submit twice")
    command = [
        "/usr/local/bin/condor_submit_bid", str(bid),
        "sub/moe_ns97m_salaad.sub",
        "-append", f"+MoeBaselineDag = {baseline_dag}",
        "-append", '+MoeRole = "salaad_followup"',
    ]
    pending = dict(ready, stage="submitting", bid=bid, baseline_dag=baseline_dag)
    save_state(state_path, pending)
    try:
        result = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
            env=dict(os.environ, DEFAULT_JOB_BID=str(bid)),
        )
        match = re.search(r"1 job\(s\) submitted to cluster (\d+)\.", result.stdout)
        if not match:
            raise RuntimeError(f"Unrecognized submit output: {result.stdout} {result.stderr}")
    except Exception as exc:
        save_state(state_path, dict(pending, stage="submission_uncertain", error=str(exc)))
        raise
    submitted = dict(pending, stage="submitted", salaad_job=f"{match.group(1)}.0")
    save_state(state_path, submitted)
    return submitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dag", required=True, type=int)
    parser.add_argument("--bid", required=True, type=int)
    parser.add_argument("--minimum-step", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if min(args.baseline_dag, args.bid, args.minimum_step, args.poll_seconds) < 1:
        parser.error("Job id, bid, minimum step, and polling interval must be positive")
    state_path = ROOT / "jobs" / f"moe_salaad_after_vanilla_{args.baseline_dag}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous.get("stage") == "submitted":
            print(json.dumps(previous), flush=True)
            return
        if previous.get("stage") in ("submitting", "submission_uncertain"):
            submit_once(args.baseline_dag, args.bid, state_path, previous)
            return
        last_progress = None
        while True:
            progress = {"stage": "waiting_for_vanilla", "baseline_dag": args.baseline_dag, "bid": args.bid}
            live = query_jobs(
                f"ClusterId == {args.baseline_dag} || DAGManJobId == {args.baseline_dag}",
                history=False,
            )
            baseline = [job for job in live if job.get("DAGNodeName") == "VANILLA"]
            if not live:
                baseline = query_jobs(
                    f'DAGManJobId == {args.baseline_dag} && DAGNodeName == "VANILLA"'
                )
            if baseline:
                job = baseline[0]
                if job["JobStatus"] == 3 or (job["JobStatus"] == 4 and job.get("ExitCode") != 0):
                    raise RuntimeError("Vanilla training failed; SALAAD will not be submitted")
                run = ROOT / "data/moe/ns97m_vanilla" / f"{job['ClusterId']}_{job['ProcId']}"
                progress["baseline_job"] = f"{job['ClusterId']}.{job['ProcId']}"
                if job["JobStatus"] in (2, 4):
                    progress.update(check_training(run, args.minimum_step))
            else:
                parent = [job for job in live if job["ClusterId"] == args.baseline_dag]
                if not parent:
                    parent = query_jobs(f"ClusterId == {args.baseline_dag}")
                if parent and parent[0]["JobStatus"] in (3, 4):
                    raise RuntimeError("The vanilla workflow ended without a usable training job")
            if progress["stage"] == "ready":
                submit_once(args.baseline_dag, args.bid, state_path, progress)
                return
            if progress != last_progress:
                save_state(state_path, progress)
                last_progress = progress
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
