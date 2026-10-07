#!/usr/bin/env python3
"""Compare completed MoE runs on identical held-out tokens and/or zero-shot tasks."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from salaad_moe.checkpoint import checkpoint_metadata
from salaad_moe.config import fingerprint
from salaad_moe.data import TokenCorpus
from salaad_moe.export import load_evaluation_model
from salaad_moe.trainer import evaluate_model, seed_everything


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def preflight(args):
    """Check all run identities before loading any checkpoint tensors or using CUDA."""
    spec = yaml.safe_load(Path(args.config).read_text())
    if spec["format"] != "salaad_moe.comparison.v1" or not spec["runs"]:
        raise ValueError("Expected a nonempty salaad_moe.comparison.v1 configuration")
    names = [run["name"] for run in spec["runs"]]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in names
    ):
        raise ValueError("Run names must be unique filename-safe labels")
    if args.runs and (set(args.runs) - set(names)):
        raise ValueError(f"Unknown runs: {sorted(set(args.runs) - set(names))}")
    root = Path(args.repo_root).resolve()
    manifest_path = (root / spec["data_manifest"]).resolve()
    manifest = read_json(manifest_path)
    identity = fingerprint(manifest)
    if identity != spec["corpus_identity"]:
        raise ValueError("Corpus identity differs from the comparison configuration")
    nll_enabled = args.suite in ("nll", "all")
    sequences = manifest["splits"][args.split]["sequences"]
    count = min(args.sequences, sequences) if args.sequences else sequences
    configs, runs = {}, []
    reference = None
    for entry in spec["runs"]:
        name = entry["name"]
        if args.runs and name not in args.runs:
            continue
        path = (root / entry["checkpoint"]).resolve()
        marker = checkpoint_metadata(path)
        config = read_json(path.parent.parent / "config.resolved.json")
        metadata = read_json(path.parent.parent / "run_metadata.json")
        if marker["step"] != spec["checkpoint_step"]:
            raise ValueError(f"{name}: wrong checkpoint step")
        if marker["config_hash"] != entry["config_hash"] or fingerprint(config) != entry["config_hash"]:
            raise ValueError(f"{name}: checkpoint configuration hash mismatch")
        if metadata["corpus_identity"] != identity:
            raise ValueError(f"{name}: training corpus differs from evaluation corpus")
        if config["training"]["weight_decay"] != entry["weight_decay"]:
            raise ValueError(f"{name}: weight decay differs from run label")
        if config["salaad"]["enabled"] != entry["salaad_enabled"]:
            raise ValueError(f"{name}: SALAAD setting differs from run label")
        if (args.matrix_norms and entry["salaad_enabled"]
                and config["salaad"].get("residual_mode") != "dense"):
            raise ValueError(f"{name}: matrix norm inspection requires dense expert-specific X_e")
        # SALAAD and weight decay are the intended experimental variables.
        common = {k: config[k] for k in ("seed", "model", "data", "parallel", "evaluation")}
        common["training"] = {k: v for k, v in config["training"].items() if k != "weight_decay"}
        if reference is not None:
            for key in common:
                if common[key] != reference[key]:
                    raise ValueError(f"{name}: incomparable {key} settings")
        reference = common
        configs[name] = config
        runs.append({
            **entry, "checkpoint": str(path), "checkpoint_step": marker["step"],
            "training_world_size": marker["world_size"],
        })
    config = configs[runs[0]["name"]]
    plan = {
        "format": "salaad_moe.comparison_results.v1",
        "comparison_config": str(Path(args.config).resolve()),
        "mode": "raw", "suite": args.suite,
        "data_manifest": str(manifest_path), "corpus_identity": identity,
        "corpus_provenance": manifest["provenance"],
        "split": args.split if nll_enabled else None,
        "available_sequences": sequences if nll_enabled else None,
        "requested_sequences": args.sequences if nll_enabled else None,
        "evaluated_sequences": count if nll_enabled else None,
        "sequence_length": config["data"]["seq_length"],
        "batch_size": args.batch_size if nll_enabled else None,
        "task_limit": args.task_limit,
        "tasks": config["evaluation"]["tasks"] if args.suite in ("tasks", "all") else [],
        "task_seed": config["evaluation"]["task_seed"],
        "num_fewshot": config["evaluation"]["num_fewshot"],
        "tokenizer_directory": args.tokenizer_directory,
        "task_precision": config["training"]["task_precision"],
        "device": args.device,
        "matrix_norms_requested": args.matrix_norms,
        "debug": bool((nll_enabled and args.sequences) or args.task_limit),
        "runs": runs,
    }
    return plan, configs


def save_results(output, plan, records):
    """Keep partial results usable if a subsequent run or task download fails."""
    for record in records:
        write_json(output / (record["name"] + ".json"), record)
    write_json(output / "summary.json", {**plan, "results": records})
    fields = [
        "name", "status", "weight_decay", "salaad_enabled", "checkpoint_step",
        "mode", "split", "debug", "nll", "perplexity", "sequences", "prediction_tokens",
        "six_task_mean_acc",
    ]
    fields += [f"{task}/{metric}" for task in plan["tasks"] for metric in ("acc", "acc_norm")]
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields if key in record}
            row.update(record.get("nll_metrics", {}))
            tasks = record.get("task_metrics", {})
            row["six_task_mean_acc"] = tasks.get("six_task_mean_acc")
            for task, metrics in tasks.get("results", {}).items():
                for metric in ("acc", "acc_norm"):
                    row[f"{task}/{metric}"] = metrics.get(metric + ",none")
            writer.writerow(row)
    temporary.replace(output / "summary.csv")


def run_comparison(args, plan, configs):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on a GPU worker or pass --device cpu")
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    plan = {**plan, "started_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": args.cpu_threads,
    }}
    write_json(output / "plan.json", plan)
    corpus = None
    if args.suite in ("nll", "all"):
        print("Checking token shard hashes once for all runs...", flush=True)
        corpus = TokenCorpus(plan["data_manifest"], configs[plan["runs"][0]["name"]])
        if corpus.identity != plan["corpus_identity"]:
            raise ValueError("Corpus changed after preflight")
    records = []
    for index, entry in enumerate(plan["runs"], 1):
        print(f"[{index}/{len(plan['runs'])}] {entry['name']}: loading raw checkpoint", flush=True)
        record = {
            **entry, "mode": "raw", "split": plan["split"], "debug": plan["debug"],
            "corpus_identity": plan["corpus_identity"],
            "task_precision": plan["task_precision"], "status": "running",
        }
        records.append(record)
        model = None
        started = time.perf_counter()
        try:
            seed_everything(configs[entry["name"]]["seed"])
            model, config = load_evaluation_model(entry["checkpoint"], "raw", device)
            if fingerprint(config) != entry["config_hash"]:
                raise ValueError("Loaded checkpoint differs from preflight configuration")
            if corpus is not None:
                print(f"  {args.split}: {plan['evaluated_sequences']} sequences", flush=True)
                record["nll_metrics"] = evaluate_model(
                    model, corpus, args.split, args.sequences, args.batch_size,
                    config, device, distributed=False,
                )
                save_results(output, plan, records)
                print(f"  NLL={record['nll_metrics']['nll']:.6f}, "
                      f"PPL={record['nll_metrics']['perplexity']:.6f}", flush=True)
            if args.suite in ("tasks", "all"):
                from scripts.evaluate_moe_tasks import evaluate_tasks

                print("  Running zero-shot tasks...", flush=True)
                result = evaluate_tasks(
                    model, config, device, limit=args.task_limit,
                    tokenizer_directory=args.tokenizer_directory,
                )
                result["comparison_run"] = entry
                write_json(output / (entry["name"] + ".tasks.json"), result)
                record["task_metrics"] = {
                    **result["moe_salaad"], "results": result["results"],
                }
            record["status"] = "complete"
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            save_results(output, plan, records)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Complete all held-out evaluations before inspecting auxiliary matrices.
    # Completed NLL results remain saved even if this optional analysis fails.
    if args.matrix_norms:
        from scripts.analyze_moe_matrix_norms import analyze_checkpoint

        for record in records:
            if not record["salaad_enabled"]:
                continue
            norm_output = output / "matrix_norms" / record["name"]
            print(f"{record['name']}: inspecting checkpoint matrix norms", flush=True)
            report = analyze_checkpoint(
                record["checkpoint"], norm_output,
                expected_config_hash=record["config_hash"], expected_step=record["checkpoint_step"],
            )
            record["matrix_norms"] = {
                "status": "complete", "report": str(norm_output / "matrix_norms.json"),
                "groups_count": report["groups_count"], "per_expert_rows": report["per_expert_rows"],
            }
            save_results(output, plan, records)
    print(f"Saved {output / 'summary.csv'}", flush=True)
    return {**plan, "results": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/eval_ns97m_comparison.yaml"))
    parser.add_argument("--repo-root", default=str(ROOT), help="Root for paths in the comparison YAML")
    parser.add_argument("--output", help="New or empty results directory (required except for --dry-run)")
    parser.add_argument("--suite", choices=("nll", "tasks", "all"), default="nll")
    parser.add_argument("--runs", nargs="+", help="Optional subset of run names from the YAML")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sequences", type=int, default=0, help="0: full split; positive: debug prefix")
    parser.add_argument("--batch-size", type=int, default=4, help="NLL batch size; task scorer uses 1")
    parser.add_argument("--task-limit", type=int, help="Debug-only examples per downstream task")
    parser.add_argument("--tokenizer-directory", help="Optional local pinned Pythia tokenizer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--matrix-norms", action="store_true",
                        help="After evaluation, inspect X/X_e/W_e/P_e for dense SALAAD runs")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata only; no tensors or GPU")
    args = parser.parse_args(argv)
    if args.sequences < 0 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("--sequences must be >= 0; --batch-size and --cpu-threads must be >= 1")
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("--task-limit must be >= 1")
    if args.suite == "nll" and args.task_limit is not None:
        parser.error("--task-limit requires --suite tasks or all")
    if args.suite == "tasks" and args.sequences:
        parser.error("--sequences applies only to --suite nll or all")
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is given")
    plan, configs = preflight(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    return run_comparison(args, plan, configs)


if __name__ == "__main__":
    main()
