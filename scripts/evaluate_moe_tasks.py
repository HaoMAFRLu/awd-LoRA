#!/usr/bin/env python3
"""Run the six planned zero-shot tasks through the custom likelihood adapter."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salaad_moe.export import load_evaluation_model


def evaluate_tasks(model, config, device, *, mode="raw", limit=None, tokenizer_directory=None):
    """Evaluate an already loaded model; also used by the multi-checkpoint runner."""
    if limit is not None and limit < 1:
        raise ValueError("Task limit must be positive, or omitted for full evaluation")
    if importlib.metadata.version("lm_eval") != "0.4.9.1":
        raise RuntimeError("Use lm-eval==0.4.9.1 for the pinned task definitions")
    from transformers import AutoTokenizer
    from lm_eval import evaluator
    from salaad_moe.lm_eval_adapter import MoEHarnessLM

    d = config["data"]
    evaluation = config["evaluation"]
    if evaluation["num_fewshot"] != 0 or evaluation["primary_task_metric"] != "acc":
        raise ValueError("This runner expects zero-shot tasks scored with acc")
    seed = evaluation["task_seed"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory or d["tokenizer"],
        revision=None if tokenizer_directory else d["tokenizer_revision"],
        trust_remote_code=False,
    )
    if len(tokenizer) != d["tokenizer_length"] or tokenizer.eos_token_id != d["eod_id"]:
        raise ValueError("Evaluation tokenizer disagrees with config")
    adapter = MoEHarnessLM(model, tokenizer, config, device)
    result = evaluator.simple_evaluate(
        model=adapter,
        tasks=evaluation["tasks"],
        num_fewshot=0,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=True,
    )
    accuracies = [result["results"][task]["acc,none"] for task in evaluation["tasks"]]
    result["moe_salaad"] = {
        "mode": mode,
        "six_task_mean_acc": sum(accuracies) / len(accuracies),
        "debug_limit": limit,
        "task_seed": seed,
        "tokenizer_source": tokenizer_directory or d["tokenizer"],
        "tokenizer_revision": d["tokenizer_revision"],
        "scoring_batch_size": 1,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="raw", choices=("raw", "reconstructed", "exported"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, help="Debug-only sample limit; recorded in results")
    parser.add_argument(
        "--tokenizer-directory", help="Otherwise fetch tokenizer only at the pinned revision"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    model, config = load_evaluation_model(args.model, args.mode, args.device)
    result = evaluate_tasks(
        model, config, args.device, mode=args.mode, limit=args.limit,
        tokenizer_directory=args.tokenizer_directory,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["moe_salaad"], indent=2))


if __name__ == "__main__":
    main()
