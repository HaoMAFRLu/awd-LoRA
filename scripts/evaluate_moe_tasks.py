#!/usr/bin/env python3
"""Run the six planned zero-shot tasks through the custom likelihood adapter."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from salaad_moe.export import load_evaluation_model


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
    if importlib.metadata.version("lm_eval") != "0.4.9.1":
        raise RuntimeError("Use lm-eval==0.4.9.1 for the pinned task definitions")
    from transformers import AutoTokenizer
    from lm_eval import evaluator
    from salaad_moe.lm_eval_adapter import MoEHarnessLM

    model, config = load_evaluation_model(args.model, args.mode, args.device)
    d = config["data"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_directory or d["tokenizer"],
        revision=None if args.tokenizer_directory else d["tokenizer_revision"],
        trust_remote_code=False,
    )
    if len(tokenizer) != d["tokenizer_length"] or tokenizer.eos_token_id != d["eod_id"]:
        raise ValueError("Evaluation tokenizer disagrees with config")
    adapter = MoEHarnessLM(model, tokenizer, config, args.device)
    result = evaluator.simple_evaluate(
        model=adapter,
        tasks=config["evaluation"]["tasks"],
        num_fewshot=0,
        limit=args.limit,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
        log_samples=True,
    )
    accuracies = [result["results"][task]["acc,none"] for task in config["evaluation"]["tasks"]]
    result["moe_salaad"] = {
        "mode": args.mode,
        "six_task_mean_acc": sum(accuracies) / len(accuracies),
        "debug_limit": args.limit,
        "tokenizer_revision": d["tokenizer_revision"],
        "scoring_batch_size": 1,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["moe_salaad"], indent=2))


if __name__ == "__main__":
    main()
