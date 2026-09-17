#!/usr/bin/env python3
"""Token-weighted NLL/PPL for raw, reconstructed, or actual exported weights."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from salaad_moe.data import TokenCorpus
from salaad_moe.export import load_evaluation_model
from salaad_moe.trainer import evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Checkpoint directory, or export .pt when mode=exported"
    )
    parser.add_argument("--mode", choices=("raw", "reconstructed", "exported"), default="raw")
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--sequences", type=int, default=0, help="0 evaluates every sequence in the split"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    model, config = load_evaluation_model(args.model, args.mode, args.device)
    corpus = TokenCorpus(args.data_manifest, config)
    result = {
        "mode": args.mode,
        "corpus_identity": corpus.identity,
        "corpus_provenance": corpus.manifest["provenance"],
        **evaluate_model(
            model,
            corpus,
            args.split,
            args.sequences,
            args.batch_size,
            config,
            torch.device(args.device),
            distributed=False,
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
