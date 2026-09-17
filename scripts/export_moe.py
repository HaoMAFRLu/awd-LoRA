#!/usr/bin/env python3
"""Export trained shared/L/S values without pruning or precision conversion."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from salaad_moe.export import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New .pt artifact path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    result = export_checkpoint(args.checkpoint, args.output, device=args.device)
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
