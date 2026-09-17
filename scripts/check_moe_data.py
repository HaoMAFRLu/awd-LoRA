#!/usr/bin/env python3
"""Check packed corpus sizes before a cluster workflow requests training GPUs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salaad_moe.config import load_config, parameter_counts, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-manifest", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    manifest_path = Path(args.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    seq = config["data"]["seq_length"]
    if manifest["sequence_length"] != seq or manifest["provenance"]["kind"] != "text":
        raise ValueError("Expected real text packed to the configured sequence length")
    required = {
        "train": parameter_counts(config)["prediction_tokens"],
        "validation": config["data"]["validation_sequences"] * seq,
        "test": config["data"]["test_sequences"] * seq,
    }
    for split, prediction_tokens in required.items():
        info = manifest["splits"][split]
        if info["tokens"] - 1 < prediction_tokens:
            raise ValueError(
                f"{split}: need {prediction_tokens:,} prediction tokens, "
                f"found {info['tokens'] - 1:,}"
            )
        path = manifest_path.parent / info["path"]
        if path.stat().st_size != info["tokens"] * 4:
            raise ValueError(f"Token file size differs from the manifest: {path}")
        print(f"{split}: {info['tokens'] - 1:,} prediction tokens", flush=True)
    # The training entry point verifies hashes when it opens the corpus.


if __name__ == "__main__":
    main()
