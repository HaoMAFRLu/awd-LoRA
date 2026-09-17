#!/usr/bin/env python3
"""Fetch pinned tokenizer assets and optionally an explicit list of DCLM shards."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download, snapshot_download
from salaad_moe.config import load_config
from salaad_moe.data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New source directory")
    parser.add_argument(
        "--shard-list",
        help="JSON array of exact dataset repository filenames; omitted means tokenizer only",
    )
    args = parser.parse_args()
    d = load_config(args.config)["data"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    token_dir = output / "tokenizer"
    token_dir.mkdir()
    snapshot = Path(
        snapshot_download(
            d["tokenizer"],
            revision=d["tokenizer_revision"],
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "config.json",
            ],
        )
    )
    files = []
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
    ):
        source = snapshot / filename
        if source.is_file():
            shutil.copyfile(source, token_dir / filename)
            files.append({"path": filename, "sha256": file_sha256(token_dir / filename)})
    (token_dir / "source_manifest.json").write_text(
        json.dumps(
            {"repository": d["tokenizer"], "revision": d["tokenizer_revision"], "files": files},
            indent=2,
        )
        + "\n"
    )
    shards = []
    if args.shard_list:
        names = json.loads(Path(args.shard_list).read_text())
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise ValueError("--shard-list must be a nonempty JSON array of filenames")
        for filename in sorted(set(names)):
            cached = Path(
                hf_hub_download(
                    d["dataset"], filename, repo_type="dataset", revision=d["dataset_revision"]
                )
            )
            # Absolute cache paths avoid duplicating large immutable raw shards.
            shards.append(
                {
                    "repository_path": filename,
                    "local_path": str(cached.absolute()),
                    "bytes": cached.stat().st_size,
                    "sha256": file_sha256(cached),
                }
            )
    manifest = {
        "dataset": d["dataset"],
        "revision": d["dataset_revision"],
        "ordered_shards": shards,
    }
    (output / "shards.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tokenizer_directory": str(token_dir),
                "shard_manifest": str(output / "shards.json"),
                "shards": len(shards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
