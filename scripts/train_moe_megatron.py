#!/usr/bin/env python3
"""Launch the pinned Megatron trainer with FP32 SALAAD optimizer hooks."""
from pathlib import Path
import argparse
import json
import os
import runpy
import shlex
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from salaad_moe.config import fingerprint, load_config, validate_config
from salaad_moe.data import TokenCorpus, file_sha256
from salaad_moe.megatron import environment_report, install_training_integration, megatron_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument("--data-directory")
    parser.add_argument("--tokenizer-directory")
    parser.add_argument("--output")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--resume", help="Megatron save root, including its salaad subdirectory")
    restore.add_argument(
        "--branch-from", help="Vanilla Megatron save root at the target SALAAD initialization step"
    )
    parser.add_argument(
        "--stop-after", type=int, help="Exit/checkpoint interval for a schedule-preserving pilot"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated flags; do not import Megatron"
    )
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.check_environment:
        report = environment_report(args.megatron_path)
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(1)
        return
    if not all((args.data_directory, args.tokenizer_directory, args.output)):
        parser.error("Require --data-directory, --tokenizer-directory and --output")
    native_args = megatron_arguments(
        config,
        args.data_directory,
        args.tokenizer_directory,
        args.output,
        args.resume or args.branch_from,
    )
    if args.stop_after is not None:
        if args.stop_after < 1:
            parser.error("--stop-after must be positive")
        native_args += ["--exit-interval", str(args.stop_after)]
    if args.dry_run:
        print(shlex.join([str(Path(args.megatron_path) / "pretrain_gpt.py"), *native_args]))
        return
    report = environment_report(args.megatron_path)
    if not report["ready"]:
        raise RuntimeError("Megatron environment is not ready:\n" + json.dumps(report, indent=2))
    if int(os.environ.get("WORLD_SIZE", "1")) != config["parallel"]["world_size"]:
        raise ValueError("Use torchrun with the configured DP size")
    primary = int(os.environ.get("RANK", 0)) == 0
    corpus = TokenCorpus(Path(args.data_directory) / "manifest.json", config, verify_hashes=primary)
    if not corpus.manifest["provenance"].get("megatron_indexed"):
        raise ValueError("Prepare real text with --write-megatron-indexed first")
    for info in corpus.manifest["provenance"]["indexed_files"]:
        path = Path(args.data_directory) / info["path"]
        if (
            not path.is_file()
            or path.stat().st_size != info["bytes"]
            or (primary and file_sha256(path) != info["sha256"])
        ):
            raise ValueError(f"Indexed corpus changed: {path}")
    token_meta = json.loads((Path(args.tokenizer_directory) / "source_manifest.json").read_text())
    if (
        token_meta["repository"] != config["data"]["tokenizer"]
        or token_meta["revision"] != config["data"]["tokenizer_revision"]
    ):
        raise ValueError("Tokenizer provenance differs from the pinned configuration")
    for info in token_meta["files"]:
        if file_sha256(Path(args.tokenizer_directory) / info["path"]) != info["sha256"]:
            raise ValueError("Local tokenizer assets changed")
    prepared_tokenizer = corpus.manifest["provenance"]["tokenizer"]
    if "local_json_sha256" in prepared_tokenizer:
        if prepared_tokenizer["local_json_sha256"] != file_sha256(
            Path(args.tokenizer_directory) / "tokenizer.json"
        ):
            raise ValueError("Prepared corpus used a different tokenizer JSON")
    elif (
        prepared_tokenizer["repository"] != token_meta["repository"]
        or prepared_tokenizer["revision"] != token_meta["revision"]
    ):
        raise ValueError("Prepared corpus used a different tokenizer revision")
    output = Path(args.output)
    if primary:
        output.mkdir(parents=True, exist_ok=True)
        resolved = output / "moe_salaad_config.json"
        if resolved.exists() and not args.resume:
            raise FileExistsError("Existing Megatron run requires --resume or a new output")
        if resolved.exists() and fingerprint(json.loads(resolved.read_text())) != fingerprint(
            config
        ):
            raise ValueError("Existing Megatron run uses a different configuration")
        resolved.write_text(json.dumps(config, indent=2) + "\n")
        (output / "moe_salaad_environment.json").write_text(json.dumps(report, indent=2) + "\n")
    # Dependency injection at the pinned entrypoint, no on-disk monkey patch.
    sys.path.insert(0, str(Path(args.megatron_path).resolve()))
    import megatron.training.training as training

    install_training_integration(
        training,
        config,
        fingerprint({"corpus": corpus.identity, "tokenizer": token_meta}),
        branch_from_vanilla=bool(args.branch_from),
    )
    sys.argv = [str(Path(args.megatron_path) / "pretrain_gpt.py"), *native_args]
    upstream = runpy.run_path(sys.argv[0], run_name="_moe_salaad_pinned_pretrain")
    provider = upstream["train_valid_test_datasets_provider"]
    original_dataset_config = provider.__globals__["core_gpt_dataset_config_from_args"]

    def dataset_config(native_args):
        result = original_dataset_config(native_args)
        result.random_seed = config["data"]["corpus_order_seed"]
        return result

    provider.__globals__["core_gpt_dataset_config_from_args"] = dataset_config
    provider.is_distributed = True
    upstream["pretrain"](
        provider,
        upstream["model_provider"],
        upstream["ModelType"].encoder_or_decoder,
        upstream["forward_step"],
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


if __name__ == "__main__":
    main()
