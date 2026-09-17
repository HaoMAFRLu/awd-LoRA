"""Prepare the device, data parallelism, output directory, and token corpus.

This module handles setup before training. The training algorithm lives in
trainer.py, and the SALAAD decomposition lives in solver.py.
"""
from datetime import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .data import TokenCorpus, make_synthetic_corpus
from .distributed import agree_or_raise, rank


def initialize_device(device_name=None, cpu_threads=4):
    """Set up the device and a real process group, including one-rank debugging."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    device_name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if device_name == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if int(os.environ.get("WORLD_SIZE", 1)) == 1:
            # A real one-rank group exercises broadcasts and reductions in the IDE.
            # An in-process store needs no master address or rendezvous port.
            dist.init_process_group(backend, store=dist.HashStore(), rank=0, world_size=1)
        else:
            # torchrun provides the rendezvous settings for multiple ranks.
            dist.init_process_group(backend)
    return device


def resolve_output_directory(config_path, folder, output, resume, device):
    """Choose the directory on rank 0 and broadcast it to avoid timestamp mismatches."""
    paths, error = [None], None
    if rank() == 0:
        try:
            if output:
                path = Path(output)
            elif resume:
                # Checkpoints live at <run>/checkpoints/step_xxxxxxxx.
                path = Path(resume).resolve().parents[1]
            else:
                path = (
                    Path(__file__).resolve().parents[1]
                    / "data" / folder / Path(config_path).stem
                    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
            paths[0] = str(path.resolve())
        except Exception as exc:
            error = exc
    agree_or_raise(error, device, "Output path")
    if dist.is_initialized():
        dist.broadcast_object_list(paths, src=0)
    return Path(paths[0])


def prepare_corpus(config, output, device, data_manifest=None, resume=None,
                   branch_from=None, allow_synthetic=False):
    """Load packed tokens; only smoke configs may generate a random corpus automatically."""
    manifest = data_manifest or config["data"].get("data_manifest")
    error = None
    try:
        source = resume or branch_from
        if not manifest and source:
            source_root = Path(source).resolve().parents[1]
            metadata = source_root / "run_metadata.json"
            if metadata.is_file():
                manifest = json.loads(metadata.read_text()).get("data_manifest")
            if not manifest and (source_root / "data/manifest.json").is_file():
                manifest = str(source_root / "data/manifest.json")
        if not manifest and config["data"].get("synthetic_smoke", False):
            manifest = str(output / "data/manifest.json")
            if rank() == 0 and not Path(manifest).is_file():
                make_synthetic_corpus(config, output / "data")
        if not manifest:
            raise ValueError("Training requires --data-manifest (automatic data is smoke-only)")
    except Exception as exc:
        error = exc
    # This collective check also ensures rank 0 finishes writing the smoke
    # corpus before other ranks start reading it.
    agree_or_raise(error, device, "Data preparation")

    corpus, error = None, None
    try:
        corpus = TokenCorpus(manifest, config, verify_hashes=rank() == 0)
        if (
            corpus.manifest["provenance"]["kind"] == "synthetic_smoke_only"
            and not allow_synthetic
            and not config["data"].get("synthetic_smoke", False)
        ):
            raise ValueError("Synthetic data is smoke-only; pass --allow-synthetic explicitly")
    except Exception as exc:
        error = exc
    agree_or_raise(error, device, "Corpus validation")
    return corpus
