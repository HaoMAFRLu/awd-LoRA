"""Immutable, deduplicated text manifests and deterministic packed token streams."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_split(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest, "big") % 10000
    return digest, "validation" if bucket < 100 else "test" if bucket < 200 else "train"


def documents(path, text_key="text"):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=[text_key]):
            yield from batch.column(0).to_pylist()
    elif path.suffix in (".jsonl", ".json", ".gz", ".zst", ".zstd"):
        if path.suffix in (".zst", ".zstd"):
            import zstandard

            opener = zstandard.open
        elif path.suffix == ".gz":
            import gzip

            opener = gzip.open
        else:
            opener = open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)[text_key]
    else:
        raise ValueError(f"Expected JSONL, compressed JSONL (.gz/.zst/.zstd), or Parquet: {path}")


def build_corpus(
    config, inputs, output, tokenizer_json=None, text_key="text", write_megatron=False,
    max_train_tokens=None,
):
    """Streaming preparation with disk-backed exact document deduplication.

    Input paths are sorted and content-hashed BEFORE processing. The source
    manifest is then rechecked, preventing unnoticed source changes mid-build.
    No raw dataset or model weights are downloaded by this function.
    """
    if max_train_tokens is not None and (
        not isinstance(max_train_tokens, int)
        or max_train_tokens < config["data"]["seq_length"]
        or max_train_tokens % config["data"]["seq_length"]
    ):
        raise ValueError("max_train_tokens must be a positive multiple of seq_length")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve filename extensions for HF cache symlinks (the blob target is
    # named by a hash and has no .parquet/.jsonl suffix).
    paths = sorted({Path(p).absolute() for p in inputs})
    if not paths:
        raise ValueError("Explicit input shards are required")
    source = [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in paths]
    d, m = config["data"], config["model"]
    if tokenizer_json:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
        tokenizer_info = {
            "local_json_sha256": file_sha256(tokenizer_json),
            "length": tokenizer.get_vocab_size(with_added_tokens=True),
        }
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            d["tokenizer"], revision=d["tokenizer_revision"], trust_remote_code=False, use_fast=True
        )
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        tokenizer_info = {
            "repository": d["tokenizer"],
            "revision": d["tokenizer_revision"],
            "length": len(tokenizer),
        }
        if len(tokenizer) != d["tokenizer_length"] or tokenizer.eos_token_id != d["eod_id"]:
            raise ValueError("Pinned tokenizer vocabulary/EOD disagrees with config")
    if (
        tokenizer_info["length"] != d["tokenizer_length"]
        or tokenizer_info["length"] > m["padded_vocab_size"]
    ):
        raise ValueError("Tokenizer vocabulary differs from config or exceeds the model vocabulary")
    source_manifest = {
        "dataset": d["dataset"],
        "dataset_revision": d["dataset_revision"],
        "ordered_shards": source,
        "tokenizer": tokenizer_info,
        "text_key": text_key,
        "split": "sha256_utf8_mod_10000_98_1_1",
        "deduplication": "sha256_utf8_exact",
        "eod_id": d["eod_id"],
    }
    if max_train_tokens is not None:
        source_manifest["max_train_prediction_tokens"] = max_train_tokens
    (output / "sources.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    database = sqlite3.connect(str(output / "document_hashes.sqlite"))
    database.execute("CREATE TABLE seen (hash BLOB PRIMARY KEY) WITHOUT ROWID")
    splits = ("train", "validation", "test")
    handles = {s: (output / f"{s}.tokens.bin").open("wb") for s in splits}
    limits = {
        # Keep one extra token to form the final next-token label.
        "train": None if max_train_tokens is None else max_train_tokens + 1,
        "validation": d["validation_sequences"] * d["seq_length"] + 1,
        "test": d["test_sequences"] * d["seq_length"] + 1,
    }
    counts = {s: {"tokens": 0, "documents": 0, "max_token_id": 0} for s in splits}

    def all_splits_full():
        return all(
            limit is not None and counts[split]["tokens"] >= limit
            for split, limit in limits.items()
        )

    builders = {}
    if write_megatron:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

        for split in splits:
            builders[split] = IndexedDatasetBuilder(
                str(output / f"{split}_text_document.bin"), dtype=np.int32
            )
    duplicates, examined = 0, 0
    try:
        for path in paths:
            for text in documents(path, text_key):
                if all_splits_full():
                    break
                if not isinstance(text, str):
                    raise ValueError(f"Non-string text in {path}")
                examined += 1
                digest, split = text_split(text)
                cursor = database.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,))
                if cursor.rowcount == 0:
                    duplicates += 1
                    continue
                if examined % 10000 == 0:
                    database.commit()
                remaining = (
                    None if limits[split] is None else limits[split] - counts[split]["tokens"]
                )
                if remaining is not None and remaining <= 0:
                    continue
                tokens = encode(text) + [d["eod_id"]]
                if any(t < 0 or t >= m["padded_vocab_size"] for t in tokens):
                    raise ValueError("Tokenizer emitted an out-of-vocabulary token")
                if remaining is not None:
                    tokens = tokens[:remaining]
                values = np.asarray(tokens, dtype="<u4")
                handles[split].write(values.tobytes())
                counts[split]["tokens"] += len(values)
                counts[split]["documents"] += 1
                counts[split]["max_token_id"] = max(counts[split]["max_token_id"], max(tokens))
                if builders:
                    builders[split].add_document(tokens, [len(tokens)])
            if all_splits_full():
                break
        database.commit()
    finally:
        database.close()
        for handle in handles.values():
            handle.close()
    for entry in source:
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source changed while preparing data: {entry['path']}")
    for split, builder in builders.items():
        builder.finalize(str(output / f"{split}_text_document.idx"))
    indexed = [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in sorted(output.glob("*_text_document.*"))
    ]
    return finish_manifest(
        output,
        config,
        counts,
        {
            "kind": "text",
            "sources_sha256": file_sha256(output / "sources.json"),
            "tokenizer": tokenizer_info,
            "examined_documents": examined,
            "duplicate_documents": duplicates,
            "megatron_indexed": write_megatron,
            "indexed_files": indexed,
        },
    )


def finish_manifest(output, config, counts, provenance):
    output = Path(output)
    seq = config["data"]["seq_length"]
    for split, values in counts.items():
        path = output / f"{split}.tokens.bin"
        values.update(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "sequences": max(0, (values["tokens"] - 1) // seq),
            }
        )
        if values["sequences"] == 0:
            raise ValueError(
                f"No full {split} sequence. Supply more input shards; no manifest was published."
            )
    manifest = {
        "format": "salaad_moe.tokens.v1",
        "dtype": "<u4",
        "sequence_length": seq,
        "eod_id": config["data"]["eod_id"],
        "padded_vocab_size": config["model"]["padded_vocab_size"],
        "provenance": provenance,
        "splits": counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def make_synthetic_corpus(config, output, sequences=128):
    """Explicit smoke-test fixture. Its manifest permanently records synthetic."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config["seed"])
    counts = {}
    for split in ("train", "validation", "test"):
        n = (sequences if split == "train" else min(sequences, 32)) * config["data"][
            "seq_length"
        ] + 1
        values = rng.integers(0, config["model"]["padded_vocab_size"], n, dtype=np.uint32).astype(
            "<u4"
        )
        values.tofile(output / f"{split}.tokens.bin")
        counts[split] = {"tokens": n, "documents": 0, "max_token_id": int(values.max())}
    return finish_manifest(
        output, config, counts, {"kind": "synthetic_smoke_only", "seed": config["seed"]}
    )


class TokenCorpus:
    """Read-only packed tokens with identity, splits, lengths, and checksums fixed by a manifest."""
    def __init__(self, manifest_path, config, verify_hashes=True):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.identity = fingerprint(self.manifest)
        self.sequence_length = config["data"]["seq_length"]
        if self.manifest["format"] != "salaad_moe.tokens.v1" or self.manifest["dtype"] != "<u4":
            raise ValueError("Unsupported token corpus format")
        if (
            self.manifest["sequence_length"] != self.sequence_length
            or self.manifest["eod_id"] != config["data"]["eod_id"]
        ):
            raise ValueError("Corpus sequence length/EOD differs from config")
        self.arrays, self.lengths = {}, {}
        for split, info in self.manifest["splits"].items():
            path = (self.path.parent / info["path"]).resolve()
            if self.path.parent not in path.parents:
                raise ValueError("Token shard must be inside its manifest directory")
            if path.stat().st_size != info["tokens"] * 4 or path.stat().st_size != info["bytes"]:
                raise ValueError(f"Truncated or changed token shard: {split}")
            if info["max_token_id"] >= config["model"]["padded_vocab_size"]:
                raise ValueError("Corpus token IDs exceed model vocabulary")
            if verify_hashes and file_sha256(path) != info["sha256"]:
                raise ValueError(f"Token shard hash mismatch: {split}")
            self.arrays[split] = np.memmap(path, mode="r", dtype="<u4")
            self.lengths[split] = (info["tokens"] - 1) // self.sequence_length
            if self.lengths[split] < 1 or self.lengths[split] != info["sequences"]:
                raise ValueError(f"Invalid sequence count: {split}")

    def batch(self, split, indices, device):
        # Read T+1 tokens per row to produce inputs and next-token labels
        # that each have length T.
        rows = []
        for index in indices:
            if not 0 <= index < self.lengths[split]:
                raise IndexError(index)
            start = index * self.sequence_length
            rows.append(
                np.asarray(
                    self.arrays[split][start : start + self.sequence_length + 1], dtype=np.int64
                )
            )
        tokens = torch.from_numpy(np.stack(rows)).to(device)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class GlobalBatchReader:
    """Use a deterministic permutation and global cursor to preserve sample order on resume."""

    def __init__(self, corpus, config):
        self.corpus, self.config, self.cursor = corpus, config, 0
        self._permutations = {}

    def sample_index(self, position):
        n = self.corpus.lengths["train"]
        epoch, offset = divmod(position, n)
        if epoch not in self._permutations:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.config["data"]["corpus_order_seed"], epoch])
            )
            a = int(rng.integers(1, max(n, 2)))
            while math.gcd(a, n) != 1:
                a += 1
            self._permutations = {epoch: (a, int(rng.integers(0, n)))}
        a, b = self._permutations[epoch]
        return (a * offset + b) % n

    def batch(self, microstep, rank, world_size, device):
        # Split one global batch by [microstep, rank, micro-batch] so ranks
        # receive distinct sample positions.
        micro = self.config["training"]["micro_batch_sequences_per_rank"]
        start = self.cursor + (microstep * world_size + rank) * micro
        return self.corpus.batch(
            "train", [self.sample_index(start + i) for i in range(micro)], device
        )

    def advance(self):
        # Advance only after the full optimizer step succeeds; checkpoints
        # store this same global position.
        self.cursor += self.config["training"]["global_batch_sequences"]

    def state_dict(self):
        return {
            "global_cursor": self.cursor,
            "corpus_identity": self.corpus.identity,
            "order_seed": self.config["data"]["corpus_order_seed"],
        }

    def load_state_dict(self, state):
        if (
            state["corpus_identity"] != self.corpus.identity
            or state["order_seed"] != self.config["data"]["corpus_order_seed"]
        ):
            raise ValueError("Resume requires the same immutable corpus and order seed")
        self.cursor = int(state["global_cursor"])
