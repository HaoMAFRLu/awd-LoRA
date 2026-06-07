"""Tokenizer-aware nearest-neighbor purity for input embeddings.

This script compares local neighborhoods of input-embedding rows from two
checkpoints:

1. baseline dense embedding and its post-hoc RPCA decomposition;
2. training-time SALAAD embedding decomposition.

The metric is intentionally conservative and does not require external NLP
resources. It uses the training tokenizer to identify readable word-like tokens
and scores whether nearest neighbors are also readable words and whether they
are lexically related by simple string/stem rules.
"""

import csv
import gc
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from salad.utils import mkdir


TOKENIZER_NAME = "t5-base"
VOCAB_SIZE = 32000

BASELINE_FOLDER = "data/baseline_fp32/llama_350m/20251203_102315"
HEAD_FOLDER = "data/head_fp32/llama_350m/20251204_134313"
VANILLA_BF16_FOLDER = "data/vanilla_bf16/llama_350m/20251209_233045"

BASELINE_MODEL = os.path.join(BASELINE_FOLDER, "model.pth")
BASELINE_RPCA = os.path.join(BASELINE_FOLDER, "rpca_X_embed_tokens.pkl")
HEAD_SALAAD_MATRIX = os.path.join(HEAD_FOLDER, "matrix_rank0.pkl")
VANILLA_BF16_MODEL = os.path.join(VANILLA_BF16_FOLDER, "model.pth")
VANILLA_BF16_RPCA = os.path.join(VANILLA_BF16_FOLDER, "rpca_X_embed_tokens.pkl")

OUTPUT_DIR = "data/figures/comparison_embedding/neighbor_purity"

TOP_KS = (5, 10, 20)
MAX_K = max(TOP_KS)
MAX_QUERY_TOKENS = 2000
QUERY_MIN_WORD_LEN = 4
CHUNK_SIZE = 192

REPRESENTATIVE_WORDS = [
    "doctor",
    "medical",
    "hospital",
    "legal",
    "computer",
    "science",
    "government",
    "music",
    "school",
    "market",
    "energy",
    "language",
    "research",
    "family",
    "city",
    "water",
    "history",
    "system",
    "network",
    "training",
]

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def normalize_piece(piece: str) -> str:
    return piece.replace("▁", "").strip().lower()


def simple_stem(word: str) -> str:
    if len(word) <= 4:
        return word
    irregular = {
        "studies": "study",
        "cities": "city",
        "families": "family",
        "companies": "company",
    }
    if word in irregular:
        return irregular[word]
    suffix_rules = [
        ("ization", "ize"),
        ("isations", "ise"),
        ("ation", "ate"),
        ("fulness", "ful"),
        ("iveness", "ive"),
        ("ingly", ""),
        ("edly", ""),
        ("ments", "ment"),
        ("ities", "ity"),
        ("ies", "y"),
        ("ing", ""),
        ("ers", ""),
        ("er", ""),
        ("ed", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ]
    for suffix, repl in suffix_rules:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + repl
    return word


def char_ngrams(word: str, n: int = 3) -> set:
    if len(word) < n:
        return {word}
    return {word[i : i + n] for i in range(len(word) - n + 1)}


def lexical_relation(a: str, b: str) -> Tuple[bool, str]:
    if not a or not b or a == b:
        return bool(a and b and a == b), "same_lower" if a and b and a == b else "none"

    sa = simple_stem(a)
    sb = simple_stem(b)
    if sa == sb and len(sa) >= 4:
        return True, "same_stem"

    shorter = min(len(a), len(b))
    common_prefix = os.path.commonprefix([a, b])
    if len(common_prefix) >= 5 and len(common_prefix) / shorter >= 0.6:
        return True, "prefix_family"

    common_suffix = os.path.commonprefix([a[::-1], b[::-1]])[::-1]
    if len(common_suffix) >= 5 and len(common_suffix) / shorter >= 0.6:
        return True, "suffix_family"

    ga = char_ngrams(a)
    gb = char_ngrams(b)
    jaccard = len(ga & gb) / max(1, len(ga | gb))
    if jaccard >= 0.45 and shorter >= 5:
        return True, "char_ngram"

    return False, "none"


def categorize_token(idx: int, piece: str, special_ids: set) -> Dict[str, object]:
    text = piece.replace("▁", "").strip()
    norm = text.lower()
    has_boundary = piece.startswith("▁")

    if idx in special_ids:
        category = "special"
    elif has_boundary and re.fullmatch(r"[A-Za-z]+", text or "") and len(text) >= 3:
        category = "function_word" if norm in STOPWORDS else "content_word"
    elif re.fullmatch(r"[A-Za-z]+", text or ""):
        category = "alpha_fragment"
    elif re.fullmatch(r"\d+", text or ""):
        category = "number"
    elif re.fullmatch(r"[\W_]+", text or ""):
        category = "punct"
    else:
        category = "other"

    return {
        "id": idx,
        "piece": piece,
        "text": text,
        "norm": norm,
        "has_boundary": has_boundary,
        "category": category,
    }


def build_token_metadata() -> List[Dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    metadata = []
    for idx in range(VOCAB_SIZE):
        piece = tokenizer.convert_ids_to_tokens(idx)
        metadata.append(categorize_token(idx, piece, special_ids))
    return metadata


def select_query_ids(metadata: List[Dict[str, object]]) -> List[int]:
    candidates = [
        item["id"]
        for item in metadata
        if item["category"] == "content_word"
        and len(item["norm"]) >= QUERY_MIN_WORD_LEN
        and item["norm"] not in STOPWORDS
    ]

    if len(candidates) <= MAX_QUERY_TOKENS:
        selected = candidates
    else:
        # SentencePiece ids roughly follow token frequency. Quantile sampling keeps
        # both common and rare readable words without requiring corpus statistics.
        positions = np.linspace(0, len(candidates) - 1, MAX_QUERY_TOKENS)
        selected = [candidates[int(round(pos))] for pos in positions]

    by_norm = defaultdict(list)
    for item in metadata:
        if item["category"] == "content_word":
            by_norm[item["norm"]].append(item["id"])

    for word in REPRESENTATIVE_WORDS:
        ids = by_norm.get(word)
        if ids:
            selected.append(ids[0])

    return sorted(set(selected))


def load_state_dict_embedding(path: str) -> torch.Tensor:
    state = torch.load(path, map_location="cpu")
    for key in (
        "model.embed_tokens.weight",
        "module.model.embed_tokens.weight",
        "embed_tokens.weight",
    ):
        if key in state:
            return state[key].detach().float().cpu()
    matches = [key for key in state if key.endswith("embed_tokens.weight")]
    if not matches:
        raise KeyError(f"No embed_tokens weight found in {path}")
    return state[matches[0]].detach().float().cpu()


def load_pickle_embedding(path: str, mode: str) -> torch.Tensor:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    L = obj["LL"]["embed_tokens"].detach().float().cpu()
    if mode == "L":
        return L
    if mode == "L_plus_S":
        S = obj["SS"]["embed_tokens"].detach().float().cpu()
        return L + S
    raise ValueError(f"Unsupported mode: {mode}")


def iter_variants() -> Iterable[Tuple[str, str, callable]]:
    yield (
        "baseline_dense_E",
        BASELINE_MODEL,
        lambda: load_state_dict_embedding(BASELINE_MODEL),
    )
    yield (
        "baseline_posthoc_rpca_L",
        BASELINE_RPCA,
        lambda: load_pickle_embedding(BASELINE_RPCA, "L"),
    )
    yield (
        "baseline_posthoc_rpca_L_plus_S",
        BASELINE_RPCA,
        lambda: load_pickle_embedding(BASELINE_RPCA, "L_plus_S"),
    )
    yield (
        "training_salaad_L",
        HEAD_SALAAD_MATRIX,
        lambda: load_pickle_embedding(HEAD_SALAAD_MATRIX, "L"),
    )
    yield (
        "training_salaad_L_plus_S",
        HEAD_SALAAD_MATRIX,
        lambda: load_pickle_embedding(HEAD_SALAAD_MATRIX, "L_plus_S"),
    )
    yield (
        "vanilla_bf16_dense_E",
        VANILLA_BF16_MODEL,
        lambda: load_state_dict_embedding(VANILLA_BF16_MODEL),
    )
    yield (
        "vanilla_bf16_posthoc_rpca_L",
        VANILLA_BF16_RPCA,
        lambda: load_pickle_embedding(VANILLA_BF16_RPCA, "L"),
    )
    yield (
        "vanilla_bf16_posthoc_rpca_L_plus_S",
        VANILLA_BF16_RPCA,
        lambda: load_pickle_embedding(VANILLA_BF16_RPCA, "L_plus_S"),
    )


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix[:VOCAB_SIZE].float()
    norms = torch.linalg.vector_norm(matrix, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
    return matrix / norms


def topk_neighbors(
    matrix: torch.Tensor,
    query_ids: List[int],
    candidate_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    emb = normalize_rows(matrix)
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_pos = {idx: pos for pos, idx in enumerate(candidate_ids)}
    candidate_emb_t = emb[candidate_tensor].T.contiguous()
    query_tensor = torch.tensor(query_ids, dtype=torch.long)
    top_ids = []
    top_scores = []

    for start in range(0, len(query_ids), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(query_ids))
        chunk_ids = query_tensor[start:end]
        sims = emb[chunk_ids] @ candidate_emb_t
        for row, query_id in enumerate(chunk_ids.tolist()):
            pos = candidate_pos.get(query_id)
            if pos is not None:
                sims[row, pos] = -float("inf")
        values, indices = torch.topk(sims, k=MAX_K, dim=1)
        actual_ids = candidate_tensor[indices.reshape(-1)].reshape(indices.shape)
        top_ids.append(actual_ids.cpu().numpy())
        top_scores.append(values.cpu().numpy())
        del sims, values, indices, actual_ids

    del emb, candidate_emb_t
    gc.collect()
    return np.vstack(top_ids), np.vstack(top_scores)


def score_variant(
    variant: str,
    query_ids: List[int],
    top_ids: np.ndarray,
    top_scores: np.ndarray,
    metadata: List[Dict[str, object]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    per_query = []
    aggregate = {
        k: {
            "lexical_purity": [],
            "content_word_rate": [],
            "word_like_rate": [],
            "artifact_rate": [],
            "mean_similarity": [],
            "relation_counts": Counter(),
            "neighbor_category_counts": Counter(),
        }
        for k in TOP_KS
    }

    for row_idx, query_id in enumerate(query_ids):
        q_meta = metadata[query_id]
        q_norm = q_meta["norm"]
        row = {
            "variant": variant,
            "query_id": query_id,
            "query_piece": q_meta["piece"],
            "query_text": q_meta["text"],
            "query_norm": q_norm,
        }

        neighbor_records = []
        for pos, neighbor_id in enumerate(top_ids[row_idx].tolist(), start=1):
            n_meta = metadata[neighbor_id]
            related, reason = lexical_relation(q_norm, n_meta["norm"])
            neighbor_records.append(
                {
                    "rank": pos,
                    "id": neighbor_id,
                    "piece": n_meta["piece"],
                    "text": n_meta["text"],
                    "norm": n_meta["norm"],
                    "category": n_meta["category"],
                    "score": float(top_scores[row_idx, pos - 1]),
                    "related": related,
                    "relation": reason,
                }
            )

        for k in TOP_KS:
            subset = neighbor_records[:k]
            related_count = sum(item["related"] for item in subset)
            content_count = sum(item["category"] == "content_word" for item in subset)
            word_like_count = sum(
                item["category"] in {"content_word", "function_word"} for item in subset
            )
            category_counts = Counter(item["category"] for item in subset)
            relation_counts = Counter(item["relation"] for item in subset)

            lexical_purity = related_count / k
            content_rate = content_count / k
            word_like_rate = word_like_count / k
            artifact_rate = 1.0 - word_like_rate
            mean_similarity = float(np.mean([item["score"] for item in subset]))

            row[f"lexical_purity@{k}"] = lexical_purity
            row[f"content_word_rate@{k}"] = content_rate
            row[f"word_like_rate@{k}"] = word_like_rate
            row[f"artifact_rate@{k}"] = artifact_rate
            row[f"mean_similarity@{k}"] = mean_similarity

            aggregate[k]["lexical_purity"].append(lexical_purity)
            aggregate[k]["content_word_rate"].append(content_rate)
            aggregate[k]["word_like_rate"].append(word_like_rate)
            aggregate[k]["artifact_rate"].append(artifact_rate)
            aggregate[k]["mean_similarity"].append(mean_similarity)
            aggregate[k]["neighbor_category_counts"].update(category_counts)
            aggregate[k]["relation_counts"].update(relation_counts)

        row["neighbors"] = neighbor_records
        per_query.append(row)

    summary = {}
    for k, values in aggregate.items():
        summary[f"lexical_purity@{k}"] = float(np.mean(values["lexical_purity"]))
        summary[f"content_word_rate@{k}"] = float(np.mean(values["content_word_rate"]))
        summary[f"word_like_rate@{k}"] = float(np.mean(values["word_like_rate"]))
        summary[f"artifact_rate@{k}"] = float(np.mean(values["artifact_rate"]))
        summary[f"mean_similarity@{k}"] = float(np.mean(values["mean_similarity"]))
        denom = len(query_ids) * k
        summary[f"neighbor_category_distribution@{k}"] = {
            name: count / denom for name, count in values["neighbor_category_counts"].items()
        }
        summary[f"relation_distribution@{k}"] = {
            name: count / denom for name, count in values["relation_counts"].items()
        }

    return summary, per_query


def write_per_query_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "query_id",
        "query_piece",
        "query_text",
        "query_norm",
    ]
    for k in TOP_KS:
        fieldnames.extend(
            [
                f"lexical_purity@{k}",
                f"content_word_rate@{k}",
                f"word_like_rate@{k}",
                f"artifact_rate@{k}",
                f"mean_similarity@{k}",
            ]
        )
    fieldnames.extend(["top_neighbor_ids", "top_neighbor_pieces", "top_neighbor_categories", "top_neighbor_related"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            compact = {key: row.get(key) for key in fieldnames if key in row}
            neighbors = row["neighbors"]
            compact["top_neighbor_ids"] = " ".join(str(item["id"]) for item in neighbors)
            compact["top_neighbor_pieces"] = " ".join(item["piece"] for item in neighbors)
            compact["top_neighbor_categories"] = " ".join(item["category"] for item in neighbors)
            compact["top_neighbor_related"] = " ".join("1" if item["related"] else "0" for item in neighbors)
            writer.writerow(compact)


def build_qualitative_examples(
    all_rows: Dict[str, List[Dict[str, object]]],
    metadata: List[Dict[str, object]],
) -> Dict[str, object]:
    by_variant_and_query = {
        variant: {row["query_id"]: row for row in rows}
        for variant, rows in all_rows.items()
    }
    norm_to_id = {}
    for item in metadata:
        if item["category"] == "content_word" and item["norm"] not in norm_to_id:
            norm_to_id[item["norm"]] = item["id"]

    examples = {}
    for word in REPRESENTATIVE_WORDS:
        query_id = norm_to_id.get(word)
        if query_id is None:
            continue
        examples[word] = {}
        for variant, rows_by_query in by_variant_and_query.items():
            row = rows_by_query.get(query_id)
            if row is None:
                continue
            examples[word][variant] = [
                {
                    "rank": item["rank"],
                    "piece": item["piece"],
                    "text": item["text"],
                    "category": item["category"],
                    "score": item["score"],
                    "related": item["related"],
                    "relation": item["relation"],
                }
                for item in row["neighbors"][:10]
            ]

    if "baseline_dense_E" in by_variant_and_query and "training_salaad_L" in by_variant_and_query:
        diffs = []
        base = by_variant_and_query["baseline_dense_E"]
        salaad = by_variant_and_query["training_salaad_L"]
        for query_id, base_row in base.items():
            if query_id not in salaad:
                continue
            diff = salaad[query_id]["lexical_purity@10"] - base_row["lexical_purity@10"]
            diffs.append((diff, query_id))
        diffs.sort(reverse=True)
        examples["largest_purity10_gains_salaad_L_vs_dense"] = [
            {
                "query_id": query_id,
                "query_piece": metadata[query_id]["piece"],
                "query_norm": metadata[query_id]["norm"],
                "delta": float(diff),
                "dense_neighbors": [
                    item["piece"] for item in base[query_id]["neighbors"][:10]
                ],
                "salaad_L_neighbors": [
                    item["piece"] for item in salaad[query_id]["neighbors"][:10]
                ],
            }
            for diff, query_id in diffs[:25]
        ]
        examples["largest_purity10_losses_salaad_L_vs_dense"] = [
            {
                "query_id": query_id,
                "query_piece": metadata[query_id]["piece"],
                "query_norm": metadata[query_id]["norm"],
                "delta": float(diff),
                "dense_neighbors": [
                    item["piece"] for item in base[query_id]["neighbors"][:10]
                ],
                "salaad_L_neighbors": [
                    item["piece"] for item in salaad[query_id]["neighbors"][:10]
                ],
            }
            for diff, query_id in diffs[-25:]
        ]

    return examples


def plot_summary(pool_name: str, pool_summary: Dict[str, object], output_dir: str) -> str:
    variants = list(pool_summary["variants"].keys())
    labels = [name.replace("_", "\n") for name in variants]
    x = np.arange(len(variants))
    width = 0.22

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for offset, k in zip((-width, 0.0, width), TOP_KS):
        axs[0].bar(
            x + offset,
            [pool_summary["variants"][name][f"lexical_purity@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )
        axs[1].bar(
            x + offset,
            [pool_summary["variants"][name][f"content_word_rate@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )
        axs[2].bar(
            x + offset,
            [pool_summary["variants"][name][f"artifact_rate@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )

    axs[0].set_title(f"Lexical neighbor purity ({pool_name})")
    axs[1].set_title(f"Content-word neighbor rate ({pool_name})")
    axs[2].set_title(f"Artifact neighbor rate ({pool_name})")
    for ax in axs:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    fig_path = os.path.join(output_dir, f"neighbor_purity_summary_{pool_name}.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def main() -> None:
    mkdir(OUTPUT_DIR)
    metadata = build_token_metadata()
    query_ids = select_query_ids(metadata)
    category_counts = Counter(item["category"] for item in metadata)
    candidate_pools = {
        "all_tokens": list(range(VOCAB_SIZE)),
        "content_words_only": [
            item["id"] for item in metadata if item["category"] == "content_word"
        ],
    }

    print(f"[tokens] category_counts={dict(category_counts)}")
    print(f"[tokens] selected_queries={len(query_ids)}")

    summary = {
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": VOCAB_SIZE,
        "top_ks": list(TOP_KS),
        "max_query_tokens": MAX_QUERY_TOKENS,
        "query_min_word_len": QUERY_MIN_WORD_LEN,
        "query_count": len(query_ids),
        "token_category_counts": dict(category_counts),
        "candidate_pool_sizes": {
            name: len(ids) for name, ids in candidate_pools.items()
        },
        "query_tokens": [
            {
                "id": idx,
                "piece": metadata[idx]["piece"],
                "text": metadata[idx]["text"],
                "norm": metadata[idx]["norm"],
            }
            for idx in query_ids
        ],
        "candidate_pools": {
            name: {"candidate_count": len(ids), "variants": {}}
            for name, ids in candidate_pools.items()
        },
    }

    all_rows_by_pool = {name: {} for name in candidate_pools}
    for variant, source_path, loader in iter_variants():
        print(f"[variant] loading {variant}: {source_path}", flush=True)
        matrix = loader()
        print(f"[variant] matrix shape={tuple(matrix.shape)} dtype={matrix.dtype}", flush=True)

        for pool_name, candidate_ids in candidate_pools.items():
            print(f"[variant] scoring {variant} with candidate_pool={pool_name}", flush=True)
            top_ids, top_scores = topk_neighbors(matrix, query_ids, candidate_ids)
            variant_summary, rows = score_variant(
                variant, query_ids, top_ids, top_scores, metadata
            )
            variant_summary["source_path"] = source_path
            variant_summary["candidate_pool"] = pool_name
            summary["candidate_pools"][pool_name]["variants"][variant] = variant_summary
            all_rows_by_pool[pool_name][variant] = rows

            csv_path = os.path.join(OUTPUT_DIR, f"{variant}_{pool_name}_per_query.csv")
            write_per_query_csv(csv_path, rows)
            print(f"[variant] saved per-query CSV: {csv_path}", flush=True)

        del matrix
        gc.collect()

    for pool_name, rows_by_variant in all_rows_by_pool.items():
        summary["candidate_pools"][pool_name]["qualitative_examples"] = (
            build_qualitative_examples(rows_by_variant, metadata)
        )

    summary_path = os.path.join(OUTPUT_DIR, "neighbor_purity_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fig_paths = [
        plot_summary(pool_name, pool_summary, OUTPUT_DIR)
        for pool_name, pool_summary in summary["candidate_pools"].items()
    ]
    print(f"[saved] {summary_path}")
    for fig_path in fig_paths:
        print(f"[saved] {fig_path}")
    compact = {
        key: value
        for key, value in summary.items()
        if key != "query_tokens"
    }
    for pool_summary in compact["candidate_pools"].values():
        pool_summary.pop("qualitative_examples", None)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
