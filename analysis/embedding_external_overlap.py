"""External-embedding nearest-neighbor overlap for input embeddings.

This analysis uses a locally cached pretrained Llama-3.2-1B input embedding as
an external semantic reference. It maps readable T5 tokens to words, keeps only
words that are single tokens under the Llama tokenizer, and compares nearest
neighbor word sets:

    model embedding neighbors vs pretrained Llama embedding neighbors

This is stronger than string-only purity because the external reference can put
words like "doctor", "nurse", and "hospital" near one another even when their
strings are not morphologically similar.
"""

import csv
import gc
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.embedding_neighbor_purity import (
    BASELINE_MODEL,
    BASELINE_RPCA,
    CHUNK_SIZE,
    HEAD_SALAAD_MATRIX,
    MAX_QUERY_TOKENS,
    OUTPUT_DIR as STRING_OUTPUT_DIR,
    QUERY_MIN_WORD_LEN,
    REPRESENTATIVE_WORDS,
    STOPWORDS,
    TOP_KS,
    VOCAB_SIZE,
    build_token_metadata,
    load_pickle_embedding,
    load_state_dict_embedding,
    normalize_rows,
)
from salad.utils import mkdir


LLAMA_SNAPSHOT = (
    "/home/hao/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B/"
    "snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
)
LLAMA_SAFETENSORS = os.path.join(LLAMA_SNAPSHOT, "model.safetensors")

OUTPUT_DIR = "data/figures/comparison_embedding/external_overlap"
MAX_K = max(TOP_KS)


def iter_variants():
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


def llama_single_token_map(words: List[str]) -> Dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_SNAPSHOT, local_files_only=True)
    word_to_llama_id = {}
    for word in sorted(set(words)):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            word_to_llama_id[word] = int(ids[0])
    return word_to_llama_id


def build_word_inventory(metadata):
    norm_to_t5_ids = defaultdict(list)
    for item in metadata:
        if (
            item["category"] == "content_word"
            and len(item["norm"]) >= QUERY_MIN_WORD_LEN
            and item["norm"] not in STOPWORDS
        ):
            norm_to_t5_ids[item["norm"]].append(item["id"])

    word_to_llama_id = llama_single_token_map(list(norm_to_t5_ids))
    valid_words = sorted(word_to_llama_id)
    valid_word_set = set(valid_words)

    candidate_ids = [
        item["id"]
        for item in metadata
        if item["category"] == "content_word" and item["norm"] in valid_word_set
    ]

    query_words = valid_words
    if len(query_words) > MAX_QUERY_TOKENS:
        positions = np.linspace(0, len(query_words) - 1, MAX_QUERY_TOKENS)
        query_words = [query_words[int(round(pos))] for pos in positions]

    for word in REPRESENTATIVE_WORDS:
        if word in valid_word_set:
            query_words.append(word)
    query_words = sorted(set(query_words))

    query_ids = [norm_to_t5_ids[word][0] for word in query_words]

    return {
        "norm_to_t5_ids": dict(norm_to_t5_ids),
        "word_to_llama_id": word_to_llama_id,
        "valid_words": valid_words,
        "candidate_ids": candidate_ids,
        "query_words": query_words,
        "query_ids": query_ids,
    }


def load_llama_external_vectors(word_to_llama_id: Dict[str, int], words: List[str]) -> torch.Tensor:
    llama_ids = torch.tensor([word_to_llama_id[word] for word in words], dtype=torch.long)
    print(f"[llama] loading embedding from {LLAMA_SAFETENSORS}", flush=True)
    with safe_open(LLAMA_SAFETENSORS, framework="pt", device="cpu") as f:
        full = f.get_tensor("model.embed_tokens.weight").float()
    vectors = full[llama_ids].contiguous()
    del full
    gc.collect()
    return vectors


def topk_external_words(vectors: torch.Tensor, words: List[str], query_words: List[str]):
    emb = normalize_rows(vectors)
    word_to_pos = {word: pos for pos, word in enumerate(words)}
    query_pos = torch.tensor([word_to_pos[word] for word in query_words], dtype=torch.long)
    all_t = emb.T.contiguous()
    top_words = {}
    top_scores = {}

    for start in range(0, len(query_words), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(query_words))
        qpos = query_pos[start:end]
        sims = emb[qpos] @ all_t
        sims[torch.arange(end - start), qpos] = -float("inf")
        values, indices = torch.topk(sims, k=MAX_K, dim=1)
        for row, word in enumerate(query_words[start:end]):
            ids = indices[row].tolist()
            top_words[word] = [words[i] for i in ids]
            top_scores[word] = [float(x) for x in values[row].tolist()]
        del sims, values, indices

    del emb, all_t
    gc.collect()
    return top_words, top_scores


def topk_model_words(
    matrix: torch.Tensor,
    query_ids: List[int],
    candidate_ids: List[int],
    metadata,
) -> Tuple[Dict[int, List[str]], Dict[int, List[float]]]:
    emb = normalize_rows(matrix)
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_emb_t = emb[candidate_tensor].T.contiguous()

    norm_to_candidate_positions = defaultdict(list)
    for pos, token_id in enumerate(candidate_ids):
        norm_to_candidate_positions[metadata[token_id]["norm"]].append(pos)

    query_tensor = torch.tensor(query_ids, dtype=torch.long)
    top_words = {}
    top_scores = {}

    # Retrieve more than MAX_K token neighbors because duplicate word forms
    # such as "Research" and "research" are collapsed to one normalized word.
    retrieve_k = min(MAX_K * 4, len(candidate_ids) - 1)
    for start in range(0, len(query_ids), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(query_ids))
        chunk_ids = query_tensor[start:end]
        sims = emb[chunk_ids] @ candidate_emb_t

        for row, query_id in enumerate(chunk_ids.tolist()):
            q_norm = metadata[query_id]["norm"]
            for pos in norm_to_candidate_positions.get(q_norm, []):
                sims[row, pos] = -float("inf")

        values, indices = torch.topk(sims, k=retrieve_k, dim=1)
        actual_ids = candidate_tensor[indices.reshape(-1)].reshape(indices.shape)

        for row, query_id in enumerate(chunk_ids.tolist()):
            words = []
            scores = []
            seen = set()
            for token_id, score in zip(actual_ids[row].tolist(), values[row].tolist()):
                word = metadata[token_id]["norm"]
                if word in seen:
                    continue
                seen.add(word)
                words.append(word)
                scores.append(float(score))
                if len(words) == MAX_K:
                    break
            top_words[query_id] = words
            top_scores[query_id] = scores

        del sims, values, indices, actual_ids

    del emb, candidate_emb_t
    gc.collect()
    return top_words, top_scores


def score_overlap(
    variant: str,
    query_ids: List[int],
    query_words: List[str],
    model_neighbors: Dict[int, List[str]],
    model_scores: Dict[int, List[float]],
    external_neighbors: Dict[str, List[str]],
    external_scores: Dict[str, List[float]],
    metadata,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rows = []
    summary_values = {
        k: {
            "overlap": [],
            "recall": [],
            "jaccard": [],
            "mean_similarity": [],
        }
        for k in TOP_KS
    }

    for query_id, word in zip(query_ids, query_words):
        m_words = model_neighbors[query_id]
        e_words = external_neighbors[word]
        row = {
            "variant": variant,
            "query_id": query_id,
            "query_piece": metadata[query_id]["piece"],
            "query_word": word,
            "model_neighbors": m_words,
            "external_neighbors": e_words,
            "model_scores": model_scores[query_id],
            "external_scores": external_scores[word],
        }

        for k in TOP_KS:
            m_set = set(m_words[:k])
            e_set = set(e_words[:k])
            inter = m_set & e_set
            union = m_set | e_set
            overlap = len(inter) / k
            recall = len(inter) / max(1, len(e_set))
            jaccard = len(inter) / max(1, len(union))
            mean_similarity = float(np.mean(model_scores[query_id][:k]))

            row[f"external_overlap@{k}"] = overlap
            row[f"external_recall@{k}"] = recall
            row[f"external_jaccard@{k}"] = jaccard
            row[f"mean_model_similarity@{k}"] = mean_similarity
            row[f"overlap_words@{k}"] = sorted(inter)

            summary_values[k]["overlap"].append(overlap)
            summary_values[k]["recall"].append(recall)
            summary_values[k]["jaccard"].append(jaccard)
            summary_values[k]["mean_similarity"].append(mean_similarity)

        rows.append(row)

    summary = {}
    for k, values in summary_values.items():
        summary[f"external_overlap@{k}"] = float(np.mean(values["overlap"]))
        summary[f"external_recall@{k}"] = float(np.mean(values["recall"]))
        summary[f"external_jaccard@{k}"] = float(np.mean(values["jaccard"]))
        summary[f"mean_model_similarity@{k}"] = float(np.mean(values["mean_similarity"]))
    return summary, rows


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    fields = [
        "variant",
        "query_id",
        "query_piece",
        "query_word",
    ]
    for k in TOP_KS:
        fields.extend(
            [
                f"external_overlap@{k}",
                f"external_recall@{k}",
                f"external_jaccard@{k}",
                f"mean_model_similarity@{k}",
                f"overlap_words@{k}",
            ]
        )
    fields.extend(["model_neighbors", "external_neighbors"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compact = {key: row.get(key) for key in fields if key in row}
            compact["model_neighbors"] = " ".join(row["model_neighbors"])
            compact["external_neighbors"] = " ".join(row["external_neighbors"])
            for k in TOP_KS:
                compact[f"overlap_words@{k}"] = " ".join(row[f"overlap_words@{k}"])
            writer.writerow(compact)


def build_examples(rows_by_variant):
    examples = {}
    for word in REPRESENTATIVE_WORDS:
        examples[word] = {}
        for variant, rows in rows_by_variant.items():
            match = next((row for row in rows if row["query_word"] == word), None)
            if match is None:
                continue
            examples[word][variant] = {
                "model_neighbors": match["model_neighbors"][:10],
                "external_neighbors": match["external_neighbors"][:10],
                "overlap@10": match["overlap_words@10"],
            }
        if not examples[word]:
            del examples[word]
    return examples


def plot_summary(summary):
    variants = list(summary["variants"])
    labels = [name.replace("_", "\n") for name in variants]
    x = np.arange(len(variants))
    width = 0.22
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for offset, k in zip((-width, 0.0, width), TOP_KS):
        axs[0].bar(
            x + offset,
            [summary["variants"][name][f"external_overlap@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )
        axs[1].bar(
            x + offset,
            [summary["variants"][name][f"external_jaccard@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )
        axs[2].bar(
            x + offset,
            [summary["variants"][name][f"mean_model_similarity@{k}"] for name in variants],
            width,
            label=f"@{k}",
        )

    axs[0].set_title("Overlap with Llama neighbors")
    axs[1].set_title("Jaccard with Llama neighbors")
    axs[2].set_title("Model neighbor cosine")
    for ax in axs:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.legend()
    axs[0].set_ylim(0, max(0.05, axs[0].get_ylim()[1]))
    axs[1].set_ylim(0, max(0.05, axs[1].get_ylim()[1]))

    path = os.path.join(OUTPUT_DIR, "external_embedding_overlap_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    mkdir(OUTPUT_DIR)
    metadata = build_token_metadata()
    inventory = build_word_inventory(metadata)
    print(
        "[inventory]",
        {
            "valid_words": len(inventory["valid_words"]),
            "candidate_ids": len(inventory["candidate_ids"]),
            "query_words": len(inventory["query_words"]),
        },
        flush=True,
    )

    external_vectors = load_llama_external_vectors(
        inventory["word_to_llama_id"], inventory["valid_words"]
    )
    external_neighbors, external_scores = topk_external_words(
        external_vectors, inventory["valid_words"], inventory["query_words"]
    )
    del external_vectors
    gc.collect()

    summary = {
        "external_reference": "meta-llama/Llama-3.2-1B input embedding",
        "external_snapshot": LLAMA_SNAPSHOT,
        "tokenizer_under_test": "t5-base",
        "vocab_size_under_test": VOCAB_SIZE,
        "top_ks": list(TOP_KS),
        "valid_external_word_count": len(inventory["valid_words"]),
        "candidate_t5_token_count": len(inventory["candidate_ids"]),
        "query_count": len(inventory["query_words"]),
        "query_words": inventory["query_words"],
        "variants": {},
    }

    rows_by_variant = {}
    for variant, source_path, loader in iter_variants():
        print(f"[variant] loading {variant}: {source_path}", flush=True)
        matrix = loader()
        model_neighbors, model_scores = topk_model_words(
            matrix,
            inventory["query_ids"],
            inventory["candidate_ids"],
            metadata,
        )
        del matrix
        gc.collect()

        variant_summary, rows = score_overlap(
            variant,
            inventory["query_ids"],
            inventory["query_words"],
            model_neighbors,
            model_scores,
            external_neighbors,
            external_scores,
            metadata,
        )
        variant_summary["source_path"] = source_path
        summary["variants"][variant] = variant_summary
        rows_by_variant[variant] = rows

        csv_path = os.path.join(OUTPUT_DIR, f"{variant}_external_overlap_per_query.csv")
        write_rows(csv_path, rows)
        print(f"[variant] saved {csv_path}", flush=True)

    summary["qualitative_examples"] = build_examples(rows_by_variant)

    summary_path = os.path.join(OUTPUT_DIR, "external_embedding_overlap_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    fig_path = plot_summary(summary)

    compact = {k: v for k, v in summary.items() if k not in {"query_words", "qualitative_examples"}}
    print(f"[saved] {summary_path}")
    print(f"[saved] {fig_path}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
