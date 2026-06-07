"""External-embedding overlap on common C4-style content words.

This script is a narrower version of embedding_external_overlap.py.  Instead of
sampling readable words across the T5 vocabulary, it uses a hand-curated set of
high-frequency English content words that are typical of C4/web text and mostly
have low T5 SentencePiece ids.  The goal is to test whether rare query words are
driving the external-overlap result.
"""

import gc
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import analysis.embedding_external_overlap as ext
from salad.utils import mkdir


OUTPUT_DIR = "data/figures/comparison_embedding/external_overlap_common_words"

COMMON_C4_PROXY_WORDS = [
    "time",
    "people",
    "work",
    "first",
    "help",
    "best",
    "years",
    "good",
    "year",
    "home",
    "information",
    "business",
    "right",
    "life",
    "place",
    "world",
    "high",
    "long",
    "service",
    "different",
    "data",
    "last",
    "using",
    "free",
    "available",
    "company",
    "experience",
    "site",
    "system",
    "important",
    "services",
    "online",
    "provide",
    "team",
    "things",
    "support",
    "number",
    "family",
    "water",
    "better",
    "design",
    "local",
    "small",
    "full",
    "process",
    "public",
    "order",
    "quality",
    "working",
    "game",
    "today",
    "week",
    "website",
    "days",
    "school",
    "state",
    "government",
    "health",
    "news",
    "research",
    "science",
    "market",
    "computer",
    "technology",
    "community",
    "country",
    "city",
    "children",
    "history",
    "energy",
    "language",
    "medical",
    "legal",
    "music",
    "money",
    "food",
    "problem",
    "example",
    "development",
    "program",
    "question",
    "social",
    "internet",
    "product",
]


def build_common_inventory(metadata) -> Dict[str, object]:
    inventory = ext.build_word_inventory(metadata)
    norm_to_t5_ids = inventory["norm_to_t5_ids"]
    word_to_llama_id = inventory["word_to_llama_id"]

    query_words = []
    skipped_words = []
    for word in COMMON_C4_PROXY_WORDS:
        if word in norm_to_t5_ids and word in word_to_llama_id:
            query_words.append(word)
        else:
            skipped_words.append(word)

    query_words = sorted(dict.fromkeys(query_words))
    query_ids = [norm_to_t5_ids[word][0] for word in query_words]

    inventory["query_words"] = query_words
    inventory["query_ids"] = query_ids
    inventory["skipped_common_words"] = skipped_words
    return inventory


def build_common_examples(rows_by_variant: Dict[str, List[Dict[str, object]]]):
    selected = [
        "people",
        "world",
        "information",
        "system",
        "government",
        "health",
        "research",
        "computer",
        "language",
        "market",
    ]
    examples = {}
    for word in selected:
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


def plot_common_summary(summary: Dict[str, object]) -> str:
    variants = list(summary["variants"])
    labels = [name.replace("_", "\n") for name in variants]
    x = np.arange(len(variants))
    width = 0.22
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for offset, k in zip((-width, 0.0, width), ext.TOP_KS):
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

    path = os.path.join(OUTPUT_DIR, "external_embedding_overlap_common_words_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    mkdir(OUTPUT_DIR)
    metadata = ext.build_token_metadata()
    inventory = build_common_inventory(metadata)
    print(
        "[inventory]",
        {
            "valid_words": len(inventory["valid_words"]),
            "candidate_ids": len(inventory["candidate_ids"]),
            "query_words": len(inventory["query_words"]),
            "skipped_common_words": inventory["skipped_common_words"],
        },
        flush=True,
    )

    external_vectors = ext.load_llama_external_vectors(
        inventory["word_to_llama_id"], inventory["valid_words"]
    )
    external_neighbors, external_scores = ext.topk_external_words(
        external_vectors, inventory["valid_words"], inventory["query_words"]
    )
    del external_vectors
    gc.collect()

    summary = {
        "external_reference": "meta-llama/Llama-3.2-1B input embedding",
        "external_snapshot": ext.LLAMA_SNAPSHOT,
        "tokenizer_under_test": "t5-base",
        "vocab_size_under_test": ext.VOCAB_SIZE,
        "query_selection": "hand-curated common C4/web-content English words",
        "top_ks": list(ext.TOP_KS),
        "valid_external_word_count": len(inventory["valid_words"]),
        "candidate_t5_token_count": len(inventory["candidate_ids"]),
        "query_count": len(inventory["query_words"]),
        "query_words": inventory["query_words"],
        "skipped_common_words": inventory["skipped_common_words"],
        "variants": {},
    }

    rows_by_variant = {}
    for variant, source_path, loader in ext.iter_variants():
        print(f"[variant] loading {variant}: {source_path}", flush=True)
        matrix = loader()
        model_neighbors, model_scores = ext.topk_model_words(
            matrix,
            inventory["query_ids"],
            inventory["candidate_ids"],
            metadata,
        )
        del matrix
        gc.collect()

        variant_summary, rows = ext.score_overlap(
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

        csv_path = os.path.join(OUTPUT_DIR, f"{variant}_external_overlap_common_words_per_query.csv")
        ext.write_rows(csv_path, rows)
        print(f"[variant] saved {csv_path}", flush=True)

    summary["qualitative_examples"] = build_common_examples(rows_by_variant)

    summary_path = os.path.join(
        OUTPUT_DIR, "external_embedding_overlap_common_words_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    fig_path = plot_common_summary(summary)

    compact = {k: v for k, v in summary.items() if k not in {"query_words", "qualitative_examples"}}
    print(f"[saved] {summary_path}")
    print(f"[saved] {fig_path}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
