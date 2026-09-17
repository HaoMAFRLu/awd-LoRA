#!/bin/bash
# Run preparation and its final size check in the same CPU job environment.
set -euo pipefail

moe_python=/lustre/home/hma2/projects/awd-LoRA/myenv/bin/python
moe_corpus=/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916

"$moe_python" -u scripts/prepare_moe_data.py \
    --config configs/ns97m_vanilla.yaml \
    --shard-manifest "$moe_corpus/sources/shards.json" \
    --tokenizer-json "$moe_corpus/sources/tokenizer/tokenizer.json" \
    --max-train-tokens 4404019200 \
    --output "$moe_corpus/tokens_ns97m"

# A failed preparation or an undersized corpus must not start GPU training.
"$moe_python" scripts/check_moe_data.py \
    --config configs/ns97m_vanilla.yaml \
    --data-manifest "$moe_corpus/tokens_ns97m/manifest.json"
