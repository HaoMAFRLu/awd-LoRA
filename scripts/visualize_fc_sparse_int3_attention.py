#!/usr/bin/env python3
"""Render lizard attention after symmetric INT3 quantization of FC sparse weights."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visualize_fc_sparse_int4_attention import main  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "figures"
    / "salaad_vision"
    / "fc_sparse_int3_attention"
    / "52_green_lizard"
)


if __name__ == "__main__":
    main(
        default_quantization_mode="int3",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
