#!/usr/bin/env python3
"""Render lizard attention after binary-quantizing every FC sparse matrix.

Each FC sparse matrix receives one scale
``alpha = mean(S[S > 0])``.  Negative entries map to ``-alpha`` and all
non-negative entries, including zeros, map to ``+alpha``.  QKV and attention
output projections retain their full-precision SALAAD ``L + S`` weights.
"""

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
    / "fc_sparse_int1_attention"
    / "52_green_lizard"
)


if __name__ == "__main__":
    main(
        default_quantization_mode="int1_positive_mean",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
