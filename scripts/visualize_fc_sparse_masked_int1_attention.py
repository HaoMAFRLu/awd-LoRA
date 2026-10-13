#!/usr/bin/env python3
"""Render lizard attention with sparse-mask-preserving binary FC-S weights.

Entries with ``abs(S) < epsilon`` remain zero.  Each retained nonzero uses one
sign bit and becomes ``-alpha`` or ``+alpha``, where each FC matrix has
``alpha = mean(S[S >= epsilon])``.
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
    / "fc_sparse_masked_int1_attention"
    / "52_green_lizard"
)


if __name__ == "__main__":
    main(
        default_quantization_mode="int1_sparse_positive_mean",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
