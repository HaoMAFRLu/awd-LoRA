"""Hugging Face backed Qwen3-VL model construction.

This repository keeps local implementations for some small language models, but
Qwen3-VL is large and actively maintained upstream.  We therefore instantiate it
through Transformers from a local config, which keeps SALAAD focused on weight
structure rather than reimplementing multimodal modeling code.
"""

from __future__ import annotations


def build_qwen3_vl_from_config(path_cfg_model: str):
    """Build a Qwen3-VL conditional generation model from a local config file."""
    try:
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL support requires transformers with Qwen3-VL classes. "
            "Install transformers>=4.57.0."
        ) from exc

    cfg = AutoConfig.from_pretrained(path_cfg_model)
    if cfg.model_type != "qwen3_vl":
        raise ValueError(f"Expected model_type='qwen3_vl', got {cfg.model_type!r}")

    return Qwen3VLForConditionalGeneration(cfg)
