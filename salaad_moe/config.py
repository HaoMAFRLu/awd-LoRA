"""Configuration shared by the reference trainer and the Megatron adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def config_for_version(version):
    """Match the repository's --cfg_version -> configs/<name>.yaml convention."""
    if not version or Path(version).name != version or version in (".", ".."):
        raise ValueError("cfg_version must be a filename in configs/, without directories")
    return CONFIG_ROOT / (version if version.endswith(".yaml") else version + ".yaml")


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            deep_merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def load_config(path, _seen=()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("inherits", None)
    if parent:
        config = deep_merge(load_config(path.parent / parent, (*_seen, path)), config)
    return config


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_config(c: dict, world_size=None) -> None:
    # This framework accepts only the MoE schema; legacy dense LLM/Vision
    # configs are not used to dispatch training.
    sections = ("model", "data", "training", "parallel", "salaad")
    if any(not isinstance(c.get(section), dict) for section in sections):
        raise ValueError("Expected a MoE configuration with model/data/training/parallel/salaad sections")
    if c["model"].get("family") != "llama_style_no_shared_expert":
        raise ValueError("Only the no-shared-expert MoE architecture is supported")
    m, t, p, s = (c[k] for k in ("model", "training", "parallel", "salaad"))
    if not isinstance(c.get("is_wandb", False), bool):
        raise ValueError("is_wandb must be a boolean")
    if c.get("is_wandb") and (
        not isinstance(c.get("wandb_project"), str) or not c["wandb_project"].strip()
    ):
        raise ValueError("Enabled W&B requires a nonempty wandb_project")
    if not isinstance(c["data"].get("synthetic_smoke", False), bool):
        raise ValueError("data.synthetic_smoke must be a boolean")
    for key in (
        "num_layers",
        "hidden_size",
        "expert_ffn_hidden_size",
        "num_attention_heads",
        "num_experts",
        "padded_vocab_size",
    ):
        if not isinstance(m[key], int) or m[key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if not 1 <= m["router_topk"] <= m["num_experts"]:
        raise ValueError("router_topk must be in [1, num_experts]")
    if (
        m["hidden_size"] % m["num_attention_heads"]
        or (m["hidden_size"] // m["num_attention_heads"]) % 2
    ):
        raise ValueError("RoPE requires an even, integral attention head dimension")
    required = {
        "num_shared_experts": 0,
        "num_query_groups": m["num_attention_heads"],
        "moe_layer_pattern": "all",
        "linear_bias": False,
        "tie_embeddings": False,
        "router_pre_softmax": False,
        "router_score_function": "softmax",
        "capacity_factor": None,
        "token_dropping": False,
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "swiglu": True,
        "mirror_routing_mu": 0.0,
        "router_dtype": "float32",
        "selected_weights_sum_to_one": True,
    }
    for key, value in required.items():
        if m.get(key) != value:
            raise ValueError(f"This implementation requires model.{key}={value!r}")
    for key, value in {
        "expert_implementation": "sequential_mlp_reference",
        "output_projection_init": "std_div_sqrt_2depth",
    }.items():
        if m[key] != value:
            raise ValueError(f"Unsupported model.{key}={m[key]!r}")
    for key in (
        "expert_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
    ):
        if p[key] != 1:
            raise ValueError(f"Only DP is implemented; parallel.{key} must be 1")
    if any(
        p[k]
        for k in ("distributed_optimizer", "overlap_gradient_reduce", "overlap_parameter_gather")
    ):
        raise ValueError("Optimizer sharding and communication overlap are not supported")
    if p["data_parallel_size"] != p["world_size"] or (
        world_size is not None and p["world_size"] != world_size
    ):
        raise ValueError("Configured DP/world size must equal the launched world size")
    for key in (
        "total_optimizer_steps",
        "global_batch_sequences",
        "micro_batch_sequences_per_rank",
        "accumulation_steps",
    ):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if (
        t["global_batch_sequences"]
        != t["micro_batch_sequences_per_rank"] * t["accumulation_steps"] * p["world_size"]
    ):
        raise ValueError("global batch must equal micro batch * accumulation * DP")
    if (
        not isinstance(t["warmup_steps"], int)
        or not 0 <= t["warmup_steps"] < t["total_optimizer_steps"]
    ):
        raise ValueError("warmup_steps must be an integer in [0, total_optimizer_steps)")
    if not 0 <= t["min_learning_rate"] <= t["learning_rate"] or t["learning_rate"] <= 0:
        raise ValueError("Invalid learning rates")
    if t["task_precision"] not in ("float32", "bfloat16"):
        raise ValueError("Only float32 and bfloat16 compute are supported")
    if t["optimizer"] != "AdamW" or t["schedule"] != "cosine":
        raise ValueError("Only AdamW with linear warmup and cosine decay is implemented")
    for key, value in {
        "load_balancing_reduction": "mean_over_rank_microbatches_sum_over_layers",
        "load_balancing_assignment_denominator": "num_tokens_times_topk",
        "z_loss_reduction": "mean_over_tokens_sum_over_layers",
        "expert_task_gradient_rescale": 1.0,
        "no_weight_decay": ["normalization", "bias"],
    }.items():
        if t[key] != value:
            raise ValueError(f"Unsupported training.{key}={t[key]!r}")
    if not isinstance(t["checkpoint_interval_steps"], int) or t["checkpoint_interval_steps"] < 1:
        raise ValueError("training.checkpoint_interval_steps must be a positive integer")
    if t["keep_latest_checkpoints"] < 1:
        raise ValueError("Keep at least one latest checkpoint")
    if (
        not all(0 <= t[key] < 1 for key in ("beta1", "beta2"))
        or t["epsilon"] <= 0
        or t["gradient_clip_global_norm"] <= 0
    ):
        raise ValueError("Invalid AdamW moments/epsilon or clipping norm")
    for key in ("weight_decay", "load_balancing_coefficient", "router_z_loss_coefficient"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"Invalid training.{key}")
    for key in ("master_parameter_precision", "gradient_precision"):
        if t[key] != "float32":
            raise ValueError(f"{key} must be float32")
    if t["activation_recompute"] not in ("none", "selective_attention"):
        raise ValueError("Unsupported activation recomputation")
    d = c["data"]
    if d["seq_length"] < 1 or not 0 <= d["eod_id"] < m["padded_vocab_size"]:
        raise ValueError("Invalid sequence length or EOD token")
    if d["indexed_sample_length"] != d["seq_length"] + 1:
        raise ValueError("Samples must contain seq_length+1 tokens (one label shift)")
    if d["validation_sequences"] < 1 or d["test_sequences"] < 1:
        raise ValueError("Invalid held-out sequence counts")
    if (
        d["packing"] != "fixed_length_no_padding"
        or d["split"] != "sha256_text_mod_10000_val_0_99_test_100_199_train_200_9999"
    ):
        raise ValueError("Unsupported packing or document split")
    if not 0 <= c["seed"] < 2**32 or d["corpus_order_seed"] < 0:
        raise ValueError("Invalid random seed")
    for key, expected in {
        "append_eod": True,
        "append_bos": False,
        "cross_document_attention": True,
        "reset_position_ids_at_eod": False,
        "reset_attention_at_eod": False,
        "include_eod_in_loss": True,
    }.items():
        if d[key] != expected:
            raise ValueError(f"Unsupported data.{key}={d[key]!r}")
    if s["enabled"]:
        if not math.isfinite(s["rho"]) or s["rho"] <= 0:
            raise ValueError("A single finite positive rho is required")
        required_salaad = {
            "consensus_scope": "same_layer_same_projection_all_experts",
            "rho_scope": "global_fixed_all_experts_all_layers",
            "penalty_reduction": "sum_all_entries_all_expert_matrices",
            "gradient_injection": "after_dp_reduce_and_prepare_grads_before_global_clip",
            "structure_order": ["shared", "low_rank", "sparse", "dual"],
            "refresh_anchor_after_structure": True,
            "auxiliary_dtype": "float32",
            "auxiliary_owner": "deterministic_group_id_mod_dp",
            "broadcast_anchor_dtype": "float32",
        }
        for key, value in required_salaad.items():
            if s[key] != value:
                raise ValueError(f"Unsupported salaad.{key}={s[key]!r}")
        if not 0 <= s["state_initialization_step"] < t["total_optimizer_steps"]:
            raise ValueError("SALAAD initialization must precede the end of training")
        if (
            s["guidance_period_optimizer_steps"] < 1
            or s["structure_inner_steps"] < 1
            or s["svd_chunk_size"] < 1
        ):
            raise ValueError("SALAAD periods and SVD chunk size must be positive")
        if (
            not s["projections"]
            or len(set(s["projections"])) != len(s["projections"])
            or not set(s["projections"]) <= {"gate", "up", "down"}
        ):
            raise ValueError("Invalid SALAAD projections")
        ctl = s["controller"]
        if ctl["rank_statistic"] != "linear_singular_value_mass" or not 0 < ctl["gamma"] <= 1:
            raise ValueError("SALAAD uses linear singular-value mass, with gamma in (0,1]")
        if (
            ctl["zero_matrix_rank_ratio"] != 0
            or ctl["nonnegative_projection"] is not False
            or ctl["threshold_gain_convention"] != "tau_alpha_delta_equals_gain_times_rank_error"
        ):
            raise ValueError("Unsupported controller conventions")
        for key in ("target_rank_ratio", "target_density"):
            if not 0 <= ctl[key] <= 1:
                raise ValueError(f"Invalid controller.{key}")
        for key in ("alpha_init", "beta_init", "gain_alpha", "gain_beta"):
            if not math.isfinite(ctl[key]) or ctl[key] < 0:
                raise ValueError(f"controller.{key} must be finite and nonnegative")
        if s.get("shared_mode", "learned") not in ("learned", "fixed", "none"):
            raise ValueError("shared_mode must be learned, fixed, or none")
        if not s.get("low_rank_enabled", True) and not s.get("sparse_enabled", True):
            raise ValueError("At least one residual component is required")


def parameter_counts(c: dict) -> dict:
    m = c["model"]
    d, f, layers, k, q, v = (
        m[x]
        for x in (
            "hidden_size",
            "expert_ffn_hidden_size",
            "num_layers",
            "num_experts",
            "router_topk",
            "padded_vocab_size",
        )
    )
    experts = 3 * layers * k * d * f
    other = 2 * v * d + layers * (4 * d * d + k * d + 2 * d) + d
    return {
        "expert_parameters": experts,
        "other_parameters": other,
        "total_parameters": experts + other,
        "active_parameters_convention": other + 3 * layers * q * d * f,
        "prediction_tokens": c["training"]["total_optimizer_steps"]
        * c["training"]["global_batch_sequences"]
        * c["data"]["seq_length"],
    }


def learning_rate(config: dict, update: int) -> float:
    """One-based optimizer update: update 1 warms up; update M reaches min LR."""
    t = config["training"]
    if not 1 <= update <= t["total_optimizer_steps"]:
        raise ValueError("LR update index is outside the configured full schedule")
    peak, low, warm = t["learning_rate"], t["min_learning_rate"], t["warmup_steps"]
    if warm and update <= warm:
        return peak * update / warm
    # Use the full remaining budget, even when this invocation pauses early.
    progress = (update - warm) / (t["total_optimizer_steps"] - warm)
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * progress))
