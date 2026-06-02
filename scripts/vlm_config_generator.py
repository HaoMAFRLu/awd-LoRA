"""Generate training YAML files for Qwen3-VL-style VLM experiments.

Edit the dictionaries at the bottom of this file and run:

    python scripts/vlm_config_generator.py
"""

import copy
import json
import os
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LANGUAGE_ATTN_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
LANGUAGE_MLP_KEYS = [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
VISION_KEYS = [
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
]
PROJECTOR_KEYS = [
    "visual.merger.linear_fc1",
    "visual.merger.linear_fc2",
]


def salad_params(
    *,
    rate_rank,
    rate_sparsity,
    init_energy,
    rho,
    alpha_init=1e-7,
    beta_init=1e-7,
    alpha_decay=0.02,
    beta_decay=0.02,
    beta_mode="hard_cut",
):
    return {
        "energy": 0.999,
        "init_energy": init_energy,
        "is_init": False,
        "iter_max": 1,
        "tol": 0.001,
        "rate_rank": rate_rank,
        "rate_sparsity": rate_sparsity,
        "alpha_dict": {
            "init": alpha_init,
            "mode": "adaptive",
            "rate_decay": alpha_decay,
        },
        "beta_dict": {
            "init": beta_init,
            "mode": beta_mode,
            "rate_decay": beta_decay,
        },
        "rho_dict": {
            "rho": rho,
            "mode": "fixed",
            "start_epoch": 2,
            "coeff_rho": 0.1,
            "coeff_rho_min": 0.01,
            "coeff_rho_max": 1500.0,
            "rho_rate": 1.0,
        },
    }


DEFAULT_PARAMS = {
    "language_attn": salad_params(rate_rank=0.15, rate_sparsity=0.05, init_energy=0.15, rho=1e-6),
    "language_mlp": salad_params(
        rate_rank=0.35,
        rate_sparsity=0.05,
        init_energy=0.35,
        rho=1e-6,
        alpha_init=1e-9,
    ),
    "vision_attn": salad_params(rate_rank=0.20, rate_sparsity=0.05, init_energy=0.20, rho=1e-6),
    "vision_mlp": salad_params(
        rate_rank=0.35,
        rate_sparsity=0.05,
        init_energy=0.35,
        rho=1e-6,
        alpha_init=1e-9,
    ),
    "projector": salad_params(rate_rank=0.30, rate_sparsity=0.05, init_energy=0.30, rho=1e-6),
    "embedding": salad_params(
        rate_rank=0.15,
        rate_sparsity=0.05,
        init_energy=0.45,
        rho=1e-6,
        alpha_init=0.0,
        beta_init=0.0,
    ),
    "lm_head": salad_params(
        rate_rank=0.15,
        rate_sparsity=0.05,
        init_energy=0.15,
        rho=1e-6,
        alpha_init=0.0,
        beta_init=0.0,
    ),
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _represent_none(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


def _represent_float(dumper, value):
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


NoAliasDumper.add_representer(type(None), _represent_none)
NoAliasDumper.add_representer(float, _represent_float)


def load_model_config(model_config):
    path_model_config = os.path.join(ROOT, "configs", model_config)
    with open(path_model_config, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_indices(total, count):
    if count < 0:
        return range(total)
    return range(min(total, count))


def add_language_layers(layers, num_layers, params, *, layer_count, include_embedding, include_lm_head):
    if include_embedding:
        layers.append({
            "name": "language_model.embed_tokens",
            "params": copy.deepcopy(params["embedding"]),
        })

    for i in _select_indices(num_layers, layer_count):
        base = f"language_model.layers.{i}"
        for key in LANGUAGE_ATTN_KEYS:
            layers.append({
                "name": f"{base}.{key}",
                "params": copy.deepcopy(params["language_attn"]),
            })
        for key in LANGUAGE_MLP_KEYS:
            layers.append({
                "name": f"{base}.{key}",
                "params": copy.deepcopy(params["language_mlp"]),
            })

    if include_lm_head:
        layers.append({
            "name": "lm_head",
            "params": copy.deepcopy(params["lm_head"]),
        })


def add_vision_layers(layers, vision_depth, params, *, layer_count):
    for i in _select_indices(vision_depth, layer_count):
        base = f"visual.blocks.{i}"
        for key in VISION_KEYS:
            param_key = "vision_attn" if key.startswith("attn.") else "vision_mlp"
            layers.append({
                "name": f"{base}.{key}",
                "params": copy.deepcopy(params[param_key]),
            })


def add_projector_layers(layers, deepstack_indexes, params):
    for key in PROJECTOR_KEYS:
        layers.append({
            "name": key,
            "params": copy.deepcopy(params["projector"]),
        })

    for i in range(len(deepstack_indexes)):
        for key in ("linear_fc1", "linear_fc2"):
            layers.append({
                "name": f"visual.deepstack_merger_list.{i}.{key}",
                "params": copy.deepcopy(params["projector"]),
            })


def generate_vlm_config(
    *,
    name="qwen3_vl_500m",
    model_config="qwen3_vl_500m_model.json",
    processor_name="Qwen/Qwen3-VL-2B-Instruct",
    training_mode="salad",
    seed=42,
    num_total_iters=1000,
    num_freq=10,
    lr=3e-4,
    weight_decay=0.1,
    batch_size=1,
    max_length=1024,
    warmup_steps=20,
    min_lr_ratio=0.1,
    num_workers=0,
    seed_for_shuffle=42,
    save_interval=50,
    is_clip=1.0,
    is_wandb=False,
    is_monitor=False,
    data_name="HuggingFaceM4/Docmatix",
    data_subset="pdf",
    data_split="train",
    data_streaming=True,
    image_column="images",
    text_column="texts",
    include_language=True,
    include_vision=False,
    include_projector=False,
    include_embedding=False,
    include_lm_head=False,
    language_layers=2,
    vision_layers=0,
    output_path=None,
):
    cfg_model = load_model_config(model_config)
    text_cfg = cfg_model["text_config"]
    vision_cfg = cfg_model["vision_config"]

    layers = []
    if training_mode == "salad":
        if include_language:
            add_language_layers(
                layers,
                text_cfg["num_hidden_layers"],
                DEFAULT_PARAMS,
                layer_count=language_layers,
                include_embedding=include_embedding,
                include_lm_head=include_lm_head,
            )
        if include_vision:
            add_vision_layers(
                layers,
                vision_cfg["depth"],
                DEFAULT_PARAMS,
                layer_count=vision_layers,
            )
        if include_projector:
            add_projector_layers(
                layers,
                vision_cfg.get("deepstack_visual_indexes", []),
                DEFAULT_PARAMS,
            )

    cfg = {
        "name": name,
        "model_type": "qwen3_vl",
        "model_config": model_config,
        "processor_name": processor_name,
        "training_mode": training_mode,
        "seed": seed,
        "num_total_iters": num_total_iters,
        "num_freq": num_freq,
        "gradient": "coupled",
        "is_asyn": False,
        "is_init": False,
        "is_clip": is_clip,
        "is_wandb": is_wandb,
        "is_monitor": is_monitor,
        "save_interval": save_interval,
        "max_length": max_length,
        "seed_for_shuffle": seed_for_shuffle,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data": {
            "type": "vlm",
            "name": data_name,
            "subset": data_subset,
            "split": data_split,
            "streaming": data_streaming,
            "processor_name": processor_name,
            "image_column": image_column,
            "text_column": text_column,
            "ignore_visual_tokens": True,
        },
        "scheduler": {
            "name": "cosine",
            "params": {
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "params": {
                "lr": lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        },
        "layers": layers,
    }

    if output_path is None:
        output_path = os.path.join(ROOT, "configs", f"{name}.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True)

    print(f"Configuration written to {output_path}")
    print(f"Generated {len(layers)} SALAAD layers.")


if __name__ == "__main__":
    cfg_qwen3_vl_500m = dict(
        name="qwen3_vl_500m",
        model_config="qwen3_vl_500m_model.json",
        processor_name="Qwen/Qwen3-VL-2B-Instruct",
        training_mode="salad",
        seed=42,
        num_total_iters=1000,
        num_freq=10,
        lr=3e-4,
        weight_decay=0.1,
        batch_size=1,
        max_length=1024,
        warmup_steps=20,
        min_lr_ratio=0.1,
        num_workers=0,
        seed_for_shuffle=42,
        save_interval=50,
        is_clip=1.0,
        is_wandb=False,
        is_monitor=False,
        data_name="HuggingFaceM4/Docmatix",
        data_subset="pdf",
        data_split="train",
        data_streaming=True,
        image_column="images",
        text_column="texts",
        include_language=True,
        include_vision=False,
        include_projector=False,
        include_embedding=False,
        include_lm_head=False,
        language_layers=2,  # -1 means all language layers
        vision_layers=0,    # -1 means all vision layers
    )

    cfg_qwen3_vl_500m_vanilla = copy.deepcopy(cfg_qwen3_vl_500m)
    cfg_qwen3_vl_500m_vanilla.update(
        name="qwen3_vl_500m_vanilla",
        training_mode="vanilla",
    )

    cfg_qwen3_vl_500m_projector = copy.deepcopy(cfg_qwen3_vl_500m)
    cfg_qwen3_vl_500m_projector.update(
        name="qwen3_vl_500m_projector",
        include_projector=True,
    )

    cfg_qwen3_vl_500m_vision_debug = copy.deepcopy(cfg_qwen3_vl_500m)
    cfg_qwen3_vl_500m_vision_debug.update(
        name="qwen3_vl_500m_vision_debug",
        include_vision=True,
        vision_layers=1,
        include_projector=True,
    )

    cfg_qwen3_vl_tiny = copy.deepcopy(cfg_qwen3_vl_500m)
    cfg_qwen3_vl_tiny.update(
        name="qwen3_vl_tiny",
        model_config="qwen3_vl_tiny_model.json",
        num_total_iters=20,
        num_freq=2,
        batch_size=1,
        max_length=256,
        warmup_steps=2,
        save_interval=10,
        include_language=True,
        language_layers=1,
        include_projector=True,
        include_vision=False,
        vision_layers=0,
    )

    cfg_qwen3_vl_tiny_vanilla = copy.deepcopy(cfg_qwen3_vl_tiny)
    cfg_qwen3_vl_tiny_vanilla.update(
        name="qwen3_vl_tiny_vanilla",
        training_mode="vanilla",
    )

    cfg_qwen3_vl_micro = copy.deepcopy(cfg_qwen3_vl_tiny)
    cfg_qwen3_vl_micro.update(
        name="qwen3_vl_micro",
        model_config="qwen3_vl_micro_model.json",
        num_total_iters=10,
        max_length=128,
        save_interval=5,
        include_language=True,
        language_layers=1,
        include_projector=False,
        include_vision=False,
        vision_layers=0,
    )

    cfg_qwen3_vl_micro_vanilla = copy.deepcopy(cfg_qwen3_vl_micro)
    cfg_qwen3_vl_micro_vanilla.update(
        name="qwen3_vl_micro_vanilla",
        training_mode="vanilla",
    )

    generate_vlm_config(**cfg_qwen3_vl_micro)
