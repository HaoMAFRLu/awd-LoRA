"""Generate the DINO ViT-B/8 training YAML used by train_salad.py.

Edit the dictionary at the bottom of this file and run:

    python scripts/vit_config_generator.py
"""

import copy
import json
import os

import yaml

try:
    from .vit_params import projection
except ImportError:
    from vit_params import projection


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ATTENTION_KEYS = [
    "attn.qkv",
    "attn.proj",
]
MLP_KEYS = [
    "mlp.fc1",
    "mlp.fc2",
]

PROJECTION_PARAMS = projection()

VIT_MIXED_RHO_BY_SUFFIX = {
    "attn.qkv": 5e-6,
    "attn.proj": 5e-8,
    "mlp.fc1": 5e-8,
    "mlp.fc2": 5e-8,
}

# Attention-only structured sparsity: fused QKV uses whole-row sparse groups
# and attention output projection uses whole-column sparse groups. MLP suffixes
# are intentionally absent from this mapping, so included FC1/FC2 layers retain
# element-wise SALAAD sparsity.
VIT_BMIXED_BLOCK_SHAPES = {
    "attn.qkv": {"block_p": 1, "block_q": "full"},
    "attn.proj": {"block_p": "full", "block_q": 1},
}
VIT_BMIXED_ALPHA_RATE_DECAY = 0.05
VIT_BMIXED_BETA_RATE_DECAY = 0.02
VIT_BMIXED_SMOKE_EXCLUDED_SUFFIXES = (
    "attn.proj",
    "mlp.fc1",
    "mlp.fc2",
)


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
    with open(path_model_config, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _select_indices(total, count):
    if count < 0:
        return range(total)
    return range(min(total, count))


def add_vit_layers(
    layers,
    num_hidden_layers,
    params,
    *,
    layer_count,
    include_attention,
    include_mlp,
    rho_by_suffix,
    block_shape_by_suffix,
    excluded_suffixes,
    alpha_rate_decay,
    beta_rate_decay,
):
    """Add bare-student ViT Linear layers in deterministic block order."""
    for block_index in _select_indices(num_hidden_layers, layer_count):
        base = f"backbone.blocks.{block_index}"
        if include_attention:
            for key in ATTENTION_KEYS:
                if key in excluded_suffixes:
                    continue
                layer_params = copy.deepcopy(params[key])
                if key in rho_by_suffix:
                    layer_params["rho_dict"]["rho"] = rho_by_suffix[key]
                if key in block_shape_by_suffix:
                    layer_params.update(block_shape_by_suffix[key])
                if alpha_rate_decay is not None:
                    layer_params["alpha_dict"]["rate_decay"] = alpha_rate_decay
                if beta_rate_decay is not None:
                    layer_params["beta_dict"]["rate_decay"] = beta_rate_decay
                layers.append(
                    {
                        "name": f"{base}.{key}",
                        "params": layer_params,
                    }
                )
        if include_mlp:
            for key in MLP_KEYS:
                if key in excluded_suffixes:
                    continue
                layer_params = copy.deepcopy(params[key])
                if key in rho_by_suffix:
                    layer_params["rho_dict"]["rho"] = rho_by_suffix[key]
                if key in block_shape_by_suffix:
                    layer_params.update(block_shape_by_suffix[key])
                if alpha_rate_decay is not None:
                    layer_params["alpha_dict"]["rate_decay"] = alpha_rate_decay
                if beta_rate_decay is not None:
                    layer_params["beta_dict"]["rate_decay"] = beta_rate_decay
                layers.append(
                    {
                        "name": f"{base}.{key}",
                        "params": layer_params,
                    }
                )


def _validate_rho_by_suffix(rho_by_suffix):
    if rho_by_suffix is None:
        return {}
    if not isinstance(rho_by_suffix, dict):
        raise TypeError("rho_by_suffix must be a dictionary or null")
    valid_suffixes = set(ATTENTION_KEYS + MLP_KEYS)
    unknown = set(rho_by_suffix) - valid_suffixes
    if unknown:
        raise ValueError(f"unknown rho override suffixes: {sorted(unknown)}")
    normalized = {}
    for suffix, rho in rho_by_suffix.items():
        if isinstance(rho, bool) or not isinstance(rho, (int, float)):
            raise TypeError(f"rho override for {suffix} must be a number")
        if rho <= 0:
            raise ValueError(f"rho override for {suffix} must be positive")
        normalized[suffix] = float(rho)
    return normalized


def _validate_block_shape_by_suffix(block_shape_by_suffix):
    if block_shape_by_suffix is None:
        return {}
    if not isinstance(block_shape_by_suffix, dict):
        raise TypeError("block_shape_by_suffix must be a dictionary or null")

    valid_suffixes = set(ATTENTION_KEYS + MLP_KEYS)
    unknown = set(block_shape_by_suffix) - valid_suffixes
    if unknown:
        raise ValueError(f"unknown block-shape suffixes: {sorted(unknown)}")

    def normalize_dimension(value, name, suffix):
        if isinstance(value, str):
            if value.lower() != "full":
                raise ValueError(
                    f"{name} for {suffix} must be a positive integer or 'full'"
                )
            return "full"
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} for {suffix} must be a positive integer or 'full'"
            )
        if value <= 0:
            raise ValueError(f"{name} for {suffix} must be positive")
        return value

    normalized = {}
    for suffix, block_shape in block_shape_by_suffix.items():
        if not isinstance(block_shape, dict):
            raise TypeError(f"block shape for {suffix} must be a dictionary")
        missing = {"block_p", "block_q"} - set(block_shape)
        if missing:
            raise ValueError(
                f"block shape for {suffix} is missing {sorted(missing)}"
            )
        extra = set(block_shape) - {"block_p", "block_q"}
        if extra:
            raise ValueError(
                f"unknown block-shape keys for {suffix}: {sorted(extra)}"
            )
        normalized[suffix] = {
            "block_p": normalize_dimension(
                block_shape["block_p"], "block_p", suffix
            ),
            "block_q": normalize_dimension(
                block_shape["block_q"], "block_q", suffix
            ),
        }
    return normalized


def _validate_excluded_suffixes(excluded_suffixes):
    if excluded_suffixes is None:
        return frozenset()
    if isinstance(excluded_suffixes, str) or not isinstance(
        excluded_suffixes,
        (list, tuple, set, frozenset),
    ):
        raise TypeError("excluded_suffixes must be a sequence or set")
    valid_suffixes = set(ATTENTION_KEYS + MLP_KEYS)
    normalized = frozenset(excluded_suffixes)
    unknown = normalized - valid_suffixes
    if unknown:
        raise ValueError(f"unknown excluded suffixes: {sorted(unknown)}")
    return normalized


def _validate_optional_rate_decay(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number or null")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def generate_vit_config(
    *,
    name="vit_b8",
    model_config="vit_b8_model.json",
    seed=42,
    training_mode="salad",
    lr=1e-5,
    num_freq=5,
    weight_decay=0.0,
    optimizer_name="AdamW",
    gradient="coupled",
    is_asyn=False,
    is_init=False,
    is_wandb=True,
    is_monitor=True,
    save_interval=200,
    seed_for_shuffle=42,
    is_clip=1.0,
    num_total_iters=200,
    batch_size=1,
    warmup_steps=20,
    scheduler_total_steps=None,
    num_workers=0,
    scheduler_type="cosine",
    min_lr_ratio=0.1,
    precision="bfloat16",
    runtime="local",
    data_location="local_smoke",
    data_root=os.path.join(
        ROOT,
        "data",
        "salaad_vision",
        "smoke",
        "imagenet_val64_parquet",
    ),
    data_cache_dir=os.path.join(
        ROOT,
        "data",
        "salaad_vision",
        "hf_cache_smoke",
        "datasets",
    ),
    data_split="validation",
    data_streaming=True,
    data_shuffle=True,
    shuffle_buffer_size=10_000,
    distillation_initialization="random_init",
    distillation_loss="mse",
    global_weight=1.0,
    patch_weight=1.0,
    include_attention=True,
    include_mlp=True,
    vit_layers=1,
    rho_by_suffix=None,
    block_shape_by_suffix=None,
    excluded_suffixes=None,
    alpha_rate_decay=None,
    beta_rate_decay=None,
    output_path=None,
):
    """Generate one ViT task config selected through --cfg_version."""
    cfg_model = load_model_config(model_config)
    if cfg_model.get("model_type") != "dino_vitb8":
        raise ValueError(
            f"Expected dino_vitb8 model config, got {cfg_model.get('model_type')!r}"
        )
    num_hidden_layers = cfg_model["num_hidden_layers"]
    rho_by_suffix = _validate_rho_by_suffix(rho_by_suffix)
    block_shape_by_suffix = _validate_block_shape_by_suffix(
        block_shape_by_suffix
    )
    excluded_suffixes = _validate_excluded_suffixes(excluded_suffixes)
    alpha_rate_decay = _validate_optional_rate_decay(
        alpha_rate_decay,
        "alpha_rate_decay",
    )
    beta_rate_decay = _validate_optional_rate_decay(
        beta_rate_decay,
        "beta_rate_decay",
    )

    layers = []
    if training_mode == "salad":
        add_vit_layers(
            layers,
            num_hidden_layers,
            PROJECTION_PARAMS,
            layer_count=vit_layers,
            include_attention=include_attention,
            include_mlp=include_mlp,
            rho_by_suffix=rho_by_suffix,
            block_shape_by_suffix=block_shape_by_suffix,
            excluded_suffixes=excluded_suffixes,
            alpha_rate_decay=alpha_rate_decay,
            beta_rate_decay=beta_rate_decay,
        )

    data = {
        "type": "vision",
        "dataset": "ILSVRC/imagenet-1k",
        "location": data_location,
        "root": data_root,
        "cache_dir": data_cache_dir,
        "split": data_split,
        "streaming": data_streaming,
        "shuffle": data_shuffle,
        "shuffle_buffer_size": shuffle_buffer_size,
    }

    cfg = {
        "seed": seed,
        "name": name,
        "model_config": model_config,
        "training_mode": training_mode,
        "model_type": "dino_vitb8",
        "task": "dino_feature_distillation",
        "runtime": runtime,
        "precision": precision,
        "num_total_iters": num_total_iters,
        "num_freq": num_freq,
        "gradient": gradient,
        "is_asyn": is_asyn,
        "is_init": is_init,
        "is_wandb": is_wandb,
        "is_monitor": is_monitor,
        "save_interval": save_interval,
        "is_clip": is_clip,
        "seed_for_shuffle": seed_for_shuffle,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "data": data,
        "distillation": {
            "initialization": distillation_initialization,
            "loss": distillation_loss,
            "global_weight": global_weight,
            "patch_weight": patch_weight,
        },
        "scheduler": {
            "name": scheduler_type,
            "params": {
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "total_steps": (
                    num_total_iters
                    if scheduler_total_steps is None
                    else scheduler_total_steps
                ),
            },
        },
        "optimizer": {
            "name": optimizer_name,
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
    with open(output_path, "w", encoding="utf-8") as config_file:
        yaml.dump(
            cfg,
            config_file,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Configuration written to {output_path}")
    print(f"Generated {len(layers)} SALAAD layers.")
    return cfg


if __name__ == "__main__":
    cfg_vit_b8 = dict(
        name="vit_b8",
        model_config="vit_b8_model.json",
        seed=42,
        training_mode="salad",
        lr=1e-4,
        num_freq=20,
        weight_decay=0.0,
        optimizer_name="AdamW",
        gradient="coupled",
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=5_000,
        seed_for_shuffle=42,
        is_clip=1.0,
        num_total_iters=120_000,
        batch_size=64,
        warmup_steps=2_000,
        scheduler_total_steps=120_000,
        num_workers=2,
        scheduler_type="cosine",
        min_lr_ratio=0.1,
        precision="bfloat16",
        runtime="cluster",
        data_location="cluster_snapshot",
        data_root=(
            "/lustre/fast/fast/hma2/data/imagenet2012/hf_cache/hub/"
            "datasets--ILSVRC--imagenet-1k/snapshots/"
            "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
        ),
        data_cache_dir="/lustre/home/hma2/hf/datasets",
        data_split="train",
        data_streaming=True,
        data_shuffle=True,
        shuffle_buffer_size=10_000,
        distillation_initialization="random_init",
        distillation_loss="mse",
        global_weight=1.0,
        patch_weight=1.0,
        include_attention=True,
        include_mlp=True,
        vit_layers=-1,  # all 12 blocks: qkv, proj, fc1, and fc2
    )

    cfg_vit_b8_vanilla = copy.deepcopy(cfg_vit_b8)
    cfg_vit_b8_vanilla.update(
        name="vit_b8_vanilla",
        training_mode="vanilla",
        num_total_iters=120_000,
    )

    cfg_vit_b8_vanilla_throughput = copy.deepcopy(cfg_vit_b8_vanilla)
    cfg_vit_b8_vanilla_throughput.update(
        name="vit_b8_vanilla_throughput",
        num_total_iters=400,
        num_freq=10,
        save_interval=400,
    )

    cfg_vit_b8_mixed_rho = copy.deepcopy(cfg_vit_b8)
    cfg_vit_b8_mixed_rho.update(
        name="vit_b8_all_qkv_rho5e6_fc_rho5e8",
        rho_by_suffix=VIT_MIXED_RHO_BY_SUFFIX,
    )

    cfg_vit_b8_bmixed = copy.deepcopy(cfg_vit_b8_mixed_rho)
    cfg_vit_b8_bmixed.update(
        name="vit_b8_bmixed",
        block_shape_by_suffix=VIT_BMIXED_BLOCK_SHAPES,
        alpha_rate_decay=VIT_BMIXED_ALPHA_RATE_DECAY,
        beta_rate_decay=VIT_BMIXED_BETA_RATE_DECAY,
    )

    cfg_vit_b8_bmixed_smoke = dict(
        name="vit_b8_bmixed_smoke",
        model_config="vit_b8_model.json",
        training_mode="salad",
        lr=1e-5,
        num_freq=2,
        is_wandb=False,
        is_monitor=True,
        save_interval=2,
        num_total_iters=10,
        batch_size=1,
        warmup_steps=1,
        scheduler_total_steps=10,
        num_workers=0,
        precision="bfloat16",
        runtime="local",
        data_location="local_smoke",
        data_root="data/salaad_vision/smoke/imagenet_val64_parquet",
        data_cache_dir="data/salaad_vision/hf_cache_smoke/datasets",
        data_split="validation",
        data_streaming=True,
        data_shuffle=False,
        shuffle_buffer_size=64,
        distillation_initialization="random_init",
        include_attention=True,
        include_mlp=True,
        vit_layers=1,
        block_shape_by_suffix=VIT_BMIXED_BLOCK_SHAPES,
        excluded_suffixes=VIT_BMIXED_SMOKE_EXCLUDED_SUFFIXES,
        alpha_rate_decay=VIT_BMIXED_ALPHA_RATE_DECAY,
        beta_rate_decay=VIT_BMIXED_BETA_RATE_DECAY,
    )

    generate_vit_config(**cfg_vit_b8)
    generate_vit_config(**cfg_vit_b8_vanilla)
    generate_vit_config(**cfg_vit_b8_vanilla_throughput)
    generate_vit_config(**cfg_vit_b8_mixed_rho)
    generate_vit_config(**cfg_vit_b8_bmixed)
    generate_vit_config(**cfg_vit_b8_bmixed_smoke)
