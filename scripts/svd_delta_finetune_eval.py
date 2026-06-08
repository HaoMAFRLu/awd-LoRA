"""Fine-tune fixed SALAAD singular directions by learning delta singular values.

For each target linear layer, this script replaces the dense layer with

    W = U diag(sigma0 + delta_sigma) Vh + S

where U, sigma0, Vh come from the SVD of the saved SALAAD low-rank component L,
S is the saved sparse component, and only delta_sigma is trainable.

Default use case:

    CHECKPOINT_FOLDER=data/head_bf16/llama_350m/20260102_233510 \
    myenv/bin/python -u scripts/svd_delta_finetune_eval.py

By default the target set is every configured transformer block matrix in the
checkpoint, excluding embed_tokens and lm_head. The output saves compact
delta_sigma tensors and summaries, not a materialized dense model.
"""

import gc
import json
import math
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINT_FOLDER = os.path.join(
    ROOT,
    "data",
    "head_bf16",
    "llama_350m",
    "20260102_233510",
)

MODEL_TYPE = os.environ.get("MODEL_TYPE", "llama_350m")
CHECKPOINT_FOLDER = os.path.abspath(
    os.environ.get("CHECKPOINT_FOLDER", DEFAULT_CHECKPOINT_FOLDER)
)
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "").strip()
TOKENIZER_NAME_OR_PATH = os.environ.get(
    "TOKENIZER_NAME_OR_PATH",
    BASE_MODEL_DIR if BASE_MODEL_DIR else "t5-base",
)

OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    os.path.join(os.path.dirname(CHECKPOINT_FOLDER), "svd_delta_finetune"),
)
CACHE_DIR = os.path.join(OUTPUT_ROOT, "hf_cache")

os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, ROOT)
from salad.register import get_model
from salad.utils import load_model, mkdir


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_list(name: str, default: Sequence[str]) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def env_int_list(name: str, default: Sequence[int]) -> List[int]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def env_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    return int(value) if value else None


TRAIN_TASK = os.environ.get("TRAIN_TASK", "boolq")
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "train")
EVAL_TASKS = env_list(
    "EVAL_TASKS",
    ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "copa"],
)
FEWSHOTS = env_int_list("FEWSHOTS", [0])

TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "1000000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "1"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
SEED = int(os.environ.get("SEED", "42"))

EXCLUDE_LAYERS = set(env_list("EXCLUDE_LAYERS", ["embed_tokens", "lm_head"]))
TARGET_LAYERS_ENV = env_list("TARGET_LAYERS", [])
TARGET_BLOCKS = env_int_list("TARGET_BLOCKS", [])
MAX_TARGET_LAYERS = env_optional_int("MAX_TARGET_LAYERS")
STRICT_TARGETS = env_bool("STRICT_TARGETS", True)

RUN_EVAL = env_bool("RUN_EVAL", True)
SAVE_FULL_RESULTS = env_bool("SAVE_FULL_RESULTS", False)
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
EVAL_LIMIT = env_optional_int("EVAL_LIMIT")
RUN_EVAL_LOSS = env_bool("RUN_EVAL_LOSS", True)
EVAL_LOSS_SPLIT = os.environ.get("EVAL_LOSS_SPLIT", "validation")
EVAL_LOSS_BATCH_SIZE = int(os.environ.get("EVAL_LOSS_BATCH_SIZE", str(EVAL_BATCH_SIZE)))

USE_WANDB = env_bool("USE_WANDB", False)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SALAD_llama_350m_fine_tune")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "hao-ma-eth-z-rich").strip() or None

DEVICE_ENV = os.environ.get("DEVICE", "")
SVD_DEVICE = os.environ.get("SVD_DEVICE", "cpu")
TRAIN_DTYPE = os.environ.get("TRAIN_DTYPE", "bfloat16")
TOKENIZER_LOCAL_ONLY = env_bool("TOKENIZER_LOCAL_ONLY", False)
DRY_RUN = env_bool("DRY_RUN", False)

RUN_NAME = os.environ.get(
    "RUN_NAME",
    (
        f"{os.path.basename(CHECKPOINT_FOLDER)}_{TRAIN_TASK}_svd_delta_"
        f"{TOKEN_BUDGET}tok_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ),
)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)
RESULTS_DIR = os.path.join(OUTPUT_DIR, "lm_harness_results")
DELTA_STATE_PATH = os.path.join(OUTPUT_DIR, "svd_delta_state.pt")
ACTIVE_WANDB_RUN = None


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported TRAIN_DTYPE={name!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_layer_name(name: str) -> str:
    name = name.strip()
    if name.startswith("model."):
        return name[len("model.") :]
    return name


def full_module_name(layer_name: str) -> str:
    if layer_name == "lm_head" or layer_name.startswith("lm_head."):
        return layer_name
    if layer_name == "embed_tokens" or layer_name.startswith("embed_tokens."):
        return "model." + layer_name
    return "model." + layer_name


def parent_and_child(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def checkpoint_yaml_path() -> str:
    return os.path.join(CHECKPOINT_FOLDER, f"{MODEL_TYPE}.yaml")


def checkpoint_model_json_path() -> str:
    return os.path.join(CHECKPOINT_FOLDER, f"{MODEL_TYPE}_model.json")


def checkpoint_model_path() -> str:
    return os.path.join(CHECKPOINT_FOLDER, "model.pth")


def matrix_paths() -> List[str]:
    files = [
        os.path.join(CHECKPOINT_FOLDER, name)
        for name in os.listdir(CHECKPOINT_FOLDER)
        if name.startswith("matrix_rank") and name.endswith(".pkl")
    ]
    return sorted(files)


def load_config() -> Dict[str, Any]:
    with open(checkpoint_yaml_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config() -> Dict[str, Any]:
    with open(checkpoint_model_json_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def select_target_layers() -> List[str]:
    if TARGET_LAYERS_ENV:
        layers = [normalize_layer_name(name) for name in TARGET_LAYERS_ENV]
    else:
        cfg = load_config()
        layers = [entry["name"] for entry in cfg.get("layers", [])]

    layers = [name for name in layers if name not in EXCLUDE_LAYERS]
    if TARGET_BLOCKS:
        prefixes = tuple(f"layers.{idx}." for idx in TARGET_BLOCKS)
        layers = [name for name in layers if name.startswith(prefixes)]
    if MAX_TARGET_LAYERS is not None:
        layers = layers[:MAX_TARGET_LAYERS]
    if not layers:
        raise ValueError("No target layers selected.")
    return layers


def estimate_sv_params_from_config(layers: Iterable[str]) -> Optional[int]:
    try:
        cfg = load_model_config()
    except FileNotFoundError:
        return None
    hidden = int(cfg["hidden_size"])
    vocab = int(cfg.get("vocab_size", hidden))
    total = 0
    for name in layers:
        if name in {"embed_tokens", "lm_head"}:
            total += min(vocab, hidden)
        elif name.startswith("layers.") and (
            name.endswith("q_proj")
            or name.endswith("k_proj")
            or name.endswith("v_proj")
            or name.endswith("o_proj")
            or name.endswith("gate_proj")
            or name.endswith("up_proj")
            or name.endswith("down_proj")
        ):
            total += hidden
        else:
            return None
    return total


def env_snapshot(target_layers: Sequence[str]) -> Dict[str, Any]:
    return {
        "model_type": MODEL_TYPE,
        "checkpoint_folder": CHECKPOINT_FOLDER,
        "base_model_dir": BASE_MODEL_DIR,
        "tokenizer_name_or_path": TOKENIZER_NAME_OR_PATH,
        "train_task": TRAIN_TASK,
        "train_split": TRAIN_SPLIT,
        "eval_tasks": EVAL_TASKS,
        "fewshots": FEWSHOTS,
        "token_budget": TOKEN_BUDGET,
        "max_length": MAX_LENGTH,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "exclude_layers": sorted(EXCLUDE_LAYERS),
        "target_blocks": TARGET_BLOCKS,
        "target_layers": list(target_layers),
        "strict_targets": STRICT_TARGETS,
        "run_eval": RUN_EVAL,
        "save_full_results": SAVE_FULL_RESULTS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_limit": EVAL_LIMIT,
        "run_eval_loss": RUN_EVAL_LOSS,
        "eval_loss_split": EVAL_LOSS_SPLIT,
        "eval_loss_batch_size": EVAL_LOSS_BATCH_SIZE,
        "use_wandb": USE_WANDB,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "device": DEVICE_ENV,
        "svd_device": SVD_DEVICE,
        "train_dtype": TRAIN_DTYPE,
        "output_dir": OUTPUT_DIR,
    }


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.dtype):
            return str(obj)
    except Exception:
        pass
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def preferred_metric(task_result: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    preferred_keys = [
        "acc_norm,none",
        "acc,none",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "f1,none",
    ]
    for key in preferred_keys:
        value = task_result.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in task_result.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr,none"):
            return key, float(value)
    return None


def init_wandb(target_layers: Sequence[str]):
    global ACTIVE_WANDB_RUN
    if not USE_WANDB:
        return None
    try:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        settings = None
        try:
            settings = wandb.Settings(
                disable_code=True,
                disable_git=True,
                _disable_stats=True,
            )
        except Exception:
            try:
                settings = wandb.Settings(disable_code=True, disable_git=True)
            except Exception:
                settings = None
        init_kwargs = {
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "name": RUN_NAME,
            "config": {},
        }
        if settings is not None:
            init_kwargs["settings"] = settings
        run = wandb.init(**init_kwargs)
        ACTIVE_WANDB_RUN = run
        print(f"[wandb] initialized project={WANDB_PROJECT} run={RUN_NAME}", flush=True)
        return run
    except Exception as exc:
        print(f"[wandb] disabled after init failure: {exc}", flush=True)
        ACTIVE_WANDB_RUN = None
        return None


def finish_wandb() -> None:
    global ACTIVE_WANDB_RUN
    if ACTIVE_WANDB_RUN is None:
        return
    try:
        ACTIVE_WANDB_RUN.finish()
    finally:
        ACTIVE_WANDB_RUN = None


def load_tokenizer() -> AutoTokenizer:
    kwargs = {"model_max_length": MAX_LENGTH}
    if TOKENIZER_LOCAL_ONLY:
        kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def boolq_examples(split: str) -> datasets.Dataset:
    data = datasets.load_dataset("super_glue", "boolq", split=split)

    def fmt(row: Dict[str, Any]) -> Dict[str, str]:
        answer = " yes" if bool(row["label"]) else " no"
        prompt = f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer:"
        return {"prompt": prompt, "target": answer}

    return data.map(fmt, remove_columns=data.column_names)


def arc_examples(split: str, challenge: bool) -> datasets.Dataset:
    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    data = datasets.load_dataset("allenai/ai2_arc", subset, split=split)

    def fmt(row: Dict[str, Any]) -> Dict[str, str]:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer_key = row["answerKey"]
        try:
            idx = labels.index(answer_key)
        except ValueError:
            idx = int(answer_key) - 1
        choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        prompt = f"Question: {row['question']}\nChoices:\n{choices}\nAnswer:"
        return {"prompt": prompt, "target": " " + texts[idx]}

    return data.map(fmt, remove_columns=data.column_names)


def piqa_examples(split: str) -> datasets.Dataset:
    data = datasets.load_dataset("piqa", split=split)

    def fmt(row: Dict[str, Any]) -> Dict[str, str]:
        target = row["sol1"] if int(row["label"]) == 0 else row["sol2"]
        prompt = f"Goal: {row['goal']}\nChoices:\nA. {row['sol1']}\nB. {row['sol2']}\nAnswer:"
        return {"prompt": prompt, "target": " " + target}

    return data.map(fmt, remove_columns=data.column_names)


def load_train_examples(task: str, split: str) -> datasets.Dataset:
    task = task.lower()
    if task == "boolq":
        return boolq_examples(split)
    if task == "arc_easy":
        return arc_examples(split, challenge=False)
    if task == "arc_challenge":
        return arc_examples(split, challenge=True)
    if task == "piqa":
        return piqa_examples(split)
    raise ValueError(f"Unsupported TRAIN_TASK={task!r}; expected boolq, arc_easy, arc_challenge, or piqa")


def tokenize_example(tokenizer: AutoTokenizer, prompt: str, target: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > MAX_LENGTH:
        overflow = len(input_ids) - MAX_LENGTH
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]

    attention_mask = [1] * len(input_ids)
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_task_dataset(tokenizer: AutoTokenizer, split: str, *, shuffle: bool) -> datasets.Dataset:
    data = load_train_examples(TRAIN_TASK, split)

    def tok(row: Dict[str, str]) -> Dict[str, List[int]]:
        return tokenize_example(tokenizer, row["prompt"], row["target"])

    if shuffle:
        data = data.shuffle(seed=SEED)
    return data.map(tok, remove_columns=data.column_names)


def build_dataset(tokenizer: AutoTokenizer) -> datasets.Dataset:
    return build_task_dataset(tokenizer, TRAIN_SPLIT, shuffle=True)


def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor([example[key] for example in batch], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


class SingularValueDeltaLinear(nn.Module):
    """Frozen U/Vh/S linear layer with trainable additive singular-value deltas."""

    def __init__(
        self,
        *,
        U: torch.Tensor,
        sigma0: torch.Tensor,
        Vh: torch.Tensor,
        S: torch.Tensor,
        bias: Optional[torch.Tensor],
        layer_name: str,
    ) -> None:
        super().__init__()
        if U.dim() != 2 or Vh.dim() != 2 or S.dim() != 2 or sigma0.dim() != 1:
            raise ValueError("U, Vh, S, and sigma0 must have SVD-compatible ranks.")
        out_features, rank = U.shape
        rank2, in_features = Vh.shape
        if rank != rank2 or sigma0.numel() != rank:
            raise ValueError(f"Bad SVD shapes for {layer_name}.")
        if tuple(S.shape) != (out_features, in_features):
            raise ValueError(f"S shape {tuple(S.shape)} does not match {layer_name}.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.layer_name = layer_name

        self.register_buffer("U", U, persistent=False)
        self.register_buffer("sigma0", sigma0.float(), persistent=False)
        self.register_buffer("Vh", Vh, persistent=False)
        self.register_buffer("S", S, persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias, persistent=False)
        self.delta_sigma = nn.Parameter(torch.zeros(rank, dtype=torch.float32, device=sigma0.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.Vh)
        scale = (self.sigma0 + self.delta_sigma).to(dtype=z.dtype)
        y_l = F.linear(z * scale, self.U)
        return y_l + F.linear(x, self.S, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, layer_name={self.layer_name!r}"
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def svd_delta_modules(model: nn.Module) -> Dict[str, SingularValueDeltaLinear]:
    modules: Dict[str, SingularValueDeltaLinear] = {}
    for _, module in model.named_modules():
        if isinstance(module, SingularValueDeltaLinear):
            modules[module.layer_name] = module
    return modules


def wandb_log(payload: Dict[str, Any], step: Optional[int] = None) -> None:
    if ACTIVE_WANDB_RUN is None:
        return
    try:
        import wandb

        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
    except Exception as exc:
        print(f"[wandb] log failed: {exc}", flush=True)


def load_base_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    if BASE_MODEL_DIR:
        if not os.path.exists(os.path.join(BASE_MODEL_DIR, "config.json")):
            raise FileNotFoundError(f"BASE_MODEL_DIR has no config.json: {BASE_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=dtype)
    else:
        model = get_model(checkpoint_model_json_path())
        load_model(model, checkpoint_model_path())
        model.to(dtype=dtype)
    model.config.use_cache = False
    model.to(device)
    return model


def replace_layer_with_svd_delta(
    *,
    model: nn.Module,
    layer_name: str,
    L: torch.Tensor,
    S: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    svd_device: torch.device,
) -> Dict[str, Any]:
    module_name = full_module_name(layer_name)
    old_module = model.get_submodule(module_name)
    if not isinstance(old_module, nn.Linear):
        raise TypeError(f"Target {module_name} is {type(old_module)}, not nn.Linear.")
    if tuple(old_module.weight.shape) != tuple(L.shape) or tuple(S.shape) != tuple(L.shape):
        raise ValueError(
            f"Shape mismatch for {layer_name}: weight={tuple(old_module.weight.shape)} "
            f"L={tuple(L.shape)} S={tuple(S.shape)}"
        )

    L_work = L.detach().float().to(svd_device)
    U, sigma0, Vh = torch.linalg.svd(L_work, full_matrices=False)
    bias = old_module.bias.detach().to(device=device, dtype=model_dtype) if old_module.bias is not None else None
    new_module = SingularValueDeltaLinear(
        U=U.to(device=device, dtype=model_dtype),
        sigma0=sigma0.to(device=device, dtype=torch.float32),
        Vh=Vh.to(device=device, dtype=model_dtype),
        S=S.detach().to(device=device, dtype=model_dtype),
        bias=bias,
        layer_name=layer_name,
    )
    parent, child = parent_and_child(model, module_name)
    setattr(parent, child, new_module)

    info = {
        "name": layer_name,
        "shape": list(L.shape),
        "num_singular_values": int(sigma0.numel()),
        "sigma0_min": float(sigma0.min().detach().cpu()),
        "sigma0_max": float(sigma0.max().detach().cpu()),
        "sigma0_mean": float(sigma0.mean().detach().cpu()),
    }

    del old_module, L_work, U, sigma0, Vh
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return info


def patch_model_with_svd_delta(
    model: nn.Module,
    target_layers: Sequence[str],
    device: torch.device,
    model_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    paths = matrix_paths()
    if not paths:
        raise FileNotFoundError(f"No matrix_rank*.pkl files found in {CHECKPOINT_FOLDER}")

    target_set = set(target_layers)
    remaining = set(target_layers)
    metadata: List[Dict[str, Any]] = []
    svd_device = torch.device(SVD_DEVICE)

    for matrix_path in paths:
        print(f"[patch] loading {matrix_path}", flush=True)
        with open(matrix_path, "rb") as f:
            obj = pickle.load(f)
        LL = obj["LL"]
        SS = obj["SS"]
        names = [name for name in target_layers if name in LL and name in target_set]
        for name in names:
            print(
                f"[patch] {len(metadata) + 1}/{len(target_layers)} {name} "
                f"shape={tuple(LL[name].shape)}",
                flush=True,
            )
            info = replace_layer_with_svd_delta(
                model=model,
                layer_name=name,
                L=LL[name],
                S=SS[name],
                device=device,
                model_dtype=model_dtype,
                svd_device=svd_device,
            )
            metadata.append(info)
            remaining.discard(name)

        del obj, LL, SS
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if remaining and STRICT_TARGETS:
        raise KeyError(f"Missing target layers in matrix shards: {sorted(remaining)[:20]}")

    return metadata


def save_delta_state(model: nn.Module, target_layers: Sequence[str], path: str) -> None:
    modules = svd_delta_modules(model)
    delta_sigma = {}
    sigma0 = {}
    for name in target_layers:
        module = modules[name]
        delta_sigma[name] = module.delta_sigma.detach().cpu()
        sigma0[name] = module.sigma0.detach().cpu()
    torch.save(
        {
            "delta_sigma": delta_sigma,
            "sigma0": sigma0,
            "target_layers": list(target_layers),
            "run_config": env_snapshot(target_layers),
        },
        path,
    )


def train_svd_delta() -> Tuple[nn.Module, AutoTokenizer, Dict[str, Any]]:
    target_layers = select_target_layers()
    mkdir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(env_snapshot(target_layers), f, indent=2)
    wandb_run = init_wandb(target_layers)

    set_seed(SEED)
    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

    train_data = build_dataset(tokenizer)
    lengths = [int(sum(example["attention_mask"])) for example in train_data]
    avg_tokens_per_example = sum(lengths) / max(1, len(lengths))
    loader = DataLoader(
        train_data,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    device = torch.device(DEVICE_ENV or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = dtype_from_name(TRAIN_DTYPE, device)
    print(
        f"[model] loading base device={device} dtype={model_dtype} "
        f"checkpoint={CHECKPOINT_FOLDER}",
        flush=True,
    )
    model = load_base_model(device, model_dtype)
    original_params = count_parameters(model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layer_metadata = patch_model_with_svd_delta(model, target_layers, device, model_dtype)
    model.train()

    trainable = count_trainable_parameters(model)
    print(
        f"[model] svd_delta_layers={len(layer_metadata)} trainable={trainable:,} "
        f"original_params={original_params:,} ratio={trainable / original_params:.8f}",
        flush=True,
    )

    tokens_per_optimizer_step = max(1.0, avg_tokens_per_example * MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)
    estimated_steps = max(1, math.ceil(TOKEN_BUDGET / tokens_per_optimizer_step))
    warmup_steps = max(1, int(estimated_steps * WARMUP_RATIO))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=estimated_steps,
    )

    seen_tokens = 0
    optimizer_steps = 0
    micro_steps = 0
    running_loss = 0.0

    while seen_tokens < TOKEN_BUDGET:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            supervised_tokens = int((batch["labels"] != -100).sum().item())
            total_tokens = int(batch["attention_mask"].sum().item())

            loss = model(**batch).loss
            loss_value = float(loss.detach().cpu())
            (loss / GRAD_ACCUM_STEPS).backward()
            running_loss += loss_value
            seen_tokens += total_tokens
            micro_steps += 1

            if micro_steps % GRAD_ACCUM_STEPS == 0 or seen_tokens >= TOKEN_BUDGET:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if optimizer_steps % 10 == 0 or seen_tokens >= TOKEN_BUDGET:
                    avg_loss = running_loss / max(1, micro_steps)
                    wandb_log({"train/loss": avg_loss}, step=optimizer_steps)
                    print(
                        f"[train] opt_step={optimizer_steps} micro_step={micro_steps} "
                        f"tokens={seen_tokens}/{TOKEN_BUDGET} supervised_tokens={supervised_tokens} "
                        f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )

            if seen_tokens >= TOKEN_BUDGET:
                break

    save_delta_state(model, target_layers, DELTA_STATE_PATH)
    train_summary = {
        "delta_state_path": DELTA_STATE_PATH,
        "seen_tokens": seen_tokens,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "mean_loss": running_loss / max(1, micro_steps),
        "avg_tokens_per_example": avg_tokens_per_example,
        "estimated_optimizer_steps": estimated_steps,
        "target_layer_count": len(target_layers),
        "layer_metadata": layer_metadata,
        "trainable_params": trainable,
        "original_params": original_params,
        "trainable_ratio_vs_original": trainable / original_params,
    }
    with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    if wandb_run is not None:
        wandb_run.summary["train/loss"] = train_summary["mean_loss"]
    print(f"[train] saved delta state: {DELTA_STATE_PATH}", flush=True)
    return model, tokenizer, train_summary


def compute_eval_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    optimizer_steps: int,
) -> Optional[Dict[str, Any]]:
    if not RUN_EVAL_LOSS:
        print("[eval_loss] skipped because RUN_EVAL_LOSS=0", flush=True)
        return None

    eval_data = build_task_dataset(tokenizer, EVAL_LOSS_SPLIT, shuffle=False)
    loader = DataLoader(
        eval_data,
        batch_size=EVAL_LOSS_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    supervised_token_count = 0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            supervised_tokens = int((batch["labels"] != -100).sum().item())
            if supervised_tokens == 0:
                continue
            loss = model(**batch).loss
            loss_sum += float(loss.detach().cpu()) * supervised_tokens
            supervised_token_count += supervised_tokens
            batch_count += 1

    if supervised_token_count == 0:
        raise RuntimeError(f"No supervised tokens found for eval split {EVAL_LOSS_SPLIT!r}.")

    eval_loss = loss_sum / supervised_token_count
    summary = {
        "task": TRAIN_TASK,
        "split": EVAL_LOSS_SPLIT,
        "batch_size": EVAL_LOSS_BATCH_SIZE,
        "num_examples": len(eval_data),
        "num_batches": batch_count,
        "supervised_tokens": supervised_token_count,
        "loss": eval_loss,
    }
    path = os.path.join(OUTPUT_DIR, "eval_loss_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    wandb_log({"eval/loss": eval_loss}, step=optimizer_steps)
    if ACTIVE_WANDB_RUN is not None:
        ACTIVE_WANDB_RUN.summary["eval/loss"] = eval_loss
    print(
        f"[eval_loss] split={EVAL_LOSS_SPLIT} examples={len(eval_data)} "
        f"supervised_tokens={supervised_token_count} loss={eval_loss:.4f}",
        flush=True,
    )
    return summary


def run_lm_harness(model: nn.Module, tokenizer: AutoTokenizer) -> None:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    mkdir(RESULTS_DIR)
    model.eval()
    device = str(next(model.parameters()).device)
    dtype = "bfloat16" if device.startswith("cuda") and TRAIN_DTYPE in {"bf16", "bfloat16"} else "float32"
    summary_rows: List[Dict[str, Any]] = []

    for num_fewshot in FEWSHOTS:
        print(f"[eval] tasks={EVAL_TASKS} fewshot={num_fewshot}", flush=True)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            batch_size=EVAL_BATCH_SIZE,
        )
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=EVAL_TASKS,
            num_fewshot=num_fewshot,
            limit=EVAL_LIMIT,
        )

        for task, task_result in results.get("results", {}).items():
            metric = preferred_metric(task_result)
            if metric is None:
                continue
            metric_name, metric_value = metric
            summary_rows.append(
                {
                    "checkpoint": os.path.basename(CHECKPOINT_FOLDER),
                    "run_name": RUN_NAME,
                    "train_task": TRAIN_TASK,
                    "fewshot": num_fewshot,
                    "task": task,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

        if SAVE_FULL_RESULTS:
            out_dir = os.path.join(RESULTS_DIR, f"fewshot_{num_fewshot}")
            mkdir(out_dir)
            with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(jsonable(results), f, indent=2)

        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(RESULTS_DIR, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[eval] saved summary: {summary_path}", flush=True)


def dry_run() -> None:
    target_layers = select_target_layers()
    estimate = estimate_sv_params_from_config(target_layers)
    print("[dry_run]", flush=True)
    print(f"checkpoint_folder={CHECKPOINT_FOLDER}", flush=True)
    print(f"target_layer_count={len(target_layers)}", flush=True)
    print(f"estimated_trainable_params={estimate}", flush=True)
    print("first_target_layers=", target_layers[:10], flush=True)
    print("last_target_layers=", target_layers[-10:], flush=True)


def main() -> None:
    try:
        if DRY_RUN:
            dry_run()
            return

        model, tokenizer, train_summary = train_svd_delta()
        print("[train_summary]", train_summary, flush=True)
        eval_loss_summary = compute_eval_loss(
            model,
            tokenizer,
            int(train_summary["optimizer_steps"]),
        )
        if eval_loss_summary is not None:
            print("[eval_loss_summary]", eval_loss_summary, flush=True)
        if RUN_EVAL:
            run_lm_harness(model, tokenizer)
        else:
            print("[eval] skipped because RUN_EVAL=0", flush=True)
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
