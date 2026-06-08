"""LoRA fine-tuning and optional LM Harness evaluation.

This script is intentionally edit-friendly. It fine-tunes a local HF-exported
LLaMA-style checkpoint with LoRA on a small task dataset, then optionally runs
LM Evaluation Harness on the resulting adapter.

Default use case:

    BASE_VARIANT=vanilla_dense TRAIN_TASK=boolq TOKEN_BUDGET=1000000 \
    myenv/bin/python -u scripts/lora_finetune_eval.py

The default evaluation writes only a compact summary. Set SAVE_FULL_RESULTS=1
only when per-sample debugging is needed; full harness outputs are large.
"""

import json
import math
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(ROOT, "data", "structured_salaad", "llama_350m")
HF_MODEL_ROOT = os.path.join(BASE_DIR, "downstream_eval", "hf_models")
OUTPUT_ROOT = os.path.join(BASE_DIR, "lora_finetune")
CACHE_DIR = os.path.join(OUTPUT_ROOT, "hf_cache")

os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:
    raise SystemExit("peft is required. Install it with `pip install peft`.") from exc

sys.path.insert(0, ROOT)
from salad.utils import mkdir


BASE_VARIANT = os.environ.get("BASE_VARIANT", "vanilla_dense")
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", os.path.join(HF_MODEL_ROOT, BASE_VARIANT))

TRAIN_TASK = os.environ.get("TRAIN_TASK", "boolq")
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "train")
EVAL_TASKS = [x.strip() for x in os.environ.get(
    "EVAL_TASKS",
    "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,copa",
).split(",") if x.strip()]
FEWSHOTS = [int(x.strip()) for x in os.environ.get("FEWSHOTS", "0").split(",") if x.strip()]

TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "1000000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
SEED = int(os.environ.get("SEED", "42"))

LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
TARGET_MODULES = [x.strip() for x in os.environ.get(
    "TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).split(",") if x.strip()]

RUN_EVAL = os.environ.get("RUN_EVAL", "1").strip().lower() in {"1", "true", "yes", "y"}
SAVE_FULL_RESULTS = os.environ.get("SAVE_FULL_RESULTS", "0").strip().lower() in {"1", "true", "yes", "y"}
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
EVAL_LIMIT = os.environ.get("EVAL_LIMIT")
EVAL_LIMIT = int(EVAL_LIMIT) if EVAL_LIMIT else None

RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"{BASE_VARIANT}_{TRAIN_TASK}_lora_r{LORA_R}_{TOKEN_BUDGET}tok_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapter")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "lm_harness_results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_snapshot() -> Dict[str, Any]:
    return {
        "base_variant": BASE_VARIANT,
        "base_model_dir": BASE_MODEL_DIR,
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
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "run_eval": RUN_EVAL,
        "save_full_results": SAVE_FULL_RESULTS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_limit": EVAL_LIMIT,
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


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, model_max_length=MAX_LENGTH)
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


def build_dataset(tokenizer: AutoTokenizer) -> datasets.Dataset:
    data = load_train_examples(TRAIN_TASK, TRAIN_SPLIT)

    def tok(row: Dict[str, str]) -> Dict[str, List[int]]:
        return tokenize_example(tokenizer, row["prompt"], row["target"])

    data = data.shuffle(seed=SEED)
    return data.map(tok, remove_columns=data.column_names)


def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor([example[key] for example in batch], dtype=torch.long)
        for key in ("input_ids", "attention_mask", "labels")
    }


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_lora() -> Dict[str, Any]:
    if not os.path.exists(os.path.join(BASE_MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"Missing HF-exported base model at {BASE_MODEL_DIR}. "
            "Run scripts/structured_salaad_downstream_eval.py with RUN_PREPARE=1 RUN_EVAL=0 first."
        )

    mkdir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(env_snapshot(), f, indent=2)

    set_seed(SEED)
    tokenizer = load_tokenizer()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=dtype)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.train()

    trainable, total = count_trainable_parameters(model)
    print(
        f"[model] trainable={trainable:,} total={total:,} "
        f"ratio={trainable / total:.6f}",
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
            (loss / GRAD_ACCUM_STEPS).backward()
            running_loss += float(loss.detach().cpu())
            seen_tokens += total_tokens
            micro_steps += 1

            if micro_steps % GRAD_ACCUM_STEPS == 0 or seen_tokens >= TOKEN_BUDGET:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if optimizer_steps % 10 == 0 or seen_tokens >= TOKEN_BUDGET:
                    avg_loss = running_loss / max(1, micro_steps)
                    print(
                        f"[train] opt_step={optimizer_steps} micro_step={micro_steps} "
                        f"tokens={seen_tokens}/{TOKEN_BUDGET} supervised_tokens={supervised_tokens} "
                        f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )

            if seen_tokens >= TOKEN_BUDGET:
                break

    mkdir(ADAPTER_DIR)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    train_summary = {
        "adapter_dir": ADAPTER_DIR,
        "seen_tokens": seen_tokens,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "mean_loss": running_loss / max(1, micro_steps),
        "avg_tokens_per_example": avg_tokens_per_example,
        "estimated_optimizer_steps": estimated_steps,
        "trainable_params": trainable,
        "total_params_with_adapter": total,
        "trainable_ratio": trainable / total,
    }
    with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    print(f"[train] saved adapter: {ADAPTER_DIR}", flush=True)
    return train_summary


def run_lm_harness() -> None:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    mkdir(RESULTS_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    summary_rows: List[Dict[str, Any]] = []

    for num_fewshot in FEWSHOTS:
        print(f"[eval] tasks={EVAL_TASKS} fewshot={num_fewshot}", flush=True)
        lm = HFLM(
            pretrained=BASE_MODEL_DIR,
            peft=ADAPTER_DIR,
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
                    "base_variant": BASE_VARIANT,
                    "adapter": RUN_NAME,
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


def main() -> None:
    train_summary = train_lora()
    print("[train_summary]", train_summary, flush=True)
    if RUN_EVAL:
        run_lm_harness()
    else:
        print("[eval] skipped because RUN_EVAL=0", flush=True)


if __name__ == "__main__":
    main()
