"""Evaluate structured SALAAD checkpoints with LM Evaluation Harness.

This script is intentionally edit-friendly: change the constants below or set
environment variables before running. It evaluates only these variants:

- SALAAD X
- SALAAD L+S
- paired vanilla dense

It does not evaluate vanilla post-hoc recovered L+S.
"""

import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(ROOT, "data", "structured_salaad", "llama_350m")

SALAAD_FOLDER = os.path.join(BASE_DIR, "20260607_171954")
VANILLA_FOLDER = os.path.join(BASE_DIR, "20260607_180343")

MODEL_TYPE = "llama_350m"
OUTPUT_ROOT = os.path.join(BASE_DIR, "downstream_eval")
MODEL_EXPORT_DIR = os.path.join(OUTPUT_ROOT, "hf_models")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "lm_harness_results")
CACHE_DIR = os.path.join(OUTPUT_ROOT, "hf_cache")

os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ROOT)
from salad.register import get_model
from salad.utils import load_model, mkdir

TASKS = [
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "copa",
]
FEWSHOTS = [0, 5]

BATCH_SIZE = 16
OVERWRITE_EXPORTS = False


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_list(name: str, default: List[str]) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def env_int_list(name: str, default: List[int]) -> List[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def env_limit() -> Optional[int]:
    value = os.environ.get("LIMIT")
    if not value:
        return None
    return int(value)


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
    except Exception:
        pass
    return obj


def load_l_plus_s(matrix_path: str) -> torch.Tensor:
    with open(matrix_path, "rb") as f:
        obj = pickle.load(f)
    return (obj["LL"]["embed_tokens"] + obj["SS"]["embed_tokens"]).detach().float().cpu()


def load_t5_tokenizer(model_max_length: int = 256):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
            model_max_length=model_max_length,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
            model_max_length=model_max_length,
        )
    return tokenizer


def export_variant(
    *,
    variant_name: str,
    checkpoint_folder: str,
    replace_embed_l_plus_s: bool,
) -> str:
    output_dir = os.path.join(MODEL_EXPORT_DIR, variant_name)
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path) and not OVERWRITE_EXPORTS:
        print(f"[export] reuse {variant_name}: {output_dir}", flush=True)
        return output_dir

    mkdir(output_dir)
    cfg_model = os.path.join(checkpoint_folder, f"{MODEL_TYPE}_model.json")
    cfg_train = os.path.join(checkpoint_folder, f"{MODEL_TYPE}.yaml")
    model_path = os.path.join(checkpoint_folder, "model.pth")

    print(f"[export] loading {variant_name} from {checkpoint_folder}", flush=True)
    model = get_model(cfg_model)
    load_model(model, model_path)

    if replace_embed_l_plus_s:
        matrix_path = os.path.join(checkpoint_folder, "matrix_rank0.pkl")
        l_plus_s = load_l_plus_s(matrix_path)
        embedding = model.get_input_embeddings()
        if tuple(embedding.weight.shape) != tuple(l_plus_s.shape):
            raise ValueError(
                f"Embedding shape mismatch for {variant_name}: "
                f"{tuple(embedding.weight.shape)} vs {tuple(l_plus_s.shape)}"
            )
        embedding.weight.data.copy_(l_plus_s.to(dtype=embedding.weight.dtype))
        print(f"[export] replaced input embedding with training-time L+S", flush=True)

    model.to(torch.bfloat16)
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = load_t5_tokenizer(model_max_length=256)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "variant_name": variant_name,
        "checkpoint_folder": checkpoint_folder,
        "model_path": model_path,
        "training_config": cfg_train,
        "replace_embed_l_plus_s": replace_embed_l_plus_s,
    }
    with open(os.path.join(output_dir, "variant_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[export] saved {variant_name}: {output_dir}", flush=True)
    return output_dir


def export_all_variants() -> Dict[str, str]:
    mkdir(MODEL_EXPORT_DIR)
    return {
        "salaad_X": export_variant(
            variant_name="salaad_X",
            checkpoint_folder=SALAAD_FOLDER,
            replace_embed_l_plus_s=False,
        ),
        "salaad_L_plus_S": export_variant(
            variant_name="salaad_L_plus_S",
            checkpoint_folder=SALAAD_FOLDER,
            replace_embed_l_plus_s=True,
        ),
        "vanilla_dense": export_variant(
            variant_name="vanilla_dense",
            checkpoint_folder=VANILLA_FOLDER,
            replace_embed_l_plus_s=False,
        ),
    }


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


def run_lm_harness(model_dirs: Dict[str, str]) -> None:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    tasks = env_list("TASKS", TASKS)
    fewshots = env_int_list("FEWSHOTS", FEWSHOTS)
    limit = env_limit()
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = os.environ.get("EVAL_DTYPE", "bfloat16" if device.startswith("cuda") else "float32")
    batch_size = int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE)))

    mkdir(RESULTS_DIR)
    summary_rows = []

    print(
        "[eval]",
        {
            "tasks": tasks,
            "fewshots": fewshots,
            "limit": limit,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        },
        flush=True,
    )

    for variant_name, model_dir in model_dirs.items():
        for num_fewshot in fewshots:
            print(f"[eval] {variant_name} fewshot={num_fewshot}", flush=True)
            lm = HFLM(
                pretrained=model_dir,
                dtype=dtype,
                device=device,
                batch_size=batch_size,
            )
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
            )

            out_dir = os.path.join(RESULTS_DIR, variant_name, f"fewshot_{num_fewshot}")
            mkdir(out_dir)
            with open(os.path.join(out_dir, "results.pkl"), "wb") as f:
                pickle.dump(results, f)
            with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(jsonable(results), f, indent=2)

            for task, task_result in results.get("results", {}).items():
                metric = preferred_metric(task_result)
                if metric is None:
                    continue
                metric_name, metric_value = metric
                summary_rows.append(
                    {
                        "variant": variant_name,
                        "fewshot": num_fewshot,
                        "task": task,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

            del lm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[eval] saved summary: {summary_path}", flush=True)


def main() -> None:
    run_prepare = env_bool("RUN_PREPARE", True)
    run_eval = env_bool("RUN_EVAL", True)

    if run_prepare:
        model_dirs = export_all_variants()
    else:
        model_dirs = {
            "salaad_X": os.path.join(MODEL_EXPORT_DIR, "salaad_X"),
            "salaad_L_plus_S": os.path.join(MODEL_EXPORT_DIR, "salaad_L_plus_S"),
            "vanilla_dense": os.path.join(MODEL_EXPORT_DIR, "vanilla_dense"),
        }

    print("[models]", model_dirs, flush=True)

    if run_eval:
        run_lm_harness(model_dirs)
    else:
        print("[eval] skipped because RUN_EVAL=0", flush=True)


if __name__ == "__main__":
    main()
