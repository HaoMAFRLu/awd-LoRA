"""
"""
import json
import torch
import math
import os, sys
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import datasets
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import transformers
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Llama import LlamaForCausalLM
from models.qwen3_vl import build_qwen3_vl_from_config
from dataloaders.Llama_dataloader import PreprocessedIterableDataset
from dataloaders.Qwen3VL_dataloader import Qwen3VLIterableDataset

def get_model(cfg: str, training_config: Optional[dict] = None):
    """
    Get the model based on the configuration.
    """
    with open(cfg, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    model_type = raw_cfg.get("model_type")
    if model_type == "qwen3_vl":
        return build_qwen3_vl_from_config(cfg)

    model_cfg = AutoConfig.from_pretrained(cfg)
    if model_cfg.model_type != "llama":
        raise ValueError(f"Unsupported model_type={model_cfg.model_type!r}")

    if training_config and training_config.get("training_mode") == "loop":
        loop_override = training_config.get("loop", {})
        if not isinstance(loop_override, dict):
            raise TypeError("training config 'loop' must be a dictionary")
        model_loop = dict(getattr(model_cfg, "loop", {}) or {})
        # Sampling and stability are training policies, not part of the
        # inference architecture. Persist only architectural/default-depth
        # overrides.
        model_loop.update({
            key: value for key, value in loop_override.items()
            if key not in {"sampling", "stability"}
        })
        model_cfg.loop = model_loop

    return LlamaForCausalLM(model_cfg)

def get_tokenizer(max_length: int = 1024, config: Optional[dict] = None):
    """
    Get the tokenizer for the model.
    """
    config = config or {}
    if config.get("model_type") == "qwen3_vl" or config.get("data", {}).get("type") == "vlm":
        return get_processor(max_length=max_length, config=config)
    tokenizer_name = config.get("tokenizer_name", "t5-base")
    return AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length)

def get_processor(max_length: int = 1024, config: Optional[dict] = None):
    config = config or {}
    processor_name = config.get("processor_name") or config.get("data", {}).get("processor_name")
    if not processor_name:
        processor_name = "Qwen/Qwen3-VL-2B-Instruct"
    return AutoProcessor.from_pretrained(processor_name, model_max_length=max_length)

def get_data(config_or_seed=42, split: str = 'train'):
    if isinstance(config_or_seed, dict):
        data_cfg = config_or_seed.get("data", {})
        seed_for_shuffle = config_or_seed.get("seed_for_shuffle", 42)
        split = data_cfg.get("split", split)
        name = data_cfg.get("name", "allenai/c4")
        subset = data_cfg.get("subset")
        streaming = data_cfg.get("streaming", True)
    else:
        seed_for_shuffle = config_or_seed
        name = "allenai/c4"
        subset = "en"
        streaming = True

    if subset:
        data = datasets.load_dataset(name, subset, split=split, streaming=streaming)
    else:
        data = datasets.load_dataset(name, split=split, streaming=streaming)
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    return data

def get_preprocessed_dataset(data, processor, config: dict, batch_size: int):
    data_cfg = config.get("data", {})
    if data_cfg.get("type", "text") == "vlm":
        return Qwen3VLIterableDataset(
            data,
            processor,
            batch_size=batch_size,
            max_length=config.get("max_length", 1024),
            image_column=data_cfg.get("image_column", "images"),
            text_column=data_cfg.get("text_column", "texts"),
            question_column=data_cfg.get("question_column", "question"),
            answer_column=data_cfg.get("answer_column", "answer"),
            system_prompt=data_cfg.get("system_prompt"),
            ignore_visual_tokens=data_cfg.get("ignore_visual_tokens", True),
        )

    return PreprocessedIterableDataset(
        data,
        processor,
        batch_size=batch_size,
        max_length=config.get("max_length", 256),
    )

def get_scheduler(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")

def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


# def get_alpha_scheduler(current_step: int=1, 
#                         total_steps: int=1000, 
#                         min_lr_ratio: float=0.2):
#     assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

#     progress = float(current_step) / float(max(1, cycle_length - num_warmup_steps))
#     cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
#     return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
