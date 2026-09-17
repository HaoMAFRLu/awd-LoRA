"""Integration with the pinned FLAME Megatron fork, without editing its checkout.

Only synchronous DP, EP=TP=PP=CP=1, and a nonsharded master optimizer are
supported. The hook wraps prepare_grads, so the upstream global clipping and
optimizer step retain their original implementation (including ChainedOptimizer).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import atomic_save, checkpoint_metadata, load_torch, validate_vanilla_branch
from .config import fingerprint, learning_rate, validate_config
from .distributed import agree_or_raise, all_finite, rank, world_size
from .groups import master, megatron_groups
from .solver import ConsensusManager

MEGATRON_REVISION = "cbaf684c5d03997e0fdd5347c5e2d371c381a3d8"
FLAME_REVISION = "e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78"


def environment_report(checkout):
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformer_engine", "apex", "transformers", "einops")
    }
    return {
        "checkout": str(checkout),
        "revision": revision,
        "required_revision": MEGATRON_REVISION,
        "tracked_checkout_clean": not bool(dirty),
        "dependencies": dependencies,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": str(torch.__version__),
        "ready": revision == MEGATRON_REVISION
        and not dirty
        and all(dependencies.values())
        and torch.cuda.is_available(),
    }


def megatron_arguments(config, data_directory, tokenizer_directory, output, resume=None):
    """Native flags checked against arguments.py at the exact pinned revision."""
    validate_config(config)
    m, t, d = (config[k] for k in ("model", "training", "data"))
    args = []

    def flag(name, value=None):
        args.append("--" + name)
        if value is not None:
            args.append(str(value))

    values = {
        "num-layers": m["num_layers"],
        "hidden-size": m["hidden_size"],
        "ffn-hidden-size": m["expert_ffn_hidden_size"],
        "moe-ffn-hidden-size": m["expert_ffn_hidden_size"],
        "num-attention-heads": m["num_attention_heads"],
        "num-query-groups": m["num_query_groups"],
        "kv-channels": m["hidden_size"] // m["num_attention_heads"],
        "num-experts": m["num_experts"],
        "moe-layer-freq": 1,
        "moe-router-topk": m["router_topk"],
        "moe-router-score-function": "softmax",
        "moe-router-dtype": "fp32",
        "moe-router-load-balancing-type": "aux_loss",
        "moe-aux-loss-coeff": t["load_balancing_coefficient"],
        "moe-z-loss-coeff": t["router_z_loss_coefficient"],
        "moe-token-dispatcher-type": "alltoall",
        "normalization": "RMSNorm",
        "norm-epsilon": m["norm_epsilon"],
        "position-embedding-type": "rope",
        "rotary-base": m["rotary_base"],
        "rotary-percent": 1.0,
        "init-method-std": m["init_std"],
        "attention-dropout": 0.0,
        "hidden-dropout": 0.0,
        "seq-length": d["seq_length"],
        "max-position-embeddings": d["seq_length"],
        "tokenizer-type": "HuggingFaceTokenizer",
        "tokenizer-model": str(Path(tokenizer_directory).resolve()),
        "make-vocab-size-divisible-by": 128,
        "micro-batch-size": t["micro_batch_sequences_per_rank"],
        "global-batch-size": t["global_batch_sequences"],
        "train-iters": t["total_optimizer_steps"],
        "lr": t["learning_rate"],
        "min-lr": t["min_learning_rate"],
        "lr-decay-style": "WSD",
        "lr-wsd-decay-style": "cosine",
        "lr-warmup-iters": t["warmup_steps"],
        "lr-wsd-decay-iters": t["decay_steps"],
        "adam-beta1": t["beta1"],
        "adam-beta2": t["beta2"],
        "adam-eps": t["epsilon"],
        "weight-decay": t["weight_decay"],
        "clip-grad": t["gradient_clip_global_norm"],
        "optimizer": "adam",
        "transformer-impl": "transformer_engine",
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "seed": config["seed"],
        "data-cache-path": str(Path(output).resolve() / "data_cache"),
        "save": str(Path(output).resolve()),
        "save-interval": t["checkpoint_interval_steps"],
        "eval-interval": t["validation_interval_steps"],
        "eval-iters": max(1, d["monitor_validation_sequences"] // t["global_batch_sequences"]),
        "log-interval": 1,
        "ckpt-format": "torch",
        "num-workers": 2,
        "dataloader-type": "single",
    }
    for name, value in values.items():
        flag(name, value)
    for split, native in (("train", "train"), ("validation", "valid"), ("test", "test")):
        flag(f"{native}-data-path", str(Path(data_directory).resolve() / f"{split}_text_document"))
    for name in (
        "swiglu",
        "disable-bias-linear",
        "untie-embeddings-and-output-weights",
        "accumulate-allreduce-grads-in-fp32",
        "attention-softmax-in-fp32",
        "moe-per-layer-logging",
        "no-one-logger",
    ):
        flag(name)
    if t["task_precision"] == "bfloat16":
        flag("bf16")
    if t["activation_recompute"] == "selective_attention":
        flag("recompute-granularity", "selective")
    if resume:
        flag("load", str(Path(resume).resolve()))
    return args


def validate_runtime(args, config):
    validate_config(config, world_size())
    expected = {
        "expert_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "use_distributed_optimizer": False,
        "overlap_grad_reduce": False,
        "overlap_param_gather": False,
        "async_save": False,
        "moe_shared_expert_intermediate_size": None,
        "moe_router_pre_softmax": False,
        "moe_expert_capacity_factor": None,
        "moe_router_enable_expert_bias": False,
        "moe_router_topk": config["model"]["router_topk"],
        "num_experts": config["model"]["num_experts"],
        "padded_vocab_size": config["model"]["padded_vocab_size"],
        "ckpt_format": "torch",
    }
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            raise ValueError(
                f"Unsupported Megatron setting {key}={getattr(args, key, None)!r}; expected {value!r}"
            )
    for key in (
        "finetune",
        "no_load_optim",
        "no_load_rng",
        "use_precision_aware_optimizer",
        "use_torch_fsdp2",
        "use_custom_fsdp",
        "moe_layer_recompute",
    ):
        if getattr(args, key, False):
            raise ValueError(f"The initial adapter does not support {key}")


class MegatronOptimizerHook:
    def __init__(self, optimizer, groups, config, step=0):
        self.optimizer, self.config, self.step = optimizer, config, step
        self.manager = ConsensusManager(groups, config) if config["salaad"]["enabled"] else None
        self.device = groups[0].weight().device
        self.last_metrics = {}
        self.original_prepare, self.original_step = optimizer.prepare_grads, optimizer.step
        if getattr(optimizer, "_moe_salaad_hook", None) is not None:
            raise ValueError("SALAAD optimizer hook was already installed")
        optimizer.prepare_grads = self.prepare_grads
        optimizer.step = self.optimizer_step
        optimizer._moe_salaad_hook = self
        if self.manager and step == 0 and config["salaad"]["state_initialization_step"] == 0:
            self.manager.initialize(0)

    def parameters(self):
        seen, parameters = set(), []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in seen:
                    parameters.append(p)
                    seen.add(id(p))
        return parameters

    @torch.no_grad()
    def prepare_grads(self):
        found_inf = self.original_prepare()
        parameters = self.parameters()
        error = (
            FloatingPointError("Megatron found nonfinite task gradients")
            if found_inf or not all_finite([p.grad for p in parameters])
            else None
        )
        agree_or_raise(error, self.device, "Megatron task gradients")
        task_norm = (
            torch.stack([p.grad.float().square().sum() for p in parameters if p.grad is not None])
            .sum()
            .sqrt()
            .item()
        )
        constraint_norm = self.manager.inject_gradients() if self.manager else 0.0
        error = (
            None
            if all_finite([p.grad for p in parameters])
            else FloatingPointError("Nonfinite SALAAD gradient")
        )
        agree_or_raise(error, self.device, "Megatron constraint injection")
        self.last_metrics = {
            "task_gradient_norm": task_norm,
            "constraint_gradient_norm": constraint_norm,
            "constraint_task_gradient_ratio": constraint_norm / max(task_norm, 1e-30),
        }
        return False

    @torch.no_grad()
    def optimizer_step(self):
        lr = learning_rate(self.config, self.step + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_mult", 1.0)
        result = self.original_step()  # calls patched prepare, then original clip
        agree_or_raise(
            None
            if result[0] and all_finite(self.parameters())
            else FloatingPointError("Megatron optimizer update failed"),
            self.device,
            "Megatron optimizer",
        )
        next_step = self.step + 1
        changed = (
            self.manager.after_step(
                next_step, final=next_step == self.config["training"]["total_optimizer_steps"]
            )
            if self.manager
            else False
        )
        self.step = next_step
        self.last_metrics.update(step=self.step, learning_rate=lr)
        if changed:
            self.last_metrics["salaad"] = self.manager.metrics()
        return result

    def uninstall(self):
        self.optimizer.prepare_grads, self.optimizer.step = (
            self.original_prepare,
            self.original_step,
        )
        del self.optimizer._moe_salaad_hook


@torch.no_grad()
def native_weights_from_megatron(chunks, config):
    """Convert full-MHA TP=1 weights, including TE's fused norm and QKV layout."""
    model = chunks[0]
    while hasattr(model, "module"):
        model = model.module

    def value(parameter):
        return master(parameter).detach().cpu().clone()

    weights = {
        "embedding.weight": value(model.embedding.word_embeddings.weight),
        "lm_head.weight": value(model.output_layer.weight),
        "norm.weight": value(model.decoder.final_layernorm.weight),
    }
    d, h = config["model"]["hidden_size"], config["model"]["num_attention_heads"]
    for i, layer in enumerate(model.decoder.layers):
        prefix = f"layers.{i}."
        qkv = layer.self_attention.linear_qkv
        norm = (
            qkv.layer_norm_weight
            if hasattr(qkv, "layer_norm_weight")
            else layer.input_layernorm.weight
        )
        weights[prefix + "attention_norm.weight"] = value(norm)
        weights[prefix + "attention.qkv.weight"] = (
            value(qkv.weight).view(h, 3, d // h, d).transpose(0, 1).reshape(3 * d, d).contiguous()
        )
        weights[prefix + "attention.out.weight"] = value(layer.self_attention.linear_proj.weight)
        weights[prefix + "moe_norm.weight"] = value(layer.pre_mlp_layernorm.weight)
        weights[prefix + "moe.router.weight"] = value(layer.mlp.router.weight)
    full = copy.deepcopy(config)
    full["salaad"]["projections"] = ["gate", "up", "down"]
    weights.update({g.name: g.weight().cpu().clone() for g in megatron_groups(chunks, full)})
    return weights


def prune_megatron_checkpoints(root, config, current_step):
    root = Path(root)
    entries = []
    for path in (root / "salaad").glob("iter_*"):
        if path.is_symlink() or not (path / "complete.json").is_file():
            continue
        metadata = checkpoint_metadata(path)
        if metadata["config_hash"] == fingerprint(config):
            entries.append((path, metadata))
    entries.sort(key=lambda item: item[1]["step"])
    latest = max(1, config["training"]["keep_latest_checkpoints"])
    keep = {p for p, _ in entries[-latest:]}
    scored = sorted(
        ((p, m) for p, m in entries if m["validation_nll"] is not None),
        key=lambda item: item[1]["validation_nll"],
    )
    keep.update(p for p, _ in scored[: config["training"]["keep_best_checkpoints"]])
    for path, metadata in entries:
        if path not in keep and metadata["step"] < current_step:
            native = root / path.name
            if native.is_dir() and not native.is_symlink():
                shutil.rmtree(native)
            shutil.rmtree(path)


def install_training_integration(
    training_module, config, data_identity, corpus=None, branch_from_vanilla=False
):
    """Install scoped entrypoint hooks; the upstream source tree stays untouched."""
    original_setup, original_save = (
        training_module.setup_model_and_optimizer,
        training_module.save_checkpoint,
    )
    original_train_step = training_module.train_step
    original_evaluate = training_module.evaluate
    original_print_evaluation = training_module.evaluate_and_print_results
    active = {}

    def setup(*args, **kwargs):
        chunks, optimizer, scheduler = original_setup(*args, **kwargs)
        native_args = training_module.get_args()
        validate_runtime(native_args, config)
        hook = MegatronOptimizerHook(
            optimizer, megatron_groups(chunks, config), config, step=native_args.iteration
        )
        active.update(hook=hook, chunks=chunks)
        if native_args.iteration:
            directory = Path(native_args.load) / "salaad" / f"iter_{native_args.iteration:07d}"
            error, shards = None, []
            try:
                metadata = checkpoint_metadata(directory)
                if (
                    (not branch_from_vanilla and metadata["config_hash"] != fingerprint(config))
                    or metadata["data_identity"] != data_identity
                    or metadata["world_size"] != world_size()
                ):
                    raise ValueError("Resume config/data/DP size differs from the saved run")
                if branch_from_vanilla:
                    source_config = load_torch(directory / "training.pt")["config"]
                    validate_vanilla_branch(source_config, config, native_args.iteration)
                elif hook.manager:
                    shards = [
                        load_torch(directory / f"rank_{i:05d}.pt")["salaad"]
                        for i in range(metadata["world_size"])
                    ]
            except Exception as exc:
                error = exc
            agree_or_raise(error, hook.device, "Megatron SALAAD resume")
            if branch_from_vanilla:
                hook.manager.initialize(native_args.iteration)
            elif hook.manager:
                hook.manager.load_shards(shards)
        elif branch_from_vanilla:
            raise ValueError("Megatron did not load the requested vanilla prefix checkpoint")
        return chunks, optimizer, scheduler

    def train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        hook = active["hook"]
        error = None
        if rank() == 0:
            try:
                with (Path(training_module.get_args().save) / "salaad_metrics.jsonl").open(
                    "a"
                ) as handle:
                    handle.write(json.dumps(hook.last_metrics, allow_nan=False) + "\n")
            except Exception as exc:
                error = exc
        agree_or_raise(error, hook.device, "Megatron SALAAD log")
        return result

    def evaluate(*args, **kwargs):
        if corpus is None or "hook" not in active:
            return original_evaluate(*args, **kwargs)
        # A fixed monitor set, independent of Megatron's advancing validation
        # iterator. Training still uses the upstream indexed GPTDataset reader.
        from .model import MoELanguageModel
        from .trainer import evaluate_model

        hook = active["hook"]
        weights = native_weights_from_megatron(active["chunks"], config)
        if config["training"]["task_precision"] == "bfloat16":
            weights = {name: tensor.bfloat16() for name, tensor in weights.items()}
        with torch.random.fork_rng(devices=[]):
            model = MoELanguageModel(config, initialize=False)
            model.load_state_dict(weights, strict=True)
        model = model.to(hook.device).eval()
        split = active.get("evaluation_split", "validation")
        full = active.get("full_evaluation", False)
        count = config["data"][
            "test_sequences"
            if split == "test"
            else "validation_sequences"
            if full
            else "monitor_validation_sequences"
        ]
        micro = config["training"]["micro_batch_sequences_per_rank"]
        raw = evaluate_model(model, corpus, split, count, micro, config, hook.device)
        evaluation = {"step": hook.step, "split": split, "full_split_evaluation": full, "raw": raw}
        if split == "validation" and not full:
            active["validation"] = evaluation
        if hook.manager and hook.manager.initialized and config["export"]["reconstructed_eval"]:
            with torch.no_grad():
                parameters = dict(model.named_parameters())
                for i, group in enumerate(hook.manager.groups):
                    value = (
                        hook.manager.states[group.name].reconstruction().contiguous()
                        if i % world_size() == rank()
                        else torch.empty_like(group.weight())
                    )
                    if dist.is_initialized():
                        dist.broadcast(value, src=i % world_size())
                    if config["training"]["task_precision"] == "bfloat16":
                        value = value.bfloat16()
                    parameters[group.name].copy_(value)
            evaluation["reconstructed"] = evaluate_model(
                model, corpus, split, count, micro, config, hook.device
            )
        error = None
        if rank() == 0:
            try:
                with (Path(training_module.get_args().save) / "salaad_validation.jsonl").open(
                    "a"
                ) as handle:
                    handle.write(json.dumps(evaluation, allow_nan=False) + "\n")
            except Exception as exc:
                error = exc
        agree_or_raise(error, hook.device, "Megatron fixed-set evaluation log")
        return {"lm loss": torch.tensor(raw["nll"], device=hook.device)}, None, False

    def print_evaluation(prefix, *args, **kwargs):
        # Prefixes are fixed in training.py at the required upstream commit.
        active["evaluation_split"] = "test" if " on test set" in prefix else "validation"
        active["full_evaluation"] = " on test set" in prefix or " on validation set" in prefix
        try:
            return original_print_evaluation(prefix, *args, **kwargs)
        finally:
            active.pop("evaluation_split", None)
            active.pop("full_evaluation", None)

    def save(iteration, chunks, optimizer, scheduler, *args, **kwargs):
        if "hook" not in active:
            return original_save(iteration, chunks, optimizer, scheduler, *args, **kwargs)
        hook = active["hook"]
        if iteration != hook.step or kwargs.get("non_persistent_ckpt", False):
            raise ValueError("Checkpoint iteration must match successful SALAAD steps")
        directory = Path(training_module.get_args().save) / "salaad" / f"iter_{iteration:07d}"
        error = None
        if rank() == 0:
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except Exception as exc:
                error = exc
        agree_or_raise(error, hook.device, "Megatron auxiliary checkpoint directory")
        error = None
        try:
            atomic_save(
                {"salaad": hook.manager.local_state_dict() if hook.manager else None},
                directory / f"rank_{rank():05d}.pt",
            )
            if rank() == 0:
                atomic_save(
                    {
                        "format": "salaad_moe.megatron_evaluation.v1",
                        "config": config,
                        "config_hash": fingerprint(config),
                        "step": iteration,
                        "model": native_weights_from_megatron(chunks, config),
                    },
                    directory / "training.pt",
                )
        except Exception as exc:
            error = exc
        agree_or_raise(error, hook.device, "Megatron auxiliary checkpoint write")
        result = original_save(iteration, chunks, optimizer, scheduler, *args, **kwargs)
        error = None
        if rank() == 0:
            try:
                validation = active.get("validation", {})
                nll = validation["raw"]["nll"] if validation.get("step") == iteration else None
                (directory / "complete.json").write_text(
                    json.dumps(
                        {
                            "format": "salaad_moe.checkpoint.v1",
                            "step": iteration,
                            "world_size": world_size(),
                            "config_hash": fingerprint(config),
                            "data_identity": data_identity,
                            "validation_nll": nll,
                            "resume_backend": "megatron_only",
                            "upstream_revision": MEGATRON_REVISION,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                prune_megatron_checkpoints(training_module.get_args().save, config, iteration)
            except Exception as exc:
                error = exc
        agree_or_raise(error, hook.device, "Megatron auxiliary checkpoint publication")
        return result

    training_module.setup_model_and_optimizer = setup
    training_module.train_step = train_step
    training_module.save_checkpoint = save
    training_module.evaluate = evaluate
    training_module.evaluate_and_print_results = print_evaluation
    return active
