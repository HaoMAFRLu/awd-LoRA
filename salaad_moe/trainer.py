"""MoE task trainer with SALAAD as a separate framework for structural constraints.

Start with the update order in Trainer.train_step(), then read _backward_task()
and solver.py. Forward passes always use the full weights X; shared/L/S/dual
are auxiliary states outside autograd.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from .config import fingerprint, learning_rate, parameter_counts, validate_config
from .data import GlobalBatchReader
from .distributed import agree_or_raise, all_finite, average_gradients, rank, world_size
from .groups import native_groups
from .model import MoELanguageModel
from .solver import ConsensusManager
from .tracking import tracking_run


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_context(config, device):
    # Autocast lowers compute precision for eligible operations; model
    # parameters, gradients, and Adam states remain FP32.
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=config["training"]["task_precision"] == "bfloat16",
    )


@torch.no_grad()
def evaluate_model(model, corpus, split, sequences, batch_size, config, device, distributed=True):
    """Compute token-mean NLL, excluding router auxiliary losses and SALAAD penalties."""
    model.eval()
    count = min(sequences, corpus.lengths[split]) if sequences else corpus.lengths[split]
    r, w = (rank(), world_size()) if distributed else (0, 1)
    indices = list(range(r, count, w))
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for start in range(0, len(indices), batch_size):
        inputs, labels = corpus.batch(split, indices[start : start + batch_size], device)
        with compute_context(config, device):
            output = model(inputs, labels)
        total[0] += output.lm_loss.double() * labels.numel()
        total[1] += labels.numel()
    if distributed and dist.is_initialized():
        dist.all_reduce(total)
    nll = (total[0] / total[1]).item()
    if not math.isfinite(nll):
        raise FloatingPointError("Nonfinite validation NLL")
    return {
        "nll": nll,
        "perplexity": math.exp(nll),
        "prediction_tokens": int(total[1].item()),
        "sequences": count,
    }


class Trainer:
    """Use one MoE training loop; salaad.enabled toggles structural constraints."""

    def __init__(self, config, corpus, device):
        validate_config(config, world_size())
        self.config = copy.deepcopy(config)
        self.corpus = corpus
        self.device = torch.device(device)
        for split, key in (("validation", "validation_sequences"), ("test", "test_sequences")):
            if corpus.lengths[split] < config["data"][key]:
                raise ValueError(
                    f"The corpus has too few {split} sequences for this training configuration"
                )
        seed_everything(config["seed"])
        # Each DP rank builds the full model; broadcast initial parameters
        # so all replicas start with identical weights.
        self.model = MoELanguageModel(config).to(self.device)
        if dist.is_initialized():
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)
        self.optimizer = self._build_optimizer()
        self.reader = GlobalBatchReader(corpus, config)
        self.step = 0  # Successfully completed optimizer steps, not micro-batches.

        # One SALAAD group contains all experts for one projection in one layer.
        # The manager handles shared/L/S/dual, constraint gradients, and periodic
        # alternating direction method of multipliers (ADMM) updates.
        self.manager = (
            ConsensusManager(native_groups(self.model, config["salaad"]["projections"]), config)
            if config["salaad"]["enabled"]
            else None
        )
        if self.manager and config["salaad"]["state_initialization_step"] == 0:
            self.manager.initialize(0)

    def _build_optimizer(self):
        t = self.config["training"]
        # ndim >= 2: linear/embedding matrices and [expert, out, in] expert weights.
        # ndim < 2: vectors such as normalization scales. Both groups are trained,
        # but weight decay is applied only to the first group.
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.ndim < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": t["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=t["learning_rate"],
            betas=(t["beta1"], t["beta2"]),
            eps=t["epsilon"],
        )

    def train_step(self):
        """Update one global batch: task gradients -> DP mean -> constraint -> AdamW -> ADMM."""
        t = self.config["training"]
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 1. Accumulate task gradients on each rank: LM + balance + z-loss.
        # The structural penalty is added later.
        losses, counts, entropy = self._backward_task()

        # 2. Average gradients across DP ranks once after accumulation.
        # No DDP wrapper is used, so gradient synchronization is explicit.
        average_gradients(self.model.parameters())
        task_norm = (
            torch.stack([p.grad.square().sum() for p in self.model.parameters()])
            .sum()
            .sqrt()
            .item()
        )
        # 3. Add rho * (X - Q) once for all experts, with Q = shared + L + S - U.
        # Add it after DP averaging, never separately for each micro-batch.
        constraint_norm = self.manager.inject_gradients() if self.manager else 0.0
        error = (
            None
            if all_finite([p.grad for p in self.model.parameters()])
            else FloatingPointError("Nonfinite combined gradient")
        )
        agree_or_raise(error, self.device, "Gradient injection")

        # 4. Clip the combined task + structural gradient, then run AdamW
        # using the learning rate from the full training schedule.
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), t["gradient_clip_global_norm"], error_if_nonfinite=True
        ).item()
        lr = learning_rate(self.config, self.step + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        error = (
            None
            if all_finite(self.model.parameters())
            else FloatingPointError(
                "Nonfinite parameter after AdamW; abort and resume the last complete checkpoint"
            )
        )
        agree_or_raise(error, self.device, "Optimizer result")

        # 5. Initialize auxiliary states or run the scheduled shared -> L -> S -> U
        # update, then broadcast the new Q. Reconstructed weights are not copied
        # back into the model parameters X.
        next_step = self.step + 1
        changed = (
            self.manager.after_step(next_step, final=next_step == t["total_optimizer_steps"])
            if self.manager
            else False
        )
        # 6. Advance the step and global data cursor only after all updates succeed.
        # A failed structural update requires aborting and resuming from a checkpoint.
        self.step = next_step
        self.reader.advance()

        # 7. Aggregate metrics: average loss/entropy across DP ranks and sum
        # routing assignment counts across ranks.
        if dist.is_initialized():
            dist.all_reduce(losses)
            losses /= world_size()
            dist.all_reduce(counts)
            dist.all_reduce(entropy)
            entropy /= world_size()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        router = {
            "assignment_counts": counts.cpu().tolist(),
            "assignment_cv": (counts.std(-1, unbiased=False) / counts.mean(-1).clamp_min(1))
            .cpu()
            .tolist(),
            "mean_token_entropy": entropy.cpu().tolist(),
            "effective_experts_from_entropy": entropy.exp().cpu().tolist(),
        }
        record = {
            "step": self.step,
            "consumed_sequences": self.reader.cursor,
            "prediction_tokens": self.reader.cursor * self.config["data"]["seq_length"],
            "learning_rate": lr,
            "lm_nll": losses[0].item(),
            "load_balancing_loss": losses[1].item(),
            "router_z_loss": losses[2].item(),
            "task_gradient_norm": task_norm,
            "constraint_gradient_norm": constraint_norm,
            "constraint_task_gradient_ratio": constraint_norm / max(task_norm, 1e-30),
            "combined_gradient_norm_before_clip": norm,
            "step_seconds": time.perf_counter() - started,
            "router": router,
        }
        if changed:
            record["salaad"] = self.manager.metrics()
        return record

    def _backward_task(self):
        """Accumulate this rank's micro-batches from one global batch and return metrics."""
        accumulation = self.config["training"]["accumulation_steps"]
        layers = self.config["model"]["num_layers"]
        experts = self.config["model"]["num_experts"]
        losses = torch.zeros(3, device=self.device)  # LM NLL, balance loss, z-loss.
        counts = torch.zeros(layers, experts, device=self.device)
        entropy = torch.zeros(layers, device=self.device)
        error = None
        try:
            for microstep in range(accumulation):
                inputs, labels = self.reader.batch(microstep, rank(), world_size(), self.device)
                with compute_context(self.config, self.device):
                    output = self.model(inputs, labels)
                    # Divide each micro-batch loss by the accumulation count
                    # so the final gradient represents the batch mean.
                    loss = output.task_loss(self.config) / accumulation
                loss.backward()
                losses += torch.stack((
                    output.lm_loss.detach(),
                    output.balance_loss.detach(),
                    output.z_loss.detach(),
                )) / accumulation
                for i, detail in enumerate(output.router_stats):
                    counts[i] += detail["counts"]
                    entropy[i] += detail["entropy"] / accumulation
            if not all_finite([losses, *(p.grad for p in self.model.parameters())]):
                raise FloatingPointError("Nonfinite task loss/gradient")
        except Exception as exc:
            error = exc
        # Forward/backward passes contain no collectives. If any rank fails,
        # all ranks exit together before gradient synchronization starts.
        agree_or_raise(error, self.device, "Task step failed before optimizer update")
        return losses, counts, entropy

    def evaluate(self, reconstructed=False):
        # Evaluate a separate model copy so reconstruction leaves the training
        # weights and Adam states intact for the next step.
        evaluation_model = copy.deepcopy(self.model).eval()
        if reconstructed:
            if not self.manager or not self.manager.initialized:
                raise ValueError("Reconstruction requires initialized SALAAD state")
            with torch.no_grad():
                eval_groups = {
                    g.name: g
                    for g in native_groups(evaluation_model, self.config["salaad"]["projections"])
                }
                for i, group in enumerate(self.manager.groups):
                    value = (
                        self.manager.states[group.name].reconstruction().contiguous()
                        if i % world_size() == rank()
                        else torch.empty_like(group.weight())
                    )
                    if dist.is_initialized():
                        dist.broadcast(value, src=i % world_size())
                    eval_groups[group.name].parameter.copy_(value)
        return evaluate_model(
            evaluation_model,
            self.corpus,
            "validation",
            self.config["data"]["monitor_validation_sequences"],
            self.config["training"]["micro_batch_sequences_per_rank"],
            self.config,
            self.device,
        )

    def run(self, output, resume=None, stop_after=None, branch_from=None):
        """Orchestrate metadata, state restoration, training, evaluation, saving, and logging."""
        output = Path(output)
        self._prepare_output(output, resume, branch_from)
        if resume:
            # Restore the model, Adam, SALAAD states, data cursor, and RNG per rank.
            load_checkpoint(self, resume)
        elif branch_from:
            # Reuse the same vanilla training prefix before creating SALAAD
            # states, allowing fair paired experiments.
            load_checkpoint(self, branch_from, branch_from_vanilla=True)
        total = self.config["training"]["total_optimizer_steps"]
        end = total if stop_after is None else min(stop_after, total)
        if end <= self.step:
            raise ValueError("Requested stop step must exceed the resumed step")
        # stop_after only sets the pause point for this run; the full learning
        # rate and controller schedules remain unchanged.
        with tracking_run(
            self.config, output, self.device, resume=bool(resume), step=self.step
        ) as tracker:
            return self._run_steps(output, end, tracker)

    def _prepare_output(self, output, resume, branch_from):
        """Write metadata on rank 0 and propagate failures through a collective check."""
        error = None
        if rank() == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                config_path = output / "config.resolved.json"
                if config_path.exists():
                    if not resume:
                        raise FileExistsError(
                            "Output already contains a run; specify --resume or use a new directory"
                        )
                    if fingerprint(json.loads(config_path.read_text())) != fingerprint(self.config):
                        raise ValueError("Existing output belongs to a different configuration")
                config_path.write_text(json.dumps(self.config, indent=2) + "\n")
                metadata_path = output / "run_metadata.json"
                previous_metadata = (
                    json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "backend": "native_pytorch_sequential_experts",
                            "torch_version": str(torch.__version__),
                            "cuda_version": torch.version.cuda,
                            "device": str(self.device),
                            "world_size": world_size(),
                            "corpus_identity": self.corpus.identity,
                            "data_manifest": str(self.corpus.path),
                            "corpus_provenance": self.corpus.manifest["provenance"],
                            "parameter_counts": parameter_counts(self.config),
                            "resume_from": str(resume) if resume else None,
                            "initial_branch_from": previous_metadata.get("initial_branch_from")
                            or (str(branch_from) if branch_from else None),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            except Exception as exc:
                error = exc
        agree_or_raise(error, self.device, "Run directory")

    def _run_steps(self, output, end, tracker):
        """Train, evaluate, and save on all ranks; write JSONL and W&B logs on rank 0."""
        checkpoint = None
        while self.step < end:
            record = self.train_step()
            t = self.config["training"]
            validation_nll = None
            if self.step % t["validation_interval_steps"] == 0 or self.step == end:
                record["validation_raw"] = self.evaluate()
                validation_nll = record["validation_raw"]["nll"]
                if (
                    self.manager
                    and self.manager.initialized
                    and self.config["export"]["reconstructed_eval"]
                ):
                    record["validation_reconstructed"] = self.evaluate(reconstructed=True)
                    record["reconstruction_nll_gap"] = (
                        record["validation_reconstructed"]["nll"] - validation_nll
                    )
            if self.step % t["checkpoint_interval_steps"] == 0 or self.step == end:
                checkpoint = save_checkpoint(self, output / "checkpoints", validation_nll)
                error = None
                if rank() == 0:
                    try:
                        prune_checkpoints(output / "checkpoints", self.config)
                    except Exception as exc:
                        error = exc
                agree_or_raise(error, self.device, "Checkpoint retention")
                record["checkpoint"] = str(checkpoint)
            error = None
            if rank() == 0:
                try:
                    with (output / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    if tracker is not None:
                        tracker.log(record)
                    print(
                        json.dumps(
                            {k: v for k, v in record.items() if k not in ("router", "salaad")}
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    error = exc
            agree_or_raise(error, self.device, "Training log")
        return checkpoint
