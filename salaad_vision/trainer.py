"""Shared training loop for downstream vision tasks."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

_ROOT = Path(__file__).resolve().parents[1]


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config requires a {name!r} mapping")
    return section


def _optimizer(parameters: Iterable[nn.Parameter], config: Mapping[str, Any]):
    params = list(parameters)
    if not params:
        raise ValueError("the vision task has no trainable parameters")

    name = config.get("name")
    lr = config.get("lr")
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError("optimizer.lr must be positive")
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(lr),
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"unsupported optimizer: {name!r}")


def _scheduler(optimizer, config: Mapping[str, Any], epochs: int):
    name = config.get("name", "none")
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"unsupported scheduler: {name!r}")


class VisionTrainer:
    """Train a task module on top of a configurable vision backbone."""

    def __init__(
        self,
        model: nn.Module,
        task: nn.Module,
        train_data,
        validation_data,
        config: Mapping[str, Any],
        device: torch.device,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.task = task.to(device)
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if self.distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("world_size > 1 requires an initialized process group")
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise ValueError("rank/world_size do not match the process group")

        training = _section(config, "training")
        self.epochs = training.get("epochs")
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("training.epochs must be a positive integer")
        self.max_train_steps = self._steps(
            training.get("max_steps_per_epoch"),
            "training.max_steps_per_epoch",
        )
        validation = config.get("validation", {})
        if not isinstance(validation, Mapping):
            raise ValueError("config 'validation' must be a mapping")
        self.max_validation_steps = self._steps(
            validation.get("max_steps"),
            "validation.max_steps",
        )

        precision = training.get("precision", "float32")
        if precision not in {"float32", "bfloat16"}:
            raise ValueError("training.precision must be 'float32' or 'bfloat16'")
        self.use_autocast = precision == "bfloat16"
        self.model_frozen = all(
            not parameter.requires_grad for parameter in self.model.parameters()
        )
        if self.distributed and not self.model_frozen:
            raise ValueError(
                "distributed linear probing requires model.freeze=true; "
                "only the task head is wrapped in DDP"
            )
        if self.distributed:
            device_ids = [device.index] if device.type == "cuda" else None
            self.task_model = DDP(
                self.task,
                device_ids=device_ids,
                broadcast_buffers=False,
            )
        else:
            self.task_model = self.task

        trainable = [
            parameter
            for module in (self.model, self.task)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = _optimizer(trainable, _section(config, "optimizer"))
        scheduler_config = config.get("scheduler", {"name": "none"})
        if not isinstance(scheduler_config, Mapping):
            raise ValueError("config 'scheduler' must be a mapping")
        self.scheduler = _scheduler(
            self.optimizer,
            scheduler_config,
            self.epochs,
        )
        self.wandb_run = self._init_wandb()

    def _init_wandb(self):
        wandb_config = self.config.get("wandb", {})
        if not isinstance(wandb_config, Mapping):
            raise ValueError("config 'wandb' must be a mapping")
        enabled = wandb_config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("wandb.enabled must be true or false")
        if not enabled or self.rank != 0:
            return None

        project = wandb_config.get("project", "SALAAD_VISION_DOWNSTREAM")
        if not isinstance(project, str) or not project:
            raise ValueError("wandb.project must be a non-empty string")

        init_kwargs: Dict[str, Any] = {
            "project": project,
            "config": dict(self.config),
            "name": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        for name in ("entity", "group", "job_type", "mode"):
            value = wandb_config.get(name)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"wandb.{name} must be a non-empty string")
            init_kwargs[name] = value

        tags = wandb_config.get("tags")
        if tags is not None:
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) and tag for tag in tags
            ):
                raise ValueError("wandb.tags must be a list of non-empty strings")
            init_kwargs["tags"] = tags

        import wandb

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init did not return a run")
        print(
            f"W&B run    : {run.name} (id={run.id}, project={project})",
            flush=True,
        )
        return run

    @staticmethod
    def _steps(value: Any, name: str) -> Optional[int]:
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer or null")
        return value

    @staticmethod
    def _set_epoch(loader, epoch: int) -> None:
        dataset = getattr(loader, "dataset", None)
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _batch(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("vision batches must be mappings")
        images = batch.get("pixel_values")
        labels = batch.get("labels")
        if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("vision batches require tensor pixel_values and labels")
        return (
            images.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
        )

    def _forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_autocast,
        ):
            if self.model_frozen:
                with torch.no_grad():
                    features = self.model(images)
            else:
                features = self.model(images)
            logits = self.task_model(features)
            loss = self.task.loss(logits, labels)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError(f"task produced an invalid loss: {loss}")
        return logits, loss

    @staticmethod
    def _add_stats(total: Dict[str, Tensor], batch: Mapping[str, Tensor]) -> None:
        for name, value in batch.items():
            value = value.detach()
            total[name] = value.clone() if name not in total else total[name] + value

    def _reduce(
        self,
        total_loss: float,
        total_weight: int,
        total_stats: Dict[str, Tensor],
        steps: int,
    ) -> tuple[float, int, Dict[str, Tensor], int]:
        if not self.distributed:
            if steps == 0:
                raise RuntimeError("dataloader yielded no batches")
            return total_loss, total_weight, total_stats, steps

        minimum_steps = torch.tensor(steps, dtype=torch.int64, device=self.device)
        dist.all_reduce(minimum_steps, op=dist.ReduceOp.MIN)
        if minimum_steps.item() == 0:
            raise RuntimeError("at least one rank received no batches")

        loss_and_weight = torch.tensor(
            [total_loss, float(total_weight)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(loss_and_weight, op=dist.ReduceOp.SUM)
        total_loss = loss_and_weight[0].item()
        total_weight = int(loss_and_weight[1].item())

        for name in sorted(total_stats):
            dist.all_reduce(total_stats[name], op=dist.ReduceOp.SUM)

        maximum_steps = torch.tensor(steps, dtype=torch.int64, device=self.device)
        dist.all_reduce(maximum_steps, op=dist.ReduceOp.MAX)
        return total_loss, total_weight, total_stats, int(maximum_steps.item())

    def _epoch(self, loader, *, train: bool, max_steps: Optional[int]) -> Dict[str, float]:
        self.model.train(train and not self.model_frozen)
        self.task_model.train(train)
        total_loss = 0.0
        total_weight = 0
        total_stats: Dict[str, Tensor] = {}
        steps = 0

        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images, labels = self._batch(batch)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                logits, loss = self._forward(images, labels)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits, loss = self._forward(images, labels)

            weight = self.task.batch_weight(labels)
            if weight <= 0:
                raise RuntimeError("task returned a non-positive batch weight")
            total_loss += loss.detach().item() * weight
            total_weight += weight
            self._add_stats(total_stats, self.task.batch_stats(logits, labels))
            steps += 1

        total_loss, total_weight, total_stats, steps = self._reduce(
            total_loss,
            total_weight,
            total_stats,
            steps,
        )

        metrics = {"loss": total_loss / total_weight}
        metrics.update(self.task.summarize(total_stats))
        metrics["steps"] = float(steps)
        return metrics

    def _save(
        self,
        epoch: int,
        history: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if self.rank != 0:
            return None
        output = self.config.get("output", {})
        if not isinstance(output, Mapping):
            raise ValueError("config 'output' must be a mapping")
        if not output.get("save", False):
            return None

        output_dir = Path(output.get("dir", "data/salaad_vision/downstream"))
        output_dir = output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = _ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "epoch": epoch,
            "task": self.task.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": history,
            "config": dict(self.config),
        }
        if not self.model_frozen:
            state["model"] = self.model.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        checkpoint = output_dir / "checkpoint.pth"
        temporary = output_dir / "checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        return checkpoint

    @staticmethod
    def _format_metric(name: str, value: float) -> str:
        if name == "steps":
            return str(int(value))
        if name == "loss":
            return f"{value:.6f}"
        return f"{value:.2f}"

    def _print_epoch(
        self,
        result: Mapping[str, Any],
        checkpoint: Optional[Path],
    ) -> None:
        train_metrics = result["train"]
        validation_metrics = result["validation"]
        preferred = ["loss", "top1", "top5"]
        available = set(train_metrics) | set(validation_metrics)
        metric_names = [name for name in preferred if name in available]
        metric_names.extend(
            sorted(available - set(metric_names) - {"steps"})
        )
        if "steps" in available:
            metric_names.append("steps")

        labels = {
            "loss": "Loss",
            "top1": "Top-1 (%)",
            "top5": "Top-5 (%)",
            "steps": "Steps",
        }
        widths = {
            name: max(12, len(labels.get(name, name.replace("_", " ").title())))
            for name in metric_names
        }

        def row(split: str, metrics: Mapping[str, float]) -> str:
            values = [f"{split:<12}"]
            for name in metric_names:
                value = metrics.get(name)
                text = "-" if value is None else self._format_metric(name, value)
                values.append(f"{text:>{widths[name]}}")
            return "  ".join(values)

        header_values = [f"{'Split':<12}"]
        for name in metric_names:
            label = labels.get(name, name.replace("_", " ").title())
            header_values.append(f"{label:>{widths[name]}}")
        header = "  ".join(header_values)
        line = "-" * max(72, len(header))

        print(f"\n{line}")
        print(
            f"Epoch {result['epoch']}/{self.epochs}"
            f"  |  learning rate {result['lr']:.6g}"
        )
        print(line)
        print(header)
        print(line)
        print(row("Train", train_metrics))
        print(row("Validation", validation_metrics))
        if checkpoint is not None:
            print(line)
            print(f"Checkpoint  : {checkpoint}")
        print(line, flush=True)

    def _log_wandb(self, result: Mapping[str, Any]) -> None:
        if self.wandb_run is None:
            return
        payload = {
            "epoch": int(result["epoch"]),
            "train/lr": float(result["lr"]),
        }
        for split in ("train", "validation"):
            for name, value in result[split].items():
                payload[f"{split}/{name}"] = float(value)
        self.wandb_run.log(payload, step=int(result["epoch"]))

    def fit(self) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        try:
            for epoch in range(self.epochs):
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self._set_epoch(self.train_data, epoch)
                train_metrics = self._epoch(
                    self.train_data,
                    train=True,
                    max_steps=self.max_train_steps,
                )
                validation_metrics = self._epoch(
                    self.validation_data,
                    train=False,
                    max_steps=self.max_validation_steps,
                )
                if self.scheduler is not None:
                    self.scheduler.step()

                result = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "lr": learning_rate,
                }
                history.append(result)
                checkpoint = self._save(epoch + 1, history)
                if self.rank == 0:
                    self._print_epoch(result, checkpoint)
                    self._log_wandb(result)
                if self.distributed:
                    dist.barrier()
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()
        return history
