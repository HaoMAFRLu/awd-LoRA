"""Rank-zero W&B logging with persistent run identity and unchanged training RNG."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import numbers
import os
from pathlib import Path
import uuid

from .checkpoint import restore_rng, rng_state
from .distributed import agree_or_raise, rank


@contextmanager
def preserve_training_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def flatten_metrics(record):
    result = {"optimizer_step": record["step"]}

    def visit(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items():
                # Per-expert raw counts remain in metrics.jsonl; log layer summaries to W&B.
                if key not in ("group", "assignment_counts"):
                    visit(item, f"{prefix}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                label = item.get("group", str(index)) if isinstance(item, dict) else str(index)
                visit(item, f"{prefix}/{label}")
        elif isinstance(value, numbers.Real):
            result[prefix] = value

    for key, value in record.items():
        if key == "salaad":
            # Use exactly two sections across all layers. Keep layer/projection/
            # expert identifiers after the only slash so they do not form sections.
            for name, metrics in value.items():
                _, layer, _, _, projection, expert = name.split(".")
                matrix = f"layer_{layer}_{projection}_{expert}"
                for metric in ("diff", "density", "effective_rank_ratio"):
                    result[f"salaad_structure/{matrix}_{metric}"] = metrics[metric]
                for metric in ("alpha", "beta"):
                    result[f"salaad_hyperparameters/{matrix}_{metric}"] = metrics[metric]
                # Rho is global and fixed; emit one metric shared by all layers.
                result["salaad_hyperparameters/rho"] = metrics["rho"]
        elif key != "step":
            if isinstance(value, dict):
                prefix = key
            else:
                section = "train" if key in (
                    "lm_nll", "load_balancing_loss", "router_z_loss", "learning_rate"
                ) else "train_stats"
                prefix = f"{section}/{key}"
            visit(value, prefix)
    return result


class WandbTracker:
    def __init__(self, config, output, resume=False, step=0):
        self.run = None
        self.path = Path(output).resolve() / "wandb_run.json"
        project = config["wandb_project"]
        entity = config.get("wandb_entity")
        previous = json.loads(self.path.read_text()) if resume and self.path.is_file() else {}
        if previous and (previous["project"] != project or previous["entity"] != entity):
            raise ValueError("The resumed W&B project/entity differs from the original run")
        # Replaying an older checkpoint needs a separate history, not dropped W&B steps.
        continuing = bool(previous) and step >= previous.get("last_logged_step", 0)
        self.state = {
            "id": previous["id"] if continuing else uuid.uuid4().hex[:8],
            "project": project,
            "entity": entity,
            "last_logged_step": step,
            "parent_run_id": previous.get("id") if previous and not continuing else None,
        }
        try:
            with preserve_training_rng():
                import wandb

                mode = os.environ.get("WANDB_MODE") or config.get("wandb_mode", "online")
                kwargs = {
                    "project": project,
                    "entity": entity,
                    "id": self.state["id"],
                    "name": config.get("wandb_name")
                    or f"{config['experiment']}-{self.path.parent.name}",
                    "config": copy.deepcopy(config),
                    "dir": str(self.path.parent),
                    "mode": mode,
                    "tags": [
                        "native-dp",
                        "salaad" if config["salaad"]["enabled"] else "vanilla",
                        "synthetic-smoke"
                        if config["data"].get("synthetic_smoke")
                        else "pretraining",
                    ],
                }
                if continuing and mode == "online":
                    kwargs["resume"] = "allow"
                self.run = wandb.init(**kwargs)
                if self.run is None:
                    raise RuntimeError("wandb.init did not return a run")
                if continuing and mode == "online" and self.run.step > step + 1:
                    raise ValueError(
                        "W&B history is ahead of this checkpoint; use a new output directory"
                    )
                self.run.define_metric("optimizer_step")
                self.run.define_metric("*", step_metric="optimizer_step")
                self.run.summary.update(
                    {
                        "backend": "native_pytorch_dp",
                        "salaad_enabled": config["salaad"]["enabled"],
                        "resume_optimizer_step": step,
                        "parent_run_id": self.state["parent_run_id"],
                    }
                )
                self.state["mode"] = mode
                self.state["url"] = self.run.get_url() if mode == "online" else None
            self._save_state()
            print(f"W&B: project={project}, run={self.state['id']}, mode={mode}", flush=True)
        except BaseException:
            if self.run is not None:
                try:
                    self.finish(exit_code=1)
                except Exception:
                    pass
            raise

    def _save_state(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.path)

    def log(self, record):
        with preserve_training_rng():
            self.run.log(flatten_metrics(record), step=record["step"])
        self.state["last_logged_step"] = record["step"]
        self._save_state()

    def finish(self, exit_code=0):
        if self.run is not None:
            run, self.run = self.run, None
            with preserve_training_rng():
                run.finish(exit_code=exit_code)


@contextmanager
def tracking_run(config, output, device, resume=False, step=0):
    if not config.get("is_wandb", False):
        yield None
        return
    tracker, error = None, None
    if rank() == 0:
        try:
            tracker = WandbTracker(config, output, resume=resume, step=step)
        except Exception as exc:
            error = exc
    agree_or_raise(error, device, "W&B initialization")
    try:
        yield tracker
    except BaseException:
        if tracker is not None:
            try:
                tracker.finish(exit_code=1)
            except Exception:
                pass  # Preserve the original, already coordinated training error.
        raise
    else:
        error = None
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as exc:
                error = exc
        agree_or_raise(error, device, "W&B finish")
