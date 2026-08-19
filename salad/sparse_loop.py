"""ADMM updates for loop-specific sparse residuals."""

from typing import Any, Dict, Mapping, Union

import torch
from torch import nn

from models.sparse_loop import SparseLoopLinear
from salad.adaptive_param import PARAM
from salad.consensus import soft_threshold


_CONTROLLER_FLOAT_FIELDS = (
    "target_rate",
    "value",
    "rate_decay",
    "drate",
    "dvalue",
    "pre_rate",
)
_CONTROLLER_INT_FIELDS = ("nr_updates", "start_epoch")


def _make_beta_controller(
    config: Mapping[str, Any], target_rate: float
) -> PARAM:
    controller_config = config.get("beta_dict")
    if not isinstance(controller_config, Mapping):
        raise ValueError("specific_sparsity.beta_dict must be a dictionary")
    controller_config = dict(controller_config)
    controller_config["target_rate"] = target_rate
    if controller_config.get("mode") != "adaptive":
        raise ValueError(
            "specific_sparsity.beta_dict.mode must be 'adaptive' for I-control"
        )
    for field in ("init", "rate_decay", "drate"):
        value = float(controller_config.get(field, 0.0))
        if value < 0.0:
            raise ValueError(
                f"specific_sparsity.beta_dict.{field} must be non-negative, "
                f"got {value}"
            )

    controller = PARAM(controller_config)
    start_epoch = controller_config.get("start_epoch", controller.start_epoch)
    if (
        not isinstance(start_epoch, int)
        or isinstance(start_epoch, bool)
        or start_epoch < 0
    ):
        raise ValueError(
            "specific_sparsity.beta_dict.start_epoch must be a non-negative "
            "integer"
        )
    controller.start_epoch = start_epoch
    return controller


def _controller_state(controller: PARAM) -> Dict[str, Any]:
    state = {"mode": controller.mode}
    for field in _CONTROLLER_FLOAT_FIELDS:
        state[field] = float(getattr(controller, field))
    for field in _CONTROLLER_INT_FIELDS:
        state[field] = int(getattr(controller, field))
    return state


def _load_controller_state(controller: PARAM, state: Mapping[str, Any]) -> None:
    if state.get("mode") != "adaptive":
        raise ValueError("saved beta controller mode must be 'adaptive'")
    controller.mode = "adaptive"
    for field in _CONTROLLER_FLOAT_FIELDS:
        if field in state:
            setattr(controller, field, float(state[field]))
    for field in _CONTROLLER_INT_FIELDS:
        if field in state:
            setattr(controller, field, int(state[field]))


def _mean_controller_value(controllers, field: str) -> float:
    return sum(float(getattr(controller, field)) for controller in controllers) / len(
        controllers
    )


class SparseLoopADMM:
    """Constrain each dense residual ``R_i`` to a sparse matrix ``S_i``.

    The recurrent backbone weight ``X`` belongs to ``SparseLoopLinear`` and is
    optimized only by the task loss.  This solver has no consensus variable
    and never averages weights across loops.  With scaled dual ``U_i``, the
    augmented term is ``rho / 2 * mean((R_i - S_i + U_i)^2)``. As in SALAAD,
    one beta I-controller per loop adjusts the soft-threshold strength toward
    a globally configured target non-zero rate.
    """

    def __init__(
        self,
        name: str,
        module: SparseLoopLinear,
        config: Mapping[str, Any],
    ) -> None:
        if not isinstance(module, SparseLoopLinear):
            raise TypeError(
                "module must be SparseLoopLinear, "
                f"got {type(module).__name__}"
            )
        self.name = name
        self.module = module
        self.rho = float(config.get("rho", 0.0))
        if self.rho <= 0.0:
            raise ValueError(f"rho must be positive, got {self.rho}")

        rate_sparsity = config.get("rate_sparsity")
        if isinstance(rate_sparsity, bool) or not isinstance(
            rate_sparsity, (int, float)
        ):
            raise TypeError(
                "specific_sparsity.rate_sparsity must be a real number"
            )
        self.rate_sparsity = float(rate_sparsity)
        if not 0.0 <= self.rate_sparsity <= 1.0:
            raise ValueError(
                "specific_sparsity.rate_sparsity must be in [0, 1], "
                f"got {self.rate_sparsity}"
            )

        initial = module.specific_weight.detach().float()
        self.sparse = torch.zeros_like(initial)
        self.scaled_dual = torch.zeros_like(initial)
        self.beta_controllers = [
            _make_beta_controller(config, self.rate_sparsity)
            for _ in range(self.num_loops)
        ]

    @property
    def specific_weight(self) -> torch.Tensor:
        return self.module.specific_weight

    @property
    def num_loops(self) -> int:
        return self.specific_weight.shape[0]

    def residual(self, *, detach: bool = True) -> torch.Tensor:
        specific = (
            self.specific_weight.detach()
            if detach
            else self.specific_weight
        )
        return specific.float() - self.sparse

    def penalty(self) -> torch.Tensor:
        shifted_residual = self.residual(detach=False) + self.scaled_dual
        return 0.5 * self.rho * shifted_residual.square().mean()

    def _update_controller(self, controller: PARAM, current_rate: float) -> None:
        previous_value = float(controller.value)
        controller.update(float(current_rate), self.rho)
        controller.value = max(0.0, float(controller.value))
        controller.dvalue = controller.value - previous_value

    @torch.no_grad()
    def step(self) -> None:
        sparse_input = self.specific_weight.detach().float() + self.scaled_dual
        elements_per_loop = self.sparse[0].numel()
        for loop_index, controller in enumerate(self.beta_controllers):
            self.sparse[loop_index].copy_(
                soft_threshold(
                    sparse_input[loop_index],
                    controller.value / self.rho,
                )
            )
            nonzero_rate = (
                int(torch.count_nonzero(self.sparse[loop_index]).item())
                / elements_per_loop
            )
            self._update_controller(controller, nonzero_rate)
        self.scaled_dual.add_(self.residual())

    @torch.no_grad()
    def stats(self) -> Dict[str, Union[float, int]]:
        residual = self.residual()
        residual_norm = torch.linalg.vector_norm(residual)
        effective = (
            self.module.weight.detach().float().unsqueeze(0)
            + self.specific_weight.detach().float()
        )
        effective_norm = torch.linalg.vector_norm(effective).clamp_min(1.0e-12)
        return {
            "relative_residual": float((residual_norm / effective_norm).item()),
            "nonzero": int(torch.count_nonzero(self.sparse).item()),
            "total_elements": self.sparse.numel(),
            "beta": _mean_controller_value(self.beta_controllers, "value"),
            "dbeta": _mean_controller_value(self.beta_controllers, "dvalue"),
            "rate_decay_beta": _mean_controller_value(
                self.beta_controllers, "rate_decay"
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_loops": self.num_loops,
            "rho": self.rho,
            "rate_sparsity": self.rate_sparsity,
            "beta_controllers": [
                _controller_state(controller)
                for controller in self.beta_controllers
            ],
            "sparse": self.sparse.detach().cpu(),
            "scaled_dual": self.scaled_dual.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("name") != self.name:
            raise ValueError(
                f"solver name mismatch: expected {self.name!r}, "
                f"got {state.get('name')!r}"
            )
        if int(state.get("num_loops", -1)) != self.num_loops:
            raise ValueError(
                "num_loops mismatch: "
                f"expected {self.num_loops}, got {state.get('num_loops')!r}"
            )
        if "rho" in state:
            self.rho = float(state["rho"])
        if "rate_sparsity" in state:
            self.rate_sparsity = float(state["rate_sparsity"])
        for name in ("sparse", "scaled_dual"):
            source = state.get(name)
            target = getattr(self, name)
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"saved {name} must be a tensor")
            if source.shape != target.shape:
                raise ValueError(
                    f"{name} shape mismatch: {tuple(source.shape)} != "
                    f"{tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))

        saved_controllers = state.get("beta_controllers")
        if not isinstance(saved_controllers, list):
            raise TypeError("saved beta_controllers must be a list")
        if len(saved_controllers) != self.num_loops:
            raise ValueError(
                "beta_controllers must contain one state per logical loop"
            )
        for controller, controller_state in zip(
            self.beta_controllers, saved_controllers
        ):
            _load_controller_state(controller, controller_state)


@torch.no_grad()
def apply_sparse_residuals(
    model: nn.Module,
    states: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> None:
    """Materialize saved ``S_i`` into the model's forward residuals."""
    model = model.module if hasattr(model, "module") else model
    modules = dict(model.named_modules())
    expected = {
        name
        for name, module in modules.items()
        if isinstance(module, SparseLoopLinear)
    }
    provided = set(states)
    if strict and provided != expected:
        raise KeyError(
            "specific sparsity state mismatch; "
            f"missing={sorted(expected - provided)}, "
            f"unexpected={sorted(provided - expected)}"
        )

    for name, state in states.items():
        module = modules.get(name)
        if not isinstance(module, SparseLoopLinear):
            if strict:
                raise KeyError(f"model has no SparseLoopLinear named {name!r}")
            continue
        sparse = state.get("sparse")
        if not isinstance(sparse, torch.Tensor):
            raise TypeError(f"saved sparse residual for {name!r} must be a tensor")
        if sparse.shape != module.specific_weight.shape:
            raise ValueError(
                f"{name} shape mismatch: {tuple(sparse.shape)} != "
                f"{tuple(module.specific_weight.shape)}"
            )
        module.specific_weight.copy_(
            sparse.to(
                device=module.specific_weight.device,
                dtype=module.specific_weight.dtype,
            )
        )
