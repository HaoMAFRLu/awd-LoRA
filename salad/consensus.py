"""Consensus ADMM updates for loop-specific low-rank plus sparse weights."""

from typing import Any, Dict, Iterable, Mapping, Union

import torch
from torch import nn

from models.consensus import ConsensusLinear
from salad.adaptive_param import PARAM
from salad.utils import get_energy_quantile


_CONTROLLER_FLOAT_FIELDS = (
    "target_rate",
    "value",
    "rate_decay",
    "drate",
    "dvalue",
    "pre_rate",
)
_CONTROLLER_INT_FIELDS = ("nr_updates", "start_epoch")
_DECOMPOSITION_COMPONENTS = frozenset({"shared", "low_rank", "sparse"})


def _make_integral_controller(
    config: Mapping[str, Any],
    key: str,
    target_rate: float,
) -> PARAM:
    controller_config = config.get(key)
    if not isinstance(controller_config, Mapping):
        raise ValueError(f"consensus_salaad.{key} must be a dictionary")

    controller_config = dict(controller_config)
    controller_config["target_rate"] = target_rate
    if controller_config.get("mode") != "adaptive":
        raise ValueError(
            f"consensus_salaad.{key}.mode must be 'adaptive' for I-control"
        )
    for field in ("init", "rate_decay", "drate"):
        value = float(controller_config.get(field, 0.0))
        if value < 0:
            raise ValueError(
                f"consensus_salaad.{key}.{field} must be non-negative, got {value}"
            )

    controller = PARAM(controller_config)
    start_epoch = controller_config.get("start_epoch", controller.start_epoch)
    if (
        not isinstance(start_epoch, int)
        or isinstance(start_epoch, bool)
        or start_epoch < 0
    ):
        raise ValueError(
            f"consensus_salaad.{key}.start_epoch must be a non-negative integer"
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
        raise ValueError("saved consensus controller mode must be 'adaptive'")
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


def soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Element-wise proximal operator for the l1 norm."""
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    return value.sign() * (value.abs() - threshold).clamp_min(0.0)


def singular_value_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Proximal operator for the nuclear norm of one matrix."""
    if value.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(value.shape)}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    result, _ = _shrink_singular_values(value, threshold)
    return result


def _shrink_singular_values(
    value: torch.Tensor,
    threshold: float,
):
    left, singular_values, right = torch.linalg.svd(value, full_matrices=False)
    shrunk = (singular_values - threshold).clamp_min(0.0)
    return (left * shrunk.unsqueeze(0)) @ right, shrunk


class ConsensusADMM:
    """ADMM state for one stack of loop-specific effective matrices.

    ``effective_weight[i]`` is :math:`X_i`, while ``shared`` is :math:`X`.
    The scaled dual variable is stored explicitly as ``scaled_dual`` (the
    usual :math:`U_i = Y_i / rho`).
    One update performs

    ``X <- mean_i(X_i - L_i - S_i + U_i)``, followed by the nuclear- and
    l1-proximal updates for ``L`` and ``S`` and finally the scaled-dual update. Each
    loop-specific ``L_i`` and ``S_i`` has its own integral-controller state,
    while all controllers use the same globally configured targets and gains.
    """

    def __init__(
        self,
        name: str,
        effective_weight: torch.Tensor,
        config: Mapping[str, Any],
    ) -> None:
        if effective_weight.ndim != 3:
            raise ValueError(
                "effective_weight must have shape [num_loops, out_features, "
                f"in_features], got {tuple(effective_weight.shape)}"
            )
        self.name = name
        self.effective_weight = effective_weight
        self.rho = float(config.get("rho", 0.0))
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")

        self.energy = float(config.get("energy", 0.999))
        self.rate_rank = float(config.get("rate_rank", 0.15))
        self.rate_sparsity = float(config.get("rate_sparsity", 0.05))
        if not 0.0 < self.energy <= 1.0:
            raise ValueError(f"energy must be in (0, 1], got {self.energy}")
        for name, value in (
            ("rate_rank", self.rate_rank),
            ("rate_sparsity", self.rate_sparsity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        initial = effective_weight.detach().float()
        self.shared = initial.mean(dim=0).clone()
        self.low_rank = torch.zeros_like(initial)
        self.sparse = torch.zeros_like(initial)
        self.scaled_dual = torch.zeros_like(initial)
        self.ranks = [0] * self.num_loops
        self.alpha_controllers = [
            _make_integral_controller(config, "alpha_dict", self.rate_rank)
            for _ in range(self.num_loops)
        ]
        self.beta_controllers = [
            _make_integral_controller(config, "beta_dict", self.rate_sparsity)
            for _ in range(self.num_loops)
        ]

    @property
    def num_loops(self) -> int:
        return self.effective_weight.shape[0]

    def residual(self, *, detach: bool = True) -> torch.Tensor:
        weight = self.effective_weight.detach() if detach else self.effective_weight
        return (
            weight.float()
            - self.shared.unsqueeze(0)
            - self.low_rank
            - self.sparse
        )

    def penalty(self) -> torch.Tensor:
        """Augmented-Lagrangian term whose gradient updates every ``X_i``."""
        shifted_residual = self.residual(detach=False) + self.scaled_dual
        return 0.5 * self.rho * shifted_residual.square().sum()

    def _update_controller(self, controller: PARAM, current_rate: float) -> None:
        """Apply the original SALAAD I update with a non-negative projection."""
        previous_value = float(controller.value)
        controller.update(float(current_rate), self.rho)
        controller.value = max(0.0, float(controller.value))
        controller.dvalue = controller.value - previous_value

    @torch.no_grad()
    def step(self) -> None:
        """Run one consensus, low-rank, sparse, and scaled-dual update."""
        effective = self.effective_weight.detach().float()

        self.shared.copy_(
            (
                effective
                - self.low_rank
                - self.sparse
                + self.scaled_dual
            ).mean(dim=0)
        )

        for loop_index in range(self.num_loops):
            value = (
                effective[loop_index]
                - self.shared
                - self.sparse[loop_index]
                + self.scaled_dual[loop_index]
            )
            alpha_controller = self.alpha_controllers[loop_index]
            low_rank, singular_values = _shrink_singular_values(
                value, alpha_controller.value / self.rho
            )
            self.low_rank[loop_index].copy_(low_rank)
            self.ranks[loop_index] = get_energy_quantile(
                singular_values, quantile=self.energy
            )
            rank_rate = self.ranks[loop_index] / min(value.shape)
            self._update_controller(alpha_controller, rank_rate)

        sparse_value = (
            effective
            - self.shared.unsqueeze(0)
            - self.low_rank
            + self.scaled_dual
        )
        elements_per_loop = self.sparse[0].numel()
        for loop_index in range(self.num_loops):
            beta_controller = self.beta_controllers[loop_index]
            self.sparse[loop_index].copy_(
                soft_threshold(
                    sparse_value[loop_index], beta_controller.value / self.rho
                )
            )
            nonzero_rate = (
                int(torch.count_nonzero(self.sparse[loop_index]).item())
                / elements_per_loop
            )
            self._update_controller(beta_controller, nonzero_rate)

        self.scaled_dual.add_(self.residual())

    @torch.no_grad()
    def reconstruction(self) -> torch.Tensor:
        return self.shared.unsqueeze(0) + self.low_rank + self.sparse

    @torch.no_grad()
    def stats(self) -> Dict[str, Union[float, int]]:
        residual = self.residual()
        residual_norm = torch.linalg.vector_norm(residual)
        weight_norm = torch.linalg.vector_norm(self.effective_weight.detach().float())
        return {
            "residual": float(residual_norm.item()),
            "relative_residual": float(
                (residual_norm / weight_norm.clamp_min(1.0e-12)).item()
            ),
            "rank": sum(self.ranks),
            "total_rank": self.num_loops * min(self.effective_weight.shape[-2:]),
            "nonzero": int(torch.count_nonzero(self.sparse).item()),
            "total_elements": self.sparse.numel(),
            "alpha": _mean_controller_value(self.alpha_controllers, "value"),
            "beta": _mean_controller_value(self.beta_controllers, "value"),
            "dalpha": _mean_controller_value(self.alpha_controllers, "dvalue"),
            "dbeta": _mean_controller_value(self.beta_controllers, "dvalue"),
            "rate_decay_alpha": _mean_controller_value(
                self.alpha_controllers, "rate_decay"
            ),
            "rate_decay_beta": _mean_controller_value(
                self.beta_controllers, "rate_decay"
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rho": self.rho,
            "energy": self.energy,
            "rate_rank": self.rate_rank,
            "rate_sparsity": self.rate_sparsity,
            "ranks": list(self.ranks),
            "alpha_controllers": [
                _controller_state(controller)
                for controller in self.alpha_controllers
            ],
            "beta_controllers": [
                _controller_state(controller)
                for controller in self.beta_controllers
            ],
            "shared": self.shared.detach().cpu(),
            "low_rank": self.low_rank.detach().cpu(),
            "sparse": self.sparse.detach().cpu(),
            "scaled_dual": self.scaled_dual.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("name") != self.name:
            raise ValueError(
                f"solver name mismatch: expected {self.name!r}, got {state.get('name')!r}"
            )
        for name in ("rho", "energy", "rate_rank", "rate_sparsity"):
            if name in state:
                setattr(self, name, float(state[name]))
        for name in ("shared", "low_rank", "sparse"):
            source = state[name]
            target = getattr(self, name)
            if source.shape != target.shape:
                raise ValueError(
                    f"{name} shape mismatch: {tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))

        # ``dual`` was the ambiguous key used by the initial implementation.
        # Accept it when loading an older checkpoint, but only save the clear
        # ``scaled_dual`` name going forward.
        scaled_dual = state.get("scaled_dual")
        if scaled_dual is None:
            scaled_dual = state.get("dual")
        if scaled_dual is None:
            raise KeyError("consensus state is missing 'scaled_dual'")
        if scaled_dual.shape != self.scaled_dual.shape:
            raise ValueError(
                "scaled_dual shape mismatch: "
                f"{tuple(scaled_dual.shape)} != {tuple(self.scaled_dual.shape)}"
            )
        self.scaled_dual.copy_(
            scaled_dual.to(
                device=self.scaled_dual.device,
                dtype=self.scaled_dual.dtype,
            )
        )
        ranks = state.get("ranks")
        if ranks is not None:
            if len(ranks) != self.num_loops:
                raise ValueError("ranks must contain one value per logical loop")
            self.ranks = [int(rank) for rank in ranks]

        for name, controllers in (
            ("alpha_controllers", self.alpha_controllers),
            ("beta_controllers", self.beta_controllers),
        ):
            saved_controllers = state.get(name)
            if saved_controllers is None:
                continue
            if len(saved_controllers) != self.num_loops:
                raise ValueError(f"{name} must contain one state per logical loop")
            for controller, controller_state in zip(
                controllers, saved_controllers
            ):
                _load_controller_state(controller, controller_state)


def compose_decomposition(
    state: Mapping[str, Any],
    components: Iterable[str] = ("shared", "low_rank", "sparse"),
) -> torch.Tensor:
    """Compose selected consensus components into loop-specific weights."""
    if isinstance(components, str):
        raise TypeError("components must be an iterable of component names")
    selected = frozenset(components)
    if not selected:
        raise ValueError("at least one decomposition component is required")
    unknown = selected - _DECOMPOSITION_COMPONENTS
    if unknown:
        raise ValueError(f"unknown decomposition components: {sorted(unknown)}")

    low_rank = state["low_rank"]
    sparse = state["sparse"]
    shared = state["shared"]
    if low_rank.shape != sparse.shape:
        raise ValueError(
            "low_rank and sparse shapes differ: "
            f"{tuple(low_rank.shape)} != {tuple(sparse.shape)}"
        )
    if shared.shape != low_rank.shape[1:]:
        raise ValueError(
            "shared shape does not match loop-specific matrices: "
            f"{tuple(shared.shape)} != {tuple(low_rank.shape[1:])}"
        )

    result = torch.zeros_like(low_rank)
    if "shared" in selected:
        result.add_(shared.unsqueeze(0))
    if "low_rank" in selected:
        result.add_(low_rank)
    if "sparse" in selected:
        result.add_(sparse)
    return result


@torch.no_grad()
def apply_decomposition(
    model: nn.Module,
    states: Mapping[str, Mapping[str, Any]],
    *,
    components: Iterable[str] = ("shared", "low_rank", "sparse"),
    strict: bool = True,
) -> None:
    """Materialize selected consensus components into effective weights.

    States from different DDP owners can be merged with normal dictionary
    updates before this function is called.
    """
    model = model.module if hasattr(model, "module") else model
    modules = dict(model.named_modules())
    expected = {
        name for name, module in modules.items()
        if isinstance(module, ConsensusLinear)
    }
    provided = set(states)
    if strict and provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise KeyError(
            f"consensus state mismatch; missing={missing}, unexpected={unexpected}"
        )

    for name, state in states.items():
        module = modules.get(name)
        if not isinstance(module, ConsensusLinear):
            if strict:
                raise KeyError(f"model has no ConsensusLinear named {name!r}")
            continue
        reconstructed = compose_decomposition(state, components)
        if reconstructed.shape != module.weight.shape:
            raise ValueError(
                f"{name} shape mismatch: {tuple(reconstructed.shape)} != "
                f"{tuple(module.weight.shape)}"
            )
        module.weight.copy_(
            reconstructed.to(device=module.weight.device, dtype=module.weight.dtype)
        )
