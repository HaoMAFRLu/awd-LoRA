"""Consensus ADMM updates for loop-specific low-rank plus sparse weights."""

from typing import Any, Dict, Mapping, Union

import torch
from torch import nn

from models.consensus import ConsensusLinear


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
    The scaled dual variable is denoted by ``dual`` (the usual :math:`U_i`).
    One update performs

    ``X <- mean_i(X_i - L_i - S_i + U_i)``, followed by the nuclear- and
    l1-proximal updates for ``L`` and ``S`` and finally the dual update.
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
        self.lambda_low_rank = float(config.get("lambda_low_rank", 0.0))
        self.lambda_sparse = float(config.get("lambda_sparse", 0.0))
        self.rank_tolerance = float(config.get("rank_tolerance", 1.0e-6))
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.lambda_low_rank < 0:
            raise ValueError(
                f"lambda_low_rank must be non-negative, got {self.lambda_low_rank}"
            )
        if self.lambda_sparse < 0:
            raise ValueError(
                f"lambda_sparse must be non-negative, got {self.lambda_sparse}"
            )
        if self.rank_tolerance < 0:
            raise ValueError(
                f"rank_tolerance must be non-negative, got {self.rank_tolerance}"
            )

        initial = effective_weight.detach().float()
        self.shared = initial.mean(dim=0).clone()
        self.low_rank = torch.zeros_like(initial)
        self.sparse = torch.zeros_like(initial)
        self.dual = torch.zeros_like(initial)
        self.ranks = [0] * self.num_loops

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
        shifted_residual = self.residual(detach=False) + self.dual
        return 0.5 * self.rho * shifted_residual.square().sum()

    @torch.no_grad()
    def step(self) -> None:
        """Run one consensus, low-rank, sparse, and dual update."""
        effective = self.effective_weight.detach().float()

        self.shared.copy_(
            (effective - self.low_rank - self.sparse + self.dual).mean(dim=0)
        )

        low_rank_threshold = self.lambda_low_rank / self.rho
        for loop_index in range(self.num_loops):
            value = (
                effective[loop_index]
                - self.shared
                - self.sparse[loop_index]
                + self.dual[loop_index]
            )
            low_rank, singular_values = _shrink_singular_values(
                value, low_rank_threshold
            )
            self.low_rank[loop_index].copy_(low_rank)
            scale = singular_values.max() if singular_values.numel() else 0.0
            rank_threshold = self.rank_tolerance * max(float(scale), 1.0)
            self.ranks[loop_index] = int(
                (singular_values > rank_threshold).sum().item()
            )

        sparse_value = (
            effective
            - self.shared.unsqueeze(0)
            - self.low_rank
            + self.dual
        )
        self.sparse.copy_(
            soft_threshold(sparse_value, self.lambda_sparse / self.rho)
        )
        self.dual.add_(self.residual())

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
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rho": self.rho,
            "lambda_low_rank": self.lambda_low_rank,
            "lambda_sparse": self.lambda_sparse,
            "rank_tolerance": self.rank_tolerance,
            "ranks": list(self.ranks),
            "shared": self.shared.detach().cpu(),
            "low_rank": self.low_rank.detach().cpu(),
            "sparse": self.sparse.detach().cpu(),
            "dual": self.dual.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("name") != self.name:
            raise ValueError(
                f"solver name mismatch: expected {self.name!r}, got {state.get('name')!r}"
            )
        for name in (
            "rho",
            "lambda_low_rank",
            "lambda_sparse",
            "rank_tolerance",
        ):
            if name in state:
                setattr(self, name, float(state[name]))
        for name in ("shared", "low_rank", "sparse", "dual"):
            source = state[name]
            target = getattr(self, name)
            if source.shape != target.shape:
                raise ValueError(
                    f"{name} shape mismatch: {tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        ranks = state.get("ranks")
        if ranks is not None:
            if len(ranks) != self.num_loops:
                raise ValueError("ranks must contain one value per logical loop")
            self.ranks = [int(rank) for rank in ranks]


@torch.no_grad()
def apply_decomposition(
    model: nn.Module,
    states: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> None:
    """Materialize ``X + L_i + S_i`` into a model's effective weights.

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
        reconstructed = state["shared"].unsqueeze(0) + state["low_rank"] + state["sparse"]
        if reconstructed.shape != module.weight.shape:
            raise ValueError(
                f"{name} shape mismatch: {tuple(reconstructed.shape)} != "
                f"{tuple(module.weight.shape)}"
            )
        module.weight.copy_(
            reconstructed.to(device=module.weight.device, dtype=module.weight.dtype)
        )
