"""Training utilities for looped models."""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_TIED_PARAMETER_NAMES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


class LoopSampler:
    """Sample a recurrent depth from a validated discrete distribution."""

    def __init__(
        self,
        values: Sequence[int],
        probabilities: Sequence[float],
        seed: int,
        expected_value: Optional[float] = None,
    ) -> None:
        if not values or len(values) != len(probabilities):
            raise ValueError("loop values and probabilities must have the same non-zero length")
        if len(set(values)) != len(values) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        ):
            raise ValueError("loop values must be unique positive integers")

        probabilities_tensor = torch.tensor(probabilities, dtype=torch.float64)
        if not torch.isfinite(probabilities_tensor).all() or (probabilities_tensor < 0).any():
            raise ValueError("loop probabilities must be finite and non-negative")
        probability_sum = probabilities_tensor.sum()
        if probability_sum <= 0:
            raise ValueError("at least one loop probability must be positive")

        self.values = tuple(values)
        self.probabilities = probabilities_tensor / probability_sum
        self.expected_value = sum(
            value * probability
            for value, probability in zip(self.values, self.probabilities.tolist())
        )
        if expected_value is not None and not math.isclose(
            self.expected_value, float(expected_value), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"loop distribution has expectation {self.expected_value}, "
                f"not {float(expected_value)}"
            )

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def sample(self) -> int:
        index = torch.multinomial(
            self.probabilities,
            num_samples=1,
            replacement=True,
            generator=self.generator,
        ).item()
        return self.values[index]


class LoopStabilitySampler:
    """Choose whether to run a long branch and how many loops to add."""

    def __init__(
        self,
        probability: float,
        deltas: Sequence[int],
        seed: int,
    ) -> None:
        self.probability = float(probability)
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("loop stability probability must be between 0 and 1")
        if not deltas or len(set(deltas)) != len(deltas) or not all(
            isinstance(delta, int) and not isinstance(delta, bool) and delta > 0
            for delta in deltas
        ):
            raise ValueError("loop stability deltas must be unique positive integers")

        self.deltas = tuple(deltas)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def sample(self) -> Optional[int]:
        if torch.rand((), generator=self.generator).item() >= self.probability:
            return None
        index = torch.randint(
            len(self.deltas),
            size=(),
            generator=self.generator,
        ).item()
        return self.deltas[index]


def monotonic_stability_loss(
    short_loss: torch.Tensor,
    long_loss: torch.Tensor,
) -> torch.Tensor:
    """Penalize a long execution only when it is worse than the short one."""
    if short_loss.numel() != 1 or long_loss.numel() != 1:
        raise ValueError("short_loss and long_loss must both be scalar tensors")
    return torch.relu(long_loss - short_loss.detach())


def hidden_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return one RMS Euclidean hidden-state distance per batch item."""
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError(
            "hidden states must have the same [batch, sequence, hidden] shape"
        )

    batch_size, sequence_length, hidden_size = first.shape
    if attention_mask is None:
        mask = torch.ones(
            (batch_size, sequence_length),
            device=first.device,
            dtype=torch.float32,
        )
    else:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must match the hidden-state batch and sequence dimensions"
            )
        mask = attention_mask.to(device=first.device, dtype=torch.float32)

    difference = (first.float() - second.float()) * mask.unsqueeze(-1)
    element_count = (mask.sum(dim=1) * hidden_size).clamp_min(1.0)
    return torch.linalg.vector_norm(difference, dim=(1, 2)) / element_count.sqrt()


def contraction_losses(
    loop_states: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    start_loop: int,
    gamma: float,
    ratio_epsilon: float = 1e-12,
    has_fixed_point_probe: bool = False,
) -> dict:
    """Compute trajectory contraction and an optional fixed-point loss.

    ``loop_states`` contains h[0] through h[K]. When
    ``has_fixed_point_probe`` is true, its last item is instead an auxiliary
    h[K+1] used to measure the residual at h[K]. The contraction condition
    begins with d[start_loop + 1] <= gamma * d[start_loop].
    """
    if not isinstance(start_loop, int) or isinstance(start_loop, bool) or start_loop < 1:
        raise ValueError("start_loop must be a positive integer")
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be finite and strictly between 0 and 1")
    if not math.isfinite(ratio_epsilon) or ratio_epsilon <= 0.0:
        raise ValueError("ratio_epsilon must be finite and positive")
    if not isinstance(has_fixed_point_probe, bool):
        raise TypeError("has_fixed_point_probe must be a boolean")
    if len(loop_states) < start_loop + 2:
        raise ValueError(
            "loop_states must contain enough complete-loop states for one contraction inequality"
        )

    distances = torch.stack(
        tuple(
            hidden_distance(current, previous, attention_mask)
            for previous, current in zip(loop_states[:-1], loop_states[1:])
        ),
        dim=1,
    )
    previous = distances[:, start_loop - 1 : -1]
    following = distances[:, start_loop:]
    violations = torch.relu(following - gamma * previous.detach())
    ratios = following.detach() / previous.detach().clamp_min(ratio_epsilon)
    fixed_point = (
        distances[:, -1].square().mean()
        if has_fixed_point_probe
        else distances.new_zeros(())
    )

    return {
        "contraction": violations.square().mean(),
        "fixed_point": fixed_point,
        "distances": distances,
        "ratios": ratios,
        "violation_rate": (violations.detach() > 0).float().mean(),
    }


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _get_block(model: nn.Module, block_name: str) -> nn.Module:
    model = _unwrap_model(model)
    candidates = (block_name, f"model.{block_name}")
    for candidate in candidates:
        try:
            return model.get_submodule(candidate)
        except AttributeError:
            continue
    raise KeyError(f"decoder block {block_name!r} was not found in the model")


def block_distance(
    model: nn.Module,
    source_block: str,
    target_block: str,
    parameter_names: Sequence[str] = DEFAULT_TIED_PARAMETER_NAMES,
    epsilon: float = 1e-12,
    reference_norms: Optional[Sequence[torch.Tensor]] = None,
    detach_normalizer: bool = True,
) -> torch.Tensor:
    """Return the mean normalized squared Frobenius error of named weights."""
    errors = block_parameter_errors(
        model,
        source_block,
        target_block,
        parameter_names,
        epsilon,
        reference_norms,
        detach_normalizer,
    )
    return torch.stack(tuple(errors.values())).mean()


def block_parameter_errors(
    model: nn.Module,
    source_block: str,
    target_block: str,
    parameter_names: Sequence[str] = DEFAULT_TIED_PARAMETER_NAMES,
    epsilon: float = 1e-12,
    reference_norms: Optional[Sequence[torch.Tensor]] = None,
    detach_normalizer: bool = True,
) -> dict:
    """Return each named matrix's normalized squared Frobenius error.

    Parameters are matched by their names relative to each decoder block.  Each
    matrix is normalized separately so that a large MLP matrix cannot dominate
    a smaller attention matrix. By default, the denominator is the current
    source norm. A caller can instead provide fixed reference norms measured at
    initialization, which prevents the model from gaming the denominator.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not parameter_names:
        raise ValueError("parameter_names must contain at least one name")
    if reference_norms is not None and len(reference_norms) != len(parameter_names):
        raise ValueError("reference_norms must have one value per parameter name")

    source = dict(_get_block(model, source_block).named_parameters())
    target = dict(_get_block(model, target_block).named_parameters())
    errors = {}

    for index, name in enumerate(parameter_names):
        if name not in source:
            raise KeyError(f"{source_block!r} has no parameter named {name!r}")
        if name not in target:
            raise KeyError(f"{target_block!r} has no parameter named {name!r}")

        source_parameter = source[name]
        target_parameter = target[name]
        if source_parameter.shape != target_parameter.shape:
            raise ValueError(
                f"shape mismatch for {name!r}: "
                f"{tuple(source_parameter.shape)} != {tuple(target_parameter.shape)}"
            )

        # Accumulate the regularizer in float32 even when model weights are
        # bfloat16. The casts remain differentiable.
        source_float = source_parameter.float()
        target_float = target_parameter.float()
        numerator = (source_float - target_float).square().sum()
        if reference_norms is None:
            normalizer = source_float.square().sum()
            if detach_normalizer:
                normalizer = normalizer.detach()
        else:
            normalizer = reference_norms[index]
            if normalizer.device != source_float.device:
                normalizer = normalizer.to(source_float.device)
        normalizer = normalizer.float().clamp_min(epsilon)
        errors[name] = numerator / normalizer

    return errors


def get_block_reference_norms(
    model: nn.Module,
    block_name: str,
    parameter_names: Sequence[str] = DEFAULT_TIED_PARAMETER_NAMES,
) -> tuple:
    """Capture fixed squared Frobenius norms for a decoder block."""
    parameters = dict(_get_block(model, block_name).named_parameters())
    norms = []
    for name in parameter_names:
        if name not in parameters:
            raise KeyError(f"{block_name!r} has no parameter named {name!r}")
        norms.append(parameters[name].detach().float().square().sum())
    return tuple(norms)
