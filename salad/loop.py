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
