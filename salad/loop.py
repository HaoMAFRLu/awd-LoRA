"""Shared utilities for training looped models."""

import math
from typing import Optional, Sequence

import torch


class LoopSampler:
    """Sample a logical loop count from a validated discrete distribution."""

    def __init__(
        self,
        values: Sequence[int],
        probabilities: Sequence[float],
        seed: int,
        expected_value: Optional[float] = None,
    ) -> None:
        if not values or len(values) != len(probabilities):
            raise ValueError(
                "loop values and probabilities must have the same non-zero length"
            )
        if len(set(values)) != len(values) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        ):
            raise ValueError("loop values must be unique positive integers")

        probability_tensor = torch.tensor(probabilities, dtype=torch.float64)
        if not torch.isfinite(probability_tensor).all() or (probability_tensor < 0).any():
            raise ValueError("loop probabilities must be finite and non-negative")
        probability_sum = probability_tensor.sum()
        if probability_sum <= 0:
            raise ValueError("at least one loop probability must be positive")

        self.values = tuple(values)
        self.probabilities = probability_tensor / probability_sum
        self.expected_value = sum(
            value * probability
            for value, probability in zip(self.values, self.probabilities.tolist())
        )
        if expected_value is not None and not math.isclose(
            self.expected_value,
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1.0e-12,
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
