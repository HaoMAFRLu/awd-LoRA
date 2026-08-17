"""Loop-specific linear layers used by Consensus SALAAD."""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class ConsensusLinear(nn.Module):
    """A linear layer with one effective weight matrix per logical loop.

    During training, ``weight[i]`` is the dense optimization variable
    :math:`X_i`.  The shared matrix and the low-rank/sparse residuals are kept
    by the ADMM solver; they are deliberately not part of the forward path so
    that the task loss and the augmented-Lagrangian guidance both update the
    same dense variable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_loops: int,
        bias: bool = False,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not isinstance(num_loops, int) or isinstance(num_loops, bool) or num_loops < 1:
            raise ValueError(f"num_loops must be a positive integer, got {num_loops!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_loops = num_loops
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(num_loops, out_features, in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs))
            if bias
            else None
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_loops: int) -> "ConsensusLinear":
        """Replace a linear layer while preserving its initialized function."""
        module = cls(
            linear.in_features,
            linear.out_features,
            num_loops,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight.unsqueeze(0).expand_as(module.weight))
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, inputs: torch.Tensor, loop_index: int) -> torch.Tensor:
        if not isinstance(loop_index, int) or isinstance(loop_index, bool):
            raise TypeError(f"loop_index must be an integer, got {loop_index!r}")
        if not 0 <= loop_index < self.num_loops:
            raise IndexError(
                f"loop_index {loop_index} is outside [0, {self.num_loops})"
            )
        return F.linear(inputs, self.weight[loop_index], self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_loops={self.num_loops}, bias={self.bias is not None}"
        )


def apply_linear(
    module: nn.Module,
    inputs: torch.Tensor,
    loop_index: Optional[int],
) -> torch.Tensor:
    """Call a normal or loop-specific linear layer with one code path."""
    if isinstance(module, ConsensusLinear):
        if loop_index is None:
            raise ValueError("ConsensusLinear requires a logical loop index")
        return module(inputs, loop_index)
    return module(inputs)
