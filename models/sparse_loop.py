"""Linear layers with a shared loop weight and loop-specific residuals."""

import torch
import torch.nn.functional as F
from torch import nn


class SparseLoopLinear(nn.Module):
    """Use one shared matrix plus one trainable residual per logical loop.

    ``weight`` is the ordinary weight of the recurrent physical block.  It is
    shared by every execution of that block and receives the task gradient
    directly.  ``specific_weight[i]`` is the dense primal variable ``R_i``;
    the sparse auxiliary variable used by ADMM lives in the solver rather
    than in the forward module.
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
        if (
            not isinstance(num_loops, int)
            or isinstance(num_loops, bool)
            or num_loops < 1
        ):
            raise ValueError(
                f"num_loops must be a positive integer, got {num_loops!r}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.num_loops = num_loops
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        self.specific_weight = nn.Parameter(
            torch.zeros(num_loops, out_features, in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs))
            if bias
            else None
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_loops: int) -> "SparseLoopLinear":
        """Replace ``linear`` without changing its initial function."""
        module = cls(
            linear.in_features,
            linear.out_features,
            num_loops,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
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
        effective_weight = self.weight + self.specific_weight[loop_index]
        return F.linear(inputs, effective_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_loops={self.num_loops}, bias={self.bias is not None}"
        )
