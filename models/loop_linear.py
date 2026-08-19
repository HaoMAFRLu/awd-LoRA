"""Common dispatch for ordinary and loop-specific linear layers."""

from typing import Optional

import torch
from torch import nn

from models.consensus import ConsensusLinear
from models.sparse_loop import SparseLoopLinear


def apply_linear(
    module: nn.Module,
    inputs: torch.Tensor,
    loop_index: Optional[int],
) -> torch.Tensor:
    """Call a normal or loop-specific linear layer with one code path."""
    if isinstance(module, (ConsensusLinear, SparseLoopLinear)):
        if loop_index is None:
            raise ValueError(
                f"{type(module).__name__} requires a logical loop index"
            )
        return module(inputs, loop_index)
    return module(inputs)
