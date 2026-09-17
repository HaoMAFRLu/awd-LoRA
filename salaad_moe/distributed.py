"""Small, synchronous DP collectives used by both backends."""
from __future__ import annotations

import torch
import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def agree_or_raise(error, device, context):
    """All ranks report local failures before entering subsequent collectives."""
    ok = torch.tensor(int(error is None), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not ok.item():
        errors = [None] * world_size()
        if dist.is_initialized():
            dist.all_gather_object(errors, None if error is None else str(error))
        else:
            errors[0] = str(error)
        raise RuntimeError(
            f"{context}: " + "; ".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e)
        )


@torch.no_grad()
def average_gradients(parameters):
    """Average gradients across DP ranks once after accumulation.

    Unused parameters contribute zeros so collective calls have the same order
    on every rank.
    """
    parameters = list(parameters)
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if dist.is_initialized():
            dist.all_reduce(p.grad)
            p.grad.div_(world_size())


def all_finite(tensors):
    # One host synchronization, rather than one per expert matrix.
    checks = [torch.isfinite(t).all() for t in tensors if t is not None]
    return bool(torch.stack(checks).all().item()) if checks else True
