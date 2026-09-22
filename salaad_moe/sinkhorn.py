"""Deterministic soft channel alignment in native expert coordinates."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .alignment import PROJECTIONS, initialize_alignment


@contextmanager
def full_precision(device):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def validate_transport(p, experts, channels, reference_expert=None, tolerance=1e-4):
    """P[expert, shared_channel, native_channel], with unit row/column sums."""
    if (
        not isinstance(p, torch.Tensor)
        or p.dtype != torch.float32
        or tuple(p.shape) != (experts, channels, channels)
        or not torch.isfinite(p).all()
        or (p < 0).any()
    ):
        raise ValueError("Invalid finite nonnegative FP32 soft permutation")
    if max(
        (p.sum(-1) - 1).abs().max().item(),
        (p.sum(-2) - 1).abs().max().item(),
    ) > tolerance:
        raise ValueError("Soft permutation must be doubly stochastic")
    if reference_expert is not None and not torch.equal(
        p[reference_expert], torch.eye(channels, device=p.device, dtype=p.dtype)
    ):
        raise ValueError("The reference soft permutation must stay at identity")


def log_sinkhorn(logits, settings, reference_expert):
    """Differentiable log-domain balancing; fail rather than commit invalid P.

    The reference has constant identity P and zero logits. It is not represented
    by infinite logits, so neither exponentials nor gradients encounter inf/nan.
    """
    z = logits / settings["temperature"]
    converged = False
    for iteration in range(settings["max_iterations"]):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
        if (iteration + 1) % 5 == 0 or iteration + 1 == settings["max_iterations"]:
            with torch.no_grad():
                value = z.exp()
                error = torch.maximum(
                    (value.sum(-1) - 1).abs().max(),
                    (value.sum(-2) - 1).abs().max(),
                )
                converged = bool(error <= settings["marginal_tolerance"])
            if converged:
                break
    if not converged:
        raise FloatingPointError("Sinkhorn did not reach the configured marginal tolerance")
    result = z.exp().clone()
    result[reference_expert] = torch.eye(
        logits.shape[-1], device=logits.device, dtype=logits.dtype,
    )
    return result


@torch.no_grad()
def least_squares_shared(residuals, permutation):
    """Solve all three shared normal equations with one SPD factorization."""
    with full_precision(permutation.device):
        matrix = (permutation @ permutation.mT).sum(0)
        rhs = {
            p: (permutation @ (residuals[p].mT if p == "down" else residuals[p])).sum(0)
            for p in PROJECTIONS
        }
        widths = [rhs[p].shape[1] for p in PROJECTIONS]
        # P_reference=I ensures matrix >= I; no ridge or explicit inverse.
        factor = torch.linalg.cholesky(matrix)
        solution = torch.cholesky_solve(torch.cat([rhs[p] for p in PROJECTIONS], -1), factor)
        return {
            p: (value.mT if p == "down" else value).contiguous()
            for p, value in zip(PROJECTIONS, solution.split(widths, dim=-1))
        }


@torch.no_grad()
def initialize_soft_alignment(weights, alignment):
    """Warm-start from joint hard matching, soften once, and refit shared."""
    indices, _ = initialize_alignment(weights, alignment)
    settings = alignment["sinkhorn"]
    channels = indices.shape[-1]
    p = F.one_hot(indices, channels).float().mT
    p = (1 - settings["initial_softening"]) * p + settings["initial_softening"] / channels
    logits = settings["temperature"] * p.log()
    reference = alignment["reference_expert"]
    logits[reference].zero_()
    with full_precision(logits.device):
        p = log_sinkhorn(logits, settings, reference)
        shared = least_squares_shared(weights, p)
    return p, shared, logits


@torch.no_grad()
def update_soft_alignment(residuals, shared, old_p, old_logits, alignment):
    """K_P SGD steps on the true joint reconstruction plus movement penalty.

    Optimize F/rho, with move_penalty_over_rho = lambda_move/rho. This rescales
    the whole subproblem, preserving its minimizer and leaving task rho intact.
    The Gram/cross products give the exact derivative of the three projection
    losses without retaining a dense expert-weight graph in every inner step.
    """
    settings = alignment["sinkhorn"]
    reference = alignment["reference_expert"]
    with full_precision(old_p.device):
        template = torch.cat((shared["gate"], shared["up"], shared["down"].mT), -1)
        target = torch.cat((residuals["gate"], residuals["up"], residuals["down"].mT), -1)
        gram = template @ template.mT
        cross = template @ target.mT
        logits = old_logits.detach().clone()
        for _ in range(settings["inner_steps"]):
            with torch.enable_grad():
                current = logits.detach().requires_grad_(True)
                p = log_sinkhorn(current, settings, reference)
                derivative = (
                    gram @ p.detach() - cross
                    + settings["move_penalty_over_rho"] * (p.detach() - old_p)
                )
                gradient, = torch.autograd.grad(p, current, grad_outputs=derivative)
            logits = current.detach() - settings["learning_rate"] * gradient
            logits[reference].zero_()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite Sinkhorn logits update")
        p = log_sinkhorn(logits, settings, reference)
    validate_transport(
        p, len(p), p.shape[-1], reference, settings["marginal_tolerance"],
    )
    return p.detach(), logits.detach()


@torch.no_grad()
def validate_soft_state(p, logits, alignment):
    settings = alignment["sinkhorn"]
    reference = alignment["reference_expert"]
    validate_transport(p, len(p), p.shape[-1], reference, settings["marginal_tolerance"])
    if (
        logits is None or logits.dtype != torch.float32 or logits.shape != p.shape
        or not torch.isfinite(logits).all() or torch.count_nonzero(logits[reference])
    ):
        raise ValueError("Invalid saved Sinkhorn logits")
    with full_precision(p.device):
        rebuilt = log_sinkhorn(logits, settings, reference)
    if not torch.allclose(rebuilt, p, rtol=0, atol=settings["marginal_tolerance"]):
        raise ValueError("Saved Sinkhorn logits and soft permutation disagree")
