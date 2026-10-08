"""Deterministic soft channel alignment in native expert coordinates."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .alignment import (
    PROJECTIONS, initialize_alignment, initialize_mean_cosine_alignment, validate_permutation,
)


def fixed_reference(alignment):
    """The initialization reference need not remain fixed during optimization."""
    return alignment["reference_expert"] if alignment.get("fix_reference", True) else None


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

    With reference_expert=None all experts are balanced. A fixed reference in
    the legacy mode instead has constant identity P and zero logits, without
    representing that identity with infinite logits.
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
            if converged and settings.get("early_stopping", True):
                break
    if not converged:
        raise FloatingPointError(
            f"Sinkhorn did not reach the configured marginal tolerance after {settings['max_iterations']} iterations: "
            f"maximum row/column sum error={error.item():.8g}, tolerance={settings['marginal_tolerance']:.8g}"
        )
    result = z.exp().clone()
    if reference_expert is not None:
        result[reference_expert] = torch.eye(
            logits.shape[-1], device=logits.device, dtype=logits.dtype,
        )
    return result


@torch.no_grad()
def least_squares_shared(residuals, permutation, *, allow_singular=False):
    """Solve the shared normal equations, including singular unanchored maps."""
    with full_precision(permutation.device):
        matrix = (permutation @ permutation.mT).sum(0)
        rhs = {
            p: (permutation @ (residuals[p].mT if p == "down" else residuals[p])).sum(0)
            for p in PROJECTIONS
        }
        widths = [rhs[p].shape[1] for p in PROJECTIONS]
        targets = torch.cat([rhs[p] for p in PROJECTIONS], -1)
        if allow_singular:
            # Minimum-Frobenius-norm consensus, as in the documented X update.
            solution = torch.linalg.pinv(matrix, hermitian=True) @ targets
        else:
            # A fixed P_reference=I ensures matrix >= I in the legacy mode.
            factor = torch.linalg.cholesky(matrix)
            solution = torch.cholesky_solve(targets, factor)
        return {
            p: (value.mT if p == "down" else value).contiguous()
            for p, value in zip(PROJECTIONS, solution.split(widths, dim=-1))
        }


@torch.no_grad()
def initialize_soft_alignment(weights, alignment, *, identity=False, mean_cosine=False):
    """Initialize exact maps from identity/cosine matching, or a legacy soft match."""
    if identity and mean_cosine:
        raise ValueError("Choose one exact permutation initializer")
    if mean_cosine:
        indices, shared = initialize_mean_cosine_alignment(weights)
        p = F.one_hot(indices, indices.shape[-1]).float().mT
        # Preserve the exact permutation. Finite logits are created by the
        # first closed-form update, not by softening the initial P.
        return p, shared, None
    if identity:
        experts, channels, _ = weights["gate"].shape
        p = torch.eye(channels, device=weights["gate"].device, dtype=torch.float32)
        p = p.expand(experts, -1, -1).clone()
        shared = {projection: weights[projection].mean(0) for projection in PROJECTIONS}
        # Exact zero entries need no finite log representation. The first
        # closed-form P update will create logits from its positive clipped Q.
        return p, shared, None
    indices, _ = initialize_alignment(weights, alignment)
    settings = alignment["sinkhorn"]
    channels = indices.shape[-1]
    p = F.one_hot(indices, channels).float().mT
    p = (1 - settings["initial_softening"]) * p + settings["initial_softening"] / channels
    logits = settings["temperature"] * p.log()
    reference = fixed_reference(alignment)
    if reference is not None:
        logits[reference].zero_()
    with full_precision(logits.device):
        p = log_sinkhorn(logits, settings, reference)
        shared = least_squares_shared(weights, p, allow_singular=reference is None)
    return p, shared, logits


@torch.no_grad()
def update_soft_alignment(residuals, shared, alignment):
    """Closed-form joint least squares, positive clipping, then Sinkhorn.

    With H=[X_gate, X_up, X_down.T] and D=[R_gate, R_up, R_down.T],
    solve H.T @ P ~= D.T for every expert using one shared pseudoinverse.
    This also defines the minimum-norm solution when H is rank deficient,
    without squaring its condition number through the normal equations.
    Saved logits encode the clipped solution; they are not optimized.
    """
    settings = alignment["sinkhorn"]
    reference = fixed_reference(alignment)
    with full_precision(shared["gate"].device):
        template = torch.cat((shared["gate"], shared["up"], shared["down"].mT), -1)
        target = torch.cat((residuals["gate"], residuals["up"], residuals["down"].mT), -1)
        if not torch.isfinite(template).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Nonfinite closed-form alignment inputs")
        solution = torch.linalg.pinv(template.mT) @ target.mT
        if not torch.isfinite(solution).all():
            raise FloatingPointError("Nonfinite closed-form alignment solution")
        clipped = solution.clamp_min(settings["clip_min"])
        logits = settings["temperature"] * clipped.log()
        if reference is not None:
            logits[reference].zero_()
        p = log_sinkhorn(logits, settings, reference)
    validate_transport(
        p, len(p), p.shape[-1], reference, settings["marginal_tolerance"],
    )
    return p.detach(), logits.detach()


@torch.no_grad()
def project_transport_to_permutation(transport, previous):
    """Nearest hard P in Frobenius norm; indices map native to shared channels.

    Maximize the sum of selected *transport entries*, not their logarithms or
    a reconstruction objective. Keep the previous permutation only on a tie;
    the legacy reconstruction improvement tolerance does not apply here.
    """
    from scipy.optimize import linear_sum_assignment

    if transport.ndim != 3:
        raise ValueError("Expected a batch of square soft permutations")
    experts, channels, _ = transport.shape
    validate_transport(transport, experts, channels)
    validate_permutation(previous, experts, channels)
    scores = transport.detach().to(device="cpu", dtype=torch.float64).numpy()
    result = previous.detach().cpu().clone()
    native = torch.arange(channels).numpy()
    for expert in range(experts):
        shared_rows, native_columns = linear_sum_assignment(scores[expert], maximize=True)
        candidate = result[expert].clone()
        candidate[torch.from_numpy(native_columns)] = torch.from_numpy(shared_rows)
        old_score = scores[expert, result[expert].numpy(), native].sum()
        new_score = scores[expert, candidate.numpy(), native].sum()
        if new_score > old_score:
            result[expert] = candidate
    return result.to(device=transport.device)


@torch.no_grad()
def validate_soft_state(p, logits, alignment, *, allow_identity=False, allow_permutation=False):
    settings = alignment["sinkhorn"]
    reference = fixed_reference(alignment)
    validate_transport(p, len(p), p.shape[-1], reference, settings["marginal_tolerance"])
    if logits is None:
        if allow_permutation and ((p == 0) | (p == 1)).all():
            return
        identity = torch.eye(p.shape[-1], device=p.device, dtype=p.dtype).expand_as(p)
        if not allow_identity or not torch.equal(p, identity):
            raise ValueError("Missing Sinkhorn logits outside exact identity initialization or permitted initial permutation")
        return
    if (
        logits.dtype != torch.float32 or logits.shape != p.shape
        or not torch.isfinite(logits).all()
        or (reference is not None and torch.count_nonzero(logits[reference]))
    ):
        raise ValueError("Invalid saved Sinkhorn logits")
    with full_precision(p.device):
        rebuilt = log_sinkhorn(logits, settings, reference)
    if not torch.allclose(rebuilt, p, rtol=0, atol=settings["marginal_tolerance"]):
        raise ValueError("Saved Sinkhorn logits and soft permutation disagree")
