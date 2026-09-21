"""Joint SwiGLU channel matching; permutations map native -> shared indices.

X, L, S, U and the streaming basis never leave the expert's native coordinates.
Only H uses shared coordinates. One permutation is used by all three projections.
"""
from __future__ import annotations

import re

import torch

from .distributed import all_finite


PROJECTIONS = ("gate", "up", "down")


def layer_groups(groups):
    """Collect complete SwiGLU triplets, independent of the incoming group order."""
    layers = {}
    for group in groups:
        match = re.fullmatch(r"layers\.(\d+)\.moe\.experts\.(gate|up|down)", group.name)
        if match is None:
            raise ValueError(f"Channel alignment requires a named SwiGLU group: {group.name}")
        layer, projection = int(match[1]), match[2]
        triplet = layers.setdefault(layer, {})
        if projection in triplet:
            raise ValueError(f"Duplicate channel-alignment group: {group.name}")
        triplet[projection] = group
    for layer, triplet in layers.items():
        if set(triplet) != set(PROJECTIONS):
            raise ValueError(f"Layer {layer}: channel alignment requires gate, up and down")
        shapes = {p: triplet[p].weight().shape for p in PROJECTIONS}
        gate = shapes["gate"]
        if len(gate) != 3 or shapes["up"] != gate or shapes["down"] != (gate[0], gate[2], gate[1]):
            raise ValueError(f"Layer {layer}: incompatible SwiGLU shapes: {shapes}")
    return dict(sorted(layers.items()))


def validate_permutation(permutation, experts, channels, reference_expert=None):
    if (
        not isinstance(permutation, torch.Tensor)
        or permutation.dtype != torch.int64
        or tuple(permutation.shape) != (experts, channels)
    ):
        raise ValueError("Permutation must be an int64 [experts, channels] tensor")
    identity = torch.arange(channels, device=permutation.device)
    if not torch.equal(permutation.sort(-1).values, identity.expand(experts, -1)):
        raise ValueError("Each expert permutation must be a bijection of the channel indices")
    if reference_expert is not None and not torch.equal(permutation[reference_expert], identity):
        raise ValueError("The reference expert permutation must stay at identity")


def native_shared(shared, permutation=None, channel_axis=None):
    """Return P.T @ H for gate/up, or H @ P for down, without dense P matrices."""
    if permutation is None:
        if channel_axis is not None:
            raise ValueError("A channel axis requires a permutation")
        return shared
    if channel_axis == 0:
        return shared[permutation]
    if channel_axis == 1:
        return shared.T[permutation].mT
    raise ValueError("channel_axis must be 0 (gate/up) or 1 (down)")


def aligned_mean(value, permutation, channel_axis):
    """Average P @ R (gate/up) or R @ P.T (down) in common coordinates."""
    inverse = permutation.argsort(-1)
    if channel_axis == 0:
        aligned = value.gather(1, inverse[:, :, None].expand_as(value))
    elif channel_axis == 1:
        aligned = value.gather(2, inverse[:, None, :].expand_as(value))
    else:
        raise ValueError("Invalid channel axis")
    return aligned.mean(0)


@torch.no_grad()
def joint_cost(residuals, shared):
    """Squared differences of [gate row, up row, down column], with equal weights."""
    descriptor = torch.cat(
        (residuals["gate"], residuals["up"], residuals["down"].mT), dim=-1
    )
    template = torch.cat((shared["gate"], shared["up"], shared["down"].T), dim=-1)
    # Direct distances avoid TF32 and cancellation in ||a||^2+||b||^2-2*a.b.
    # cdist does not materialize [experts, channels, channels, descriptor_size].
    with torch.autocast(device_type=descriptor.device.type, enabled=False):
        cost = torch.cdist(
            descriptor.float(), template.float().unsqueeze(0),
            compute_mode="donot_use_mm_for_euclid_dist",
        ).square_()
    if not all_finite([cost]):
        raise FloatingPointError("Nonfinite joint channel-matching cost")
    return cost


@torch.no_grad()
def match_channels(residuals, shared, permutation, settings):
    # Import only for aligned runs; unaligned training keeps its dependencies.
    from scipy.optimize import linear_sum_assignment

    cost = joint_cost(residuals, shared).cpu().numpy()
    candidate = permutation.detach().cpu().clone()
    tolerance = settings["improvement_tolerance"]
    for expert in range(len(candidate)):
        if expert == settings["reference_expert"]:
            continue
        rows, columns = linear_sum_assignment(cost[expert])
        old_cost = cost[expert, rows, candidate[expert].numpy()].sum(dtype="float64")
        new_cost = cost[expert, rows, columns].sum(dtype="float64")
        if new_cost < old_cost - tolerance * max(1.0, old_cost):
            candidate[expert, torch.from_numpy(rows)] = torch.from_numpy(columns)
    return candidate.to(device=permutation.device)


def matching_error(weights, shared, permutation):
    terms = [
        (weights[p] - native_shared(shared[p], permutation, int(p == "down"))).square().sum()
        for p in PROJECTIONS
    ]
    value = torch.stack(terms).sum()
    if not all_finite([value]):
        raise FloatingPointError("Nonfinite channel-alignment objective")
    return value.item()


@torch.no_grad()
def initialize_alignment(weights, settings):
    experts, channels, _ = weights["gate"].shape
    permutation = torch.arange(channels, device=weights["gate"].device).expand(experts, -1).clone()
    shared = {p: weights[p][settings["reference_expert"]].clone() for p in PROJECTIONS}
    for _ in range(settings["initialization_max_iterations"]):
        old_error = matching_error(weights, shared, permutation)
        permutation = match_channels(weights, shared, permutation, settings)
        shared = {
            p: aligned_mean(weights[p], permutation, int(p == "down")) for p in PROJECTIONS
        }
        new_error = matching_error(weights, shared, permutation)
        if old_error - new_error <= settings["improvement_tolerance"] * max(1.0, old_error):
            break
    return permutation, shared
