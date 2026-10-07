"""ADMM with either an unrestricted expert residual or legacy L/S residuals.

The task trainer updates network weights W; this module updates the auxiliaries.
The dual field stores U = Y_hat / rho. The training anchor is the reconstructed
weight minus U. PyTorch stores weights transposed relative to the LaTeX notes.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.distributed as dist

from .config import IDENTITY_INITIALIZATION
from .distributed import agree_or_raise, all_finite, rank, world_size
from .alignment import (
    PROJECTIONS, aligned_mean, initialize_alignment, layer_groups, match_channels,
    native_shared, validate_permutation,
)
from .sinkhorn import (
    fixed_reference, initialize_soft_alignment, least_squares_shared, update_soft_alignment,
    validate_soft_state, validate_transport,
)


def soft_threshold(x, threshold):
    # For nonnegative thresholds, this is the L1 proximal operator.
    # Negative thresholds expand nonzero entries; the sign of zero stays zero.
    return x.sign() * (x.abs() - threshold).clamp_min(0)


def streaming_svd_step(x, basis):
    """One SALAAD++ orthogonal iteration, reusing each expert's previous basis.

    The basis spans the smaller matrix dimension. Column norms approximate
    singular values; repeated updates track the changing singular vectors.
    Adapted from salaadpp/train/salad/solver.py::_streaming_svd_step.
    """
    transposed = x.shape[-1] > x.shape[-2]
    matrix = x.mT if transposed else x
    # Work with a tall matrix so the persistent basis is only k by k.
    next_basis, _ = torch.linalg.qr(matrix.mT @ (matrix @ basis), mode="reduced")
    projected = matrix @ next_basis
    sigma = projected.norm(dim=-2)
    left = projected / sigma.clamp_min(1e-12).unsqueeze(-2)
    if transposed:
        return next_basis, sigma, left.mT, next_basis
    return left, sigma, next_basis.mT, next_basis


def initial_svd_basis(x, chunk_size):
    """Seed streaming iteration with one ordinary SVD, as in SALAAD++."""
    k = min(x.shape[-2:])
    basis = x.new_empty(len(x), k, k)
    spectra = x.new_empty(len(x), k)
    with torch.autocast(device_type=x.device.type, enabled=False):
        for start in range(0, len(x), chunk_size):
            stop = min(start + chunk_size, len(x))
            u, sigma, vh = torch.linalg.svd(x[start:stop], full_matrices=False)
            basis[start:stop] = u if x.shape[-1] > x.shape[-2] else vh.mT
            spectra[start:stop] = sigma
    return basis, spectra


def linear_mass_rank(singular_values, gamma=0.999):
    """Effective rank / min(m,n), using SUM sigma (not SUM sigma squared)."""
    # Streaming column norms are not necessarily ordered.
    singular_values = singular_values.sort(dim=-1, descending=True).values
    total = singular_values.sum(-1)
    cutoff = total * gamma
    ranks = (singular_values.cumsum(-1) < cutoff[..., None]).sum(-1) + 1
    ranks = torch.where(total > 0, ranks.clamp_max(singular_values.shape[-1]), 0)
    return ranks.float() / singular_values.shape[-1]


def svt(x, thresholds, basis, chunk_size=8):
    """Threshold the streaming spectrum and return the basis for the next sweep."""
    low = torch.empty_like(x)
    spectra = x.new_empty(x.shape[0], min(x.shape[-2:]))
    next_basis = torch.empty_like(basis)
    # Disable TF32 for reconstruction of small singular components.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, len(x), chunk_size):
                stop = min(start + chunk_size, len(x))
                u, sigma, vh, updated_basis = streaming_svd_step(
                    x[start:stop], basis[start:stop]
                )
                shrunk = (sigma - thresholds[start:stop, None]).clamp_min(0)
                low[start:stop] = (u * shrunk[:, None, :]) @ vh
                spectra[start:stop] = shrunk
                next_basis[start:stop] = updated_basis
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return low, spectra, next_basis


class AuxiliaryState:
    def native_shared(self):
        return native_shared(self.shared, self.permutation, self.channel_axis)

    def anchor(self):
        # The training target Q includes the dual term; reconstructed weights
        # used for evaluation exclude it.
        return self.reconstruction() - self.dual

    def tensors(self):
        return [
            value for f in fields(self)
            if isinstance(value := getattr(self, f.name), torch.Tensor)
        ]

    def state_dict(self):
        return {
            f.name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for f in fields(self) if (value := getattr(self, f.name)) is not None
        }

    @classmethod
    def from_state_dict(cls, state, device):
        if "residual" in state:
            if any(key in state for key in ("low_rank", "sparse", "svd_basis", "tau_l", "tau_s")):
                raise ValueError("Mixed dense and low-rank/sparse auxiliary state")
            cls = DenseResidualState
        values = {
            f.name: state[f.name].to(device=device, dtype=torch.float32)
            for f in fields(cls) if f.name not in ("permutation", "channel_axis", "alignment_logits")
        }
        if cls is DenseResidualState and (
            values["residual"].ndim != 3
            or values["dual"].shape != values["residual"].shape
            or values["shared"].shape != values["residual"].shape[1:]
            or not all_finite(values.values())
        ):
            raise ValueError("Invalid dense residual checkpoint tensors")
        permutation = state.get("permutation")
        axis = state.get("channel_axis")
        logits = state.get("alignment_logits")
        if permutation is not None:
            if axis not in (0, 1) or isinstance(axis, bool):
                raise ValueError("Invalid auxiliary channel axis")
            soft = isinstance(permutation, torch.Tensor) and permutation.ndim == 3
            validator = validate_transport if soft else validate_permutation
            validator(
                permutation, values["dual"].shape[0], values["shared"].shape[axis]
            )
            if logits is not None:
                if (
                    not soft or not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
                    or logits.shape != permutation.shape or not torch.isfinite(logits).all()
                ):
                    raise ValueError("Invalid auxiliary Sinkhorn logits")
                logits = logits.to(device=device)
            permutation = permutation.to(device=device)
        elif axis is not None or logits is not None:
            raise ValueError("Auxiliary channel permutation is missing")
        return cls(**values, permutation=permutation, channel_axis=axis, alignment_logits=logits)


@dataclass
class GroupState(AuxiliaryState):
    """Legacy shared + low-rank + sparse state, with its threshold controller."""
    residual_mode = "low_rank_sparse"
    shared: torch.Tensor
    low_rank: torch.Tensor
    sparse: torch.Tensor
    dual: torch.Tensor
    tau_l: torch.Tensor
    tau_s: torch.Tensor
    rank_ratio: torch.Tensor
    density: torch.Tensor
    actual_rank: torch.Tensor
    svd_basis: torch.Tensor
    permutation: torch.Tensor | None = None
    channel_axis: int | None = None
    alignment_logits: torch.Tensor | None = None

    def reconstruction(self):
        return self.native_shared() + self.low_rank + self.sparse


@dataclass
class DenseResidualState(AuxiliaryState):
    """Documented X, X_e and U=Y_hat/rho; no L/S or threshold-controller state."""
    residual_mode = "dense"
    shared: torch.Tensor
    residual: torch.Tensor
    dual: torch.Tensor
    permutation: torch.Tensor | None = None
    channel_axis: int | None = None
    alignment_logits: torch.Tensor | None = None

    def reconstruction(self):
        return self.native_shared() + self.residual


def initial_state(x, config, *, shared=None, permutation=None, channel_axis=None, alignment_logits=None):
    # Initialize shared, an exact expert remainder, and U=0. The legacy mode
    # represents that remainder with L/S and also initializes its SVD basis.
    # The initial constraint residual is zero, avoiding a sudden large penalty
    # gradient when the auxiliary states are initialized.
    s = config["salaad"]
    if shared is None:
        shared = torch.zeros_like(x[0]) if s.get("shared_mode", "learned") == "none" else x.mean(0)
    remainder = x - native_shared(shared, permutation, channel_axis)
    if s.get("residual_mode", "low_rank_sparse") == "dense":
        return DenseResidualState(
            shared, remainder, torch.zeros_like(x), permutation, channel_axis, alignment_logits,
        )
    ctl = s["controller"]
    use_sparse = s.get("sparse_enabled", True)
    low = torch.zeros_like(x) if use_sparse else remainder
    sparse = remainder if use_sparse else torch.zeros_like(x)
    basis, sigma = initial_svd_basis(remainder, s["svd_chunk_size"])
    if use_sparse:
        sigma = torch.zeros_like(sigma)
    return GroupState(
        shared,
        low,
        sparse,
        torch.zeros_like(x),
        x.new_full((len(x),), ctl["alpha_init"] / s["rho"]),
        x.new_full((len(x),), ctl["beta_init"] / s["rho"]),
        linear_mass_rank(sigma, ctl["gamma"]),
        sparse.ne(0).float().mean((-1, -2)),
        sigma.gt(0).sum(-1).float(),
        basis,
        permutation,
        channel_axis,
        alignment_logits,
    )


@torch.no_grad()
def structure_sweep(x, old, config, *, shared=None, permutation=None, alignment_logits=None):
    """Hold W fixed and update consensus, expert residual(s), then U."""
    s = config["salaad"]
    if s.get("residual_mode", "low_rank_sparse") == "dense":
        if shared is None:
            if old.permutation is not None:
                raise ValueError("Aligned shared must be updated jointly across gate/up/down")
            shared = (x - old.residual + old.dual).mean(0)
        mapped = native_shared(shared, permutation, old.channel_axis)
        # Step 4: X_e(new) = W(new) - mapped X(new) + Y_hat(old)/rho.
        residual = x - mapped + old.dual
        # Step 5 in scaled units. For a free residual, U(new) is zero up to
        # roundoff; calculate the update rather than silently resetting U.
        dual = old.dual + (x - mapped - residual)
        new = DenseResidualState(
            shared, residual, dual, permutation, old.channel_axis, alignment_logits,
        )
        if not all_finite([*new.tensors(), mapped + residual - dual]):
            raise FloatingPointError("Nonfinite auxiliary update; old state was retained")
        return new
    ctl = s["controller"]
    mode = s.get("shared_mode", "learned")
    # 1. Unregularized consensus: average over experts within this layer and
    # projection. Groups never mix layers or projection types.
    if shared is None:
        if old.permutation is not None:
            raise ValueError("Aligned shared must be updated jointly across gate/up/down")
        shared = (x - old.low_rank - old.sparse + old.dual).mean(0) if mode == "learned" else old.shared
    mapped_shared = native_shared(shared, permutation, old.channel_axis)
    # 2. Low-rank residual: threshold the spectrum of X - shared - S_old + U_old.
    if s.get("low_rank_enabled", True):
        low, spectrum, basis = svt(
            x - mapped_shared - old.sparse + old.dual,
            old.tau_l,
            old.svd_basis,
            s["svd_chunk_size"],
        )
    else:
        low = torch.zeros_like(x)
        spectrum = x.new_zeros(len(x), min(x.shape[-2:]))
        basis = old.svd_basis
    # 3. Sparse residual: threshold entries of X - shared - L_new + U_old.
    sparse = (
        soft_threshold(x - mapped_shared - low + old.dual, old.tau_s[:, None, None])
        if s.get("sparse_enabled", True)
        else torch.zeros_like(x)
    )
    # 4. Accumulate constraint violations so later structural updates keep
    # tracking the dense weights X.
    dual = old.dual + x - mapped_shared - low - sparse
    ratio = linear_mass_rank(spectrum, ctl["gamma"])
    density = sparse.ne(0).float().mean((-1, -2))
    tau_l, tau_s = old.tau_l.clone(), old.tau_s.clone()
    # Update thresholds on every sweep against fixed rank/density targets.
    # Keep signed thresholds so overshoot can produce a negative control output.
    if s.get("low_rank_enabled", True):
        tau_l = tau_l + ctl["gain_alpha"] * (ratio - ctl["target_rank_ratio"])
    if s.get("sparse_enabled", True):
        tau_s = tau_s + ctl["gain_beta"] * (density - ctl["target_density"])
    new = GroupState(
        shared, low, sparse, dual, tau_l, tau_s, ratio, density,
        spectrum.gt(0).sum(-1).float(), basis, permutation, old.channel_axis, alignment_logits,
    )
    if not all_finite([*new.tensors(), mapped_shared + low + sparse - dual]):
        raise FloatingPointError("Nonfinite auxiliary update; old state was retained")
    return new


@torch.no_grad()
def aligned_structure_sweep(weights, old, config, rematch):
    """Update consensus/P in the configured order, then expert residual(s) and U."""
    dense = config["salaad"].get("residual_mode", "low_rank_sparse") == "dense"
    residuals = {
        p: (weights[p] - old[p].residual + old[p].dual if dense else
            weights[p] - old[p].low_rank - old[p].sparse + old[p].dual)
        for p in PROJECTIONS
    }
    permutation = old["gate"].permutation
    logits = old["gate"].alignment_logits
    soft = config["salaad"]["channel_alignment"].get("method") == "sinkhorn"
    shared_first = config["salaad"]["structure_order"][:2] == ["shared", "permutation"]
    if soft:
        if rematch:
            permutation, logits = update_soft_alignment(
                residuals, {p: old[p].shared for p in PROJECTIONS},
                config["salaad"]["channel_alignment"],
            )
        shared = least_squares_shared(
            residuals, permutation,
            allow_singular=fixed_reference(config["salaad"]["channel_alignment"]) is None,
        )
    elif shared_first:
        # Freeze R = W(new) - X_e(old) + U(old) for both updates. Average
        # using P(old), then match against X(new); do not average again.
        shared = {
            p: aligned_mean(residuals[p], permutation, int(p == "down")) for p in PROJECTIONS
        }
        if rematch:
            permutation = match_channels(
                residuals, shared, permutation, config["salaad"]["channel_alignment"],
            )
    else:
        if rematch:
            permutation = match_channels(
                residuals, {p: old[p].shared for p in PROJECTIONS}, permutation,
                config["salaad"]["channel_alignment"],
            )
        shared = {
            p: aligned_mean(residuals[p], permutation, int(p == "down")) for p in PROJECTIONS
        }
    return {
        p: structure_sweep(
            weights[p], old[p], config, shared=shared[p], permutation=permutation, alignment_logits=logits,
        )
        for p in PROJECTIONS
    }


class ConsensusManager:
    """Connect task training and ADMM through group states, gradients, scheduling, and DP owners."""
    def __init__(self, groups, config):
        self.groups, self.config = groups, config
        self.residual_mode = config["salaad"].get("residual_mode", "low_rank_sparse")
        self.states, self.anchors = {}, {}
        self.initialized = False
        self.last_structure_step = None
        self.last_matching_step = None
        self.sweeps = 0
        self.device = groups[0].weight().device
        self.alignment = config["salaad"].get("channel_alignment", {})
        self.aligned = self.alignment.get("enabled", False)
        self.alignment_method = self.alignment.get("method", "hungarian") if self.aligned else "none"
        self.soft_aligned = self.alignment_method == "sinkhorn"
        self.layers = layer_groups(groups) if self.aligned else {}
        self.group_layers = {
            group.name: layer for layer, triplet in self.layers.items() for group in triplet.values()
        }

    def owner(self, index):
        key = self.group_layers[self.groups[index].name] if self.aligned else index
        return key % world_size()

    def owned(self):
        # Owners split auxiliary storage and SVD work across ranks; every rank
        # still uses the full MoE model for forward passes.
        return [(i, g) for i, g in enumerate(self.groups) if self.owner(i) == rank()]

    def owned_layers(self):
        return [(layer, triplet) for layer, triplet in self.layers.items() if layer % world_size() == rank()]

    @torch.no_grad()
    def initialize(self, step):
        if self.initialized:
            raise RuntimeError("SALAAD state is already initialized")
        staged, error = {}, None
        try:
            if self.aligned:
                for _, triplet in self.owned_layers():
                    weights = {p: triplet[p].weight() for p in PROJECTIONS}
                    logits = None
                    if self.soft_aligned:
                        permutation, shared, logits = initialize_soft_alignment(
                            weights, self.alignment,
                            identity=self.config["salaad"]["initialization"] == IDENTITY_INITIALIZATION,
                        )
                    else:
                        permutation, shared = initialize_alignment(
                            weights, self.alignment,
                            identity=(self.residual_mode == "dense" and
                                      self.config["salaad"]["initialization"] == IDENTITY_INITIALIZATION),
                        )
                    for p in PROJECTIONS:
                        staged[triplet[p].name] = initial_state(
                            weights[p], self.config, shared=shared[p],
                            permutation=permutation, channel_axis=int(p == "down"),
                            alignment_logits=logits,
                        )
            else:
                for _, group in self.owned():
                    staged[group.name] = initial_state(group.weight(), self.config)
            for name, state in staged.items():
                if not all_finite([*state.tensors(), state.anchor()]):
                    raise FloatingPointError(f"Nonfinite initial state: {name}")
        except Exception as exc:
            error = exc
        agree_or_raise(error, self.device, "SALAAD initialization failed")
        self.states, self.initialized, self.last_structure_step = staged, True, step
        self.last_matching_step = step if self.aligned else None
        self.refresh_anchors()

    @torch.no_grad()
    def refresh_anchors(self):
        for i, group in enumerate(self.groups):
            owner = self.owner(i)
            if owner == rank():
                value = self.states[group.name].anchor().contiguous()
            else:
                value = torch.empty_like(group.weight(), memory_format=torch.contiguous_format)
            if dist.is_initialized():
                dist.broadcast(value, src=owner)
            self.anchors[group.name] = value

    @torch.no_grad()
    def inject_gradients(self, task_norm):
        """Add rho * (X-Q); return its norm and cosine with the original task gradient."""
        if not self.initialized:
            return 0.0, None
        # Accumulate the squared structure norm and the dot product directly.
        # Measuring before addition avoids cancellation when the penalty is small.
        statistics = torch.zeros(2, device=self.device)
        rho = self.config["salaad"]["rho"]
        for group in self.groups:
            delta = (group.weight() - self.anchors[group.name]) * rho
            statistics[1] += group.add_gradient(delta)
            statistics[0] += delta.square().sum()
        statistics[0].sqrt_()
        constraint_norm, dot = statistics.tolist()
        if task_norm == 0 or constraint_norm == 0:
            return constraint_norm, None  # A zero vector has no direction.
        # The task norm includes all model parameters; structural gradients are
        # zero outside the expert groups. Clamp only floating-point roundoff.
        cosine = max(-1.0, min(1.0, dot / (task_norm * constraint_norm)))
        return constraint_norm, cosine

    @torch.no_grad()
    def update(self, step):
        staged, error = {}, None
        interval = self.alignment.get("match_every_optimizer_steps", 0)
        # Count matching intervals from initialization, including a vanilla prefix.
        # For initialization at 100 and interval 200, rematch at 300, 500, ... .
        elapsed = step - self.config["salaad"]["state_initialization_step"]
        rematch = (
            self.aligned and interval > 0 and elapsed > 0
            and elapsed % interval == 0 and step > self.last_matching_step
        )
        # Dense ADMM and legacy soft P match on every structure sweep,
        # including an off-period final flush. Legacy hard schedules stay intact.
        if self.soft_aligned or (self.aligned and self.residual_mode == "dense"):
            rematch = step > self.last_matching_step
        try:
            if self.aligned:
                for _, triplet in self.owned_layers():
                    states = aligned_structure_sweep(
                        {p: triplet[p].weight() for p in PROJECTIONS},
                        {p: self.states[triplet[p].name] for p in PROJECTIONS},
                        self.config, rematch,
                    )
                    staged.update({triplet[p].name: states[p] for p in PROJECTIONS})
            else:
                for _, group in self.owned():
                    x, state = group.weight(), self.states[group.name]
                    for _ in range(self.config["salaad"]["structure_inner_steps"]):
                        state = structure_sweep(x, state, self.config)
                    staged[group.name] = state
        except Exception as exc:
            error = exc
        # Nothing has modified the current states or anchors before consensus.
        agree_or_raise(error, self.device, "SALAAD sweep failed (no state committed)")
        self.states = staged
        self.last_structure_step = step
        if rematch:
            self.last_matching_step = step
        self.sweeps += self.config["salaad"]["structure_inner_steps"]
        self.refresh_anchors()

    def after_step(self, step, final=False):
        # The prefix uses ordinary task training. After state initialization,
        # structural updates run at fixed intervals.
        s = self.config["salaad"]
        if not self.initialized:
            if step >= s["state_initialization_step"]:
                self.initialize(step)
                return True
            return False
        due = step > self.last_structure_step and (
            (step - s["state_initialization_step"]) % s["guidance_period_optimizer_steps"] == 0
            or (final and s["final_structure_flush"])
        )
        if due:
            self.update(step)
        return due

    @torch.no_grad()
    def metrics(self):
        """Report each expert's structure and coefficients, including the shared rho."""
        if not self.initialized:
            return {}
        records = {}
        rho = self.config["salaad"]["rho"]
        for _, group in self.owned():
            state, x = self.states[group.name], group.weight()
            # Include the shared matrix in the MoE reconstruction residual.
            diff = (x - state.reconstruction()).flatten(1).norm(dim=1)
            if self.residual_mode == "dense":
                values = torch.stack((
                    diff, state.residual.flatten(1).norm(dim=1),
                    rho * state.dual.flatten(1).norm(dim=1),
                ), dim=1).tolist()
                for expert, (error, residual_norm, multiplier_norm) in enumerate(values):
                    records[f"{group.name}.expert_{expert}"] = {
                        "diff": error, "residual_norm": residual_norm,
                        "multiplier_norm": multiplier_norm, "rho": rho,
                    }
                continue
            # Report alpha/beta in their original units: tau = coefficient / rho.
            values = torch.stack(
                (diff, state.density, state.rank_ratio, rho * state.tau_l, rho * state.tau_s),
                dim=1,
            ).tolist()
            for expert, (diff, density, effective_rank_ratio, alpha, beta) in enumerate(values):
                records[f"{group.name}.expert_{expert}"] = {
                    "diff": diff,
                    "density": density,
                    # Use the current MoE controller's per-expert rank statistic.
                    "effective_rank_ratio": effective_rank_ratio,
                    "rho": rho,
                    "alpha": alpha,
                    "beta": beta,
                }
        if dist.is_initialized():
            parts = [None] * world_size()
            dist.all_gather_object(parts, records)
            records = {name: values for part in parts for name, values in part.items()}
        return dict(sorted(records.items()))

    def local_state_dict(self):
        return {
            "initialized": self.initialized,
            "last_structure_step": self.last_structure_step,
            "sweeps": self.sweeps,
            "channel_alignment": self.aligned,
            "alignment_method": self.alignment_method,
            "residual_mode": self.residual_mode,
            "last_matching_step": self.last_matching_step,
            "states": {name: state.state_dict() for name, state in self.states.items()},
        }

    def load_shards(self, shards):
        """Restore native states and P; assign complete layers to aligned DP owners."""
        error, staged, metadata = None, {}, None
        try:
            if not shards:
                raise ValueError("No SALAAD state shards")
            metadata = [
                (
                    s["initialized"], s["last_structure_step"], s["sweeps"],
                    s.get("channel_alignment", False), s.get("last_matching_step"),
                    s.get("alignment_method", "hungarian" if s.get("channel_alignment", False) else "none"),
                    s.get("residual_mode", "low_rank_sparse"),
                )
                for s in shards
            ]
            if len(set(metadata)) != 1:
                raise ValueError("Inconsistent SALAAD checkpoint shards")
            initialized, last_structure, sweeps, aligned, last_matching, method, residual_mode = metadata[0]
            if residual_mode != self.residual_mode:
                raise ValueError("Checkpoint residual mode differs from configuration")
            if aligned != self.aligned or method != self.alignment_method:
                raise ValueError("Checkpoint channel-alignment mode differs from configuration")
            if initialized and self.aligned and (
                type(last_matching) is not int or not 0 <= last_matching <= last_structure
            ):
                raise ValueError("Invalid checkpoint channel-matching step")
            if metadata[0][0]:
                available = {}
                for shard in shards:
                    if available.keys() & shard["states"].keys():
                        raise ValueError("Duplicate auxiliary group in checkpoint")
                    available.update(shard["states"])
                if set(available) != {g.name for g in self.groups}:
                    raise ValueError("Auxiliary checkpoint groups do not match the model")
                for _, group in self.owned():
                    state = GroupState.from_state_dict(available[group.name], self.device)
                    shape = group.weight().shape
                    valid_shapes = (
                        state.residual_mode == self.residual_mode
                        and state.shared.shape == shape[1:]
                        and state.dual.shape == shape
                    )
                    if self.residual_mode == "dense":
                        valid_shapes = (
                            valid_shapes and isinstance(state, DenseResidualState)
                            and state.residual.shape == shape
                        )
                    else:
                        valid_shapes = valid_shapes and isinstance(state, GroupState) and (
                            state.svd_basis.shape == (shape[0], min(shape[1:]), min(shape[1:]))
                            and all(
                                getattr(state, key).shape == shape
                                for key in ("low_rank", "sparse")
                            )
                            and all(
                                getattr(state, key).shape == (shape[0],)
                                for key in ("tau_l", "tau_s", "rank_ratio", "density", "actual_rank")
                            )
                        )
                    if not valid_shapes or not all_finite([*state.tensors(), state.anchor()]):
                        raise ValueError(f"Invalid auxiliary checkpoint: {group.name}")
                    if (state.permutation is not None) != self.aligned:
                        raise ValueError(f"Missing or unexpected channel permutation: {group.name}")
                    soft = state.permutation is not None and state.permutation.ndim == 3
                    if soft != self.soft_aligned or (state.alignment_logits is not None and not soft):
                        raise ValueError(f"Unexpected channel-permutation representation: {group.name}")
                    if self.residual_mode != "dense" and (
                        (state.rank_ratio < 0).any()
                        or (state.rank_ratio > 1).any()
                        or (state.density < 0).any()
                        or (state.density > 1).any()
                    ):
                        raise ValueError(f"Invalid auxiliary controller state: {group.name}")
                    staged[group.name] = state
                if self.aligned:
                    validate_layer_states(
                        staged, self.alignment["reference_expert"], self.alignment,
                        allow_identity=(
                            sweeps == 0 and self.config["salaad"]["initialization"] == IDENTITY_INITIALIZATION
                        ),
                    )
        except Exception as exc:
            error = exc
        agree_or_raise(error, self.device, "Auxiliary checkpoint load")
        self.initialized, self.last_structure_step, self.sweeps, _, self.last_matching_step, _, _ = metadata[0]
        self.states = staged
        if self.initialized:
            self.refresh_anchors()


def validate_layer_states(states, reference_expert, alignment=None, *, allow_identity=False):
    """Check saved triplets and share one immutable P tensor within each layer."""
    if alignment is not None:
        reference_expert = fixed_reference(alignment)
    layers = {}
    for name, state in states.items():
        layer, projection = name.rsplit(".", 1)
        layers.setdefault(layer, {})[projection] = state
    for layer, triplet in layers.items():
        if set(triplet) != set(PROJECTIONS):
            raise ValueError(f"Incomplete channel-alignment checkpoint layer: {layer}")
        permutation = triplet["gate"].permutation
        logits = triplet["gate"].alignment_logits
        soft = alignment is not None and alignment.get("method") == "sinkhorn"
        if soft:
            validate_soft_state(permutation, logits, alignment, allow_identity=allow_identity)
        elif logits is not None:
            raise ValueError("Missing soft alignment configuration")
        for p, state in triplet.items():
            if state.channel_axis != int(p == "down"):
                raise ValueError(f"Wrong saved channel axis: {layer}.{p}")
            validator = validate_transport if soft else validate_permutation
            validator(
                state.permutation, len(state.dual), state.shared.shape[state.channel_axis],
                reference_expert,
            )
            if not torch.equal(state.permutation, permutation):
                raise ValueError(f"gate/up/down permutations disagree: {layer}")
            if (state.alignment_logits is None) != (logits is None) or (
                logits is not None and not torch.equal(state.alignment_logits, logits)
            ):
                raise ValueError(f"gate/up/down Sinkhorn logits disagree: {layer}")
        for state in triplet.values():
            state.permutation = permutation
            state.alignment_logits = logits
