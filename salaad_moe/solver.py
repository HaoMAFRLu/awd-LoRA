"""SALAAD structural constraints: X_i = shared + L_i + S_i within each expert group.

The task trainer updates X with AdamW; this module updates separate auxiliary
states with ADMM. The dual field stores the scaled dual U = Y / rho, and the
anchor is Q = shared + L + S - U.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.distributed as dist

from .distributed import agree_or_raise, all_finite, rank, world_size


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


@dataclass
class GroupState:
    # shared: [out, in]; low_rank/sparse/dual: [experts, out, in].
    # tau_l/tau_s: [experts], equal to alpha/rho and beta/rho, respectively.
    shared: torch.Tensor
    low_rank: torch.Tensor
    sparse: torch.Tensor
    dual: torch.Tensor
    tau_l: torch.Tensor
    tau_s: torch.Tensor
    rank_ratio: torch.Tensor
    density: torch.Tensor
    # Positive streaming components; this is an approximate rank diagnostic.
    actual_rank: torch.Tensor
    # Persistent QR basis: [experts, k, k], where k = min(out, in).
    # This is distinct from the training anchor Q and must survive checkpoints.
    svd_basis: torch.Tensor

    def anchor(self):
        # The training target Q includes the dual term; reconstructed weights
        # used for evaluation exclude it.
        return self.shared + self.low_rank + self.sparse - self.dual

    def reconstruction(self):
        return self.shared + self.low_rank + self.sparse

    def state_dict(self):
        return {f.name: getattr(self, f.name).detach().cpu().clone() for f in fields(self)}

    @classmethod
    def from_state_dict(cls, state, device):
        return cls(
            **{f.name: state[f.name].to(device=device, dtype=torch.float32) for f in fields(cls)}
        )


def initial_state(x, config):
    # By default: shared = expert mean, L = 0, S = X - shared, U = 0.
    # The initial constraint residual is zero, avoiding a sudden large penalty
    # gradient when the auxiliary states are initialized.
    s, ctl = config["salaad"], config["salaad"]["controller"]
    shared = torch.zeros_like(x[0]) if s.get("shared_mode", "learned") == "none" else x.mean(0)
    remainder = x - shared
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
    )


@torch.no_grad()
def structure_sweep(x, old, config):
    """Hold the current model weights X fixed and run one shared -> L -> S -> U update."""
    s, ctl = config["salaad"], config["salaad"]["controller"]
    mode = s.get("shared_mode", "learned")
    # 1. Unregularized consensus: average over experts within this layer and
    # projection. Groups never mix layers or projection types.
    shared = (x - old.low_rank - old.sparse + old.dual).mean(0) if mode == "learned" else old.shared
    # 2. Low-rank residual: threshold the spectrum of X - shared - S_old + U_old.
    if s.get("low_rank_enabled", True):
        low, spectrum, basis = svt(
            x - shared - old.sparse + old.dual,
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
        soft_threshold(x - shared - low + old.dual, old.tau_s[:, None, None])
        if s.get("sparse_enabled", True)
        else torch.zeros_like(x)
    )
    # 4. Accumulate constraint violations so later structural updates keep
    # tracking the dense weights X.
    dual = old.dual + x - shared - low - sparse
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
        spectrum.gt(0).sum(-1).float(), basis,
    )
    if not all_finite([getattr(new, f.name) for f in fields(new)]):
        raise FloatingPointError("Nonfinite auxiliary update; old state was retained")
    return new


class ConsensusManager:
    """Connect task training and ADMM through group states, gradients, scheduling, and DP owners."""
    def __init__(self, groups, config):
        self.groups, self.config = groups, config
        self.states, self.anchors = {}, {}
        self.initialized = False
        self.last_structure_step = None
        self.sweeps = 0
        self.device = groups[0].weight().device

    def owned(self):
        # Owners split auxiliary storage and SVD work across ranks; every rank
        # still uses the full MoE model for forward passes.
        return [(i, g) for i, g in enumerate(self.groups) if i % world_size() == rank()]

    @torch.no_grad()
    def initialize(self, step):
        if self.initialized:
            raise RuntimeError("SALAAD state is already initialized")
        staged, error = {}, None
        try:
            for _, group in self.owned():
                state = initial_state(group.weight(), self.config)
                if not all_finite([state.anchor()]):
                    raise FloatingPointError(f"Nonfinite initial state: {group.name}")
                staged[group.name] = state
        except Exception as exc:
            error = exc
        agree_or_raise(error, self.device, "SALAAD initialization failed")
        self.states, self.initialized, self.last_structure_step = staged, True, step
        self.refresh_anchors()

    @torch.no_grad()
    def refresh_anchors(self):
        for i, group in enumerate(self.groups):
            owner = i % world_size()
            if owner == rank():
                value = self.states[group.name].anchor().contiguous()
            else:
                value = torch.empty_like(group.weight(), memory_format=torch.contiguous_format)
            if dist.is_initialized():
                dist.broadcast(value, src=owner)
            self.anchors[group.name] = value

    @torch.no_grad()
    def inject_gradients(self):
        """Add rho * (X-Q) to existing gradients and return its norm over all parameters."""
        if not self.initialized:
            return 0.0
        squared = torch.zeros((), device=self.device)
        rho = self.config["salaad"]["rho"]
        for group in self.groups:
            delta = (group.weight() - self.anchors[group.name]) * rho
            group.add_gradient(delta)
            squared += delta.square().sum()
        return squared.sqrt().item()

    @torch.no_grad()
    def update(self, step):
        staged, error = {}, None
        try:
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
        """Report diff, density, effective rank ratio, alpha and beta per expert."""
        if not self.initialized:
            return {}
        records = {}
        rho = self.config["salaad"]["rho"]
        for _, group in self.owned():
            state, x = self.states[group.name], group.weight()
            # Include the shared matrix in the MoE reconstruction residual.
            diff = (x - state.reconstruction()).flatten(1).norm(dim=1)
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
            "states": {name: state.state_dict() for name, state in self.states.items()},
        }

    def load_shards(self, shards):
        """Read complete owner shards; reassign whole groups to the current DP."""
        error, staged, metadata = None, {}, None
        try:
            if not shards:
                raise ValueError("No SALAAD state shards")
            metadata = [(s["initialized"], s["last_structure_step"], s["sweeps"]) for s in shards]
            if len(set(metadata)) != 1:
                raise ValueError("Inconsistent SALAAD checkpoint shards")
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
                        state.shared.shape == shape[1:]
                        and state.svd_basis.shape == (shape[0], min(shape[1:]), min(shape[1:]))
                        and all(
                            getattr(state, key).shape == shape
                            for key in ("low_rank", "sparse", "dual")
                        )
                        and all(
                            getattr(state, key).shape == (shape[0],)
                            for key in ("tau_l", "tau_s", "rank_ratio", "density", "actual_rank")
                        )
                    )
                    if not valid_shapes or not all_finite(
                        [getattr(state, f.name) for f in fields(state)]
                    ):
                        raise ValueError(f"Invalid auxiliary checkpoint: {group.name}")
                    if (
                        (state.rank_ratio < 0).any()
                        or (state.rank_ratio > 1).any()
                        or (state.density < 0).any()
                        or (state.density > 1).any()
                    ):
                        raise ValueError(f"Invalid auxiliary controller state: {group.name}")
                    staged[group.name] = state
        except Exception as exc:
            error = exc
        agree_or_raise(error, self.device, "Auxiliary checkpoint load")
        self.initialized, self.last_structure_step, self.sweeps = metadata[0]
        self.states = staged
        if self.initialized:
            self.refresh_anchors()
