"""Direct spectral balancing for projected optimization.

This mode deliberately does not implement ADMM.  After an optimizer step, it
balances a configured weight's singular-value energies and writes the result
back into the same ``Parameter``.  The writeback runs under ``no_grad`` so the
next forward pass starts from the projected weight without differentiating
through the spectral decomposition.
"""

from __future__ import annotations

import torch

from salad.utils import get_energy_quantile


def validate_balance_rate(value: float) -> float:
    """Validate and normalize a conservative spectral balance rate."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError(
            "energy_balance_rate must be in (0, 1) so one update "
            f"cannot flatten the spectrum immediately, got {value!r}"
        )
    return float(value)


def balance_singular_energies(
    singular_values: torch.Tensor,
    balance_rate: float,
) -> torch.Tensor:
    """Move every singular-value energy toward the spectrum's mean.

    For ``e_i = sigma_i**2``, the update is

    ``e_i' = (1 - rate) * e_i + rate * mean(e)``.

    It preserves total Frobenius energy while gradually suppressing strong
    directions and lifting weak directions.
    """
    rate = validate_balance_rate(balance_rate)
    energies = singular_values.square()
    mean_energy = energies.mean()
    balanced_energies = (1.0 - rate) * energies + rate * mean_energy
    return balanced_energies.clamp_min(0.0).sqrt()


class SpectralWriteback:
    """Per-layer state for optimizer-step followed by spectral writeback."""

    def __init__(
        self,
        layer_name: str,
        params: dict,
        X: torch.Tensor,
        nr_layers: int,
        is_full: bool,
        precision: torch.dtype = torch.float32,
    ) -> None:
        del nr_layers, precision
        self.layer_name = layer_name
        self.X_with_grad = X.t() if "lm_head" in layer_name else X
        self.energy = params.get("energy", 0.999)
        if (
            isinstance(self.energy, bool)
            or not isinstance(self.energy, (int, float))
            or not 0.0 < float(self.energy) <= 1.0
        ):
            raise ValueError(f"energy must be in (0, 1], got {self.energy!r}")
        self.energy = float(self.energy)
        self.energy_balance_rate = validate_balance_rate(
            params.get("energy_balance_rate", 5e-4)
        )

        rows, cols = self.X_with_grad.shape
        self.nr_total_rank = min(rows, cols)
        self.nr_elements = self.X_with_grad.numel()
        self.layer_gpu_map = -1
        self.layer_idx = -1
        self._cuda_linalg_ready = False

        # Only the owner rank computes these values; other ranks retain the
        # lightweight metadata needed for deterministic collectives/logging.
        if is_full:
            self._clear_statistics()

    def _clear_statistics(self) -> None:
        self.pre_rank = 0
        self.nr_rank = 0
        self.spectrum_cv_before = 0.0
        self.spectrum_cv_after = 0.0
        self.projection_relative_change = 0.0

    @torch.no_grad()
    def _warmup_cuda_linalg(self) -> None:
        """Create the lazy cuSOLVER handle when gradients have been released."""
        if not self.X_with_grad.is_cuda or self._cuda_linalg_ready:
            return
        # cuSOLVER allocates outside PyTorch's caching allocator.  Releasing
        # unused cached blocks once leaves room for its persistent handle on
        # small GPUs without adding an empty_cache call to every projection.
        torch.cuda.empty_cache()
        probe = torch.ones(
            (1, 1),
            dtype=torch.float32,
            device=self.X_with_grad.device,
        )
        torch.linalg.eigh(probe)
        self._cuda_linalg_ready = True

    @staticmethod
    def _energy_cv(singular_values: torch.Tensor) -> torch.Tensor:
        energies = singular_values.square()
        mean = energies.mean()
        if mean.item() == 0.0:
            return torch.zeros((), device=energies.device)
        return energies.std(unbiased=False) / mean

    @staticmethod
    def _project_with_svd(
        X: torch.Tensor,
        balance_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact fallback for a numerically rank-deficient Gram matrix."""
        U, singular_values, Vt = torch.linalg.svd(X, full_matrices=False)
        balanced = balance_singular_energies(
            singular_values,
            balance_rate,
        )
        projected = (U * balanced.unsqueeze(0)) @ Vt
        return projected, singular_values, balanced

    @staticmethod
    def _project_matrix(
        X: torch.Tensor,
        balance_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the spectral map using the smaller FP32 Gram matrix.

        For a full-rank tall matrix ``X = U diag(s) Vt``, this evaluates
        ``X V diag(s'/s) Vt``; for a wide matrix it uses the symmetric left
        form.  This is algebraically the same spectral writeback as a compact
        SVD, while avoiding a large cuSOLVER SVD workspace.  The exact SVD
        path remains as a fallback if FP32 Gram eigenvalues lose a zero mode.
        """
        rows, cols = X.shape
        if rows >= cols:
            gram = X.mT @ X
            eigenvalues, basis = torch.linalg.eigh(gram)
            eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
            basis = basis.flip(1)
            singular_values = eigenvalues.sqrt()
            if torch.count_nonzero(singular_values).item() != cols:
                return SpectralWriteback._project_with_svd(X, balance_rate)
            balanced = balance_singular_energies(
                singular_values,
                balance_rate,
            )
            rotated = X @ basis
            projected = (
                rotated * (balanced / singular_values).unsqueeze(0)
            ) @ basis.mT
        else:
            gram = X @ X.mT
            eigenvalues, basis = torch.linalg.eigh(gram)
            eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
            basis = basis.flip(1)
            singular_values = eigenvalues.sqrt()
            if torch.count_nonzero(singular_values).item() != rows:
                return SpectralWriteback._project_with_svd(X, balance_rate)
            balanced = balance_singular_energies(
                singular_values,
                balance_rate,
            )
            rotated = basis.mT @ X
            projected = basis @ (
                (balanced / singular_values).unsqueeze(1) * rotated
            )
        return projected, singular_values, balanced

    @torch.no_grad()
    def project_and_writeback(self) -> None:
        """Balance the current weight and copy it back in place.

        ``copy_`` preserves the original ``Parameter`` identity, optimizer
        state, and DDP hooks.  The spectral decomposition and reconstruction
        use FP32 for numerical stability; the result is converted back to the
        parameter dtype.
        """
        self._warmup_cuda_linalg()
        X = self.X_with_grad.detach().float()
        projected, singular_values, balanced = self._project_matrix(
            X,
            self.energy_balance_rate,
        )

        denominator = torch.linalg.vector_norm(singular_values)
        change = torch.linalg.vector_norm(balanced - singular_values)
        if denominator.item() != 0.0:
            change = change / denominator

        self.pre_rank = get_energy_quantile(
            singular_values,
            quantile=self.energy,
        )
        self.nr_rank = get_energy_quantile(
            balanced,
            quantile=self.energy,
        )
        self.spectrum_cv_before = self._energy_cv(singular_values).item()
        self.spectrum_cv_after = self._energy_cv(balanced).item()
        self.projection_relative_change = change.item()

        self.X_with_grad.copy_(projected.to(self.X_with_grad.dtype))

    def init_T(self, nr_layers: int, K: int = 12) -> None:
        """Allocate the trainer-compatible distributed statistics buffer."""
        if K < 8:
            raise ValueError("Spectral writeback statistics require 8 columns")
        self.T = torch.zeros(
            nr_layers,
            K,
            dtype=torch.float32,
            device=self.X_with_grad.device,
        )

    @torch.no_grad()
    def cal_results(self) -> None:
        """Write the latest pre/post projection statistics to this layer row."""
        row = self.T[self.layer_idx]
        row.zero_()
        row[0] = self.pre_rank
        row[1] = self.nr_rank
        row[2] = self.spectrum_cv_before
        row[3] = self.spectrum_cv_after
        row[4] = self.energy_balance_rate
        row[5] = self.projection_relative_change
        row[6] = self.nr_total_rank
        row[7] = self.nr_elements
