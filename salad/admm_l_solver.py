"""ADMM solver for a dense weight constrained to equal one balanced matrix.

The solver implements the split problem

    minimize_X,L  task_loss(X)
    subject to    X = L

One update moves all singular-value energies a small step toward their mean
while preserving their sum.  This suppresses dominant directions and lifts
weak directions gradually instead of adding a constant to every singular
value.  No rank range is selected and no singular-value tail is truncated.
The solver deliberately has no sparse variable and no alpha/beta controller.
"""

from __future__ import annotations

import torch

from salad.adaptive_rho import RHO
from salad.spectral_writeback import (
    balance_singular_energies,
    validate_balance_rate,
)
from salad.utils import get_energy_quantile


class ADMM_L:
    """Per-layer ADMM state for the equality constraint ``X = L``."""

    def __init__(
        self,
        layer_name: str,
        params: dict,
        X: torch.Tensor,
        nr_layers: int,
        is_full: bool,
        precision: torch.dtype = torch.float32,
    ) -> None:
        self.layer_name = layer_name
        self.X_with_grad = X.t() if "lm_head" in layer_name else X
        self.precision = precision

        self.energy = params.get("energy", 0.999)
        self.energy_balance_rate = params.get("energy_balance_rate", 0.05)
        self._validate_fraction(self.energy, "energy")
        self.energy_balance_rate = validate_balance_rate(
            self.energy_balance_rate
        )

        rho_cfg = dict(params.get("rho_dict", {}))
        rows, cols = self.X_with_grad.shape
        rho_cfg.update(
            row=rows,
            col=cols,
            nr_layers=nr_layers,
            X_norm=(
                torch.norm(self.X_with_grad.detach().float(), p="fro")
                .cpu()
                .numpy()
            ),
        )
        self.rho_solver = RHO(rho_cfg)
        self.rho = self.rho_solver.rho

        self.nr_elements = self.X_with_grad.numel()
        self.nr_total_rank = min(rows, cols)
        self.nr_epoch = 0
        self.ema_r = None
        self.ema_s = None

        if is_full:
            singular_values = torch.linalg.svdvals(
                self.X_with_grad.detach().float()
            )
            # X=L is the feasible initial point requested by this mode.
            self.L = self.X_with_grad.detach().clone().to(self.precision)
            self.Y = torch.zeros_like(self.L)
            self.nr_rank = get_energy_quantile(
                singular_values,
                quantile=self.energy,
            )

        self.reset()

    @staticmethod
    def _validate_fraction(value: float, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be in (0, 1], got {value!r}")

    def reset(self) -> None:
        self.total_loss = 0.0
        self.nr_cals = 0

    @torch.no_grad()
    def get_diff(self) -> torch.Tensor:
        """Return and accumulate the primal residual ``||X-L||_F``."""
        loss = torch.norm(
            self.X_with_grad.detach().float() - self.L.float(),
            p="fro",
        )
        self.nr_cals += 1
        self.total_loss += loss.item()
        return loss

    def get_penalty(self) -> torch.Tensor:
        """Return ``rho/2 * ||X-L+Y/rho||_F^2`` with gradient to X."""
        residual = (
            self.X_with_grad
            - self.L
            + self.Y / self.rho
        )
        return self.rho / 2 * torch.norm(residual, p="fro") ** 2

    def get_gradient(self) -> torch.Tensor:
        """Closed-form gradient of the augmented term with respect to X."""
        return self.rho * (
            self.X_with_grad.detach()
            - self.L
            + self.Y / self.rho
        )

    @staticmethod
    def _update_L(
        X: torch.Tensor,
        Y: torch.Tensor,
        rho: float,
        energy_balance_rate: float,
        energy_quantile: float,
    ) -> tuple[torch.Tensor, int]:
        U, singular_values, Vt = torch.linalg.svd(
            X + Y / rho,
            full_matrices=False,
        )
        balanced = balance_singular_energies(
            singular_values,
            energy_balance_rate,
        )
        nr_rank = get_energy_quantile(
            balanced,
            quantile=energy_quantile,
        )
        L = (U * balanced.unsqueeze(0)) @ Vt
        return L, nr_rank

    @torch.no_grad()
    def update_L(self) -> None:
        L, self.nr_rank = self._update_L(
            self.X_with_grad.detach().float(),
            self.Y.float(),
            self.rho,
            self.energy_balance_rate,
            self.energy,
        )
        self.L = L.to(self.precision)

    @torch.no_grad()
    def update_Y(self) -> None:
        self.Y = (
            self.Y
            + self.rho * (self.X_with_grad.detach() - self.L)
        ).to(self.precision)

    def update_rho(self) -> None:
        self.nr_epoch += 1
        self.rho = self.rho_solver.get_rho(
            self.nr_epoch,
            self.ema_r,
            self.ema_s,
        )

    def init_T(self, nr_layers: int, K: int = 12) -> None:
        """Allocate the trainer-compatible statistics buffer."""
        if K < 12:
            raise ValueError("ADMM_L statistics require at least 12 columns")
        self.T = torch.zeros(
            nr_layers,
            K,
            dtype=torch.float32,
            device=self.X_with_grad.device,
        )

    def cal_results(self) -> None:
        """Write this layer's values into the shared statistics layout.

        Slots associated with SALAAD alpha/S/beta remain zero; they are
        transport padding only and are not exposed as ADMM_L state.
        """
        row = self.T[self.layer_idx]
        row.zero_()
        row[4] = self.rho
        row[7] = (
            self.total_loss / self.nr_cals
            if self.nr_cals
            else 0.0
        )
        row[8] = self.nr_rank
        row[10] = self.nr_total_rank
        row[11] = self.nr_elements
