from __future__ import annotations

import copy
import unittest

import torch

from salad.salad_solver import (
    SALAD,
    _block_norms,
    _block_soft_threshold,
    _resolve_block_shape,
)
from salad.utils import soft_threshold


def _solver_params(*, beta_mode="fixed", block_shape=None):
    params = {
        "energy": 0.999,
        "init_energy": 0.5,
        "is_init": False,
        "iter_max": 1,
        "tol": 0.001,
        "rate_rank": 0.5,
        "rate_sparsity": 0.25,
        "alpha_dict": {
            "init": 0.0,
            "mode": "fixed",
        },
        "beta_dict": {
            "init": 0.0,
            "mode": beta_mode,
            "rate_decay": 0.5,
            "drate": 0.0,
        },
        "rho_dict": {
            "rho": 1.0,
            "mode": "fixed",
        },
    }
    if block_shape is not None:
        params.update(block_shape)
    return params


class BlockSparseProxTest(unittest.TestCase):
    def test_row_group_soft_threshold_removes_a_whole_row(self) -> None:
        matrix = torch.tensor([[3.0, 4.0], [0.3, 0.4]])

        actual = _block_soft_threshold(matrix, threshold=1.0, p=1, q=2)

        expected = torch.tensor([[2.4, 3.2], [0.0, 0.0]])
        torch.testing.assert_close(actual, expected)

    def test_column_group_soft_threshold_removes_a_whole_column(self) -> None:
        matrix = torch.tensor([[3.0, 0.3], [4.0, 0.4]])

        actual = _block_soft_threshold(matrix, threshold=1.0, p=2, q=1)

        expected = torch.tensor([[2.4, 0.0], [3.2, 0.0]])
        torch.testing.assert_close(actual, expected)

    def test_block_norms_follow_the_configured_partition(self) -> None:
        matrix = torch.tensor(
            [
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 12.0],
            ]
        )

        norms = _block_norms(matrix, p=1, q=2)

        torch.testing.assert_close(
            norms,
            torch.tensor([[5.0, 0.0], [0.0, 13.0]]),
        )

    def test_full_dimensions_resolve_to_row_and_column_groups(self) -> None:
        self.assertEqual(_resolve_block_shape((6, 4), 1, "full"), (1, 4))
        self.assertEqual(_resolve_block_shape((6, 4), "FULL", 1), (6, 1))
        self.assertIsNone(_resolve_block_shape((6, 4)))

    def test_invalid_block_shapes_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            _resolve_block_shape((6, 4), 1, None)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            _resolve_block_shape((6, 4), 4, 2)
        with self.assertRaisesRegex(TypeError, "positive integer"):
            _resolve_block_shape((6, 4), True, 1)
        with self.assertRaisesRegex(ValueError, "2-D"):
            _resolve_block_shape((2, 3, 4), 1, 1)

    def test_unconfigured_solver_keeps_elementwise_behavior(self) -> None:
        params = _solver_params()
        solver = SALAD(
            "linear",
            copy.deepcopy(params),
            torch.eye(2),
            nr_layers=1,
            is_full=True,
        )
        solver.beta_solver.value = 0.25
        candidate = torch.tensor([[0.1, -0.5], [1.0, -2.0]])

        actual = solver._update_S(
            candidate,
            torch.zeros_like(candidate),
            torch.zeros_like(candidate),
            rho=1.0,
        )

        torch.testing.assert_close(actual, soft_threshold(candidate, 0.25))
        self.assertIsNone(solver.block_shape)
        self.assertEqual(solver.nr_sparse_units, candidate.numel())

    def test_hard_cut_uses_block_norm_quantiles(self) -> None:
        params = _solver_params(
            beta_mode="hard_cut",
            block_shape={"block_p": 1, "block_q": "full"},
        )
        params["rate_sparsity"] = 0.5
        solver = SALAD(
            "linear",
            copy.deepcopy(params),
            torch.eye(2),
            nr_layers=1,
            is_full=True,
        )
        candidate = torch.tensor([[3.0, 4.0], [0.3, 0.4]])

        actual = solver._update_S(
            candidate,
            torch.zeros_like(candidate),
            torch.zeros_like(candidate),
            rho=1.0,
        )

        torch.testing.assert_close(solver.beta_solver.value, torch.tensor(0.5))
        torch.testing.assert_close(
            actual,
            torch.tensor([[2.7, 3.6], [0.0, 0.0]]),
        )

    def test_adaptive_beta_controller_counts_nonzero_groups(self) -> None:
        params = _solver_params(
            beta_mode="adaptive",
            block_shape={"block_p": 1, "block_q": "full"},
        )
        solver = SALAD(
            "linear",
            copy.deepcopy(params),
            torch.eye(2),
            nr_layers=1,
            is_full=True,
        )
        # One partially populated row is one nonzero group out of two.  Its
        # element density is only 1/4, so this distinguishes block counting.
        solver.S = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

        solver.update_beta()

        self.assertEqual(solver.nr_sparse_units, 2)
        torch.testing.assert_close(
            solver.beta_solver.dvalue,
            torch.tensor(0.125),
        )


if __name__ == "__main__":
    unittest.main()
