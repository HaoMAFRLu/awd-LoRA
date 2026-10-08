"""Check diagonal maxima, ties, and the row/column orientation independently."""
import unittest

import torch

from scripts.analyze_moe_pe_row_maxima import row_maxima, summarize


class PeRowMaximaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_diagonal_tie_is_not_a_strict_maximum(self):
        p = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.25, 0.25], [0.0, 0.25, 0.75]]])
        rows = row_maxima(7, p)
        result = summarize(rows)
        self.assertEqual(result["diagonal_max_rows"], 2)
        self.assertEqual(result["diagonal_strict_max_rows"], 1)
        self.assertEqual(result["diagonal_max_tie_rows"], 1)
        self.assertEqual(result["off_diagonal_wins_rows"], 1)
        self.assertEqual([row["argmax_column"] for row in rows], [0, 0, 2])
        self.assertEqual([row["diagonal_rank"] for row in rows], [1, 2, 1])
        self.assertEqual(rows[1]["diagonal_minus_off_diagonal_max"], -0.25)

    def test_non_self_inverse_permutation_preserves_row_orientation(self):
        p = torch.tensor([[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], torch.eye(3).tolist()])
        rows = row_maxima(0, p)
        self.assertEqual([row["argmax_column"] for row in rows[:3]], [1, 2, 0])
        self.assertEqual(summarize(rows[:3])["diagonal_max_rows"], 0)
        self.assertEqual(summarize(rows[3:])["diagonal_strict_max_rows"], 3)
        self.assertEqual([row["expert"] for row in rows], [0, 0, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
