import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.splines import evaluate_open_uniform_bspline_basis


class KANBsplineBasisTests(unittest.TestCase):
    def test_open_uniform_cubic_bspline_basis_is_smooth_and_partitioned(self):
        x = np.array([[-1.5], [-0.75], [0.0], [0.75], [1.5]], dtype=np.float64)
        basis = evaluate_open_uniform_bspline_basis(
            x,
            num_basis=7,
            degree=3,
            grid_range=(-1.5, 1.5),
        )
        self.assertEqual(basis.shape, (5, 1, 7))
        np.testing.assert_allclose(np.sum(basis[:, 0, :], axis=1), np.ones(5), atol=1e-10)

        interior = basis[2, 0, :]
        self.assertLess(float(np.max(interior)), 1.0)
        self.assertGreaterEqual(int(np.count_nonzero(interior > 1.0e-10)), 3)


if __name__ == "__main__":
    unittest.main()
