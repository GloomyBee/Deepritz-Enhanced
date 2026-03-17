import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_2d_trial_space_value.common import (
    assemble_poisson_penalty_system,
    compute_matrix_stats,
    gauss_legendre_1d,
    square_boundary_quadrature,
    square_domain_quadrature,
)


class TrialSpaceCommonTests(unittest.TestCase):
    def test_gauss_legendre_interval_weights_sum_to_one(self):
        _, weights = gauss_legendre_1d(order=4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_square_domain_quadrature_weights_sum_to_area(self):
        _, weights = square_domain_quadrature(order=4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_square_boundary_quadrature_weights_sum_to_perimeter(self):
        _, weights = square_boundary_quadrature(order=4)
        self.assertAlmostEqual(float(np.sum(weights)), 4.0)

    def test_assemble_poisson_penalty_system_matches_manual_result(self):
        phi_domain = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
        grad_phi_domain = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float64,
        )
        forcing = np.array([1.0, 2.0], dtype=np.float64)
        domain_weights = np.array([0.5, 0.5], dtype=np.float64)
        phi_boundary = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        boundary_values = np.array([0.0, 1.0], dtype=np.float64)
        boundary_weights = np.array([0.25, 0.25], dtype=np.float64)

        matrix, rhs = assemble_poisson_penalty_system(
            phi_domain=phi_domain,
            grad_phi_domain=grad_phi_domain,
            forcing=forcing,
            domain_weights=domain_weights,
            phi_boundary=phi_boundary,
            boundary_values=boundary_values,
            boundary_weights=boundary_weights,
            beta_bc=10.0,
        )

        expected_matrix = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        expected_rhs = np.array([0.6, 3.4], dtype=np.float64)
        np.testing.assert_allclose(matrix, expected_matrix)
        np.testing.assert_allclose(rhs, expected_rhs)

    def test_compute_matrix_stats_reports_condition_and_extremal_eigs(self):
        matrix = np.diag([2.0, 8.0])
        stats = compute_matrix_stats(matrix)
        self.assertAlmostEqual(stats["cond_k"], 4.0)
        self.assertAlmostEqual(stats["lambda_min_k"], 2.0)
        self.assertAlmostEqual(stats["lambda_max_k"], 8.0)


if __name__ == "__main__":
    unittest.main()
