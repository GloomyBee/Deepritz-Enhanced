import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_1d_validation.common import (
    analytic_derivative_moment_matrices_1d,
    analytic_moment_matrix_1d,
    build_consistency_metrics_payload as build_consistency_metrics_payload_1d,
    evaluate_rkpm_consistency_bundle_1d,
    gauss_legendre_interval_1d,
    generate_interval_nodes,
)
from examples.meshfree_kan_rkpm_2d_validation.common import (
    analytic_derivative_moment_matrices_2d,
    analytic_moment_matrix_2d,
    build_consistency_metrics_payload as build_consistency_metrics_payload_2d,
    evaluate_rkpm_consistency_bundle_2d,
    generate_square_nodes,
    square_boundary_quadrature,
    square_domain_quadrature,
)


class ConsistencyCommonTests(unittest.TestCase):
    def test_interval_gauss_weights_sum_to_length(self):
        _, weights = gauss_legendre_interval_1d(order=4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_square_domain_quadrature_weights_sum_to_area(self):
        _, weights = square_domain_quadrature(order=4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_square_boundary_quadrature_weights_sum_to_perimeter_and_normals(self):
        points, weights, normals = square_boundary_quadrature(order=4)
        self.assertEqual(points.shape, (16, 2))
        self.assertEqual(weights.shape, (16,))
        self.assertEqual(normals.shape, (16, 2))
        self.assertAlmostEqual(float(np.sum(weights)), 4.0)
        expected_normals = {
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
        }
        self.assertEqual({tuple(item) for item in normals.tolist()}, expected_normals)

    def test_analytic_moment_matrices_match_closed_form(self):
        np.testing.assert_allclose(
            analytic_moment_matrix_1d(),
            np.array([[1.0, 0.5], [0.5, 1.0 / 3.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            analytic_moment_matrix_2d(),
            np.array(
                [
                    [1.0, 0.5, 0.5],
                    [0.5, 1.0 / 3.0, 0.25],
                    [0.5, 0.25, 1.0 / 3.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_analytic_derivative_moment_matrices_match_closed_form(self):
        np.testing.assert_allclose(
            analytic_derivative_moment_matrices_1d(),
            np.array([[[0.0, 0.0], [1.0, 0.5]]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            analytic_derivative_moment_matrices_2d(),
            np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.5, 0.5],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.5, 0.5],
                    ],
                ],
                dtype=np.float64,
            ),
        )

    def test_rkpm_consistency_bundle_1d_shapes_and_finite_metrics(self):
        nodes, h = generate_interval_nodes(n_nodes=7)
        bundle = evaluate_rkpm_consistency_bundle_1d(
            nodes=nodes,
            support_radius=2.0 * h,
            quadrature_order=6,
        )
        self.assertIn("metrics", bundle)
        self.assertIn("arrays", bundle)
        self.assertEqual(bundle["arrays"]["boundary_residuals"].shape, (7, 1, 2))
        self.assertEqual(bundle["arrays"]["derivative_matrix_residual"].shape, (1, 2, 2))
        self.assertEqual(bundle["arrays"]["phi"].shape[1], 7)
        self.assertTrue(np.isfinite(bundle["metrics"]["moment_matrix_residual_fro"]))
        self.assertTrue(np.isfinite(bundle["metrics"]["derivative_boundary_residual_max"]))

    def test_rkpm_consistency_bundle_2d_shapes_and_finite_metrics(self):
        nodes, h = generate_square_nodes(n_side=5)
        bundle = evaluate_rkpm_consistency_bundle_2d(
            nodes=nodes,
            support_radius=2.5 * h,
            quadrature_order=4,
        )
        self.assertIn("metrics", bundle)
        self.assertIn("arrays", bundle)
        self.assertEqual(bundle["arrays"]["boundary_residuals"].shape, (25, 2, 3))
        self.assertEqual(bundle["arrays"]["derivative_matrix_residual"].shape, (2, 3, 3))
        self.assertEqual(bundle["arrays"]["phi"].shape[1], 25)
        self.assertTrue(np.isfinite(bundle["metrics"]["moment_matrix_residual_fro"]))
        self.assertTrue(np.isfinite(bundle["metrics"]["derivative_boundary_residual_max"]))

    def test_metrics_payload_structure_matches_between_1d_and_2d(self):
        case = {"dimension": 1, "seed": 42}
        learned = {"mass_sum_residual": 0.1, "moment_matrix_residual_fro": 0.2}
        rkpm = {"mass_sum_residual": 0.01, "moment_matrix_residual_fro": 0.02}
        payload_1d = build_consistency_metrics_payload_1d(case=case, learned=learned, rkpm=rkpm)
        payload_2d = build_consistency_metrics_payload_2d(case=case, learned=learned, rkpm=rkpm)

        self.assertEqual(set(payload_1d.keys()), {"case", "learned", "rkpm", "comparison"})
        self.assertEqual(set(payload_2d.keys()), {"case", "learned", "rkpm", "comparison"})
        self.assertEqual(set(payload_1d["comparison"].keys()), set(payload_2d["comparison"].keys()))
        self.assertTrue(
            all(
                key.startswith("learned_minus_rkpm_") or key.startswith("learned_over_rkpm_")
                for key in payload_1d["comparison"].keys()
            )
        )


if __name__ == "__main__":
    unittest.main()
