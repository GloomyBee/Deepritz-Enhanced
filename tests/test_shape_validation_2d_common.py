import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.two_d.common import (
    build_case_name,
    compute_lambda_h_stats,
    generate_square_nodes,
    merge_histories,
    rkpm_shape_matrix_2d,
    resolve_variant_config,
)


class MeshfreeKAN2DCommonTests(unittest.TestCase):
    def test_build_case_name_orders_tokens(self):
        case_name = build_case_name(
            variant="teacher",
            n_side=15,
            kappa=2.5,
            seed=42,
            tag="pilot",
        )
        self.assertEqual(case_name, "variant_teacher_ns15_k2p5_seed42_pilot")

    def test_generate_square_nodes_shape_and_spacing(self):
        nodes, h = generate_square_nodes(n_side=5)
        self.assertEqual(nodes.shape, (25, 2))
        self.assertAlmostEqual(h, 0.25)
        self.assertTrue(np.all(nodes >= 0.0))
        self.assertTrue(np.all(nodes <= 1.0))

    def test_lambda_h_stats_match_absolute_sum_definition(self):
        phi = np.array([
            [0.5, 0.5, 0.0],
            [1.2, -0.2, 0.0],
        ])
        stats = compute_lambda_h_stats(phi)
        self.assertAlmostEqual(stats["lambda_h_max"], 1.4)
        self.assertAlmostEqual(stats["lambda_h_mean"], 1.2)

    def test_rkpm_shape_matrix_reproduces_partition_and_linear_fields(self):
        nodes, h = generate_square_nodes(n_side=5)
        support_radius = 2.5 * h
        x_eval = np.array([
            [0.2, 0.3],
            [0.5, 0.5],
            [0.8, 0.4],
        ])

        phi = rkpm_shape_matrix_2d(x_eval=x_eval, nodes=nodes, support_radius=support_radius)

        self.assertEqual(phi.shape, (3, 25))
        np.testing.assert_allclose(np.sum(phi, axis=1), np.ones(3), atol=1e-8)
        np.testing.assert_allclose(phi @ nodes[:, 0], x_eval[:, 0], atol=1e-6)
        np.testing.assert_allclose(phi @ nodes[:, 1], x_eval[:, 1], atol=1e-6)

    def test_merge_histories_shifts_phase_b_steps(self):
        history_a = {"steps": [0.0, 2.0], "loss": [1.0, 0.5], "linear": [0.1, 0.05]}
        history_b = {"steps": [0.0, 3.0], "loss": [0.4, 0.2], "energy": [-1.0, -1.2]}
        merged = merge_histories(history_a, history_b)
        self.assertEqual(merged["steps"], [0.0, 2.0, 3.0, 6.0])
        self.assertEqual(merged["loss"], [1.0, 0.5, 0.4, 0.2])
        self.assertEqual(merged["linear"], [0.1, 0.05])
        self.assertEqual(merged["energy"], [-1.0, -1.2])

    def test_no_fallback_variant_config(self):
        config = resolve_variant_config("no_softplus_raw_pu_bd_no_fallback")
        self.assertFalse(config["enable_fallback"])
        self.assertFalse(config["use_softplus"])


if __name__ == "__main__":
    unittest.main()

