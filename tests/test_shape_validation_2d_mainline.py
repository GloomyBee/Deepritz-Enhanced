import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.two_d.common import generate_square_nodes, grid_points, rkpm_shape_matrix_2d
from experiments.shape_validation.two_d.shape_validation import (
    build_shape_validation_metrics_payload_2d,
    evaluate_shape_metrics_2d,
    plot_diagnostic_shape_representatives_2d,
    plot_main_figure_shape_case_2d,
    plot_summary_figure_shape_jitters_2d,
)


class ShapeValidation2DMainlineTests(unittest.TestCase):
    def _prepare_dir(self, name: str) -> Path:
        path = ROOT_DIR / "tests" / "output" / "_test_artifacts" / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_active_shape_validation_entrypoints_are_reduced_to_uniform_and_irregular(self):
        uniform_entry = (
            ROOT_DIR
            / "experiments"
            / "shape_validation"
            / "two_d"
            / "uniform_nodes"
            / "deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py"
        )
        irregular_entry = (
            ROOT_DIR
            / "experiments"
            / "shape_validation"
            / "two_d"
            / "irregular_nodes"
            / "deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py"
        )
        retired_patch_test = (
            ROOT_DIR
            / "experiments"
            / "shape_validation"
            / "two_d"
            / "patch_test"
            / "deepritz_meshfree_kan_rkpm_2d_patch_test_v1.py"
        )
        self.assertTrue(uniform_entry.is_file())
        self.assertTrue(irregular_entry.is_file())
        self.assertFalse(retired_patch_test.exists())

    def test_shape_metrics_2d_expose_representative_cuts(self):
        nodes, h = generate_square_nodes(n_side=5)
        _x_grid, _y_grid, x_eval = grid_points(21)
        phi_rkpm = rkpm_shape_matrix_2d(x_eval=x_eval, nodes=nodes, support_radius=2.5 * h)
        perturb = 5.0e-3 * np.sin(np.pi * x_eval[:, :1]) * np.cos(np.pi * x_eval[:, 1:])
        phi_learned = np.maximum(phi_rkpm + perturb, 0.0)
        phi_learned = phi_learned / np.sum(phi_learned, axis=1, keepdims=True)

        metrics, arrays = evaluate_shape_metrics_2d(
            x_eval=x_eval,
            nodes=nodes,
            phi_learned=phi_learned,
            phi_rkpm=phi_rkpm,
        )

        self.assertEqual(set(metrics["representative_labels"]), {"corner", "edge_mid", "center"})
        self.assertEqual(len(metrics["representative_node_indices"]), 3)
        self.assertEqual(arrays["cut_x"].shape[0], 3)
        self.assertEqual(arrays["cut_learned"].shape, arrays["cut_rkpm"].shape)
        self.assertEqual(arrays["cut_abs_error"].shape, arrays["cut_rkpm"].shape)
        self.assertEqual(arrays["pointwise_relative_error"].shape, (x_eval.shape[0],))

    def test_shape_validation_2d_plotting_writes_main_summary_and_diagnostics(self):
        root = self._prepare_dir("shape_validation_2d_mainline")
        nodes, h = generate_square_nodes(n_side=5)
        x_grid, y_grid, x_eval = grid_points(21)
        phi_rkpm = rkpm_shape_matrix_2d(x_eval=x_eval, nodes=nodes, support_radius=2.5 * h)
        perturb = 5.0e-3 * np.sin(np.pi * x_eval[:, :1]) * np.cos(np.pi * x_eval[:, 1:])
        phi_learned = np.maximum(phi_rkpm + perturb, 0.0)
        phi_learned = phi_learned / np.sum(phi_learned, axis=1, keepdims=True)
        learned_shape, shape_arrays = evaluate_shape_metrics_2d(
            x_eval=x_eval,
            nodes=nodes,
            phi_learned=phi_learned,
            phi_rkpm=phi_rkpm,
        )
        payload = build_shape_validation_metrics_payload_2d(
            case={"dimension": 2, "seed": 42, "layout": "uniform"},
            learned_shape=learned_shape,
            rkpm_shape={
                "global_l2": 0.0,
                "shape_relative_l2": 0.0,
                "corner_l2": 0.0,
                "edge_mid_l2": 0.0,
                "center_l2": 0.0,
                "pu_max_error": 0.0,
                "linear_x_rmse": 0.0,
                "linear_y_rmse": 0.0,
                "lambda_h_max": float(np.max(np.sum(np.abs(phi_rkpm), axis=1))),
                "representative_labels": learned_shape["representative_labels"],
                "representative_node_indices": learned_shape["representative_node_indices"],
            },
            learned_consistency={
                "mass_sum_residual": 1.0e-4,
                "moment_x_residual": 2.0e-4,
                "moment_y_residual": 2.5e-4,
                "moment_matrix_residual_fro": 1.0e-3,
                "derivative_boundary_residual_max": 1.0e-3,
                "derivative_boundary_residual_fro": 2.0e-3,
                "derivative_matrix_residual_max": 3.0e-3,
                "derivative_matrix_residual_fro": 4.0e-3,
            },
            rkpm_consistency={
                "mass_sum_residual": 1.0e-12,
                "moment_x_residual": 1.0e-12,
                "moment_y_residual": 1.0e-12,
                "moment_matrix_residual_fro": 1.0e-12,
                "derivative_boundary_residual_max": 1.0e-12,
                "derivative_boundary_residual_fro": 1.0e-12,
                "derivative_matrix_residual_max": 1.0e-12,
                "derivative_matrix_residual_fro": 1.0e-12,
            },
        )
        history = {
            "steps": [0.0, 10.0, 20.0],
            "loss": [1.0, 0.2, 0.05],
            "linear": [0.3, 0.08, 0.02],
            "pu": [0.1, 0.03, 0.01],
            "bd": [0.05, 0.02, 0.01],
        }
        plot_main_figure_shape_case_2d(
            x_eval=x_eval,
            x_grid=x_grid,
            y_grid=y_grid,
            nodes=nodes,
            phi_learned=phi_learned,
            phi_rkpm=phi_rkpm,
            history=history,
            metrics_payload=payload,
            shape_arrays=shape_arrays,
            path=root / "main_figure.png",
        )
        plot_diagnostic_shape_representatives_2d(
            x_grid=x_grid,
            y_grid=y_grid,
            phi_learned=phi_learned,
            phi_rkpm=phi_rkpm,
            representative_labels=learned_shape["representative_labels"],
            representative_indices=learned_shape["representative_node_indices"],
            diagnostics_dir=root / "diagnostics",
        )
        plot_summary_figure_shape_jitters_2d(
            [
                {"jitter": 0.0, "metrics": payload},
                {"jitter": 0.1, "metrics": payload},
            ],
            root / "summary_figure.png",
        )
        self.assertTrue((root / "main_figure.png").is_file())
        self.assertTrue((root / "summary_figure.png").is_file())
        self.assertTrue((root / "diagnostics" / "representative_corner_learned.png").is_file())
        self.assertTrue((root / "diagnostics" / "representative_center_rkpm.png").is_file())


if __name__ == "__main__":
    unittest.main()
