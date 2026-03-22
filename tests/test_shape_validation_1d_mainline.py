import shutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.trial_space_value.one_d.common import resolve_repo_root
from experiments.trial_space_value.one_d.plotting import plot_main_figure_poisson_1d
from experiments.trial_space_value.one_d.trial_space import build_poisson_summary_lines_1d, solve_fixed_basis_poisson_1d
from experiments.shape_validation.one_d.basis import (
    MeshfreeKAN1D,
    build_shape_validation_metrics_payload_1d,
    generate_interval_nodes,
    generate_nonuniform_interval_nodes,
)
from experiments.shape_validation.one_d.plotting import (
    plot_diagnostic_shape_overlay_1d,
    plot_main_figure_shape_case_1d,
    plot_summary_figure_shape_seeds_1d,
)
from experiments.shape_validation.one_d.training import (
    train_phase_a_distill,
    train_phase_a_raw_pu,
)


class MainlineShapeFigureTests(unittest.TestCase):
    def _prepare_dir(self, name: str) -> Path:
        path = ROOT_DIR / "output" / "_test_artifacts" / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_generate_nonuniform_nodes_preserves_endpoints_and_order(self):
        nodes, h = generate_nonuniform_interval_nodes(n_nodes=9, jitter_factor=0.4, seed=42)
        self.assertEqual(nodes.shape, (9,))
        self.assertAlmostEqual(nodes[0], 0.0)
        self.assertAlmostEqual(nodes[-1], 1.0)
        self.assertTrue(np.all(np.diff(nodes) > 0.0))
        self.assertAlmostEqual(h, 1.0 / 8.0)

    def test_train_phase_a_raw_pu_uses_normalized_shape_for_boundary(self):
        nodes, h = generate_interval_nodes(n_nodes=7)
        model = MeshfreeKAN1D(
            nodes=torch.tensor(nodes, dtype=torch.float64),
            support_radius=2.0 * h,
            hidden_dim=8,
            use_softplus=True,
        )
        calls = []
        original = model.compute_shape_functions

        def wrapped(self, x, *args, **kwargs):
            calls.append((tuple(x.shape), kwargs.get("return_stage", "normalized")))
            return original(x, *args, **kwargs)

        model.compute_shape_functions = types.MethodType(wrapped, model)
        train_phase_a_raw_pu(
            model=model,
            nodes=model.nodes,
            device="cpu",
            steps=1,
            batch_size=16,
            lr=1.0e-3,
            lambda_pu=0.0,
            lambda_bd=1.0,
            lambda_reg=0.0,
            pu_on_raw=True,
            log_interval=1,
        )
        self.assertIn(((2, 1), "normalized"), calls)

    def test_train_phase_a_distill_keeps_linear_and_pu_as_diagnostics(self):
        nodes, h = generate_interval_nodes(n_nodes=7)
        model = MeshfreeKAN1D(
            nodes=torch.tensor(nodes, dtype=torch.float64),
            support_radius=2.0 * h,
            hidden_dim=8,
            use_softplus=False,
        )
        history = train_phase_a_distill(
            model=model,
            nodes=model.nodes,
            device="cpu",
            steps=1,
            batch_size=16,
            lr=1.0e-3,
            lambda_teacher=0.0,
            lambda_bd=0.0,
            lambda_reg=0.0,
            log_interval=1,
        )
        self.assertAlmostEqual(history["loss"][0], 0.0, places=12)
        self.assertGreaterEqual(history["linear"][0], 0.0)
        self.assertGreaterEqual(history["pu"][0], 0.0)

    def test_resolve_repo_root_handles_nested_entry_script(self):
        repo_root = ROOT_DIR.parents[0]
        script_path = (
            repo_root
            / "experiments"
            / "shape_validation"
            / "one_d"
            / "uniform_nodes"
            / "deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py"
        )
        self.assertEqual(resolve_repo_root(script_path), repo_root)

    def test_only_uniform_and_nonuniform_remain_active_in_one_d(self):
        repo_root = ROOT_DIR.parents[0]
        active_uniform = repo_root / 'experiments' / 'shape_validation' / 'one_d' / 'uniform_nodes' / 'deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py'
        active_nonuniform = repo_root / 'experiments' / 'shape_validation' / 'one_d' / 'nonuniform_nodes' / 'deepritz_meshfree_kan_rkpm_1d_nonuniform_nodes_v1.py'
        retired_stability = repo_root / 'experiments' / 'shape_validation' / 'one_d' / 'stability_ablation'
        self.assertTrue(active_uniform.is_file())
        self.assertTrue(active_nonuniform.is_file())
        self.assertFalse(retired_stability.exists())

    def test_mainline_shape_figures_write_files(self):
        root = self._prepare_dir("mainline_shape_outputs")
        x_eval = np.linspace(0.0, 1.0, 201, dtype=np.float64)
        nodes = np.linspace(0.0, 1.0, 7, dtype=np.float64)
        phi_rkpm = np.exp(-((x_eval[:, None] - nodes[None, :]) ** 2) / 0.02)
        phi_rkpm = phi_rkpm / np.sum(phi_rkpm, axis=1, keepdims=True)
        phi_learned = phi_rkpm + 0.01 * np.sin(2.0 * np.pi * x_eval)[:, None]
        phi_learned = np.maximum(phi_learned, 0.0)
        phi_learned = phi_learned / np.sum(phi_learned, axis=1, keepdims=True)
        history = {
            "steps": [0.0, 10.0, 20.0],
            "loss": [1.0, 0.2, 0.05],
            "teacher": [0.8, 0.1, 0.02],
            "linear": [0.3, 0.08, 0.02],
            "pu": [0.1, 0.03, 0.01],
            "bd": [0.05, 0.02, 0.01],
            "reg": [0.02, 0.01, 0.005],
        }
        payload = build_shape_validation_metrics_payload_1d(
            case={"seed": 42},
            learned_shape={
                "shape_relative_l2": 1.0e-2,
                "global_l2": 9.0e-3,
                "center_l2": 8.0e-3,
                "boundary_l2": 1.2e-2,
                "pu_max_error": 2.0e-3,
                "linear_reproduction_rmse": 5.0e-3,
                "center_node_index": 3,
                "boundary_node_index": 0,
            },
            rkpm_shape={
                "shape_relative_l2": 0.0,
                "global_l2": 0.0,
                "center_l2": 0.0,
                "boundary_l2": 0.0,
                "pu_max_error": 0.0,
                "linear_reproduction_rmse": 0.0,
                "center_node_index": 3,
                "boundary_node_index": 0,
            },
            learned_consistency={
                "mass_sum_residual": 1.0e-4,
                "moment_matrix_residual_fro": 1.0e-3,
                "derivative_matrix_residual_fro": 2.0e-3,
            },
            rkpm_consistency={
                "mass_sum_residual": 1.0e-12,
                "moment_matrix_residual_fro": 1.0e-12,
                "derivative_matrix_residual_fro": 1.0e-12,
            },
        )
        plot_main_figure_shape_case_1d(
            x_eval=x_eval,
            nodes=nodes,
            phi_learned=phi_learned,
            phi_rkpm=phi_rkpm,
            history=history,
            metrics_payload=payload,
            path=root / "main_figure.png",
        )
        plot_diagnostic_shape_overlay_1d(
            x_eval=x_eval,
            phi_rkpm=phi_rkpm,
            phi_kan=phi_learned,
            path=root / "shape_subset_overlay.png",
        )
        plot_summary_figure_shape_seeds_1d(
            [
                {"metrics": payload},
                {"metrics": {**payload, "case": {"seed": 43}}},
            ],
            root / "summary_figure.png",
        )
        self.assertTrue((root / "main_figure.png").is_file())
        self.assertTrue((root / "shape_subset_overlay.png").is_file())
        self.assertTrue((root / "summary_figure.png").is_file())

    def test_poisson_main_figure_writes_file(self):
        root = self._prepare_dir("mainline_poisson_outputs")
        x_eval = np.linspace(0.0, 1.0, 201, dtype=np.float64)
        u_exact = np.sin(np.pi * x_eval)
        u_pred = u_exact + 0.01 * np.sin(3.0 * np.pi * x_eval)
        plot_main_figure_poisson_1d(
            x_eval=x_eval,
            u_pred=u_pred,
            u_exact=u_exact,
            solver_iterations=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            solver_residual_norm=np.array([1.0, 0.2, 0.04, 0.01], dtype=np.float64),
            metrics={
                "l2_error": 1.0e-2,
                "h1_semi_error": 2.0e-2,
                "boundary_error": 1.0e-6,
                "bc_residual_norm": 1.0e-12,
                "solver_residual_norm": 1.0e-9,
                "condition_number": 1.0e3,
            },
            path=root / "main_figure.png",
        )
        self.assertTrue((root / "main_figure.png").is_file())

    def test_poisson_summary_reads_shape_only_basis_quality_payload(self):
        summary_lines = build_poisson_summary_lines_1d(
            {
                "case": {"seed": 42, "method": "fixed_basis"},
                "method": "fixed_basis",
                "basis_quality": {
                    "shape_relative_l2": 1.0e-2,
                    "linear_reproduction_rmse": 5.0e-3,
                },
                "trial_space": {
                    "l2_error": 2.0e-2,
                    "h1_semi_error": 3.0e-2,
                    "boundary_error": 1.0e-6,
                    "bc_residual_norm": 1.0e-12,
                    "solver_residual_norm": 1.0e-9,
                    "condition_number": 1.0e3,
                },
            }
        )
        self.assertTrue(any("basis_shape_relative_l2" in line for line in summary_lines))
        self.assertTrue(any("basis_linear_reproduction_rmse" in line for line in summary_lines))

    def test_kkt_poisson_solver_enforces_boundary_constraints(self):
        nodes, h = generate_interval_nodes(n_nodes=9)
        model = MeshfreeKAN1D(
            nodes=torch.tensor(nodes, dtype=torch.float64),
            support_radius=2.0 * h,
            hidden_dim=8,
            use_softplus=True,
        )
        result = solve_fixed_basis_poisson_1d(
            model=model,
            quadrature_order=10,
            eval_resolution=101,
            device="cpu",
        )
        self.assertLess(result["metrics"]["bc_residual_norm"], 1.0e-8)
        self.assertTrue(np.isfinite(result["metrics"]["solver_residual_norm"]))


if __name__ == "__main__":
    unittest.main()




