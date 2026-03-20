import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.one_d.plotting import (
    ensure_legacy_figure_artifacts_1d,
    plot_diagnostic_shape_overlay_1d,
    plot_main_figure_shape_validation_1d,
)


class LegacyShapeFigureTests(unittest.TestCase):
    def _prepare_dir(self, name: str) -> Path:
        path = ROOT_DIR / "output" / "_test_artifacts" / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_ensure_legacy_figure_artifacts_creates_expected_dirs(self):
        root = self._prepare_dir("legacy_figure_dirs")
        figures_dir, diagnostics_dir = ensure_legacy_figure_artifacts_1d(root)
        self.assertTrue(figures_dir.is_dir())
        self.assertTrue(diagnostics_dir.is_dir())
        self.assertEqual(diagnostics_dir.parent, figures_dir)

    def test_shape_validation_helpers_write_main_and_diagnostic_figures(self):
        root = self._prepare_dir("legacy_figure_outputs")
        figures_dir, diagnostics_dir = ensure_legacy_figure_artifacts_1d(root)
        x_eval = np.linspace(0.0, 1.0, 201, dtype=np.float64)
        nodes = np.linspace(0.0, 1.0, 7, dtype=np.float64)
        phi_rkpm = np.exp(-((x_eval[:, None] - nodes[None, :]) ** 2) / 0.02)
        phi_rkpm = phi_rkpm / np.sum(phi_rkpm, axis=1, keepdims=True)
        phi_kan = phi_rkpm + 0.01 * np.sin(2.0 * np.pi * x_eval)[:, None]
        phi_kan = np.maximum(phi_kan, 0.0)
        phi_kan = phi_kan / np.sum(phi_kan, axis=1, keepdims=True)
        history = {
            "steps": [0, 10, 20],
            "loss": [1.0, 0.2, 0.05],
            "linear": [0.3, 0.08, 0.02],
            "pu": [0.1, 0.03, 0.01],
        }
        metrics = {
            "global_l2": 1.0e-2,
            "center_l2": 8.0e-3,
            "boundary_l2": 1.2e-2,
            "pu_max_error": 2.0e-3,
            "linear_reproduction_rmse": 5.0e-3,
        }
        plot_main_figure_shape_validation_1d(
            x_eval=x_eval,
            phi_rkpm=phi_rkpm,
            phi_kan=phi_kan,
            history=history,
            center_idx=3,
            boundary_idx=0,
            metrics=metrics,
            path=figures_dir / "main_figure.png",
        )
        plot_diagnostic_shape_overlay_1d(
            x_eval=x_eval,
            phi_rkpm=phi_rkpm,
            phi_kan=phi_kan,
            path=diagnostics_dir / "shape_subset_overlay.png",
        )
        self.assertTrue((figures_dir / "main_figure.png").is_file())
        self.assertTrue((diagnostics_dir / "shape_subset_overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()

