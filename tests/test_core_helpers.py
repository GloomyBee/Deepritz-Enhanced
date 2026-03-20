import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.io_or_artifacts import ensure_run_artifacts, ensure_trial_space_artifacts
from core.numerics.quadrature import (
    gauss_legendre_interval_1d,
    square_boundary_quadrature,
    square_domain_quadrature,
)
from core.plotting import save_figure
from core.utils import parse_float_list, parse_int_list, resolve_repo_root, safe_token


class CoreHelperTests(unittest.TestCase):
    def test_resolve_repo_root_finds_workspace_root(self):
        self.assertEqual(resolve_repo_root(Path(__file__).resolve()), ROOT_DIR)

    def test_safe_token_and_parse_helpers(self):
        self.assertEqual(safe_token("2.5 / demo"), "2p5___demo")
        self.assertEqual(parse_int_list("1, 2,3"), [1, 2, 3])
        self.assertEqual(parse_float_list("1.0, 2.5"), [1.0, 2.5])

    def test_run_and_trial_artifacts_create_expected_dirs(self):
        run_artifacts = ensure_run_artifacts(ROOT_DIR, "_test_artifacts", "run_case")
        trial_artifacts = ensure_trial_space_artifacts(ROOT_DIR, "_test_artifacts", "trial_case")
        self.assertTrue(run_artifacts.figures_dir.is_dir())
        self.assertTrue(run_artifacts.diagnostics_dir.is_dir())
        self.assertTrue(trial_artifacts.figures_dir.is_dir())
        self.assertTrue(trial_artifacts.diagnostics_dir.is_dir())
        self.assertTrue(trial_artifacts.methods_dir.is_dir())

    def test_quadrature_helpers_match_measure(self):
        _, interval_weights = gauss_legendre_interval_1d(order=4)
        _, domain_weights = square_domain_quadrature(order=4)
        _, boundary_weights, normals = square_boundary_quadrature(order=4)
        self.assertAlmostEqual(float(np.sum(interval_weights)), 1.0)
        self.assertAlmostEqual(float(np.sum(domain_weights)), 1.0)
        self.assertAlmostEqual(float(np.sum(boundary_weights)), 4.0)
        self.assertEqual(normals.shape, (16, 2))

    def test_save_figure_writes_file(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot([0, 1], [0, 1])
        path = ROOT_DIR / "output" / "_test_artifacts" / "core_plotting" / "figure.png"
        save_figure(fig, path, dpi=120)
        self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
