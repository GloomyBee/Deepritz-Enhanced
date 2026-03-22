from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.shape_validation.one_d.plotting import (
    plot_diagnostic_shape_consistency_1d,
    plot_diagnostic_shape_overlay_1d,
)
from experiments.trial_space_value.one_d.common import ROOT_DIR
from experiments.shape_validation.one_d.common import (
    AUX_COLORS,
    FIGURE_DPI,
    LEARNED_COLOR,
    MAIN_FIGURE_SIZE,
)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, *, log_scale: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def plot_main_figure_poisson_1d(
    *,
    x_eval: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    solver_iterations: np.ndarray,
    solver_residual_norm: np.ndarray,
    metrics: dict[str, Any],
    path: Path,
) -> None:
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    u_pred = np.asarray(u_pred, dtype=np.float64).reshape(-1)
    u_exact = np.asarray(u_exact, dtype=np.float64).reshape(-1)
    solver_iterations = np.asarray(solver_iterations, dtype=np.float64).reshape(-1)
    solver_residual_norm = np.asarray(solver_residual_norm, dtype=np.float64).reshape(-1)
    abs_error = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)

    ax = axes[0, 0]
    ax.plot(x_eval, u_pred, color=LEARNED_COLOR, lw=1.9)
    _style_axis(ax, "Predicted solution", "x", "u_h")

    ax = axes[0, 1]
    ax.plot(x_eval, u_exact, color=AUX_COLORS[2], lw=1.9)
    _style_axis(ax, "Exact solution", "x", "u")

    ax = axes[1, 0]
    ax.semilogy(x_eval, np.maximum(abs_error, 1.0e-16), color=AUX_COLORS[1], lw=1.9)
    _style_axis(ax, "Absolute error", "x", "|u_h-u|")
    text = (
        f"L2={float(metrics.get('l2_error', 0.0)):.2e}\n"
        f"H1-semi={float(metrics.get('h1_semi_error', 0.0)):.2e}\n"
        f"BC res={float(metrics.get('bc_residual_norm', metrics.get('boundary_error', 0.0))):.2e}\n"
        f"cond={float(metrics.get('condition_number', 0.0)):.2e}"
    )
    ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)

    ax = axes[1, 1]
    if solver_iterations.size > 1:
        ax.semilogy(
            solver_iterations,
            np.maximum(solver_residual_norm, 1.0e-16),
            "o-",
            color=LEARNED_COLOR,
            lw=1.8,
            ms=6,
        )
    else:
        ax.scatter(solver_iterations, np.maximum(solver_residual_norm, 1.0e-16), color=LEARNED_COLOR, s=36)
        ax.text(0.03, 0.88, "Direct KKT solve", transform=ax.transAxes, fontsize=9)
    _style_axis(ax, "Solver residual history", "iteration", "residual", log_scale=True)

    _save_figure(fig, path)
