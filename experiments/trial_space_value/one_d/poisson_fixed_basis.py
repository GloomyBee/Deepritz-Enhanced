from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.shape_validation.one_d.common import (
    AUX_COLORS,
    FIGURE_DPI,
    LEARNED_COLOR,
    MAIN_FIGURE_SIZE,
)
from experiments.shape_validation.one_d.basis import (
    MeshfreeKAN1D,
    compute_model_shape_and_gradients_1d,
    gauss_legendre_interval_1d,
)


def poisson_exact_solution_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.sin(np.pi * x)


def poisson_exact_gradient_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.pi * np.cos(np.pi * x)


def poisson_source_term_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return (np.pi**2) * np.sin(np.pi * x)


def solve_fixed_basis_poisson_1d(
    model: MeshfreeKAN1D,
    *,
    quadrature_order: int,
    eval_resolution: int,
    device: str,
) -> dict[str, Any]:
    quad_points, quad_weights = gauss_legendre_interval_1d(quadrature_order)
    quad_shape = compute_model_shape_and_gradients_1d(
        model=model,
        points=quad_points,
        device=device,
    )
    phi = quad_shape["normalized"]
    grad_phi = quad_shape["grad_normalized"][:, :, 0]

    stiffness = (grad_phi * quad_weights[:, None]).T @ grad_phi
    rhs = (phi * (poisson_source_term_1d(quad_points) * quad_weights)[:, None]).sum(axis=0)

    boundary_x = np.array([0.0, 1.0], dtype=np.float64)
    boundary_shape = compute_model_shape_and_gradients_1d(
        model=model,
        points=boundary_x,
        device=device,
    )
    boundary_matrix = boundary_shape["normalized"]
    boundary_values = np.zeros(2, dtype=np.float64)

    zero_bc = np.zeros((boundary_matrix.shape[0], boundary_matrix.shape[0]), dtype=np.float64)
    kkt = np.block(
        [
            [stiffness, boundary_matrix.T],
            [boundary_matrix, zero_bc],
        ]
    )
    rhs_kkt = np.concatenate([rhs, boundary_values], axis=0)

    try:
        solution = np.linalg.solve(kkt, rhs_kkt)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(kkt, rhs_kkt, rcond=1.0e-12)

    coeffs = solution[: model.n_nodes]
    lagrange = solution[model.n_nodes :]
    residual = stiffness @ coeffs + boundary_matrix.T @ lagrange - rhs
    bc_residual = boundary_matrix @ coeffs - boundary_values

    x_eval = np.linspace(0.0, 1.0, eval_resolution, dtype=np.float64)
    eval_shape = compute_model_shape_and_gradients_1d(
        model=model,
        points=x_eval,
        device=device,
    )
    u_pred = eval_shape["normalized"] @ coeffs
    du_pred = eval_shape["grad_normalized"][:, :, 0] @ coeffs
    u_exact = poisson_exact_solution_1d(x_eval)
    du_exact = poisson_exact_gradient_1d(x_eval)
    abs_error = np.abs(u_pred - u_exact)

    metrics = {
        "l2_error": float(np.sqrt(np.mean((u_pred - u_exact) ** 2))),
        "h1_semi_error": float(np.sqrt(np.mean((du_pred - du_exact) ** 2))),
        "boundary_error": float(np.linalg.norm(boundary_matrix @ coeffs - boundary_values)),
        "bc_residual_norm": float(np.linalg.norm(bc_residual)),
        "solver_residual_norm": float(np.linalg.norm(residual)),
        "condition_number": float(np.linalg.cond(kkt)),
    }
    arrays = {
        "x_eval": x_eval,
        "u_pred": u_pred,
        "u_exact": u_exact,
        "u_abs_error": abs_error,
        "du_pred": du_pred,
        "du_exact": du_exact,
        "quad_points": quad_points,
        "quad_weights": quad_weights,
        "quad_phi": phi,
        "quad_grad_phi": grad_phi,
        "boundary_x": boundary_x,
        "boundary_matrix": boundary_matrix,
        "coeffs": coeffs,
        "lagrange": lagrange,
        "solver_iterations": np.array([0.0], dtype=np.float64),
        "solver_residual_norm": np.array([metrics["solver_residual_norm"]], dtype=np.float64),
        "bc_residual_norm": np.array([metrics["bc_residual_norm"]], dtype=np.float64),
    }
    return {"metrics": metrics, "arrays": arrays}


def build_poisson_summary_lines_1d(case: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    return [
        f"case: {case}",
        f"l2_error: {metrics['l2_error']:.6e}",
        f"h1_semi_error: {metrics['h1_semi_error']:.6e}",
        f"boundary_error: {metrics['boundary_error']:.6e}",
        f"bc_residual_norm: {metrics['bc_residual_norm']:.6e}",
        f"solver_residual_norm: {metrics['solver_residual_norm']:.6e}",
        f"condition_number: {metrics['condition_number']:.6e}",
    ]


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

