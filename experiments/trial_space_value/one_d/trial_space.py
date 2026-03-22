from __future__ import annotations

from typing import Any

import numpy as np

from experiments.trial_space_value.one_d.basis import (
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


def build_poisson_summary_lines_1d(metrics_payload: dict[str, Any]) -> list[str]:
    case = metrics_payload["case"]
    basis_quality = metrics_payload["basis_quality"]
    trial_metrics = metrics_payload["trial_space"]
    return [
        f"method: {metrics_payload['method']}",
        f"case: {case}",
        f"basis_shape_relative_l2: {basis_quality['shape_relative_l2']:.6e}",
        f"basis_linear_reproduction_rmse: {basis_quality['linear_reproduction_rmse']:.6e}",
        f"l2_error: {trial_metrics['l2_error']:.6e}",
        f"h1_semi_error: {trial_metrics['h1_semi_error']:.6e}",
        f"boundary_error: {trial_metrics['boundary_error']:.6e}",
        f"bc_residual_norm: {trial_metrics['bc_residual_norm']:.6e}",
        f"solver_residual_norm: {trial_metrics['solver_residual_norm']:.6e}",
        f"condition_number: {trial_metrics['condition_number']:.6e}",
    ]
