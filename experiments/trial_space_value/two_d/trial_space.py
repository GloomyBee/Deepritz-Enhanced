from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.trial_space_value.two_d.basis import (
    MeshfreeKAN2D,
    SinusoidalPoissonProblem2D,
    evaluate_basis_quality_2d,
    grid_points,
    square_boundary_quadrature,
    square_domain_quadrature,
)
from experiments.trial_space_value.two_d.training import (
    history_to_arrays,
    init_solver_history,
    train_frozen_w_poisson,
    train_phase_b_poisson,
)


def clone_model(model: MeshfreeKAN2D) -> MeshfreeKAN2D:
    cloned = MeshfreeKAN2D(
        nodes=model.nodes.detach().clone(),
        support_radius=model.support_radius,
        hidden_dim=model.kan.hidden_dim,
        use_softplus=model.use_softplus,
        enable_fallback=model.enable_fallback,
    ).to(device=model.nodes.device, dtype=model.nodes.dtype)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    cloned.load_state_dict(state)
    return cloned


def set_coefficients(model: MeshfreeKAN2D, coeffs: np.ndarray) -> None:
    coeffs_t = torch.tensor(coeffs.reshape(-1, 1), device=model.w.device, dtype=model.w.dtype)
    with torch.no_grad():
        model.w.copy_(coeffs_t)


def compute_shape_and_gradients(
    model: MeshfreeKAN2D,
    points: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    points_t = torch.tensor(points, device=device, dtype=model.nodes.dtype, requires_grad=True)
    phi = model.compute_shape_functions(points_t)
    grad_phi = torch.zeros(
        (points_t.shape[0], model.n_nodes, points_t.shape[1]),
        device=device,
        dtype=model.nodes.dtype,
    )
    for index in range(model.n_nodes):
        retain_graph = index + 1 < model.n_nodes
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            points_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, :] = grad_i
    return phi.detach().cpu().numpy(), grad_phi.detach().cpu().numpy()


def assemble_poisson_penalty_system(
    phi_domain: np.ndarray,
    grad_phi_domain: np.ndarray,
    forcing: np.ndarray,
    domain_weights: np.ndarray,
    phi_boundary: np.ndarray,
    boundary_values: np.ndarray,
    boundary_weights: np.ndarray,
    beta_bc: float,
) -> tuple[np.ndarray, np.ndarray]:
    k_domain = np.einsum("m,mia,mja->ij", domain_weights, grad_phi_domain, grad_phi_domain)
    k_boundary = beta_bc * np.einsum("m,mi,mj->ij", boundary_weights, phi_boundary, phi_boundary)
    f_domain = np.einsum("m,m,mi->i", domain_weights, forcing, phi_domain)
    f_boundary = beta_bc * np.einsum("m,m,mi->i", boundary_weights, boundary_values, phi_boundary)
    return k_domain + k_boundary, f_domain + f_boundary


def compute_matrix_stats(matrix: np.ndarray) -> dict[str, float]:
    sym_matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(sym_matrix)
    return {
        "cond_k": float(np.linalg.cond(sym_matrix)),
        "lambda_min_k": float(np.min(eigvals)),
        "lambda_max_k": float(np.max(eigvals)),
    }


def stabilize_symmetric_system(
    matrix: np.ndarray,
    abs_eig_floor: float = 1e-12,
    rel_eig_floor: float = 1e-10,
) -> tuple[np.ndarray, float]:
    sym_matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(sym_matrix)
    lambda_min = float(np.min(eigvals))
    lambda_scale = max(float(np.max(np.abs(eigvals))), 1.0)
    target_min = max(abs_eig_floor, rel_eig_floor * lambda_scale)
    shift = max(0.0, target_min - lambda_min)
    if shift == 0.0:
        return sym_matrix, 0.0
    regularized = sym_matrix + shift * np.eye(sym_matrix.shape[0], dtype=sym_matrix.dtype)
    return regularized, shift


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    stabilized_matrix, shift = stabilize_symmetric_system(matrix)
    try:
        coeffs = np.linalg.solve(stabilized_matrix, rhs)
        solver = "solve_spd_shifted" if shift > 0.0 else "solve_spd"
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(stabilized_matrix, rhs, rcond=None)
        solver = "lstsq_spd_shifted" if shift > 0.0 else "lstsq_spd"
    residual = float(np.linalg.norm(matrix @ coeffs - rhs))
    stabilized_residual = float(np.linalg.norm(stabilized_matrix @ coeffs - rhs))
    return coeffs, {
        "solver_name": solver,
        "solver_residual": residual,
        "solver_stabilized_residual": stabilized_residual,
        "solver_shift": shift,
    }


def assemble_trial_space_system(
    model: MeshfreeKAN2D,
    problem: Any,
    device: str,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
) -> dict[str, Any]:
    domain_points, domain_weights = square_domain_quadrature(domain_order)
    boundary_points, boundary_weights = square_boundary_quadrature(boundary_order)
    phi_domain, grad_phi_domain = compute_shape_and_gradients(model, domain_points, device)
    boundary_t = torch.tensor(boundary_points, device=device, dtype=model.nodes.dtype)
    domain_t = torch.tensor(domain_points, device=device, dtype=model.nodes.dtype)
    with torch.no_grad():
        phi_boundary = model.compute_shape_functions(boundary_t).cpu().numpy()
        boundary_values = problem.exact_solution(boundary_t).squeeze(1).cpu().numpy()
        forcing = problem.source_term(domain_t).squeeze(1).cpu().numpy()
    matrix, rhs = assemble_poisson_penalty_system(
        phi_domain=phi_domain,
        grad_phi_domain=grad_phi_domain,
        forcing=forcing,
        domain_weights=domain_weights,
        phi_boundary=phi_boundary,
        boundary_values=boundary_values,
        boundary_weights=boundary_weights,
        beta_bc=beta_bc,
    )
    stats = compute_matrix_stats(matrix)
    return {
        "matrix": matrix,
        "rhs": rhs,
        "stats": stats,
    }


def evaluate_solution_metrics(
    model: MeshfreeKAN2D,
    problem: Any,
    device: str,
    resolution: int = 81,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid, y_grid, points = grid_points(resolution)
    points_t = torch.tensor(points, device=device, dtype=model.nodes.dtype, requires_grad=True)
    u_pred = model(points_t)
    grad_pred = torch.autograd.grad(u_pred, points_t, torch.ones_like(u_pred), create_graph=False)[0]
    u_exact = problem.exact_solution(points_t)
    grad_exact = problem.exact_gradient(points_t)
    diff = u_pred - u_exact
    grad_diff = grad_pred - grad_exact
    metrics = {
        "l2_error": float(torch.sqrt(torch.mean(diff**2)).item()),
        "h1_semi_error": float(torch.sqrt(torch.mean(torch.sum(grad_diff**2, dim=1))).item()),
    }
    boundary_points, _ = square_boundary_quadrature(32)
    boundary_t = torch.tensor(boundary_points, device=device, dtype=model.nodes.dtype)
    with torch.no_grad():
        u_boundary = model(boundary_t)
        metrics["boundary_l2"] = float(
            torch.sqrt(torch.mean((u_boundary - problem.exact_solution(boundary_t)) ** 2)).item()
        )
    pred_np = u_pred.detach().cpu().numpy().reshape(resolution, resolution)
    exact_np = u_exact.detach().cpu().numpy().reshape(resolution, resolution)
    return metrics, x_grid, y_grid, pred_np, exact_np


def evaluate_trial_space_metrics(
    model: MeshfreeKAN2D,
    problem: Any,
    device: str,
    grid_resolution: int,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solution_metrics, x_grid, y_grid, pred, exact = evaluate_solution_metrics(
        model=model,
        problem=problem,
        device=device,
        resolution=grid_resolution,
    )
    system_info = assemble_trial_space_system(
        model=model,
        problem=problem,
        device=device,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    metrics = {}
    metrics.update(solution_metrics)
    metrics.update(system_info["stats"])
    return metrics, x_grid, y_grid, pred, exact


def build_trial_space_summary_lines(metrics_payload: dict[str, Any]) -> list[str]:
    case = metrics_payload["case"]
    basis_quality = metrics_payload["basis_quality"]
    trial_metrics = metrics_payload["trial_space"]
    return [
        f"method: {metrics_payload['method']}",
        f"case: {case}",
        f"basis_pu_rmse: {basis_quality['pu_rmse']:.6e}",
        f"basis_linear_x_rmse: {basis_quality['linear_x_rmse']:.6e}",
        f"basis_linear_y_rmse: {basis_quality['linear_y_rmse']:.6e}",
        f"basis_lambda_h_max: {basis_quality['lambda_h_max']:.6e}",
        f"l2_error: {trial_metrics['l2_error']:.6e}",
        f"h1_semi_error: {trial_metrics['h1_semi_error']:.6e}",
        f"boundary_l2: {trial_metrics['boundary_l2']:.6e}",
        f"cond_k: {trial_metrics['cond_k']:.6e}",
        f"lambda_min_k: {trial_metrics['lambda_min_k']:.6e}",
        f"lambda_max_k: {trial_metrics['lambda_max_k']:.6e}",
        f"solver_name: {trial_metrics.get('solver_name', 'n/a')}",
        f"solver_residual: {trial_metrics.get('solver_residual', float('nan')):.6e}",
        f"solver_shift: {trial_metrics.get('solver_shift', float('nan')):.6e}",
    ]


def run_classical_solve(
    model: MeshfreeKAN2D,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
) -> tuple[dict[str, list[float]], dict[str, float | str]]:
    system_info = assemble_trial_space_system(
        model=model,
        problem=problem,
        device=device,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    coeffs, solver_metrics = solve_linear_system(system_info["matrix"], system_info["rhs"])
    set_coefficients(model, coeffs)
    return init_solver_history(), solver_metrics


def run_classical_method(
    phase_a_model: MeshfreeKAN2D,
    *,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
    grid_resolution: int,
) -> tuple[MeshfreeKAN2D, dict[str, list[float]], dict[str, Any], dict[str, np.ndarray]]:
    model = clone_model(phase_a_model)
    history, extra_metrics = run_classical_solve(
        model=model,
        problem=problem,
        device=device,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    trial_metrics, x_grid, y_grid, pred, exact = evaluate_trial_space_metrics(
        model=model,
        problem=problem,
        device=device,
        grid_resolution=grid_resolution,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    trial_metrics.update(extra_metrics)
    arrays = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pred": pred,
        "exact": exact,
        **history_to_arrays(history),
    }
    return model, history, trial_metrics, arrays


def run_frozen_w_method(
    phase_a_model: MeshfreeKAN2D,
    *,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    phase_b_steps: int,
    batch_size: int,
    lr_w: float,
    beta_bc: float,
    gamma_linear: float,
    eval_interval: int,
    eval_resolution: int,
    log_interval: int,
    domain_order: int,
    boundary_order: int,
    grid_resolution: int,
) -> tuple[MeshfreeKAN2D, dict[str, list[float]], dict[str, Any], dict[str, np.ndarray]]:
    model = clone_model(phase_a_model)
    history = train_frozen_w_poisson(
        model=model,
        problem=problem,
        device=device,
        steps=phase_b_steps,
        batch_size=batch_size,
        lr_w=lr_w,
        beta_bc=beta_bc,
        gamma_linear=gamma_linear,
        eval_interval=eval_interval,
        eval_resolution=eval_resolution,
        log_interval=log_interval,
    )
    trial_metrics, x_grid, y_grid, pred, exact = evaluate_trial_space_metrics(
        model=model,
        problem=problem,
        device=device,
        grid_resolution=grid_resolution,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    trial_metrics.update({"solver_name": "adam_w_only", "solver_residual": 0.0, "solver_shift": 0.0})
    arrays = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pred": pred,
        "exact": exact,
        **history_to_arrays(history),
    }
    return model, history, trial_metrics, arrays


def run_joint_method(
    phase_a_model: MeshfreeKAN2D,
    *,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    phase_b_steps: int,
    batch_size: int,
    lr_kan_b: float,
    lr_w: float,
    beta_bc: float,
    gamma_linear: float,
    warmup_w_steps: int,
    eval_interval: int,
    eval_resolution: int,
    log_interval: int,
    domain_order: int,
    boundary_order: int,
    grid_resolution: int,
) -> tuple[MeshfreeKAN2D, dict[str, list[float]], dict[str, Any], dict[str, np.ndarray]]:
    model = clone_model(phase_a_model)
    history = train_phase_b_poisson(
        model=model,
        problem=problem,
        device=device,
        steps=phase_b_steps,
        batch_size=batch_size,
        lr_kan=lr_kan_b,
        lr_w=lr_w,
        beta_bc=beta_bc,
        gamma_linear=gamma_linear,
        warmup_w_steps=warmup_w_steps,
        eval_interval=eval_interval,
        eval_resolution=eval_resolution,
        log_interval=log_interval,
    )
    trial_metrics, x_grid, y_grid, pred, exact = evaluate_trial_space_metrics(
        model=model,
        problem=problem,
        device=device,
        grid_resolution=grid_resolution,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    trial_metrics.update({"solver_name": "adam_joint", "solver_residual": 0.0, "solver_shift": 0.0})
    arrays = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pred": pred,
        "exact": exact,
        **history_to_arrays(history),
    }
    return model, history, trial_metrics, arrays


def run_trial_method_case(
    *,
    method: str,
    phase_a_model: MeshfreeKAN2D,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    phase_b_steps: int,
    batch_size: int,
    lr_kan_b: float,
    lr_w: float,
    beta_bc: float,
    gamma_linear: float,
    warmup_w_steps: int,
    eval_interval: int,
    eval_resolution: int,
    log_interval: int,
    domain_order: int,
    boundary_order: int,
    grid_resolution: int,
) -> tuple[MeshfreeKAN2D, dict[str, list[float]], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if method == "classical":
        model, history, trial_metrics, arrays = run_classical_method(
            phase_a_model,
            problem=problem,
            device=device,
            domain_order=domain_order,
            boundary_order=boundary_order,
            beta_bc=beta_bc,
            grid_resolution=grid_resolution,
        )
    elif method == "frozen_w":
        model, history, trial_metrics, arrays = run_frozen_w_method(
            phase_a_model,
            problem=problem,
            device=device,
            phase_b_steps=phase_b_steps,
            batch_size=batch_size,
            lr_w=lr_w,
            beta_bc=beta_bc,
            gamma_linear=gamma_linear,
            eval_interval=eval_interval,
            eval_resolution=eval_resolution,
            log_interval=log_interval,
            domain_order=domain_order,
            boundary_order=boundary_order,
            grid_resolution=grid_resolution,
        )
    elif method == "joint":
        model, history, trial_metrics, arrays = run_joint_method(
            phase_a_model,
            problem=problem,
            device=device,
            phase_b_steps=phase_b_steps,
            batch_size=batch_size,
            lr_kan_b=lr_kan_b,
            lr_w=lr_w,
            beta_bc=beta_bc,
            gamma_linear=gamma_linear,
            warmup_w_steps=warmup_w_steps,
            eval_interval=eval_interval,
            eval_resolution=eval_resolution,
            log_interval=log_interval,
            domain_order=domain_order,
            boundary_order=boundary_order,
            grid_resolution=grid_resolution,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    basis_quality = evaluate_basis_quality_2d(
        model,
        device=device,
        grid_resolution=grid_resolution,
    )
    return model, history, basis_quality, trial_metrics, arrays
