from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_2d_validation import common as validation_common


torch.set_default_dtype(torch.float64)

MeshfreeKAN2D = validation_common.MeshfreeKAN2D
RunArtifacts = validation_common.RunArtifacts
SinusoidalPoissonProblem2D = validation_common.SinusoidalPoissonProblem2D
build_case_name = validation_common.build_case_name
compute_patch_metrics = validation_common.compute_patch_metrics
evaluate_solution_metrics = validation_common.evaluate_solution_metrics
generate_square_nodes = validation_common.generate_square_nodes
grid_points = validation_common.grid_points
parse_float_list = validation_common.parse_float_list
parse_int_list = validation_common.parse_int_list
plot_solution_triplet = validation_common.plot_solution_triplet
plot_training_curves = validation_common.plot_training_curves
resolve_variant_config = validation_common.resolve_variant_config
save_json = validation_common.save_json
save_run_bundle = validation_common.save_run_bundle
save_summary = validation_common.save_summary
seed_everything = validation_common.seed_everything
train_phase_a = validation_common.train_phase_a
train_phase_b_poisson = validation_common.train_phase_b_poisson


METHOD_LABELS = {
    "classical": "Fixed basis + classical solve",
    "frozen_w": "Fixed basis + optimize w",
    "joint": "Joint Phase B",
}


@dataclass
class TrialSpaceArtifacts:
    root_dir: Path
    figures_dir: Path
    methods_dir: Path


def ensure_trial_space_artifacts(group: str, case_name: str) -> TrialSpaceArtifacts:
    root_dir = ROOT_DIR / "output" / "meshfree_kan_rkpm_2d_trial_space_value" / group / case_name
    figures_dir = root_dir / "figures"
    methods_dir = root_dir / "methods"
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    return TrialSpaceArtifacts(root_dir=root_dir, figures_dir=figures_dir, methods_dir=methods_dir)


def ensure_method_artifacts(artifacts: TrialSpaceArtifacts, method_name: str) -> RunArtifacts:
    root_dir = artifacts.methods_dir / method_name
    figures_dir = root_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(root_dir=root_dir, figures_dir=figures_dir)


def gauss_legendre_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return nodes.astype(np.float64), weights.astype(np.float64)


def square_domain_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes_1d, weights_1d = gauss_legendre_1d(order)
    x_grid, y_grid = np.meshgrid(nodes_1d, nodes_1d, indexing="xy")
    wx_grid, wy_grid = np.meshgrid(weights_1d, weights_1d, indexing="xy")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    weights = (wx_grid * wy_grid).ravel()
    return points, weights


def square_boundary_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes_1d, weights_1d = gauss_legendre_1d(order)
    left = np.column_stack([np.zeros_like(nodes_1d), nodes_1d])
    right = np.column_stack([np.ones_like(nodes_1d), nodes_1d])
    bottom = np.column_stack([nodes_1d, np.zeros_like(nodes_1d)])
    top = np.column_stack([nodes_1d, np.ones_like(nodes_1d)])
    points = np.vstack([left, right, bottom, top])
    weights = np.tile(weights_1d, 4)
    return points.astype(np.float64), weights.astype(np.float64)


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


def freeze_basis(model: MeshfreeKAN2D) -> None:
    for param in model.kan.parameters():
        param.requires_grad_(False)


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


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, str, float]:
    try:
        coeffs = np.linalg.solve(matrix, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        solver = "lstsq"
    residual = float(np.linalg.norm(matrix @ coeffs - rhs))
    return coeffs, solver, residual


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


def evaluate_trial_space_metrics(
    model: MeshfreeKAN2D,
    problem: Any,
    device: str,
    grid_resolution: int,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, eval_points = grid_points(grid_resolution)
    with torch.no_grad():
        phi_eval = model.compute_shape_functions(
            torch.tensor(eval_points, device=device, dtype=model.nodes.dtype)
        ).cpu().numpy()
    patch_metrics = compute_patch_metrics(phi_eval, model.nodes.detach().cpu().numpy(), eval_points)
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
    metrics.update(patch_metrics)
    metrics.update(solution_metrics)
    metrics.update(system_info["stats"])
    return metrics, x_grid, y_grid, pred, exact


def init_solver_history() -> dict[str, list[float]]:
    return {
        "steps": [],
        "loss": [],
        "energy": [],
        "bc": [],
        "linear": [],
        "val_l2": [],
        "val_h1": [],
    }


def train_frozen_w_poisson(
    model: MeshfreeKAN2D,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    steps: int = 2000,
    batch_size: int = 1024,
    lr_w: float = 1e-2,
    beta_bc: float = 100.0,
    gamma_linear: float = 10.0,
    eval_interval: int = 100,
    eval_resolution: int = 41,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    history = init_solver_history()
    freeze_basis(model)
    model.init_w_from_exact(problem.exact_solution)
    optimizer = torch.optim.Adam([model.w], lr=lr_w)
    nodes_x = model.nodes[:, 0:1]
    nodes_y = model.nodes[:, 1:2]
    _, _, eval_points = grid_points(eval_resolution)
    eval_points_t = torch.tensor(
        eval_points,
        device=device,
        dtype=model.nodes.dtype,
        requires_grad=True,
    )
    best_score = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for step in range(steps):
        x_domain = torch.rand(batch_size, 2, device=device, dtype=model.nodes.dtype)
        x_domain.requires_grad_(True)
        x_boundary = validation_common.sample_square_boundary(batch_size, device)
        u_domain, phi_domain = model(x_domain, return_phi=True)
        grad_u = torch.autograd.grad(u_domain, x_domain, torch.ones_like(u_domain), create_graph=True)[0]
        forcing = problem.source_term(x_domain)
        energy_density = 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True) - forcing * u_domain
        loss_energy = torch.mean(energy_density)
        u_boundary = model(x_boundary)
        loss_bc = torch.mean((u_boundary - problem.exact_solution(x_boundary)) ** 2)
        repro_x = phi_domain @ nodes_x
        repro_y = phi_domain @ nodes_y
        loss_linear = torch.mean((repro_x - x_domain[:, 0:1]) ** 2 + (repro_y - x_domain[:, 1:2]) ** 2)
        loss = loss_energy + beta_bc * loss_bc + gamma_linear * loss_linear

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_l2 = 0.0
        val_h1 = 0.0
        if step % eval_interval == 0 or step == steps - 1:
            eval_points_t = eval_points_t.detach().clone().requires_grad_(True)
            u_eval = model(eval_points_t)
            grad_eval = torch.autograd.grad(
                u_eval,
                eval_points_t,
                torch.ones_like(u_eval),
                create_graph=False,
            )[0]
            u_exact = problem.exact_solution(eval_points_t)
            grad_exact = problem.exact_gradient(eval_points_t)
            val_l2 = float(torch.sqrt(torch.mean((u_eval - u_exact) ** 2)).item())
            val_h1 = float(torch.sqrt(torch.mean(torch.sum((grad_eval - grad_exact) ** 2, dim=1))).item())
            score = val_l2 + val_h1
            if score < best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if step % log_interval == 0 or step == steps - 1:
            history["steps"].append(float(step))
            history["loss"].append(float(loss.item()))
            history["energy"].append(float(loss_energy.item()))
            history["bc"].append(float(loss_bc.item()))
            history["linear"].append(float(loss_linear.item()))
            history["val_l2"].append(val_l2)
            history["val_h1"].append(val_h1)
            print(
                f"FrozenW | step={step:4d} | loss={loss.item():.3e} | "
                f"energy={loss_energy.item():.3e} | bc={loss_bc.item():.3e} | "
                f"linear={loss_linear.item():.3e} | val_l2={val_l2:.3e} | val_h1={val_h1:.3e}"
            )

    model.load_state_dict(best_state)
    return history


def history_to_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {f"history_{key}": np.array(value, dtype=np.float64) for key, value in history.items()}


def plot_metric_bars(entries: list[dict[str, Any]], path: Path) -> None:
    methods = [item["method_label"] for item in entries]
    x_pos = np.arange(len(entries))
    l2 = np.array([item["l2_error"] for item in entries], dtype=np.float64)
    h1 = np.array([item["h1_semi_error"] for item in entries], dtype=np.float64)
    boundary = np.array([item["boundary_l2"] for item in entries], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, values, title in zip(
        axes,
        [l2, h1, boundary],
        ["L2 error", "H1 semi error", "Boundary L2"],
    ):
        ax.bar(x_pos, values, color=["#4477aa", "#66ccee", "#228833"][: len(entries)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_conditioning_bars(entries: list[dict[str, Any]], path: Path) -> None:
    methods = [item["method_label"] for item in entries]
    x_pos = np.arange(len(entries))
    cond_values = np.array([item["cond_k"] for item in entries], dtype=np.float64)
    lambda_min = np.array([item["lambda_min_k"] for item in entries], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x_pos, cond_values, color="#cc6677")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_title("Condition number")
    axes[0].grid(True, axis="y", ls="--", alpha=0.4)

    axes[1].bar(x_pos, lambda_min, color="#aa4499")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_title("Smallest eigenvalue")
    axes[1].grid(True, axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_representative_shape_functions(
    model: MeshfreeKAN2D,
    path: Path,
    resolution: int = 81,
) -> None:
    x_grid, y_grid, points = grid_points(resolution)
    points_t = torch.tensor(points, device=model.nodes.device, dtype=model.nodes.dtype)
    with torch.no_grad():
        phi = model.compute_shape_functions(points_t).cpu().numpy()
    nodes = model.nodes.detach().cpu().numpy()
    targets = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, target in zip(axes.ravel(), targets):
        node_index = int(np.argmin(np.sum((nodes - target) ** 2, axis=1)))
        field = phi[:, node_index].reshape(resolution, resolution)
        image = ax.contourf(x_grid, y_grid, field, levels=50, cmap="viridis")
        ax.scatter(nodes[node_index, 0], nodes[node_index, 1], color="white", s=25, edgecolors="black")
        ax.set_title(f"node {node_index} @ ({nodes[node_index, 0]:.2f}, {nodes[node_index, 1]:.2f})")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_classical_solve(
    model: MeshfreeKAN2D,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    domain_order: int,
    boundary_order: int,
    beta_bc: float,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    system_info = assemble_trial_space_system(
        model=model,
        problem=problem,
        device=device,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    coeffs, solver_name, residual = solve_linear_system(system_info["matrix"], system_info["rhs"])
    set_coefficients(model, coeffs)
    return init_solver_history(), {"solver_name": solver_name, "solver_residual": residual}


def run_trial_method(
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
) -> tuple[MeshfreeKAN2D, dict[str, list[float]], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = clone_model(phase_a_model)
    if method == "classical":
        history, extra_metrics = run_classical_solve(
            model=model,
            problem=problem,
            device=device,
            domain_order=domain_order,
            boundary_order=boundary_order,
            beta_bc=beta_bc,
        )
    elif method == "frozen_w":
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
        extra_metrics = {"solver_name": "adam_w_only", "solver_residual": 0.0}
    elif method == "joint":
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
        extra_metrics = {"solver_name": "adam_joint", "solver_residual": 0.0}
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics, x_grid, y_grid, pred, exact = evaluate_trial_space_metrics(
        model=model,
        problem=problem,
        device=device,
        grid_resolution=grid_resolution,
        domain_order=domain_order,
        boundary_order=boundary_order,
        beta_bc=beta_bc,
    )
    metrics.update(extra_metrics)
    return model, history, metrics, x_grid, y_grid, pred, exact
