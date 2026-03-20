from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def generate_interval_nodes(n_nodes: int) -> tuple[np.ndarray, float]:
    nodes = np.linspace(0.0, 1.0, n_nodes, dtype=np.float64)
    h = 1.0 / max(n_nodes - 1, 1)
    return nodes, h


def generate_nonuniform_interval_nodes(
    n_nodes: int,
    jitter_factor: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    nodes, h = generate_interval_nodes(n_nodes)
    if n_nodes <= 2 or jitter_factor <= 0.0:
        return nodes, h
    rng = np.random.default_rng(seed)
    interior = nodes[1:-1].copy()
    perturb = jitter_factor * h * rng.uniform(-1.0, 1.0, size=interior.shape[0])
    interior = np.clip(interior + perturb, 1.0e-6, 1.0 - 1.0e-6)
    nodes_nonuniform = np.concatenate(([0.0], np.sort(interior), [1.0]))
    return nodes_nonuniform.astype(np.float64), h


def gauss_legendre_interval_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return points.astype(np.float64), weights.astype(np.float64)


def interval_boundary_quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array([[0.0], [1.0]], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    normals = np.array([[-1.0], [1.0]], dtype=np.float64)
    return points, weights, normals


def cubic_spline_window_torch(dist: torch.Tensor, radius: float) -> torch.Tensor:
    q = dist / radius
    out = torch.zeros_like(q)
    mask_inner = q <= 0.5
    mask_outer = (q > 0.5) & (q <= 1.0)
    q_inner = q[mask_inner]
    out[mask_inner] = 2.0 / 3.0 - 4.0 * q_inner**2 + 4.0 * q_inner**3
    q_outer = q[mask_outer]
    out[mask_outer] = 4.0 / 3.0 - 4.0 * q_outer + 4.0 * q_outer**2 - (4.0 / 3.0) * q_outer**3
    return out


def cubic_spline_window_np(dist: np.ndarray, radius: float) -> np.ndarray:
    q = dist / radius
    out = np.zeros_like(q, dtype=np.float64)
    mask_inner = q <= 0.5
    mask_outer = (q > 0.5) & (q <= 1.0)
    q_inner = q[mask_inner]
    out[mask_inner] = 2.0 / 3.0 - 4.0 * q_inner**2 + 4.0 * q_inner**3
    q_outer = q[mask_outer]
    out[mask_outer] = 4.0 / 3.0 - 4.0 * q_outer + 4.0 * q_outer**2 - (4.0 / 3.0) * q_outer**3
    return out


class KANSpline1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 16,
        num_basis: int = 7,
        grid_range: tuple[float, float] = (-1.5, 1.5),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        gmin, gmax = grid_range
        grid = torch.linspace(gmin, gmax, num_basis)
        h = (gmax - gmin) / (num_basis - 1)
        self.register_buffer("grid", grid)
        self.register_buffer("h", torch.tensor(h))
        self.layer1 = nn.Linear(num_basis, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim * num_basis, 1, bias=False)

    def _hat_basis(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        batch_size, width = x.shape
        grid = self.grid.view(1, 1, self.num_basis)
        x_expanded = x.view(batch_size, width, 1)
        return torch.relu(1.0 - torch.abs(x_expanded - grid) / self.h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis_1 = self._hat_basis(x).squeeze(1)
        hidden = self.layer1(basis_1)
        basis_2 = self._hat_basis(hidden)
        return self.layer2(basis_2.reshape(x.shape[0], -1))


class MeshfreeKAN1D(nn.Module):
    def __init__(
        self,
        nodes: torch.Tensor,
        support_radius: float,
        hidden_dim: int = 16,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.register_buffer("nodes", nodes.reshape(-1))
        self.n_nodes = int(nodes.numel())
        self.support_radius = float(support_radius)
        self.kan = KANSpline1D(hidden_dim=hidden_dim).to(dtype=self.nodes.dtype)
        self.use_softplus = bool(use_softplus)
        self.softplus = nn.Softplus()

    def compute_shape_functions(
        self,
        x: torch.Tensor,
        return_stage: str = "normalized",
        eps_cov: float = 1e-14,
        knn_k: int = 3,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:, 0]

        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)
        dist = torch.abs(diff)
        phi_pre_window = self.kan((diff / self.support_radius).reshape(-1, 1)).reshape(x.shape[0], self.n_nodes)
        if self.use_softplus:
            phi_pre_window = self.softplus(phi_pre_window)

        if return_stage == "pre_window":
            return phi_pre_window

        phi_windowed = phi_pre_window * cubic_spline_window_torch(dist, self.support_radius)
        if return_stage == "windowed":
            return phi_windowed
        if return_stage != "normalized":
            raise ValueError(f"Unknown return_stage: {return_stage}")

        phi_sum = torch.sum(phi_windowed, dim=1, keepdim=True)
        orphan = torch.abs(phi_sum.squeeze(1)) < eps_cov
        phi = torch.empty_like(phi_windowed)
        normal = ~orphan
        phi[normal] = phi_windowed[normal] / (phi_sum[normal] + 1e-12)

        if orphan.any():
            k_use = min(knn_k, self.n_nodes)
            dist_orphan = dist[orphan]
            idx = torch.topk(dist_orphan, k_use, largest=False).indices
            d_knn = torch.gather(dist_orphan, 1, idx)
            alpha = 20.0 / max(self.support_radius, 1e-12)
            shifted = d_knn - torch.min(d_knn, dim=1, keepdim=True).values
            weights = torch.exp(-alpha * shifted)
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
            phi_orphan = torch.zeros((dist_orphan.shape[0], self.n_nodes), device=x.device, dtype=x.dtype)
            phi_orphan.scatter_(1, idx, weights)
            phi[orphan] = phi_orphan

        return phi


def get_model_phi_stages(model: MeshfreeKAN1D, x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "pre_window": model.compute_shape_functions(x, return_stage="pre_window"),
        "windowed": model.compute_shape_functions(x, return_stage="windowed"),
        "normalized": model.compute_shape_functions(x, return_stage="normalized"),
    }


def rkpm_shape_matrix_1d(
    x_eval: np.ndarray,
    nodes: np.ndarray,
    support_radius: float,
    cond_threshold: float = 1e12,
) -> np.ndarray:
    p0 = np.array([1.0, 0.0], dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    phi = np.zeros((x_eval.shape[0], nodes.shape[0]), dtype=np.float64)
    for ix, x in enumerate(x_eval):
        r = x - nodes
        weights = cubic_spline_window_np(np.abs(r), support_radius)
        m11 = np.sum(weights)
        m12 = np.sum(weights * r)
        m22 = np.sum(weights * r * r)
        matrix = np.array([[m11, m12], [m12, m22]], dtype=np.float64)
        if np.linalg.cond(matrix) > cond_threshold:
            matrix_inv = np.linalg.pinv(matrix, rcond=1e-12)
        else:
            matrix_inv = np.linalg.inv(matrix)
        p = np.column_stack([np.ones(nodes.shape[0]), r])
        phi[ix, :] = (p0 @ matrix_inv @ p.T) * weights
    return phi


def rkpm_shape_matrix_1d_torch(
    x_eval: torch.Tensor,
    nodes: torch.Tensor,
    support_radius: float,
    eps_det: float = 1e-12,
) -> torch.Tensor:
    if x_eval.dim() == 2:
        x_eval = x_eval[:, 0]
    if nodes.dim() != 1:
        nodes = nodes.reshape(-1)
    r = x_eval.unsqueeze(1) - nodes.unsqueeze(0)
    weights = cubic_spline_window_torch(torch.abs(r), support_radius)
    s0 = torch.sum(weights, dim=1, keepdim=True)
    s1 = torch.sum(weights * r, dim=1, keepdim=True)
    s2 = torch.sum(weights * r * r, dim=1, keepdim=True)
    det = s0 * s2 - s1 * s1
    det_safe = torch.clamp(det, min=eps_det)
    return ((s2 - s1 * r) / det_safe) * weights


def polynomial_basis_1d(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])


def polynomial_gradients_1d(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64).reshape(-1)
    gradients = np.zeros((x.shape[0], 2, 1), dtype=np.float64)
    gradients[:, 1, 0] = 1.0
    return gradients


def analytic_moment_matrix_1d() -> np.ndarray:
    return np.array([[1.0, 0.5], [0.5, 1.0 / 3.0]], dtype=np.float64)


def analytic_derivative_moment_matrices_1d() -> np.ndarray:
    return np.array([[[0.0, 0.0], [1.0, 0.5]]], dtype=np.float64)


def compute_support_metrics(windowed_sum: np.ndarray, orphan_tol: float = 1e-14) -> dict[str, float]:
    windowed_sum = np.asarray(windowed_sum, dtype=np.float64).reshape(-1)
    return {
        "orphan_ratio": float(np.mean(np.abs(windowed_sum) < orphan_tol)),
        "phi_sum_min": float(np.min(windowed_sum)),
        "phi_sum_p01": float(np.quantile(windowed_sum, 0.01)),
    }


def compute_rkpm_shape_and_gradients_1d(
    points: np.ndarray,
    nodes: np.ndarray,
    support_radius: float,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    points_t = torch.tensor(points, device=device, dtype=torch.float64, requires_grad=True)
    nodes_t = torch.tensor(nodes, device=device, dtype=torch.float64)
    phi = rkpm_shape_matrix_1d_torch(points_t, nodes_t, support_radius)
    grad_phi = torch.zeros((points_t.shape[0], nodes_t.shape[0], 1), device=device, dtype=torch.float64)
    for index in range(nodes_t.shape[0]):
        retain_graph = index + 1 < nodes_t.shape[0]
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            points_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, 0] = grad_i[:, 0]
    return phi.detach().cpu().numpy(), grad_phi.detach().cpu().numpy()


def compute_model_shape_and_gradients_1d(
    model: MeshfreeKAN1D,
    points: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float64).reshape(-1, 1)
    points_t = torch.tensor(points_np, device=device, dtype=torch.float64, requires_grad=True)
    stages = get_model_phi_stages(model, points_t)
    phi = stages["normalized"]
    grad_phi = torch.zeros((points_t.shape[0], model.n_nodes, 1), device=device, dtype=torch.float64)
    for index in range(model.n_nodes):
        retain_graph = index + 1 < model.n_nodes
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            points_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, 0] = grad_i[:, 0]
    return {
        "normalized": stages["normalized"].detach().cpu().numpy(),
        "windowed": stages["windowed"].detach().cpu().numpy(),
        "pre_window": stages["pre_window"].detach().cpu().numpy(),
        "grad_normalized": grad_phi.detach().cpu().numpy(),
    }


def compute_value_consistency_metrics_1d(
    phi: np.ndarray,
    nodes: np.ndarray,
    domain_points: np.ndarray,
    domain_weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    phi = np.asarray(phi, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    domain_points = np.asarray(domain_points, dtype=np.float64).reshape(-1, 1)
    domain_weights = np.asarray(domain_weights, dtype=np.float64)
    p_nodes = polynomial_basis_1d(nodes)
    p_eval = polynomial_basis_1d(domain_points)
    nodal_integrals = phi.T @ domain_weights
    reproduced = phi @ p_nodes
    moment_matrix = (reproduced * domain_weights[:, None]).T @ p_eval
    exact_moment = analytic_moment_matrix_1d()
    pu_residual_field = np.sum(phi, axis=1) - 1.0
    x_repro_field = reproduced[:, 1] - domain_points[:, 0]
    metrics = {
        "mass_sum_residual": float(abs(np.sum(nodal_integrals) - 1.0)),
        "moment_x_residual": float(abs(np.sum(nodes * nodal_integrals) - 0.5)),
        "int_pu_residual": float(abs(np.sum(pu_residual_field * domain_weights))),
        "int_x_repro_residual": float(abs(np.sum(x_repro_field * domain_weights))),
        "moment_matrix_residual_fro": float(np.linalg.norm(moment_matrix - exact_moment, ord="fro")),
        "moment_matrix_residual_max": float(np.max(np.abs(moment_matrix - exact_moment))),
    }
    arrays = {
        "nodal_integrals": nodal_integrals,
        "pu_residual_field": pu_residual_field,
        "x_repro_residual_field": x_repro_field,
        "moment_matrix": moment_matrix,
        "moment_matrix_residual": moment_matrix - exact_moment,
    }
    return metrics, arrays


def compute_derivative_consistency_metrics_1d(
    phi: np.ndarray,
    grad_phi: np.ndarray,
    nodes: np.ndarray,
    domain_points: np.ndarray,
    domain_weights: np.ndarray,
    boundary_points: np.ndarray,
    boundary_weights: np.ndarray,
    boundary_normals: np.ndarray,
    phi_boundary: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    phi = np.asarray(phi, dtype=np.float64)
    grad_phi = np.asarray(grad_phi, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    domain_points = np.asarray(domain_points, dtype=np.float64).reshape(-1, 1)
    domain_weights = np.asarray(domain_weights, dtype=np.float64)
    boundary_points = np.asarray(boundary_points, dtype=np.float64).reshape(-1, 1)
    boundary_weights = np.asarray(boundary_weights, dtype=np.float64)
    boundary_normals = np.asarray(boundary_normals, dtype=np.float64).reshape(-1, 1)
    phi_boundary = np.asarray(phi_boundary, dtype=np.float64)
    p_nodes = polynomial_basis_1d(nodes)
    p_eval = polynomial_basis_1d(domain_points)
    p_boundary = polynomial_basis_1d(boundary_points)
    grad_p_eval = polynomial_gradients_1d(domain_points)

    lhs = np.einsum("mna,mk,m->nak", grad_phi, p_eval, domain_weights)
    boundary_term = np.einsum("bn,bk,ba,b->nak", phi_boundary, p_boundary, boundary_normals, boundary_weights)
    correction_term = np.einsum("mn,mka,m->nak", phi, grad_p_eval, domain_weights)
    boundary_residuals = lhs - (boundary_term - correction_term)

    reproduced_grad = np.einsum("mna,nk->mak", grad_phi, p_nodes)
    derivative_matrix = np.einsum("mak,mj,m->akj", reproduced_grad, p_eval, domain_weights)
    exact_derivative_matrix = analytic_derivative_moment_matrices_1d()
    derivative_matrix_residual = derivative_matrix - exact_derivative_matrix

    metrics = {
        "derivative_boundary_residual_fro": float(np.sqrt(np.sum(boundary_residuals ** 2))),
        "derivative_boundary_residual_max": float(np.max(np.abs(boundary_residuals))),
        "derivative_matrix_residual_fro": float(np.sqrt(np.sum(derivative_matrix_residual ** 2))),
        "derivative_matrix_residual_max": float(np.max(np.abs(derivative_matrix_residual))),
    }
    arrays = {
        "boundary_residuals": boundary_residuals,
        "derivative_matrix": derivative_matrix,
        "derivative_matrix_residual": derivative_matrix_residual,
    }
    return metrics, arrays


def evaluate_consistency_bundle_1d(
    phi: np.ndarray,
    grad_phi: np.ndarray,
    nodes: np.ndarray,
    domain_points: np.ndarray,
    domain_weights: np.ndarray,
    boundary_points: np.ndarray,
    boundary_weights: np.ndarray,
    boundary_normals: np.ndarray,
    phi_boundary: np.ndarray,
    windowed_sum: np.ndarray,
) -> dict[str, Any]:
    support_metrics = compute_support_metrics(windowed_sum)
    value_metrics, value_arrays = compute_value_consistency_metrics_1d(
        phi=phi,
        nodes=nodes,
        domain_points=domain_points,
        domain_weights=domain_weights,
    )
    derivative_metrics, derivative_arrays = compute_derivative_consistency_metrics_1d(
        phi=phi,
        grad_phi=grad_phi,
        nodes=nodes,
        domain_points=domain_points,
        domain_weights=domain_weights,
        boundary_points=boundary_points,
        boundary_weights=boundary_weights,
        boundary_normals=boundary_normals,
        phi_boundary=phi_boundary,
    )
    arrays: dict[str, np.ndarray] = {
        "domain_points": np.asarray(domain_points, dtype=np.float64).reshape(-1, 1),
        "domain_weights": np.asarray(domain_weights, dtype=np.float64),
        "boundary_points": np.asarray(boundary_points, dtype=np.float64).reshape(-1, 1),
        "boundary_weights": np.asarray(boundary_weights, dtype=np.float64),
        "boundary_normals": np.asarray(boundary_normals, dtype=np.float64).reshape(-1, 1),
        "phi": np.asarray(phi, dtype=np.float64),
        "grad_phi": np.asarray(grad_phi, dtype=np.float64),
        "phi_boundary": np.asarray(phi_boundary, dtype=np.float64),
        "windowed_sum": np.asarray(windowed_sum, dtype=np.float64).reshape(-1),
    }
    arrays.update(value_arrays)
    arrays.update(derivative_arrays)
    metrics = {}
    metrics.update(support_metrics)
    metrics.update(value_metrics)
    metrics.update(derivative_metrics)
    metrics["lambda_h_max"] = float(np.max(np.sum(np.abs(phi), axis=1)))
    metrics["lambda_h_mean"] = float(np.mean(np.sum(np.abs(phi), axis=1)))
    return {"metrics": metrics, "arrays": arrays}


def evaluate_model_consistency_bundle_1d(
    model: MeshfreeKAN1D,
    quadrature_order: int,
    device: str,
) -> dict[str, Any]:
    domain_points, domain_weights = gauss_legendre_interval_1d(quadrature_order)
    boundary_points, boundary_weights, boundary_normals = interval_boundary_quadrature()
    shape_data = compute_model_shape_and_gradients_1d(model, domain_points, device=device)
    boundary_t = torch.tensor(boundary_points, device=device, dtype=torch.float64)
    with torch.no_grad():
        phi_boundary = model.compute_shape_functions(boundary_t, return_stage="normalized").cpu().numpy()
    return evaluate_consistency_bundle_1d(
        phi=shape_data["normalized"],
        grad_phi=shape_data["grad_normalized"],
        nodes=model.nodes.detach().cpu().numpy(),
        domain_points=domain_points,
        domain_weights=domain_weights,
        boundary_points=boundary_points,
        boundary_weights=boundary_weights,
        boundary_normals=boundary_normals,
        phi_boundary=phi_boundary,
        windowed_sum=np.sum(shape_data["windowed"], axis=1),
    )


def evaluate_rkpm_consistency_bundle_1d(
    nodes: np.ndarray,
    support_radius: float,
    quadrature_order: int,
    device: str = "cpu",
) -> dict[str, Any]:
    domain_points, domain_weights = gauss_legendre_interval_1d(quadrature_order)
    boundary_points, boundary_weights, boundary_normals = interval_boundary_quadrature()
    phi, grad_phi = compute_rkpm_shape_and_gradients_1d(
        points=domain_points.reshape(-1, 1),
        nodes=nodes,
        support_radius=support_radius,
        device=device,
    )
    phi_boundary = rkpm_shape_matrix_1d(boundary_points.reshape(-1), nodes, support_radius)
    return evaluate_consistency_bundle_1d(
        phi=phi,
        grad_phi=grad_phi,
        nodes=nodes,
        domain_points=domain_points,
        domain_weights=domain_weights,
        boundary_points=boundary_points,
        boundary_weights=boundary_weights,
        boundary_normals=boundary_normals,
        phi_boundary=phi_boundary,
        windowed_sum=np.sum(phi, axis=1),
    )


def build_consistency_metrics_payload(
    case: dict[str, Any],
    learned: dict[str, Any],
    rkpm: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, float] = {}
    for key in sorted(set(learned.keys()) & set(rkpm.keys())):
        value_learned = learned[key]
        value_rkpm = rkpm[key]
        if isinstance(value_learned, (int, float)) and isinstance(value_rkpm, (int, float)):
            comparison[f"learned_minus_rkpm_{key}"] = float(value_learned) - float(value_rkpm)
            if abs(float(value_rkpm)) > 1e-16:
                comparison[f"learned_over_rkpm_{key}"] = float(value_learned) / float(value_rkpm)
    return {
        "case": case,
        "learned": learned,
        "rkpm": rkpm,
        "comparison": comparison,
    }


def prefix_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": np.asarray(value) for key, value in arrays.items()}


def history_to_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {f"history_{key}": np.array(value, dtype=np.float64) for key, value in history.items()}


def evaluate_shape_metrics_1d(
    *,
    x_eval: np.ndarray,
    nodes: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
    phi_windowed_learned: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    phi_learned = np.asarray(phi_learned, dtype=np.float64)
    phi_rkpm = np.asarray(phi_rkpm, dtype=np.float64)
    diff = phi_learned - phi_rkpm
    center_idx = int(np.argmin(np.abs(nodes - 0.5)))
    boundary_idx = 0
    global_l2 = float(np.sqrt(np.mean(diff**2)))
    global_linf = float(np.max(np.abs(diff)))
    relative_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(phi_rkpm), 1.0e-16))
    relative_linf = float(global_linf / max(float(np.max(np.abs(phi_rkpm))), 1.0e-16))
    center_diff = diff[:, center_idx]
    boundary_diff = diff[:, boundary_idx]
    pointwise_relative_error = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(phi_rkpm, axis=1), 1.0e-16)
    metrics = {
        "global_l2": global_l2,
        "global_linf": global_linf,
        "shape_relative_l2": relative_l2,
        "shape_relative_linf": relative_linf,
        "center_l2": float(np.sqrt(np.mean(center_diff**2))),
        "center_linf": float(np.max(np.abs(center_diff))),
        "boundary_l2": float(np.sqrt(np.mean(boundary_diff**2))),
        "boundary_linf": float(np.max(np.abs(boundary_diff))),
        "pu_max_error": float(np.max(np.abs(np.sum(phi_learned, axis=1) - 1.0))),
        "linear_reproduction_rmse": float(
            np.sqrt(np.mean((phi_learned @ nodes.reshape(-1, 1) - x_eval.reshape(-1, 1)) ** 2))
        ),
        "center_node_index": float(center_idx),
        "boundary_node_index": float(boundary_idx),
    }
    if phi_windowed_learned is not None:
        metrics["pu_windowed_sum_rmse"] = float(np.sqrt(np.mean((np.sum(phi_windowed_learned, axis=1) - 1.0) ** 2)))
    arrays = {
        "pointwise_relative_error": pointwise_relative_error,
        "center_abs_error_curve": np.abs(center_diff),
        "boundary_abs_error_curve": np.abs(boundary_diff),
        "global_abs_error_curve": np.max(np.abs(diff), axis=1),
    }
    return metrics, arrays


def build_shape_validation_metrics_payload_1d(
    *,
    case: dict[str, Any],
    learned_shape: dict[str, Any],
    rkpm_shape: dict[str, Any],
    learned_consistency: dict[str, Any],
    rkpm_consistency: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, float] = {}
    for key, value in learned_shape.items():
        if isinstance(value, (int, float)) and key in rkpm_shape and isinstance(rkpm_shape[key], (int, float)):
            comparison[f"learned_minus_rkpm_shape_{key}"] = float(value) - float(rkpm_shape[key])
            if abs(float(rkpm_shape[key])) > 1.0e-16:
                comparison[f"learned_over_rkpm_shape_{key}"] = float(value) / float(rkpm_shape[key])
    comparison.update(
        build_consistency_metrics_payload(case={}, learned=learned_consistency, rkpm=rkpm_consistency)["comparison"]
    )
    return {
        "case": case,
        "learned": {
            "shape": learned_shape,
            "consistency": learned_consistency,
        },
        "rkpm": {
            "shape": rkpm_shape,
            "consistency": rkpm_consistency,
        },
        "comparison": comparison,
    }


def evaluate_shape_validation_case_1d(
    *,
    model: MeshfreeKAN1D,
    nodes: np.ndarray,
    support_radius: float,
    n_eval: int,
    quadrature_order: int,
    device: str,
    case_meta: dict[str, Any],
) -> dict[str, Any]:
    x_eval = np.linspace(0.0, 1.0, n_eval, dtype=np.float64)
    shape_data = compute_model_shape_and_gradients_1d(
        model=model,
        points=x_eval,
        device=device,
    )
    phi_rkpm, grad_rkpm = compute_rkpm_shape_and_gradients_1d(
        points=x_eval.reshape(-1, 1),
        nodes=nodes,
        support_radius=support_radius,
        device=device,
    )
    learned_shape_metrics, shape_arrays = evaluate_shape_metrics_1d(
        x_eval=x_eval,
        nodes=nodes,
        phi_learned=shape_data["normalized"],
        phi_rkpm=phi_rkpm,
        phi_windowed_learned=shape_data["windowed"],
    )
    rkpm_shape_metrics = {
        "global_l2": 0.0,
        "global_linf": 0.0,
        "shape_relative_l2": 0.0,
        "shape_relative_linf": 0.0,
        "center_l2": 0.0,
        "center_linf": 0.0,
        "boundary_l2": 0.0,
        "boundary_linf": 0.0,
        "pu_max_error": float(np.max(np.abs(np.sum(phi_rkpm, axis=1) - 1.0))),
        "linear_reproduction_rmse": float(
            np.sqrt(np.mean((phi_rkpm @ nodes.reshape(-1, 1) - x_eval.reshape(-1, 1)) ** 2))
        ),
        "center_node_index": learned_shape_metrics["center_node_index"],
        "boundary_node_index": learned_shape_metrics["boundary_node_index"],
    }
    learned_consistency_bundle = evaluate_model_consistency_bundle_1d(
        model=model,
        quadrature_order=quadrature_order,
        device=device,
    )
    rkpm_consistency_bundle = evaluate_rkpm_consistency_bundle_1d(
        nodes=nodes,
        support_radius=support_radius,
        quadrature_order=quadrature_order,
        device=device,
    )
    metrics = build_shape_validation_metrics_payload_1d(
        case=case_meta,
        learned_shape=learned_shape_metrics,
        rkpm_shape=rkpm_shape_metrics,
        learned_consistency=learned_consistency_bundle["metrics"],
        rkpm_consistency=rkpm_consistency_bundle["metrics"],
    )
    arrays: dict[str, np.ndarray] = {
        "x_eval": x_eval,
        "nodes": np.asarray(nodes, dtype=np.float64),
        "phi_learned": shape_data["normalized"],
        "phi_windowed_learned": shape_data["windowed"],
        "phi_pre_window_learned": shape_data["pre_window"],
        "phi_rkpm": phi_rkpm,
        "grad_phi_learned": shape_data["grad_normalized"],
        "grad_phi_rkpm": grad_rkpm,
    }
    arrays.update(shape_arrays)
    arrays.update(prefix_arrays("learned_consistency", learned_consistency_bundle["arrays"]))
    arrays.update(prefix_arrays("rkpm_consistency", rkpm_consistency_bundle["arrays"]))
    return {
        "metrics": metrics,
        "arrays": arrays,
        "learned_consistency_bundle": learned_consistency_bundle,
        "rkpm_consistency_bundle": rkpm_consistency_bundle,
    }


def build_shape_validation_summary_lines_1d(payload: dict[str, Any]) -> list[str]:
    case = payload["case"]
    learned_shape = payload["learned"]["shape"]
    learned_consistency = payload["learned"]["consistency"]
    return [
        f"case: {case}",
        f"shape_relative_l2: {learned_shape['shape_relative_l2']:.6e}",
        f"shape_relative_linf: {learned_shape['shape_relative_linf']:.6e}",
        f"global_l2: {learned_shape['global_l2']:.6e}",
        f"center_l2: {learned_shape['center_l2']:.6e}",
        f"boundary_l2: {learned_shape['boundary_l2']:.6e}",
        f"pu_max_error: {learned_shape['pu_max_error']:.6e}",
        f"linear_reproduction_rmse: {learned_shape['linear_reproduction_rmse']:.6e}",
        f"mass_sum_residual: {learned_consistency['mass_sum_residual']:.6e}",
        f"moment_matrix_residual_fro: {learned_consistency['moment_matrix_residual_fro']:.6e}",
        f"derivative_matrix_residual_fro: {learned_consistency['derivative_matrix_residual_fro']:.6e}",
    ]


def build_consistency_summary_lines(payload: dict[str, Any]) -> list[str]:
    case = payload["case"]
    learned = payload["learned"]
    rkpm = payload["rkpm"]
    return [
        f"case: {case}",
        f"learned_moment_matrix_residual_fro: {learned['moment_matrix_residual_fro']}",
        f"learned_derivative_matrix_residual_fro: {learned['derivative_matrix_residual_fro']}",
        f"rkpm_moment_matrix_residual_fro: {rkpm['moment_matrix_residual_fro']}",
        f"rkpm_derivative_matrix_residual_fro: {rkpm['derivative_matrix_residual_fro']}",
    ]
