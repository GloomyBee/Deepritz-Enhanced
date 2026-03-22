from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core.splines import evaluate_open_uniform_bspline_basis


torch.set_default_dtype(torch.float64)


def generate_square_nodes(
    n_side: int,
    jitter: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    grid = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="xy")
    nodes = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    h = 1.0 / max(n_side - 1, 1)
    if jitter <= 0.0 or n_side <= 2:
        return nodes, h

    rng = np.random.default_rng(seed)
    interior = (
        (nodes[:, 0] > 1e-12)
        & (nodes[:, 0] < 1.0 - 1e-12)
        & (nodes[:, 1] > 1e-12)
        & (nodes[:, 1] < 1.0 - 1e-12)
    )
    perturb = rng.uniform(-jitter * h, jitter * h, size=(int(interior.sum()), 2))
    nodes[interior] = np.clip(nodes[interior] + perturb, 0.0, 1.0)
    return nodes, h


def reflect_to_unit_box(x: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    x = torch.where(x > 1.0, 2.0 - x, x)
    return torch.clamp(x, 0.0, 1.0)


def sample_square_boundary(batch_size: int, device: str) -> torch.Tensor:
    n_edge = max(batch_size // 4, 1)
    zeros = torch.zeros(n_edge, device=device)
    ones = torch.ones(n_edge, device=device)
    rands = [torch.rand(n_edge, device=device) for _ in range(4)]
    left = torch.stack([zeros, rands[0]], dim=1)
    right = torch.stack([ones, rands[1]], dim=1)
    bottom = torch.stack([rands[2], zeros], dim=1)
    top = torch.stack([rands[3], ones], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)


def sample_mixed_domain_points(
    nodes: torch.Tensor,
    support_radius: float,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    n_uniform = batch_size // 2
    n_local = batch_size - n_uniform
    x_uniform = torch.rand(n_uniform, 2, device=device)
    rand_idx = torch.randint(0, nodes.shape[0], (n_local,), device=device)
    centers = nodes[rand_idx]
    noise = torch.randn(n_local, 2, device=device) * (support_radius * 0.5)
    x_local = reflect_to_unit_box(centers + noise)
    return torch.cat([x_uniform, x_local], dim=0)


def grid_points(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return x_grid, y_grid, points


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


def cubic_spline_window_torch(dist: torch.Tensor, radius: float) -> torch.Tensor:
    q = dist / radius
    out = torch.zeros_like(q)
    inner = q <= 0.5
    outer = (q > 0.5) & (q <= 1.0)
    q_inner = q[inner]
    out[inner] = 2.0 / 3.0 - 4.0 * q_inner**2 + 4.0 * q_inner**3
    q_outer = q[outer]
    out[outer] = 4.0 / 3.0 - 4.0 * q_outer + 4.0 * q_outer**2 - (4.0 / 3.0) * q_outer**3
    return out


def cubic_spline_window_np(dist: np.ndarray, radius: float) -> np.ndarray:
    q = dist / radius
    out = np.zeros_like(q, dtype=np.float64)
    inner = q <= 0.5
    outer = (q > 0.5) & (q <= 1.0)
    q_inner = q[inner]
    out[inner] = 2.0 / 3.0 - 4.0 * q_inner**2 + 4.0 * q_inner**3
    q_outer = q[outer]
    out[outer] = 4.0 / 3.0 - 4.0 * q_outer + 4.0 * q_outer**2 - (4.0 / 3.0) * q_outer**3
    return out


class KANSpline2D(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        num_basis: int = 7,
        grid_range: tuple[float, float] = (-1.5, 1.5),
        degree: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.grid_range = grid_range
        self.degree = degree
        self.layer1 = nn.ModuleList(
            [nn.Linear(num_basis, hidden_dim, bias=False) for _ in range(input_dim)]
        )
        self.layer2 = nn.Linear(hidden_dim * num_basis, 1, bias=False)

    def _basis_response(self, x: torch.Tensor) -> torch.Tensor:
        return evaluate_open_uniform_bspline_basis(
            x,
            num_basis=self.num_basis,
            degree=self.degree,
            grid_range=self.grid_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_sum = 0.0
        for index, layer in enumerate(self.layer1):
            basis = self._basis_response(x[:, index : index + 1]).squeeze(1)
            hidden_sum = hidden_sum + layer(basis)
        basis_hidden = self._basis_response(hidden_sum)
        return self.layer2(basis_hidden.reshape(x.shape[0], -1))


class MeshfreeKAN2D(nn.Module):
    def __init__(
        self,
        nodes: torch.Tensor,
        support_radius: float,
        hidden_dim: int = 16,
        use_softplus: bool = True,
        enable_fallback: bool = True,
    ):
        super().__init__()
        self.register_buffer("nodes", nodes)
        self.n_nodes = int(nodes.shape[0])
        self.support_radius = float(support_radius)
        self.use_softplus = bool(use_softplus)
        self.enable_fallback = bool(enable_fallback)
        self.kan = KANSpline2D(hidden_dim=hidden_dim)
        self.softplus = nn.Softplus()
        self.w = nn.Parameter(torch.zeros(self.n_nodes, 1, dtype=nodes.dtype, device=nodes.device))

    def init_w_from_exact(self, exact_fn) -> None:
        with torch.no_grad():
            self.w[:] = exact_fn(self.nodes)

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.use_softplus:
            return self.softplus(raw)
        return raw

    def compute_shape_functions(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
        eps_cov: float = 1e-14,
        knn_k: int = 8,
    ) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)
        dist = torch.norm(diff, dim=2)
        kan_in = (diff / self.support_radius).reshape(-1, 2)
        phi_raw = self._activate(self.kan(kan_in).reshape(x.shape[0], self.n_nodes))
        phi_windowed = phi_raw * cubic_spline_window_torch(dist, self.support_radius)
        if return_raw:
            return phi_windowed

        phi_sum = torch.sum(phi_windowed, dim=1, keepdim=True)
        orphan = torch.abs(phi_sum.squeeze(1)) < eps_cov
        phi = torch.empty_like(phi_windowed)
        normal = ~orphan
        phi[normal] = phi_windowed[normal] / (phi_sum[normal] + 1e-12)
        if orphan.any() and self.enable_fallback:
            k_use = min(knn_k, self.n_nodes)
            dist_orphan = dist[orphan]
            idx = torch.topk(dist_orphan, k_use, largest=False).indices
            d_knn = torch.gather(dist_orphan, 1, idx)
            alpha = 20.0 / max(self.support_radius, 1e-12)
            weights = torch.exp(-alpha * d_knn)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-18)
            phi_orphan = torch.zeros(
                (dist_orphan.shape[0], self.n_nodes),
                device=x.device,
                dtype=x.dtype,
            )
            phi_orphan.scatter_(1, idx, weights)
            phi[orphan] = phi_orphan
        elif orphan.any():
            phi[orphan] = phi_windowed[orphan] / (phi_sum[orphan] + 1e-12)
        return phi

    def forward(self, x: torch.Tensor, return_phi: bool = False):
        phi = self.compute_shape_functions(x)
        u = phi @ self.w
        if return_phi:
            return u, phi
        return u


class SinusoidalPoissonProblem2D:
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

    def exact_gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = math.pi * torch.cos(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])
        grad_y = math.pi * torch.sin(math.pi * x[:, 0:1]) * torch.cos(math.pi * x[:, 1:2])
        return torch.cat([grad_x, grad_y], dim=1)

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        exact = self.exact_solution(x)
        return 2.0 * math.pi**2 * exact


def rkpm_shape_matrix_2d(
    x_eval: np.ndarray,
    nodes: np.ndarray,
    support_radius: float,
    cond_threshold: float = 1e12,
) -> np.ndarray:
    m_eval = x_eval.shape[0]
    n_nodes = nodes.shape[0]
    phi = np.zeros((m_eval, n_nodes), dtype=np.float64)
    p0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for index, x in enumerate(x_eval):
        r = x[None, :] - nodes
        dist = np.linalg.norm(r, axis=1)
        weight = cubic_spline_window_np(dist, support_radius)
        p = np.column_stack([np.ones(n_nodes), r[:, 0], r[:, 1]])
        moment = np.einsum("i,ij,ik->jk", weight, p, p)
        cond = np.linalg.cond(moment)
        inverse = np.linalg.pinv(moment, rcond=1e-12) if cond > cond_threshold else np.linalg.inv(moment)
        phi[index, :] = (p0 @ inverse @ p.T) * weight
    return phi


def rkpm_shape_matrix_2d_torch(
    x_eval: torch.Tensor,
    nodes: torch.Tensor,
    support_radius: float,
    rcond: float = 1e-12,
) -> torch.Tensor:
    diff = x_eval.unsqueeze(1) - nodes.unsqueeze(0)
    dist = torch.norm(diff, dim=2)
    weight = cubic_spline_window_torch(dist, support_radius)
    ones = torch.ones_like(weight)
    p = torch.stack([ones, diff[:, :, 0], diff[:, :, 1]], dim=2)
    moment = torch.einsum("mn,mni,mnj->mij", weight, p, p)
    inverse = torch.linalg.pinv(moment, rtol=rcond)
    first_row = inverse[:, 0, :].unsqueeze(1)
    return torch.sum(first_row * p, dim=2) * weight


def compute_lambda_h_stats(phi: np.ndarray) -> dict[str, float]:
    kappa = np.sum(np.abs(phi), axis=1)
    return {
        "lambda_h_max": float(np.max(kappa)),
        "lambda_h_mean": float(np.mean(kappa)),
    }


def compute_patch_metrics(phi: np.ndarray, nodes: np.ndarray, x_eval: np.ndarray) -> dict[str, float]:
    pu = np.sum(phi, axis=1) - 1.0
    linear_x = phi @ nodes[:, 0] - x_eval[:, 0]
    linear_y = phi @ nodes[:, 1] - x_eval[:, 1]
    stats = compute_lambda_h_stats(phi)
    stats.update(
        {
            "pu_rmse": float(np.sqrt(np.mean(pu**2))),
            "pu_linf": float(np.max(np.abs(pu))),
            "linear_x_rmse": float(np.sqrt(np.mean(linear_x**2))),
            "linear_y_rmse": float(np.sqrt(np.mean(linear_y**2))),
        }
    )
    return stats


def evaluate_basis_quality_2d(
    model: MeshfreeKAN2D,
    *,
    device: str,
    grid_resolution: int,
) -> dict[str, float]:
    _, _, eval_points = grid_points(grid_resolution)
    with torch.no_grad():
        phi_eval = model.compute_shape_functions(
            torch.tensor(eval_points, device=device, dtype=model.nodes.dtype)
        ).cpu().numpy()
    nodes = model.nodes.detach().cpu().numpy()
    return compute_patch_metrics(phi_eval, nodes, eval_points)
