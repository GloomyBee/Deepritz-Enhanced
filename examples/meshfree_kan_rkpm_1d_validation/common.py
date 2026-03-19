from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

torch.set_default_dtype(torch.float64)


FIGURE_DPI = 180
MAIN_FIGURE_SIZE = (12.0, 9.0)
SUMMARY_FIGURE_SIZE = (12.0, 8.0)
DIAGNOSTIC_FIGURE_SIZE = (6.0, 4.6)
LEARNED_COLOR = "#1f4e79"
RKPM_COLOR = "#b03a2e"
AUX_COLORS = ["#1f4e79", "#b03a2e", "#2f7d32", "#7a3e9d", "#8c5a2b"]


@dataclass
class RunArtifacts:
    root_dir: Path
    figures_dir: Path
    diagnostics_dir: Path


def safe_token(value: Any) -> str:
    token = str(value).strip().replace(".", "p")
    token = token.replace(" ", "_").replace("/", "_")
    return token


def build_case_name(
    *,
    n_nodes: int | None = None,
    support_factor: float | None = None,
    seed: int | None = None,
    tag: str = "",
) -> str:
    parts: list[str] = []
    if n_nodes is not None:
        parts.append(f"nn{n_nodes}")
    if support_factor is not None:
        parts.append(f"sf{safe_token(support_factor)}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(safe_token(tag))
    return "_".join(parts)


def ensure_run_artifacts(group: str, case_name: str) -> RunArtifacts:
    root_dir = ROOT_DIR / "output" / "meshfree_kan_rkpm_1d_validation" / group / case_name
    figures_dir = root_dir / "figures"
    diagnostics_dir = figures_dir / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(root_dir=root_dir, figures_dir=figures_dir, diagnostics_dir=diagnostics_dir)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_summary(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def save_run_bundle(
    artifacts: RunArtifacts,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    arrays: dict[str, np.ndarray],
    summary_lines: list[str],
) -> None:
    save_json(artifacts.root_dir / "config.json", config)
    save_json(artifacts.root_dir / "metrics.json", metrics)
    save_summary(artifacts.root_dir / "summary.txt", summary_lines)
    np.savez(artifacts.root_dir / "curves.npz", **arrays)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_interval_nodes(n_nodes: int) -> tuple[np.ndarray, float]:
    nodes = np.linspace(0.0, 1.0, n_nodes, dtype=np.float64)
    h = 1.0 / max(n_nodes - 1, 1)
    return nodes, h


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
    def __init__(self, nodes: torch.Tensor, support_radius: float, hidden_dim: int = 16):
        super().__init__()
        self.register_buffer("nodes", nodes.reshape(-1))
        self.n_nodes = int(nodes.numel())
        self.support_radius = float(support_radius)
        self.kan = KANSpline1D(hidden_dim=hidden_dim)

    def compute_shape_functions(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
        eps_cov: float = 1e-14,
        knn_k: int = 3,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:, 0]
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)
        dist = torch.abs(diff)
        phi_raw = self.kan((diff / self.support_radius).reshape(-1, 1)).reshape(x.shape[0], self.n_nodes)
        phi_windowed = phi_raw * cubic_spline_window_torch(dist, self.support_radius)
        if return_raw:
            return phi_windowed
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
            weights = torch.exp(-alpha * d_knn)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-18)
            phi_orphan = torch.zeros((dist_orphan.shape[0], self.n_nodes), device=x.device, dtype=x.dtype)
            phi_orphan.scatter_(1, idx, weights)
            phi[orphan] = phi_orphan
        return phi


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
    sign = torch.where(det >= 0.0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = det + sign * eps_det
    return ((s2 - s1 * r) / det_safe) * weights


def init_history() -> dict[str, list[float]]:
    return {
        "steps": [],
        "loss": [],
        "teacher": [],
        "linear": [],
        "pu": [],
        "bd": [],
        "reg": [],
    }


def train_phase_a_distill(
    model: MeshfreeKAN1D,
    nodes: torch.Tensor,
    device: str,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_teacher: float = 1.0,
    lambda_bd: float = 0.1,
    lambda_reg: float = 1e-4,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nodes_col = nodes.reshape(-1, 1)
    history = init_history()
    x_boundary = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float64)
    for step in range(steps):
        x_domain = torch.rand(batch_size, 1, device=device)
        x_all = torch.cat([x_domain, x_boundary], dim=0)
        phi = model.compute_shape_functions(x_all)
        with torch.no_grad():
            phi_teacher = rkpm_shape_matrix_1d_torch(x_all, nodes, support_radius=model.support_radius)
        loss_teacher = torch.mean((phi - phi_teacher) ** 2)
        phi_boundary = phi[-2:, :]
        loss_boundary = (
            (phi_boundary[0, 0] - 1.0) ** 2
            + torch.sum(phi_boundary[0, 1:] ** 2)
            + (phi_boundary[1, -1] - 1.0) ** 2
            + torch.sum(phi_boundary[1, :-1] ** 2)
        )
        phi_raw = model.compute_shape_functions(x_all, return_raw=True)
        loss_reg = torch.mean(phi_raw ** 2)
        repro_x = phi @ nodes_col
        loss_linear = torch.mean((repro_x - x_all) ** 2)
        loss_pu = torch.mean((torch.sum(phi, dim=1, keepdim=True) - 1.0) ** 2)
        loss = lambda_teacher * loss_teacher + lambda_bd * loss_boundary + lambda_reg * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_interval == 0 or step == steps - 1:
            history["steps"].append(float(step))
            history["loss"].append(float(loss.item()))
            history["teacher"].append(float(loss_teacher.item()))
            history["linear"].append(float(loss_linear.item()))
            history["pu"].append(float(loss_pu.item()))
            history["bd"].append(float(loss_boundary.item()))
            history["reg"].append(float(loss_reg.item()))
            print(
                f"PhaseA1D | step={step:4d} | loss={loss.item():.3e} | "
                f"teacher={loss_teacher.item():.3e} | linear={loss_linear.item():.3e} | "
                f"pu={loss_pu.item():.3e} | bd={loss_boundary.item():.3e} | reg={loss_reg.item():.3e}"
            )
    return history


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_line_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, log_scale: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def _plot_history_series(ax: plt.Axes, history: dict[str, list[float]]) -> None:
    steps = np.array(history.get("steps", []), dtype=np.float64)
    if steps.size == 0:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        _style_line_axis(ax, "Training History", "step", "loss")
        return
    for key, color in zip(["loss", "teacher", "linear", "pu", "bd", "reg"], AUX_COLORS + ["#666666"]):
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0 or np.allclose(values, 0.0):
            continue
        count = min(values.size, steps.size)
        ax.semilogy(steps[:count], np.maximum(values[:count], 1e-16), lw=1.8, color=color, label=key)
    _style_line_axis(ax, "Training History", "step", "loss")
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)


def _plot_metric_point_panel(
    ax: plt.Axes,
    labels: list[str],
    learned: np.ndarray,
    rkpm: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    x_pos = np.arange(len(labels), dtype=np.float64)
    ax.plot(x_pos, np.maximum(learned, 1e-16), "o-", lw=1.8, ms=6, color=LEARNED_COLOR, label="learned")
    ax.plot(x_pos, np.maximum(rkpm, 1e-16), "s-", lw=1.8, ms=6, color=RKPM_COLOR, label="rkpm")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    _style_line_axis(ax, title, "metric", ylabel, log_scale=True)
    ax.legend(fontsize=8)


def plot_training_curves(history: dict[str, list[float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    _plot_history_series(ax, history)
    ax.set_title(title)
    _save_figure(fig, path)

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


def compute_support_metrics(phi_sum: np.ndarray, orphan_tol: float = 1e-14) -> dict[str, float]:
    phi_sum = np.asarray(phi_sum, dtype=np.float64).reshape(-1)
    return {
        "orphan_ratio": float(np.mean(np.abs(phi_sum) < orphan_tol)),
        "phi_sum_min": float(np.min(phi_sum)),
        "phi_sum_p01": float(np.quantile(phi_sum, 0.01)),
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
    phi_sum: np.ndarray,
) -> dict[str, Any]:
    support_metrics = compute_support_metrics(phi_sum)
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
        "phi_sum": np.asarray(phi_sum, dtype=np.float64).reshape(-1),
    }
    arrays.update(value_arrays)
    arrays.update(derivative_arrays)
    metrics = {}
    metrics.update(compute_support_metrics(phi_sum))
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
    domain_t = torch.tensor(domain_points.reshape(-1, 1), device=device, dtype=torch.float64, requires_grad=True)
    phi = model.compute_shape_functions(domain_t)
    raw_phi = model.compute_shape_functions(domain_t.detach(), return_raw=True)
    grad_phi = torch.zeros((domain_t.shape[0], model.n_nodes, 1), device=device, dtype=torch.float64)
    for index in range(model.n_nodes):
        retain_graph = index + 1 < model.n_nodes
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            domain_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, 0] = grad_i[:, 0]
    boundary_t = torch.tensor(boundary_points, device=device, dtype=torch.float64)
    with torch.no_grad():
        phi_boundary = model.compute_shape_functions(boundary_t).cpu().numpy()
    return evaluate_consistency_bundle_1d(
        phi=phi.detach().cpu().numpy(),
        grad_phi=grad_phi.detach().cpu().numpy(),
        nodes=model.nodes.detach().cpu().numpy(),
        domain_points=domain_points,
        domain_weights=domain_weights,
        boundary_points=boundary_points,
        boundary_weights=boundary_weights,
        boundary_normals=boundary_normals,
        phi_boundary=phi_boundary,
        phi_sum=raw_phi.detach().cpu().numpy().sum(axis=1),
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
        phi_sum=np.sum(phi, axis=1),
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


def history_to_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {f"history_{key}": np.array(value, dtype=np.float64) for key, value in history.items()}


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


def plot_value_consistency_bars(payload: dict[str, Any], path: Path) -> None:
    metric_keys = ["mass_sum_residual", "moment_x_residual", "moment_matrix_residual_fro"]
    labels = ["mass", "mx", "moment_fro"]
    learned = np.array([payload["learned"][key] for key in metric_keys], dtype=np.float64)
    rkpm = np.array([payload["rkpm"][key] for key in metric_keys], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
    _plot_metric_point_panel(ax, labels, learned, rkpm, "Value consistency", "residual")
    _save_figure(fig, path)


def plot_derivative_consistency_bars(payload: dict[str, Any], path: Path) -> None:
    metric_keys = [
        "derivative_boundary_residual_max",
        "derivative_boundary_residual_fro",
        "derivative_matrix_residual_max",
        "derivative_matrix_residual_fro",
    ]
    labels = ["bd-max", "bd-fro", "mat-max", "mat-fro"]
    learned = np.array([payload["learned"][key] for key in metric_keys], dtype=np.float64)
    rkpm = np.array([payload["rkpm"][key] for key in metric_keys], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
    _plot_metric_point_panel(ax, labels, learned, rkpm, "Derivative consistency", "residual")
    _save_figure(fig, path)


def plot_derivative_node_residuals_1d(
    learned_residuals: np.ndarray,
    rkpm_residuals: np.ndarray,
    nodes: np.ndarray,
    path: Path,
) -> None:
    learned_node = np.max(np.abs(learned_residuals), axis=(1, 2))
    rkpm_node = np.max(np.abs(rkpm_residuals), axis=(1, 2))
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.5))
    ax.semilogy(nodes, np.maximum(learned_node, 1e-16), "o-", lw=1.8, color=LEARNED_COLOR, label="learned")
    ax.semilogy(nodes, np.maximum(rkpm_node, 1e-16), "s--", lw=1.8, color=RKPM_COLOR, label="rkpm")
    _style_line_axis(ax, "Nodewise derivative boundary residual", "node x", "residual")
    ax.legend(fontsize=8)
    _save_figure(fig, path)


def plot_main_figure_consistency_1d(
    payload: dict[str, Any],
    learned_arrays: dict[str, np.ndarray],
    rkpm_arrays: dict[str, np.ndarray],
    nodes: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
) -> None:
    x_eval = np.asarray(learned_arrays["domain_points"], dtype=np.float64).reshape(-1)
    order = np.argsort(x_eval)
    x_eval = x_eval[order]
    learned_phi = np.asarray(learned_arrays["phi"], dtype=np.float64)[order]
    rkpm_phi = np.asarray(rkpm_arrays["phi"], dtype=np.float64)[order]
    diff_phi = np.abs(learned_phi - rkpm_phi)
    rep_indices = sorted(set([0, len(nodes) // 2, len(nodes) - 1]))

    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    for index in rep_indices:
        axes[0, 0].plot(x_eval, rkpm_phi[:, index], lw=1.4, ls="--", color=RKPM_COLOR, alpha=0.85)
        axes[0, 0].plot(x_eval, learned_phi[:, index], lw=1.8, color=LEARNED_COLOR, alpha=0.85)
    axes[0, 0].scatter(nodes[rep_indices], np.zeros(len(rep_indices)), s=28, color="black", zorder=3)
    _style_line_axis(axes[0, 0], "Representative shape functions", "x", "phi")
    axes[0, 0].text(0.02, 0.03, "solid: learned, dashed: RKPM", transform=axes[0, 0].transAxes, fontsize=9)

    max_error = np.max(diff_phi, axis=1)
    mean_error = np.mean(diff_phi, axis=1)
    axes[0, 1].semilogy(x_eval, np.maximum(max_error, 1e-16), color=LEARNED_COLOR, lw=1.8, label="max |phi_l-r|")
    axes[0, 1].semilogy(x_eval, np.maximum(mean_error, 1e-16), color=AUX_COLORS[2], lw=1.8, label="mean |phi_l-r|")
    _style_line_axis(axes[0, 1], "Shape error curves", "x", "error")
    axes[0, 1].legend(fontsize=8)

    value_keys = ["mass_sum_residual", "moment_x_residual", "moment_matrix_residual_fro"]
    value_labels = ["mass", "mx", "moment_fro"]
    _plot_metric_point_panel(
        axes[1, 0],
        value_labels,
        np.array([payload["learned"][key] for key in value_keys], dtype=np.float64),
        np.array([payload["rkpm"][key] for key in value_keys], dtype=np.float64),
        "Consistency indicators",
        "residual",
    )

    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)

def plot_consistency_summary(entries: list[dict[str, Any]], path: Path) -> None:
    case_labels = [item["case_label"] for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    value_learned = np.maximum(np.array([item["payload"]["learned"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    value_rkpm = np.maximum(np.array([item["payload"]["rkpm"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    derivative_learned = np.maximum(np.array([item["payload"]["learned"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    derivative_rkpm = np.maximum(np.array([item["payload"]["rkpm"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, learned_values, rkpm_values, title in zip(
        axes,
        [value_learned, derivative_learned],
        [value_rkpm, derivative_rkpm],
        ["Value consistency summary", "Derivative consistency summary"],
    ):
        ax.plot(x_pos, learned_values, "o-", lw=1.8, ms=6, color=LEARNED_COLOR, label="learned")
        ax.plot(x_pos, rkpm_values, "s-", lw=1.8, ms=6, color=RKPM_COLOR, label="rkpm")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        _style_line_axis(ax, title, "case", "residual", log_scale=True)
        ax.legend(fontsize=8)
    _save_figure(fig, path)
