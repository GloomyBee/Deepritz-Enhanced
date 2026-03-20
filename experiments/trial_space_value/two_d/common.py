from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.utils import ensure_repo_root_on_path


ROOT_DIR = ensure_repo_root_on_path(Path(__file__).resolve())

torch.set_default_dtype(torch.float64)


VARIANT_PRESETS = {
    "softplus_raw_pu_bd": {
        "use_softplus": True,
        "enable_fallback": True,
        "use_linear_loss": True,
        "use_pu_loss": True,
        "use_bd_loss": True,
        "use_teacher_loss": False,
        "lambda_pu": 0.1,
        "lambda_bd": 1.0,
        "lambda_teacher": 0.0,
        "lambda_reg": 0.0,
    },
    "no_softplus_raw_pu_bd": {
        "use_softplus": False,
        "enable_fallback": True,
        "use_linear_loss": True,
        "use_pu_loss": True,
        "use_bd_loss": True,
        "use_teacher_loss": False,
        "lambda_pu": 0.1,
        "lambda_bd": 1.0,
        "lambda_teacher": 0.0,
        "lambda_reg": 0.0,
    },
    "no_softplus_teacher": {
        "use_softplus": False,
        "enable_fallback": True,
        "use_linear_loss": False,
        "use_pu_loss": False,
        "use_bd_loss": True,
        "use_teacher_loss": True,
        "lambda_pu": 0.0,
        "lambda_bd": 0.1,
        "lambda_teacher": 1.0,
        "lambda_reg": 1e-4,
    },
    "no_softplus_teacher_reg": {
        "use_softplus": False,
        "enable_fallback": True,
        "use_linear_loss": False,
        "use_pu_loss": False,
        "use_bd_loss": True,
        "use_teacher_loss": True,
        "lambda_pu": 0.0,
        "lambda_bd": 0.1,
        "lambda_teacher": 1.0,
        "lambda_reg": 5e-4,
    },
    "no_softplus_raw_pu_bd_no_fallback": {
        "use_softplus": False,
        "enable_fallback": False,
        "use_linear_loss": True,
        "use_pu_loss": True,
        "use_bd_loss": True,
        "use_teacher_loss": False,
        "lambda_pu": 0.1,
        "lambda_bd": 1.0,
        "lambda_teacher": 0.0,
        "lambda_reg": 0.0,
    },
}


METHOD_LABELS = {
    "classical": "Fixed basis + classical solve",
    "frozen_w": "Fixed basis + optimize w",
    "joint": "Joint Phase B",
}


FIGURE_DPI = 180
MAIN_FIGURE_SIZE = (12.0, 9.0)
SUMMARY_FIGURE_SIZE = (12.0, 8.2)
DIAGNOSTIC_FIGURE_SIZE = (6.0, 5.0)
LEARNED_COLOR = "#1f4e79"
RKPM_COLOR = "#b03a2e"
AUX_COLORS = ["#1f4e79", "#b03a2e", "#2f7d32", "#7a3e9d", "#8c5a2b"]
SOLUTION_CMAP = "inferno"
ERROR_CMAP = "magma"
STRUCTURE_CMAP = "viridis"


@dataclass
class RunArtifacts:
    root_dir: Path
    figures_dir: Path
    diagnostics_dir: Path


@dataclass
class TrialSpaceArtifacts:
    root_dir: Path
    figures_dir: Path
    diagnostics_dir: Path
    methods_dir: Path


def safe_token(value: Any) -> str:
    token = str(value).strip().replace(".", "p")
    token = token.replace(" ", "_").replace("/", "_")
    return token


def build_case_name(
    *,
    variant: str | None = None,
    n_side: int | None = None,
    kappa: float | None = None,
    seed: int | None = None,
    jitter: float | None = None,
    tag: str = "",
) -> str:
    parts: list[str] = []
    if variant:
        parts.append(f"variant_{safe_token(variant)}")
    if n_side is not None:
        parts.append(f"ns{n_side}")
    if kappa is not None:
        parts.append(f"k{safe_token(kappa)}")
    if jitter is not None:
        parts.append(f"jit{safe_token(jitter)}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(safe_token(tag))
    return "_".join(parts)


def ensure_run_artifacts(group: str, case_name: str) -> RunArtifacts:
    root_dir = ROOT_DIR / "output" / "shape_validation" / "two_d" / group / case_name
    figures_dir = root_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(root_dir=root_dir, figures_dir=figures_dir)


def ensure_trial_space_artifacts(group: str, case_name: str) -> TrialSpaceArtifacts:
    root_dir = ROOT_DIR / "output" / "trial_space_value" / "two_d" / group / case_name
    figures_dir = root_dir / "figures"
    diagnostics_dir = figures_dir / "diagnostics"
    methods_dir = root_dir / "methods"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    return TrialSpaceArtifacts(root_dir=root_dir, figures_dir=figures_dir, diagnostics_dir=diagnostics_dir, methods_dir=methods_dir)


def ensure_method_artifacts(artifacts: TrialSpaceArtifacts, method_name: str) -> RunArtifacts:
    root_dir = artifacts.methods_dir / method_name
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


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    ):
        super().__init__()
        grid_min, grid_max = grid_range
        grid = torch.linspace(grid_min, grid_max, num_basis)
        h = (grid_max - grid_min) / (num_basis - 1)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.register_buffer("grid", grid)
        self.register_buffer("h", torch.tensor(h))
        self.layer1 = nn.ModuleList(
            [nn.Linear(num_basis, hidden_dim, bias=False) for _ in range(input_dim)]
        )
        self.layer2 = nn.Linear(hidden_dim * num_basis, 1, bias=False)

    def _hat_basis(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        batch_size, dim = x.shape
        grid = self.grid.view(1, 1, self.num_basis)
        x_exp = x.view(batch_size, dim, 1)
        return torch.relu(1.0 - torch.abs(x_exp - grid) / self.h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_sum = 0.0
        for index, layer in enumerate(self.layer1):
            basis = self._hat_basis(x[:, index : index + 1]).squeeze(1)
            hidden_sum = hidden_sum + layer(basis)
        basis_hidden = self._hat_basis(hidden_sum)
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


class LinearPatchProblem2D:
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 + x[:, 0:1] + 2.0 * x[:, 1:2]

    def exact_gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros_like(x)
        grad[:, 0] = 1.0
        grad[:, 1] = 2.0
        return grad

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)


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


def resolve_variant_config(name: str) -> dict[str, Any]:
    if name not in VARIANT_PRESETS:
        raise ValueError(f"Unknown variant: {name}")
    return dict(VARIANT_PRESETS[name])


def merge_histories(
    history_a: dict[str, list[float]],
    history_b: dict[str, list[float]],
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    keys = set(history_a.keys()) | set(history_b.keys())
    step_shift = history_a.get("steps", [])[-1] + 1.0 if history_a.get("steps") else 0.0
    for key in keys:
        values_a = list(history_a.get(key, []))
        values_b = list(history_b.get(key, []))
        if key == "steps":
            values_b = [item + step_shift for item in values_b]
        merged[key] = values_a + values_b
    return merged


def corner_anchor_loss(phi_corner: torch.Tensor) -> torch.Tensor:
    if phi_corner.shape[1] < 4:
        return torch.tensor(0.0, device=phi_corner.device, dtype=phi_corner.dtype)
    loss = (phi_corner[0, 0] - 1.0) ** 2 + torch.sum(phi_corner[0, 1:] ** 2)
    loss = loss + (phi_corner[1, -1] - 1.0) ** 2 + torch.sum(phi_corner[1, :-1] ** 2)
    loss = loss + (phi_corner[2, phi_corner.shape[1] // 2 - 1] * 0.0)
    return loss


def compute_corner_anchor_loss(model: MeshfreeKAN2D, x_corners: torch.Tensor) -> torch.Tensor:
    phi_corner = model.compute_shape_functions(x_corners)
    targets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    loss = torch.tensor(0.0, device=x_corners.device, dtype=x_corners.dtype)
    for row_idx, coord in enumerate(targets):
        distances = torch.sum((model.nodes - torch.tensor(coord, device=model.nodes.device)) ** 2, dim=1)
        anchor_idx = int(torch.argmin(distances).item())
        loss = loss + (phi_corner[row_idx, anchor_idx] - 1.0) ** 2
        if anchor_idx > 0:
            loss = loss + torch.sum(phi_corner[row_idx, :anchor_idx] ** 2)
        if anchor_idx + 1 < model.n_nodes:
            loss = loss + torch.sum(phi_corner[row_idx, anchor_idx + 1 :] ** 2)
    return loss


def init_history() -> dict[str, list[float]]:
    return {
        "steps": [],
        "loss": [],
        "linear": [],
        "pu": [],
        "bd": [],
        "teacher": [],
        "reg": [],
        "energy": [],
        "bc": [],
        "val_l2": [],
        "val_h1": [],
    }


def append_history(history: dict[str, list[float]], step: int, **values: float) -> None:
    history["steps"].append(float(step))
    for key, value in values.items():
        if key in history:
            history[key].append(float(value))


def train_phase_a(
    model: MeshfreeKAN2D,
    nodes: torch.Tensor,
    device: str,
    variant: str,
    steps: int = 1500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    cfg = resolve_variant_config(variant)
    history = init_history()
    optimizer = torch.optim.Adam(model.kan.parameters(), lr=lr)
    nodes_x = nodes[:, 0:1]
    nodes_y = nodes[:, 1:2]
    x_corners = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=nodes.dtype,
        device=device,
    )

    for step in range(steps):
        x_domain = sample_mixed_domain_points(nodes, model.support_radius, batch_size, device)
        phi = model.compute_shape_functions(x_domain)
        repro_x = phi @ nodes_x
        repro_y = phi @ nodes_y
        loss_linear = torch.mean((repro_x - x_domain[:, 0:1]) ** 2 + (repro_y - x_domain[:, 1:2]) ** 2)

        phi_raw = model.compute_shape_functions(x_domain, return_raw=True)
        loss_pu = torch.mean((torch.sum(phi_raw, dim=1, keepdim=True) - 1.0) ** 2)
        loss_bd = compute_corner_anchor_loss(model, x_corners)
        loss_reg = torch.mean(phi_raw**2)
        if cfg["use_teacher_loss"]:
            x_teacher = torch.cat([x_domain, x_corners], dim=0)
            phi_all = model.compute_shape_functions(x_teacher)
            with torch.no_grad():
                phi_teacher = rkpm_shape_matrix_2d_torch(x_teacher, nodes, model.support_radius)
            loss_teacher = torch.mean((phi_all - phi_teacher) ** 2)
        else:
            loss_teacher = torch.tensor(0.0, device=device, dtype=nodes.dtype)

        loss = torch.tensor(0.0, device=device, dtype=nodes.dtype)
        if cfg["use_linear_loss"]:
            loss = loss + loss_linear
        if cfg["use_pu_loss"]:
            loss = loss + cfg["lambda_pu"] * loss_pu
        if cfg["use_bd_loss"]:
            loss = loss + cfg["lambda_bd"] * loss_bd
        if cfg["use_teacher_loss"]:
            loss = loss + cfg["lambda_teacher"] * loss_teacher
        if cfg["lambda_reg"] > 0.0:
            loss = loss + cfg["lambda_reg"] * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == steps - 1:
            append_history(
                history,
                step,
                loss=loss.item(),
                linear=loss_linear.item(),
                pu=loss_pu.item(),
                bd=loss_bd.item(),
                teacher=loss_teacher.item(),
                reg=loss_reg.item(),
            )
            print(
                f"PhaseA {variant} | step={step:4d} | loss={loss.item():.3e} | "
                f"linear={loss_linear.item():.3e} | pu={loss_pu.item():.3e} | "
                f"bd={loss_bd.item():.3e} | teacher={loss_teacher.item():.3e}"
            )
    return history


def train_phase_b_poisson(
    model: MeshfreeKAN2D,
    problem: SinusoidalPoissonProblem2D,
    device: str,
    steps: int = 2000,
    batch_size: int = 1024,
    lr_kan: float = 1e-4,
    lr_w: float = 1e-2,
    beta_bc: float = 100.0,
    gamma_linear: float = 10.0,
    warmup_w_steps: int = 300,
    eval_interval: int = 100,
    eval_resolution: int = 41,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    history = init_history()
    model.init_w_from_exact(problem.exact_solution)
    optimizer = torch.optim.Adam(
        [
            {"params": model.kan.parameters(), "lr": lr_kan},
            {"params": [model.w], "lr": lr_w},
        ]
    )
    optimizer_w = torch.optim.Adam([model.w], lr=lr_w)
    nodes_x = model.nodes[:, 0:1]
    nodes_y = model.nodes[:, 1:2]
    _, _, eval_points = grid_points(eval_resolution)
    eval_points_t = torch.tensor(eval_points, device=device, dtype=model.nodes.dtype, requires_grad=True)
    best_score = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for step in range(steps):
        x_domain = torch.rand(batch_size, 2, device=device, dtype=model.nodes.dtype)
        x_domain.requires_grad_(True)
        x_boundary = sample_square_boundary(batch_size, device)
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

        current_optimizer = optimizer_w if step < warmup_w_steps else optimizer
        current_optimizer.zero_grad()
        loss.backward()
        current_optimizer.step()

        val_l2 = None
        val_h1 = None
        if step % eval_interval == 0 or step == steps - 1:
            with torch.enable_grad():
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
            append_history(
                history,
                step,
                loss=loss.item(),
                energy=loss_energy.item(),
                bc=loss_bc.item(),
                linear=loss_linear.item(),
                val_l2=val_l2 if val_l2 is not None else 0.0,
                val_h1=val_h1 if val_h1 is not None else 0.0,
            )
            print(
                f"PhaseB | step={step:4d} | loss={loss.item():.3e} | "
                f"energy={loss_energy.item():.3e} | bc={loss_bc.item():.3e} | "
                f"linear={loss_linear.item():.3e} | val_l2={(val_l2 if val_l2 is not None else float('nan')):.3e} | "
                f"val_h1={(val_h1 if val_h1 is not None else float('nan')):.3e}"
            )
    model.load_state_dict(best_state)
    return history


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
    x_boundary = sample_square_boundary(2048, device)
    with torch.no_grad():
        u_boundary = model(x_boundary)
        metrics["boundary_l2"] = float(torch.sqrt(torch.mean((u_boundary - problem.exact_solution(x_boundary)) ** 2)).item())
    pred_np = u_pred.detach().cpu().numpy().reshape(resolution, resolution)
    exact_np = u_exact.detach().cpu().numpy().reshape(resolution, resolution)
    error_np = np.abs(pred_np - exact_np)
    return metrics, x_grid, y_grid, pred_np, exact_np + 0.0 * error_np


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


def _plot_history_series(
    ax: plt.Axes,
    history: dict[str, list[float]],
    phase_split_step: float | None = None,
) -> None:
    steps = np.array(history.get("steps", []), dtype=np.float64)
    if steps.size == 0:
        ax.text(0.5, 0.56, "No iterative training", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.42, "Classical solve / direct assembly", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        _style_line_axis(ax, "Solver history", "step", "loss / metric")
        return
    for key, color in zip(["loss", "linear", "pu", "bd", "teacher", "reg", "energy", "bc", "val_l2", "val_h1"], AUX_COLORS + ["#666666", AUX_COLORS[0], AUX_COLORS[1], AUX_COLORS[2], AUX_COLORS[3]]):
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0:
            continue
        if key in {"val_l2", "val_h1"}:
            values = values[values > 0.0]
            x_values = steps[-values.size :] if values.size else np.array([], dtype=np.float64)
        else:
            count = min(values.size, steps.size)
            values = values[:count]
            x_values = steps[:count]
        if values.size == 0 or x_values.size == 0 or np.allclose(values, 0.0):
            continue
        ax.semilogy(x_values, np.maximum(values, 1e-16), lw=1.8, color=color, label=key)
    if phase_split_step is not None:
        ax.axvline(phase_split_step, color="#444444", lw=1.2, ls=":", label="phase switch")
    _style_line_axis(ax, "Training History", "step", "loss / metric")
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)


def _plot_field_on_axis(fig: plt.Figure, ax: plt.Axes, x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray, title: str, cmap: str) -> None:
    contour = ax.contourf(x_grid, y_grid, values, levels=100, cmap=cmap)
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def plot_training_curves(
    history: dict[str, list[float]],
    path: Path,
    title: str,
    phase_split_step: float | None = None,
) -> None:
    steps = np.array(history.get("steps", []), dtype=np.float64)

    def plot_series(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, label: str) -> None:
        if x_values.size == 0 or y_values.size == 0:
            return
        if np.allclose(y_values, 0.0):
            return
        ax.semilogy(x_values, np.maximum(y_values, 1e-16), lw=1.8, label=label)

    def finalize(ax: plt.Axes, fig: plt.Figure, save_path: Path, figure_title: str) -> None:
        _style_line_axis(ax, figure_title, "step", "loss / metric")
        if ax.lines:
            ax.legend(fontsize=8)
        _save_figure(fig, save_path)

    if phase_split_step is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for key in ["loss", "linear", "pu", "bd", "teacher", "reg", "energy", "bc"]:
            values = np.array(history.get(key, []), dtype=np.float64)
            count = min(len(steps), len(values))
            if count == 0:
                continue
            plot_series(ax, steps[:count], values[:count], key)
        finalize(ax, fig, path, title)
        return

    split_idx = int(np.searchsorted(steps, phase_split_step, side="left"))
    shared_keys = [("loss", "total"), ("linear", "linear")]
    phase_a_only = [("pu", "pu"), ("bd", "bd"), ("teacher", "teacher"), ("reg", "reg")]
    phase_b_only = [("energy", "energy"), ("bc", "bc"), ("val_l2", "val_l2"), ("val_h1", "val_h1")]
    phase_a_path = path.with_name(f"{path.stem}_phase_a{path.suffix}")
    phase_b_path = path.with_name(f"{path.stem}_phase_b{path.suffix}")

    fig_a, ax_a = plt.subplots(1, 1, figsize=(8, 4.5))
    for key, label in shared_keys:
        values = np.array(history.get(key, []), dtype=np.float64)
        count = min(split_idx, values.size)
        if count == 0:
            continue
        plot_series(ax_a, steps[:count], values[:count], label)
    for key, label in phase_a_only:
        values = np.array(history.get(key, []), dtype=np.float64)
        count = min(split_idx, values.size)
        if count == 0:
            continue
        plot_series(ax_a, steps[:count], values[:count], label)
    finalize(ax_a, fig_a, phase_a_path, f"{title} - Phase A")

    fig_b, ax_b = plt.subplots(1, 1, figsize=(8, 4.5))
    for key, label in shared_keys:
        values = np.array(history.get(key, []), dtype=np.float64)
        count = min(len(steps), values.size)
        if count <= split_idx:
            continue
        plot_series(ax_b, steps[split_idx:count], values[split_idx:count], label)
    for key, label in phase_b_only:
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0:
            continue
        if key in {"val_l2", "val_h1"}:
            values = values[values > 0.0]
        if values.size == 0:
            continue
        plot_series(ax_b, steps[-values.size :], values, label)
    finalize(ax_b, fig_b, phase_b_path, f"{title} - Phase B")


def plot_heatmap(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    path: Path,
    title: str,
    cmap: str = STRUCTURE_CMAP,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=DIAGNOSTIC_FIGURE_SIZE)
    _plot_field_on_axis(fig, ax, x_grid, y_grid, values, title, cmap)
    _save_figure(fig, path)


def plot_solution_triplet(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    path: Path,
    title_prefix: str,
) -> None:
    error = np.abs(pred - exact)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, values, title, cmap in [
        (axes[0], pred, f"{title_prefix} Prediction", SOLUTION_CMAP),
        (axes[1], exact, "Exact Solution", SOLUTION_CMAP),
        (axes[2], error, "Absolute Error", ERROR_CMAP),
    ]:
        _plot_field_on_axis(fig, axis, x_grid, y_grid, values, title, cmap)
    _save_figure(fig, path)


def plot_main_figure_trial_method(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    title_prefix: str,
    path: Path,
) -> None:
    error = np.abs(pred - exact)
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, pred, f"{title_prefix} prediction", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[0, 1], x_grid, y_grid, exact, "Exact field", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[1, 0], x_grid, y_grid, error, "Absolute error", ERROR_CMAP)
    if history.get("steps"):
        _plot_history_series(axes[1, 1], history)
    else:
        axes[1, 1].axis("off")
        lines = [
            "Solver information",
            "",
            f"solver: {metrics.get('solver_name', 'N/A')}",
            f"cond_k: {metrics.get('cond_k', float('nan')):.3e}",
            f"lambda_min_k: {metrics.get('lambda_min_k', float('nan')):.3e}",
            f"solver_residual: {metrics.get('solver_residual', float('nan')):.3e}",
            f"solver_shift: {metrics.get('solver_shift', float('nan')):.3e}",
        ]
        axes[1, 1].text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    _save_figure(fig, path)

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


def compose_frozen_w_loss(
    loss_energy: torch.Tensor,
    loss_bc: torch.Tensor,
    beta_bc: float,
) -> torch.Tensor:
    return loss_energy + beta_bc * loss_bc


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
        x_boundary = sample_square_boundary(batch_size, device)
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
        loss = compose_frozen_w_loss(
            loss_energy=loss_energy,
            loss_bc=loss_bc,
            beta_bc=beta_bc,
        )

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
    x_pos = np.arange(len(entries), dtype=np.float64)
    l2 = np.maximum(np.array([item["l2_error"] for item in entries], dtype=np.float64), 1e-16)
    h1 = np.maximum(np.array([item["h1_semi_error"] for item in entries], dtype=np.float64), 1e-16)
    boundary = np.maximum(np.array([item["boundary_l2"] for item in entries], dtype=np.float64), 1e-16)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, values, title, color, marker in zip(
        axes,
        [l2, h1, boundary],
        ["L2 error", "H1 semi error", "Boundary L2"],
        [LEARNED_COLOR, RKPM_COLOR, AUX_COLORS[2]],
        ["o", "s", "^"] ,
    ):
        ax.plot(x_pos, values, marker=marker, lw=1.8, ms=6, color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        _style_line_axis(ax, title, "method", "error", log_scale=True)
    _save_figure(fig, path)


def plot_conditioning_bars(entries: list[dict[str, Any]], path: Path) -> None:
    methods = [item["method_label"] for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    cond_values = np.maximum(np.array([item["cond_k"] for item in entries], dtype=np.float64), 1e-16)
    lambda_min = np.array([item["lambda_min_k"] for item in entries], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].plot(x_pos, cond_values, "o-", lw=1.8, ms=6, color=LEARNED_COLOR)
    axes[1].plot(x_pos, lambda_min, "s-", lw=1.8, ms=6, color=RKPM_COLOR)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    _style_line_axis(axes[0], "Condition number", "method", "cond_k", log_scale=True)
    _style_line_axis(axes[1], "Smallest eigenvalue", "method", "lambda_min_k")
    _save_figure(fig, path)

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



