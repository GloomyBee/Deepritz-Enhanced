import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.splines import evaluate_open_uniform_bspline_basis

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


FIGURE_DPI = 180
MAIN_FIGURE_SIZE = (12.0, 9.0)
SUMMARY_FIGURE_SIZE = (12.0, 8.2)
DIAGNOSTIC_FIGURE_SIZE = (6.0, 5.0)
LEARNED_COLOR = "#1f4e79"
RKPM_COLOR = "#b03a2e"
AUX_COLORS = ["#1f4e79", "#b03a2e", "#2f7d32", "#7a3e9d", "#8c5a2b"]
SOLUTION_CMAP = "inferno"
ERROR_CMAP = "magma"
RESIDUAL_CMAP = "coolwarm"
STRUCTURE_CMAP = "viridis"


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
    x_eval, y_eval, eval_points = grid_points(eval_resolution)
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
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        _style_line_axis(ax, "Training History", "step", "loss / metric")
        return
    color_map = {
        "loss": AUX_COLORS[0],
        "linear": AUX_COLORS[2],
        "pu": AUX_COLORS[1],
        "bd": AUX_COLORS[3],
        "teacher": AUX_COLORS[4],
        "reg": "#666666",
        "energy": AUX_COLORS[0],
        "bc": AUX_COLORS[1],
        "val_l2": AUX_COLORS[2],
        "val_h1": AUX_COLORS[3],
    }
    for key in ["loss", "linear", "pu", "bd", "teacher", "reg", "energy", "bc", "val_l2", "val_h1"]:
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0:
            continue
        if key in {"val_l2", "val_h1"}:
            mask = values > 0.0
            values = values[mask]
            x_values = steps[-values.size :] if values.size else np.array([], dtype=np.float64)
        else:
            count = min(values.size, steps.size)
            values = values[:count]
            x_values = steps[:count]
        if values.size == 0 or x_values.size == 0 or np.allclose(values, 0.0):
            continue
        ax.semilogy(x_values, np.maximum(values, 1e-16), lw=1.8, label=key, color=color_map.get(key))
    if phase_split_step is not None:
        ax.axvline(phase_split_step, color="#444444", lw=1.2, ls=":", label="phase switch")
    _style_line_axis(ax, "Training History", "step", "loss / metric")
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)


def _plot_field_on_axis(
    fig: plt.Figure,
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    symmetric: bool = False,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    kwargs: dict[str, Any] = {"levels": 100, "cmap": cmap}
    if symmetric:
        vmax = float(np.max(np.abs(values)))
        vmax = max(vmax, 1e-16)
        kwargs["vmin"] = -vmax
        kwargs["vmax"] = vmax
    contour = ax.contourf(x_grid, y_grid, values, **kwargs)
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def _plot_nodes_on_axis(ax: plt.Axes, nodes: np.ndarray, title: str) -> None:
    ax.scatter(nodes[:, 0], nodes[:, 1], s=16, color=LEARNED_COLOR, edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.25)


def _plot_metric_point_panel(
    ax: plt.Axes,
    labels: list[str],
    series: list[tuple[str, np.ndarray, str, str]],
    title: str,
    ylabel: str,
    log_scale: bool = True,
) -> None:
    x_pos = np.arange(len(labels), dtype=np.float64)
    for label, values, color, marker in series:
        arr = np.maximum(np.asarray(values, dtype=np.float64), 1e-16)
        ax.plot(x_pos, arr, marker=marker, lw=1.8, ms=6, color=color, label=label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    _style_line_axis(ax, title, "metric", ylabel, log_scale=log_scale)
    ax.legend(fontsize=8)


def _plot_metric_text_panel(ax: plt.Axes, title: str, metrics: list[tuple[str, float]]) -> None:
    ax.axis("off")
    lines = [title, ""]
    for key, value in metrics:
        lines.append(f"{key}: {value:.3e}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")


def _plot_matrix_panel(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 3:
        array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    vmax = float(np.max(np.abs(array)))
    vmax = max(vmax, 1e-16)
    image = ax.imshow(array, cmap=RESIDUAL_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def _plot_solution_triptych_cell(
    fig: plt.Figure,
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    title_prefix: str,
) -> None:
    subspec = ax.get_subplotspec().subgridspec(1, 3, wspace=0.24)
    fig.delaxes(ax)
    error = np.abs(pred - exact)
    fields = [
        (pred, f"{title_prefix} prediction", SOLUTION_CMAP, False),
        (exact, "Exact solution", SOLUTION_CMAP, False),
        (error, "Absolute error", ERROR_CMAP, False),
    ]
    for index, (values, title, cmap, symmetric) in enumerate(fields):
        inner_ax = fig.add_subplot(subspec[0, index])
        _plot_field_on_axis(fig, inner_ax, x_grid, y_grid, values, title, cmap=cmap, symmetric=symmetric)


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
    symmetric = cmap == RESIDUAL_CMAP
    _plot_field_on_axis(fig, ax, x_grid, y_grid, values, title, cmap=cmap, symmetric=symmetric)
    _save_figure(fig, path)


def plot_solution_triplet(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    path: Path,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    _plot_solution_triptych_cell(fig, ax, x_grid, y_grid, pred, exact, title_prefix)
    _save_figure(fig, path)


def plot_main_figure_patch_test(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pu_field: np.ndarray,
    linear_x_field: np.ndarray,
    linear_y_field: np.ndarray,
    lambda_field: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
    phase_split_step: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    linear_field = linear_x_field
    linear_title = "Linear reproduction x residual"
    if np.max(np.abs(linear_y_field)) > np.max(np.abs(linear_x_field)):
        linear_field = linear_y_field
        linear_title = "Linear reproduction y residual"
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, pu_field, "PU residual field", RESIDUAL_CMAP, symmetric=True)
    _plot_field_on_axis(fig, axes[0, 1], x_grid, y_grid, linear_field, linear_title, RESIDUAL_CMAP, symmetric=True)
    _plot_field_on_axis(fig, axes[1, 0], x_grid, y_grid, lambda_field, "Lambda_h field", STRUCTURE_CMAP)
    _plot_history_series(axes[1, 1], history, phase_split_step=phase_split_step)
    _save_figure(fig, path)


def plot_patch_test_summary_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    linear_x_field: np.ndarray,
    linear_y_field: np.ndarray,
    nodes: np.ndarray,
    metrics: dict[str, float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGURE_SIZE)
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, linear_x_field, "Linear reproduction x", RESIDUAL_CMAP, symmetric=True)
    _plot_field_on_axis(fig, axes[0, 1], x_grid, y_grid, linear_y_field, "Linear reproduction y", RESIDUAL_CMAP, symmetric=True)
    _plot_nodes_on_axis(axes[1, 0], nodes, "Node layout")
    _plot_metric_text_panel(
        axes[1, 1],
        "Patch metrics",
        [
            ("pu_rmse", metrics["pu_rmse"]),
            ("pu_linf", metrics["pu_linf"]),
            ("linear_x_rmse", metrics["linear_x_rmse"]),
            ("linear_y_rmse", metrics["linear_y_rmse"]),
            ("lambda_h_max", metrics["lambda_h_max"]),
            ("lambda_h_mean", metrics["lambda_h_mean"]),
        ],
    )
    _save_figure(fig, path)


def plot_main_figure_stability(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    lambda_field: np.ndarray,
    nodes: np.ndarray,
    metrics: dict[str, float],
    history: dict[str, list[float]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, lambda_field, "Lambda_h field", STRUCTURE_CMAP)
    metric_labels = ["PU", "Lin-x", "Lin-y", "Lh-mean", "Lh-max"]
    metric_values = np.array(
        [metrics["pu_rmse"], metrics["linear_x_rmse"], metrics["linear_y_rmse"], metrics["lambda_h_mean"], metrics["lambda_h_max"]],
        dtype=np.float64,
    )
    _plot_metric_point_panel(
        axes[0, 1],
        metric_labels,
        [("variant", metric_values, LEARNED_COLOR, "o")],
        "Structure indicators",
        "value",
        log_scale=True,
    )
    _plot_nodes_on_axis(axes[1, 0], nodes, "Node layout")
    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)


def plot_stability_summary_2d(entries: list[dict[str, Any]], path: Path) -> None:
    variants = [item["variant"] for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    lambda_max = np.maximum(np.array([item["lambda_h_max"] for item in entries], dtype=np.float64), 1e-16)
    linear_rmse = np.maximum(
        np.array([max(item["linear_x_rmse"], item["linear_y_rmse"]) for item in entries], dtype=np.float64),
        1e-16,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(x_pos, lambda_max, "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
    axes[1].plot(x_pos, linear_rmse, "s-", color=RKPM_COLOR, lw=1.8, ms=6)
    for ax, title, ylabel in [
        (axes[0], "Variant vs Lambda_h", "lambda_h_max"),
        (axes[1], "Variant vs linear residual", "max linear rmse"),
    ]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        _style_line_axis(ax, title, "variant", ylabel, log_scale=True)
    _save_figure(fig, path)


def plot_main_figure_poisson(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
    phase_split_step: float | None = None,
) -> None:
    error = np.abs(pred - exact)
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, pred, "Predicted field", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[0, 1], x_grid, y_grid, exact, "Exact field", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[1, 0], x_grid, y_grid, error, "Absolute error", ERROR_CMAP)
    _plot_history_series(axes[1, 1], history, phase_split_step=phase_split_step)
    _save_figure(fig, path)


def plot_poisson_convergence_summary_2d(entries: list[dict[str, Any]], path: Path) -> None:
    hs = np.array([item["h"] for item in entries], dtype=np.float64)
    l2 = np.maximum(np.array([item["l2_error"] for item in entries], dtype=np.float64), 1e-16)
    h1 = np.maximum(np.array([item["h1_semi_error"] for item in entries], dtype=np.float64), 1e-16)
    n_sides = np.array([item["n_side"] for item in entries], dtype=np.int64)
    order = np.argsort(hs)
    hs = hs[order]
    l2 = l2[order]
    h1 = h1[order]
    n_sides = n_sides[order]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    ax.loglog(hs, l2, "o-", ms=7, lw=1.8, color=LEARNED_COLOR, label="L2")
    ax.loglog(hs, h1, "s-", ms=7, lw=1.8, color=RKPM_COLOR, label="H1 semi")
    for x_value, y_value, n_side in zip(hs, l2, n_sides):
        ax.annotate(f"n={n_side}", (x_value, y_value), textcoords="offset points", xytext=(4, 5), fontsize=8)
    _style_line_axis(ax, "Poisson convergence", "h", "error")
    ax.legend()
    _save_figure(fig, path)


def plot_main_figure_irregular_nodes(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    nodes: np.ndarray,
    lambda_field: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
    phase_split_step: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    _plot_solution_triptych_cell(fig, axes[0, 0], x_grid, y_grid, pred, exact, "Deep-RKPM-KAN")
    _plot_nodes_on_axis(axes[0, 1], nodes, "Irregular node layout")
    _plot_field_on_axis(fig, axes[1, 0], x_grid, y_grid, lambda_field, "Lambda_h field", STRUCTURE_CMAP)
    _plot_history_series(axes[1, 1], history, phase_split_step=phase_split_step)
    _save_figure(fig, path)


def plot_irregular_summary_2d(entries: list[dict[str, Any]], path: Path) -> None:
    jitters = np.array([item["jitter"] for item in entries], dtype=np.float64)
    l2 = np.maximum(np.array([item["l2_error"] for item in entries], dtype=np.float64), 1e-16)
    lam = np.maximum(np.array([item["lambda_h_max"] for item in entries], dtype=np.float64), 1e-16)
    order = np.argsort(jitters)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(jitters[order], l2[order], "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
    axes[1].plot(jitters[order], lam[order], "s-", color=RKPM_COLOR, lw=1.8, ms=6)
    _style_line_axis(axes[0], "Jitter vs L2 error", "jitter", "L2 error", log_scale=True)
    _style_line_axis(axes[1], "Jitter vs Lambda_h", "jitter", "lambda_h_max", log_scale=True)
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




def gauss_legendre_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return points.astype(np.float64), weights.astype(np.float64)


def square_domain_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    points_1d, weights_1d = gauss_legendre_1d(order)
    x_grid, y_grid = np.meshgrid(points_1d, points_1d, indexing="xy")
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing="xy")
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    weights = (wx * wy).ravel()
    return points, weights.astype(np.float64)


def square_boundary_quadrature(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_points, edge_weights = gauss_legendre_1d(order)
    zeros = np.zeros_like(edge_points)
    ones = np.ones_like(edge_points)
    points = np.vstack(
        [
            np.column_stack([zeros, edge_points]),
            np.column_stack([ones, edge_points]),
            np.column_stack([edge_points, zeros]),
            np.column_stack([edge_points, ones]),
        ]
    )
    weights = np.concatenate([edge_weights, edge_weights, edge_weights, edge_weights]).astype(np.float64)
    normals = np.vstack(
        [
            np.tile(np.array([[-1.0, 0.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[0.0, -1.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (order, 1)),
        ]
    )
    return points, weights, normals


def polynomial_basis_2d(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)
    return np.column_stack([np.ones(xy.shape[0], dtype=np.float64), xy[:, 0], xy[:, 1]])


def polynomial_gradients_2d(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)
    gradients = np.zeros((xy.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 1, 0] = 1.0
    gradients[:, 2, 1] = 1.0
    return gradients


def analytic_moment_matrix_2d() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0 / 3.0, 0.25],
            [0.5, 0.25, 1.0 / 3.0],
        ],
        dtype=np.float64,
    )


def analytic_derivative_moment_matrices_2d() -> np.ndarray:
    return np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.5],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.5],
            ],
        ],
        dtype=np.float64,
    )


def compute_support_metrics(phi_sum: np.ndarray, orphan_tol: float = 1e-14) -> dict[str, float]:
    phi_sum = np.asarray(phi_sum, dtype=np.float64).reshape(-1)
    return {
        "orphan_ratio": float(np.mean(np.abs(phi_sum) < orphan_tol)),
        "phi_sum_min": float(np.min(phi_sum)),
        "phi_sum_p01": float(np.quantile(phi_sum, 0.01)),
    }


def compute_value_consistency_metrics_2d(
    phi: np.ndarray,
    nodes: np.ndarray,
    domain_points: np.ndarray,
    domain_weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    phi = np.asarray(phi, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    domain_points = np.asarray(domain_points, dtype=np.float64)
    domain_weights = np.asarray(domain_weights, dtype=np.float64)
    p_nodes = polynomial_basis_2d(nodes)
    p_eval = polynomial_basis_2d(domain_points)
    nodal_integrals = phi.T @ domain_weights
    reproduced = phi @ p_nodes
    moment_matrix = (reproduced * domain_weights[:, None]).T @ p_eval
    exact_moment = analytic_moment_matrix_2d()
    pu_residual_field = np.sum(phi, axis=1) - 1.0
    x_repro_field = reproduced[:, 1] - domain_points[:, 0]
    y_repro_field = reproduced[:, 2] - domain_points[:, 1]
    metrics = {
        "mass_sum_residual": float(abs(np.sum(nodal_integrals) - 1.0)),
        "moment_x_residual": float(abs(np.sum(nodes[:, 0] * nodal_integrals) - 0.5)),
        "moment_y_residual": float(abs(np.sum(nodes[:, 1] * nodal_integrals) - 0.5)),
        "int_pu_residual": float(abs(np.sum(pu_residual_field * domain_weights))),
        "int_x_repro_residual": float(abs(np.sum(x_repro_field * domain_weights))),
        "int_y_repro_residual": float(abs(np.sum(y_repro_field * domain_weights))),
        "moment_matrix_residual_fro": float(np.linalg.norm(moment_matrix - exact_moment, ord="fro")),
        "moment_matrix_residual_max": float(np.max(np.abs(moment_matrix - exact_moment))),
    }
    arrays = {
        "nodal_integrals": nodal_integrals,
        "pu_residual_field": pu_residual_field,
        "x_repro_residual_field": x_repro_field,
        "y_repro_residual_field": y_repro_field,
        "moment_matrix": moment_matrix,
        "moment_matrix_residual": moment_matrix - exact_moment,
    }
    return metrics, arrays


def compute_derivative_consistency_metrics_2d(
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
    nodes = np.asarray(nodes, dtype=np.float64)
    domain_points = np.asarray(domain_points, dtype=np.float64)
    domain_weights = np.asarray(domain_weights, dtype=np.float64)
    boundary_points = np.asarray(boundary_points, dtype=np.float64)
    boundary_weights = np.asarray(boundary_weights, dtype=np.float64)
    boundary_normals = np.asarray(boundary_normals, dtype=np.float64)
    phi_boundary = np.asarray(phi_boundary, dtype=np.float64)
    p_nodes = polynomial_basis_2d(nodes)
    p_eval = polynomial_basis_2d(domain_points)
    p_boundary = polynomial_basis_2d(boundary_points)
    grad_p_eval = polynomial_gradients_2d(domain_points)

    lhs = np.einsum("mna,mk,m->nak", grad_phi, p_eval, domain_weights)
    boundary_term = np.einsum("bn,bk,ba,b->nak", phi_boundary, p_boundary, boundary_normals, boundary_weights)
    correction_term = np.einsum("mn,mka,m->nak", phi, grad_p_eval, domain_weights)
    boundary_residuals = lhs - (boundary_term - correction_term)

    reproduced_grad = np.einsum("mna,nk->mak", grad_phi, p_nodes)
    derivative_matrix = np.einsum("mak,mj,m->akj", reproduced_grad, p_eval, domain_weights)
    exact_derivative_matrix = analytic_derivative_moment_matrices_2d()
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


def compute_rkpm_shape_and_gradients_2d(
    points: np.ndarray,
    nodes: np.ndarray,
    support_radius: float,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    points_t = torch.tensor(points, device=device, dtype=torch.float64, requires_grad=True)
    nodes_t = torch.tensor(nodes, device=device, dtype=torch.float64)
    phi = rkpm_shape_matrix_2d_torch(points_t, nodes_t, support_radius)
    grad_phi = torch.zeros((points_t.shape[0], nodes_t.shape[0], 2), device=device, dtype=torch.float64)
    for index in range(nodes_t.shape[0]):
        retain_graph = index + 1 < nodes_t.shape[0]
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            points_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, :] = grad_i
    return phi.detach().cpu().numpy(), grad_phi.detach().cpu().numpy()


def evaluate_consistency_bundle_2d(
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
    value_metrics, value_arrays = compute_value_consistency_metrics_2d(
        phi=phi,
        nodes=nodes,
        domain_points=domain_points,
        domain_weights=domain_weights,
    )
    derivative_metrics, derivative_arrays = compute_derivative_consistency_metrics_2d(
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
        "domain_points": domain_points,
        "domain_weights": domain_weights,
        "boundary_points": boundary_points,
        "boundary_weights": boundary_weights,
        "boundary_normals": boundary_normals,
        "phi": phi,
        "grad_phi": grad_phi,
        "phi_boundary": phi_boundary,
        "phi_sum": np.asarray(phi_sum, dtype=np.float64),
    }
    arrays.update(value_arrays)
    arrays.update(derivative_arrays)
    metrics = {}
    metrics.update(compute_lambda_h_stats(phi))
    metrics.update(support_metrics)
    metrics.update(value_metrics)
    metrics.update(derivative_metrics)
    return {"metrics": metrics, "arrays": arrays}


def evaluate_model_consistency_bundle_2d(
    model: MeshfreeKAN2D,
    quadrature_order: int,
    device: str,
) -> dict[str, Any]:
    domain_points, domain_weights = square_domain_quadrature(quadrature_order)
    boundary_points, boundary_weights, boundary_normals = square_boundary_quadrature(quadrature_order)
    domain_t = torch.tensor(domain_points, device=device, dtype=model.nodes.dtype, requires_grad=True)
    phi = model.compute_shape_functions(domain_t)
    raw_phi = model.compute_shape_functions(domain_t.detach(), return_raw=True)
    grad_phi = torch.zeros((domain_t.shape[0], model.n_nodes, 2), device=device, dtype=model.nodes.dtype)
    for index in range(model.n_nodes):
        retain_graph = index + 1 < model.n_nodes
        grad_i = torch.autograd.grad(
            phi[:, index].sum(),
            domain_t,
            retain_graph=retain_graph,
            create_graph=False,
        )[0]
        grad_phi[:, index, :] = grad_i
    boundary_t = torch.tensor(boundary_points, device=device, dtype=model.nodes.dtype)
    with torch.no_grad():
        phi_boundary = model.compute_shape_functions(boundary_t).cpu().numpy()
    return evaluate_consistency_bundle_2d(
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


def evaluate_rkpm_consistency_bundle_2d(
    nodes: np.ndarray,
    support_radius: float,
    quadrature_order: int,
    device: str = "cpu",
) -> dict[str, Any]:
    domain_points, domain_weights = square_domain_quadrature(quadrature_order)
    boundary_points, boundary_weights, boundary_normals = square_boundary_quadrature(quadrature_order)
    phi, grad_phi = compute_rkpm_shape_and_gradients_2d(
        points=domain_points,
        nodes=nodes,
        support_radius=support_radius,
        device=device,
    )
    phi_boundary = rkpm_shape_matrix_2d(boundary_points, nodes, support_radius)
    return evaluate_consistency_bundle_2d(
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
    metric_keys = [
        "mass_sum_residual",
        "moment_x_residual",
        "moment_y_residual",
        "moment_matrix_residual_fro",
    ]
    labels = ["mass", "mx", "my", "moment_fro"]
    learned = np.array([payload["learned"][key] for key in metric_keys], dtype=np.float64)
    rkpm = np.array([payload["rkpm"][key] for key in metric_keys], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.5))
    _plot_metric_point_panel(
        ax,
        labels,
        [("learned", learned, LEARNED_COLOR, "o"), ("rkpm", rkpm, RKPM_COLOR, "s")],
        "Value consistency",
        "residual",
        log_scale=True,
    )
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
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.5))
    _plot_metric_point_panel(
        ax,
        labels,
        [("learned", learned, LEARNED_COLOR, "o"), ("rkpm", rkpm, RKPM_COLOR, "s")],
        "Derivative consistency",
        "residual",
        log_scale=True,
    )
    _save_figure(fig, path)


def plot_derivative_node_residuals_2d(
    learned_residuals: np.ndarray,
    rkpm_residuals: np.ndarray,
    nodes: np.ndarray,
    path: Path,
) -> None:
    learned_node = np.max(np.abs(learned_residuals), axis=(1, 2))
    rkpm_node = np.max(np.abs(rkpm_residuals), axis=(1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, values, title in zip(
        axes,
        [learned_node, rkpm_node],
        ["Learned node residual", "RKPM node residual"],
    ):
        image = ax.scatter(nodes[:, 0], nodes[:, 1], c=np.maximum(values, 1e-16), cmap=STRUCTURE_CMAP, s=42)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.2)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def plot_main_figure_consistency_2d(
    payload: dict[str, Any],
    learned_arrays: dict[str, np.ndarray],
    rkpm_arrays: dict[str, np.ndarray],
    nodes: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    value_labels = ["mass", "mx", "my", "moment_fro"]
    value_keys = ["mass_sum_residual", "moment_x_residual", "moment_y_residual", "moment_matrix_residual_fro"]
    derivative_labels = ["bd-max", "bd-fro", "mat-max", "mat-fro"]
    derivative_keys = [
        "derivative_boundary_residual_max",
        "derivative_boundary_residual_fro",
        "derivative_matrix_residual_max",
        "derivative_matrix_residual_fro",
    ]
    _plot_metric_point_panel(
        axes[0, 0],
        value_labels,
        [
            ("learned", np.array([payload["learned"][key] for key in value_keys], dtype=np.float64), LEARNED_COLOR, "o"),
            ("rkpm", np.array([payload["rkpm"][key] for key in value_keys], dtype=np.float64), RKPM_COLOR, "s"),
        ],
        "Value consistency",
        "residual",
        log_scale=True,
    )
    _plot_metric_point_panel(
        axes[0, 1],
        derivative_labels,
        [
            ("learned", np.array([payload["learned"][key] for key in derivative_keys], dtype=np.float64), LEARNED_COLOR, "o"),
            ("rkpm", np.array([payload["rkpm"][key] for key in derivative_keys], dtype=np.float64), RKPM_COLOR, "s"),
        ],
        "Derivative consistency",
        "residual",
        log_scale=True,
    )
    learned_node = np.max(np.abs(learned_arrays["boundary_residuals"]), axis=(1, 2))
    image = axes[1, 0].scatter(nodes[:, 0], nodes[:, 1], c=np.maximum(learned_node, 1e-16), cmap=STRUCTURE_CMAP, s=46)
    axes[1, 0].set_title("Learned node residual map")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, ls=":", alpha=0.2)
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)


def plot_summary_figure_consistency_2d(
    payload: dict[str, Any],
    learned_arrays: dict[str, np.ndarray],
    rkpm_arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGURE_SIZE)
    _plot_matrix_panel(axes[0, 0], learned_arrays["moment_matrix_residual"], "Learned moment matrix residual")
    _plot_matrix_panel(axes[0, 1], learned_arrays["derivative_matrix_residual"], "Learned derivative matrix residual")
    coverage_labels = ["orphan", "phi-min", "phi-p01"]
    _plot_metric_point_panel(
        axes[1, 0],
        coverage_labels,
        [
            (
                "learned",
                np.array([
                    payload["learned"]["orphan_ratio"],
                    abs(payload["learned"]["phi_sum_min"]),
                    abs(payload["learned"]["phi_sum_p01"]),
                ], dtype=np.float64),
                LEARNED_COLOR,
                "o",
            ),
            (
                "rkpm",
                np.array([
                    payload["rkpm"]["orphan_ratio"],
                    abs(payload["rkpm"]["phi_sum_min"]),
                    abs(payload["rkpm"]["phi_sum_p01"]),
                ], dtype=np.float64),
                RKPM_COLOR,
                "s",
            ),
        ],
        "Coverage indicators",
        "value",
        log_scale=True,
    )
    lambda_labels = ["lambda_max", "lambda_mean"]
    _plot_metric_point_panel(
        axes[1, 1],
        lambda_labels,
        [
            (
                "learned",
                np.array([payload["learned"]["lambda_h_max"], payload["learned"]["lambda_h_mean"]], dtype=np.float64),
                LEARNED_COLOR,
                "o",
            ),
            (
                "rkpm",
                np.array([payload["rkpm"]["lambda_h_max"], payload["rkpm"]["lambda_h_mean"]], dtype=np.float64),
                RKPM_COLOR,
                "s",
            ),
        ],
        "Lambda_h indicators",
        "value",
        log_scale=True,
    )
    _save_figure(fig, path)

def plot_consistency_summary(entries: list[dict[str, Any]], path: Path) -> None:
    case_labels = [item["case_label"] for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    value_learned = np.maximum(
        np.array([item["payload"]["learned"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64),
        1e-16,
    )
    value_rkpm = np.maximum(
        np.array([item["payload"]["rkpm"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64),
        1e-16,
    )
    derivative_learned = np.maximum(
        np.array([item["payload"]["learned"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64),
        1e-16,
    )
    derivative_rkpm = np.maximum(
        np.array([item["payload"]["rkpm"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64),
        1e-16,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, learned_values, rkpm_values, title in zip(
        axes,
        [value_learned, derivative_learned],
        [value_rkpm, derivative_rkpm],
        ["Value consistency summary", "Derivative consistency summary"],
    ):
        ax.plot(x_pos, learned_values, "o-", color=LEARNED_COLOR, lw=1.8, ms=6, label="learned")
        ax.plot(x_pos, rkpm_values, "s-", color=RKPM_COLOR, lw=1.8, ms=6, label="rkpm")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        _style_line_axis(ax, title, "case", "residual", log_scale=True)
        ax.legend(fontsize=8)
    _save_figure(fig, path)


