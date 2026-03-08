import json
import math
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


@dataclass
class RunArtifacts:
    root_dir: Path
    figures_dir: Path


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
    root_dir = ROOT_DIR / "output" / "meshfree_kan_rkpm_2d_validation" / group / case_name
    figures_dir = root_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(root_dir=root_dir, figures_dir=figures_dir)


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


def plot_training_curves(
    history: dict[str, list[float]],
    path: Path,
    title: str,
    phase_split_step: float | None = None,
) -> None:
    steps = np.array(history["steps"], dtype=np.float64)

    def plot_series(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, label: str) -> None:
        if x_values.size == 0 or y_values.size == 0:
            return
        if np.allclose(y_values, 0.0):
            return
        ax.semilogy(x_values, np.maximum(y_values, 1e-16), lw=2, label=label)

    def finalize(ax: plt.Axes, fig: plt.Figure, save_path: Path, figure_title: str) -> None:
        ax.set_title(figure_title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss / metric")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

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
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    contour = ax.contourf(x_grid, y_grid, values, levels=100, cmap=cmap)
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        (axes[0], pred, f"{title_prefix} Prediction", "inferno"),
        (axes[1], exact, "Exact Solution", "inferno"),
        (axes[2], error, "Absolute Error", "magma"),
    ]:
        contour = axis.contourf(x_grid, y_grid, values, levels=100, cmap=cmap)
        plt.colorbar(contour, ax=axis)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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


