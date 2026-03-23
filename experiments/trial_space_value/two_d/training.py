from __future__ import annotations

import torch

from experiments.shape_validation.two_d.config import resolve_variant_config
from experiments.trial_space_value.two_d.basis import (
    MeshfreeKAN2D,
    SinusoidalPoissonProblem2D,
    grid_points,
    rkpm_shape_matrix_2d_torch,
    sample_mixed_domain_points,
    sample_square_boundary,
)


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
                f"JointPhaseB | step={step:4d} | loss={loss.item():.3e} | "
                f"energy={loss_energy.item():.3e} | bc={loss_bc.item():.3e} | "
                f"linear={loss_linear.item():.3e} | val_l2={(val_l2 if val_l2 is not None else float('nan')):.3e} | "
                f"val_h1={(val_h1 if val_h1 is not None else float('nan')):.3e}"
            )
    model.load_state_dict(best_state)
    return history


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
    del gamma_linear
    history = init_solver_history()
    for param in model.kan.parameters():
        param.requires_grad_(False)
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
    import numpy as np

    return {f"history_{key}": np.array(value, dtype=np.float64) for key, value in history.items()}
