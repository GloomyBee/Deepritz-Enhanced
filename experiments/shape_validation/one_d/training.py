from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.shape_validation.one_d.basis import (
    MeshfreeKAN1D,
    get_model_phi_stages,
    rkpm_shape_matrix_1d_torch,
)


@dataclass(frozen=True)
class ShapeTrainingVariant1D:
    name: str
    objective: str
    use_softplus: bool
    lambda_pu: float = 0.0
    lambda_bd: float = 0.0
    lambda_teacher: float = 0.0
    lambda_reg: float = 0.0
    pu_on_raw: bool = False


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


SHAPE_TRAINING_VARIANTS_1D: dict[str, ShapeTrainingVariant1D] = {
    "raw_pu": ShapeTrainingVariant1D(
        name="raw_pu",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=0.0,
        pu_on_raw=False,
    ),
    "raw_pu_fix": ShapeTrainingVariant1D(
        name="raw_pu_fix",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=0.0,
        pu_on_raw=True,
    ),
    "boundary_anchor": ShapeTrainingVariant1D(
        name="boundary_anchor",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=1.0,
        pu_on_raw=True,
    ),
    "no_softplus": ShapeTrainingVariant1D(
        name="no_softplus",
        objective="raw",
        use_softplus=False,
        lambda_pu=0.1,
        lambda_bd=1.0,
        pu_on_raw=True,
    ),
    "teacher_distill": ShapeTrainingVariant1D(
        name="teacher_distill",
        objective="distill",
        use_softplus=False,
        lambda_teacher=1.0,
        lambda_bd=0.1,
        lambda_reg=1.0e-4,
    ),
}


def get_shape_training_variant_1d(name: str) -> ShapeTrainingVariant1D:
    key = name.strip().lower()
    if key not in SHAPE_TRAINING_VARIANTS_1D:
        raise KeyError(f"Unknown 1D training variant: {name}")
    return SHAPE_TRAINING_VARIANTS_1D[key]


def build_shape_model_1d(
    nodes,
    support_radius: float,
    hidden_dim: int,
    variant_name: str,
    device: str,
) -> MeshfreeKAN1D:
    variant = get_shape_training_variant_1d(variant_name)
    nodes_t = torch.tensor(nodes, device=device, dtype=torch.float64)
    return MeshfreeKAN1D(
        nodes=nodes_t,
        support_radius=support_radius,
        hidden_dim=hidden_dim,
        use_softplus=variant.use_softplus,
    ).to(device)


def train_phase_a_raw_pu(
    model: MeshfreeKAN1D,
    nodes: torch.Tensor,
    device: str,
    *,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_pu: float = 0.1,
    lambda_bd: float = 0.0,
    lambda_reg: float = 0.0,
    pu_on_raw: bool = True,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nodes_col = nodes.reshape(-1, 1)
    history = init_history()
    x_boundary = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float64)
    for step in range(steps):
        x_domain = torch.rand(batch_size, 1, device=device)
        phi_stages = get_model_phi_stages(model, x_domain)
        phi = phi_stages["normalized"]
        phi_windowed = phi_stages["windowed"]
        phi_pre_window = phi_stages["pre_window"]

        repro_x = phi @ nodes_col
        loss_linear = torch.mean((repro_x - x_domain) ** 2)
        if pu_on_raw:
            pu_target = torch.sum(phi_windowed, dim=1, keepdim=True)
        else:
            pu_target = torch.sum(phi, dim=1, keepdim=True)
        loss_pu = torch.mean((pu_target - 1.0) ** 2)

        if lambda_bd > 0.0:
            phi_bd = model.compute_shape_functions(x_boundary, return_stage="normalized")
            loss_boundary = (
                (phi_bd[0, 0] - 1.0) ** 2
                + torch.sum(phi_bd[0, 1:] ** 2)
                + (phi_bd[1, -1] - 1.0) ** 2
                + torch.sum(phi_bd[1, :-1] ** 2)
            )
        else:
            loss_boundary = torch.zeros((), device=device, dtype=torch.float64)
        loss_reg = (
            torch.mean(phi_pre_window ** 2)
            if lambda_reg > 0.0
            else torch.zeros((), device=device, dtype=torch.float64)
        )
        loss = loss_linear + lambda_pu * loss_pu + lambda_bd * loss_boundary + lambda_reg * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_interval == 0 or step == steps - 1:
            history["steps"].append(float(step))
            history["loss"].append(float(loss.item()))
            history["teacher"].append(0.0)
            history["linear"].append(float(loss_linear.item()))
            history["pu"].append(float(loss_pu.item()))
            history["bd"].append(float(loss_boundary.item()))
            history["reg"].append(float(loss_reg.item()))
            print(
                f"PhaseA1D | step={step:4d} | loss={loss.item():.3e} | "
                f"linear(opt)={loss_linear.item():.3e} | pu(opt)={loss_pu.item():.3e} | "
                f"bd(opt)={loss_boundary.item():.3e} | reg(opt)={loss_reg.item():.3e}"
            )
    return history


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
        phi_stages = get_model_phi_stages(model, x_all)
        phi = phi_stages["normalized"]
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
        loss_reg = torch.mean(phi_stages["pre_window"] ** 2)
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
                f"teacher(opt)={loss_teacher.item():.3e} | bd(opt)={loss_boundary.item():.3e} | "
                f"reg(opt)={loss_reg.item():.3e} | linear(diag)={loss_linear.item():.3e} | "
                f"pu(diag)={loss_pu.item():.3e}"
            )
    return history


def train_shape_model_1d(
    model: MeshfreeKAN1D,
    nodes: torch.Tensor,
    *,
    variant_name: str,
    device: str,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_interval: int = 100,
) -> dict[str, list[float]]:
    variant = get_shape_training_variant_1d(variant_name)
    if variant.objective == "distill":
        return train_phase_a_distill(
            model=model,
            nodes=nodes,
            device=device,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            lambda_teacher=variant.lambda_teacher,
            lambda_bd=variant.lambda_bd,
            lambda_reg=variant.lambda_reg,
            log_interval=log_interval,
        )
    return train_phase_a_raw_pu(
        model=model,
        nodes=nodes,
        device=device,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        lambda_pu=variant.lambda_pu,
        lambda_bd=variant.lambda_bd,
        lambda_reg=variant.lambda_reg,
        pu_on_raw=variant.pu_on_raw,
        log_interval=log_interval,
    )

