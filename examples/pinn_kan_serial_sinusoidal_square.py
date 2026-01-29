"""
Example: Serial KAN PINN for sinusoidal square problem.
Coarse + fine KAN decomposition with two-phase training.
"""

import sys
sys.path.append("..")

import os
import numpy as np
import torch

from problems import SinusoidalSquare
from samplers import SquareSampler
from networks import KANSerialPINN
from visualizers import plot_square_triplet, plot_error_history


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy


def evaluate(problem, model, sampler, device, n, mode="total"):
    with torch.no_grad():
        pts = torch.tensor(sampler.sample_domain(n), dtype=torch.float32, device=device)
        if mode == "coarse":
            u_pred = model.coarse(pts)
        else:
            u_pred = model.total(pts)
        u_ref = problem.exact_solution(pts)
        l2 = torch.sqrt(torch.mean((u_pred - u_ref) ** 2)).item()

    with torch.enable_grad():
        pts_h1 = torch.tensor(sampler.sample_domain(n), dtype=torch.float32, device=device)
        pts_h1.requires_grad_(True)
        if mode == "coarse":
            u_h1 = model.coarse(pts_h1)
        else:
            u_h1 = model.total(pts_h1)
        g_pred = torch.autograd.grad(u_h1, pts_h1, torch.ones_like(u_h1), create_graph=False)[0]
        g_ref = problem.exact_gradient(pts_h1)
        h1 = torch.sqrt(torch.mean(torch.sum((g_pred - g_ref) ** 2, dim=1))).item()
    return l2, h1


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    problem = SinusoidalSquare()
    sampler = SquareSampler(x_range=(0, 1), y_range=(0, 1))
    model = KANSerialPINN().to(device)

    output_dir = "output_kan_serial"
    os.makedirs(output_dir, exist_ok=True)

    # Optimizers
    optimizer_c = torch.optim.Adam(
        list(model.kan_coarse.parameters()) + list(model.out_coarse.parameters()),
        lr=1e-2
    )
    optimizer_f = torch.optim.Adam(
        list(model.kan_fine.parameters()) + list(model.out_fine.parameters()),
        lr=1e-3
    )
    scheduler_f = torch.optim.lr_scheduler.ExponentialLR(optimizer_f, gamma=0.9)

    pretrain_steps = 500
    phase1_steps = 1000
    phase2_steps = 4000
    batch_domain = 2000
    batch_boundary = 500
    lambda_bc = 100.0
    lambda_fine_reg = 0.0
    lambda_fine_bc = 1.0
    err_every = 200
    err_samples = 2000

    err_steps, err_l2, err_h1 = [], [], []

    # Pretrain coarse on boundary
    print(f">>> Pretrain: {pretrain_steps} steps")
    model.unfreeze_coarse_params()
    for i in range(pretrain_steps + 1):
        x_bd = torch.tensor(sampler.sample_boundary(batch_boundary), dtype=torch.float32, device=device)
        u_pred = model.coarse(x_bd)
        u_exact = problem.exact_solution(x_bd)
        loss_pre = torch.mean((u_pred - u_exact) ** 2)

        optimizer_c.zero_grad()
        loss_pre.backward()
        optimizer_c.step()

        if i % 100 == 0:
            print(f"Pretrain step {i}: BC loss={loss_pre.item():.6e}")

    # Phase 1: coarse
    print(f"\n>>> Phase1: COARSE {phase1_steps} steps")
    for step in range(phase1_steps + 1):
        x_dom = torch.tensor(sampler.sample_domain(batch_domain), dtype=torch.float32, device=device)
        x_dom.requires_grad_(True)
        x_bd = torch.tensor(sampler.sample_boundary(batch_boundary), dtype=torch.float32, device=device)

        u_dom = model.coarse(x_dom)
        res = -laplacian(u_dom, x_dom) - problem.source_term(x_dom)
        loss_pde = torch.mean(res ** 2)

        u_bd = model.coarse(x_bd)
        loss_bc = torch.mean((u_bd - problem.exact_solution(x_bd)) ** 2)

        loss = loss_pde + lambda_bc * loss_bc

        optimizer_c.zero_grad()
        loss.backward()
        optimizer_c.step()

        if step % err_every == 0:
            l2, h1 = evaluate(problem, model, sampler, device, err_samples, mode="coarse")
            err_steps.append(step)
            err_l2.append(l2)
            err_h1.append(h1)
            print(f"[COARSE] step {step:5d}/{phase1_steps}: loss={loss.item():.6f}, L2={l2:.2e}")

    # Phase 2: fine with coarse frozen
    print(f"\n>>> Phase2: FINE {phase2_steps} steps")
    model.freeze_coarse_params()
    for step in range(phase2_steps + 1):
        x_dom = torch.tensor(sampler.sample_domain(batch_domain), dtype=torch.float32, device=device)
        x_dom.requires_grad_(True)
        x_bd = torch.tensor(sampler.sample_boundary(batch_boundary), dtype=torch.float32, device=device)

        u_dom_total = model.total(x_dom)
        res = -laplacian(u_dom_total, x_dom) - problem.source_term(x_dom)
        loss_pde = torch.mean(res ** 2)

        u_bd_total = model.total(x_bd)
        loss_bc = torch.mean((u_bd_total - problem.exact_solution(x_bd)) ** 2)

        u_f_dom = model.fine(x_dom)
        u_f_bd = model.fine(x_bd)
        loss_freg = torch.mean(u_f_dom ** 2)
        loss_fbc = torch.mean(u_f_bd ** 2)

        loss = loss_pde + lambda_bc * loss_bc
        loss = loss + lambda_fine_reg * loss_freg + lambda_fine_bc * loss_fbc

        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()

        if (step > 0) and (step % 1000 == 0):
            scheduler_f.step()

        if step % err_every == 0:
            l2, h1 = evaluate(problem, model, sampler, device, err_samples, mode="total")
            err_steps.append(phase1_steps + step)
            err_l2.append(l2)
            err_h1.append(h1)
            print(f"[FINE] step {step:5d}/{phase2_steps}: loss={loss.item():.6f}, L2={l2:.2e}")

    # Visualization
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model.total(pts).cpu().numpy().reshape(X.shape)
        exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

    plot_square_triplet(
        X, Y, pred, exact,
        title_prefix="KAN Serial PINN",
        filename=os.path.join(output_dir, "kan_serial_field.png")
    )

    plot_error_history(
        np.array(err_steps), np.array(err_l2), np.array(err_h1),
        filename=os.path.join(output_dir, "kan_serial_errors.png"),
        title="KAN Serial PINN Errors"
    )


if __name__ == "__main__":
    main()
