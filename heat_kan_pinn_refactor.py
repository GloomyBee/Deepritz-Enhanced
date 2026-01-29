import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Optional: reuse external viz helpers if available
try:
    from compare_viz_square import plot_square_triplet, plot_error_history
except Exception:
    plot_square_triplet = None
    plot_error_history = None


# ==========================================
# 0. Utils
# ==========================================
def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 1. KAN core components (kept compatible with your original)
# ==========================================
def extend_grid(grid: torch.Tensor, k_extend: int = 0) -> torch.Tensor:
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for _ in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


def B_batch(x: torch.Tensor, grid: torch.Tensor, k: int = 0) -> torch.Tensor:
    # Original recursive B-spline basis, lightly formatted only.
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + \
                (grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    return torch.nan_to_num(value)


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num: int = 5, k: int = 3, grid_range=(0.0, 1.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.register_buffer("grid", grid)

        num_coef = self.grid.shape[1] - k - 1
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_coef) * 0.1)
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim) * 0.1)
        self.base_fun = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_fun(x)
        y_base = torch.einsum("bi,io->bo", base, self.scale_base)
        b_splines = B_batch(x, self.grid, k=self.k)
        y_spline = torch.einsum("bik,iok->bo", b_splines, self.coef)
        return y_base + y_spline


# ==========================================
# 2. Problem + Sampling
# ==========================================
class SinusoidalProblem:
    """Poisson on [0,1]^2 with exact u = sin(pi x) sin(pi y).

    PDE (strong form): -Δu = f, where f = 2*pi^2*u
    """

    @staticmethod
    def exact_solution(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0:1], xy[:, 1:2]
        return torch.sin(np.pi * x) * torch.sin(np.pi * y)

    @staticmethod
    def exact_gradient(xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0:1], xy[:, 1:2]
        ux = np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
        uy = np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
        return torch.cat([ux, uy], dim=1)

    @staticmethod
    def source_term(xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (np.pi ** 2) * SinusoidalProblem.exact_solution(xy)


class SquareSampler:
    @staticmethod
    def sample_domain(n: int) -> np.ndarray:
        return np.random.rand(n, 2)

    @staticmethod
    def sample_boundary(n: int) -> np.ndarray:
        n_each = n // 4
        r = np.random.rand(n_each, 1)

        left = np.hstack([np.zeros((n_each, 1)), r])
        right = np.hstack([np.ones((n_each, 1)), r])
        bottom = np.hstack([r, np.zeros((n_each, 1))])
        top = np.hstack([r, np.ones((n_each, 1))])

        pts = np.vstack([left, right, bottom, top])
        if pts.shape[0] < n:
            extra = np.random.rand(n - pts.shape[0], 2)
            extra[:, 0] = np.random.choice([0.0, 1.0], size=(extra.shape[0],))
            pts = np.vstack([pts, extra])
        return pts


# ==========================================
# 3. Serial filtering KAN PINN (Coarse + Fine)
# ==========================================
class HeatKANSerialPINN(nn.Module):
    """Serial-filter KAN PINN:

    u(x) = u_c(x) + u_f(x)

    Phase1: train u_c with PDE + BC.
    Phase2: freeze coarse PARAMETERS (no update), but keep coarse INPUT graph so Δu_c is included in residual.
    """

    def __init__(self, grid_range=(-0.2, 1.2)):
        super().__init__()

        # Coarse (low-pass)
        self.kan_coarse = KANLayer(in_dim=2, out_dim=8, num=3, k=2, grid_range=grid_range)
        self.out_coarse = nn.Linear(8, 1, bias=False)

        # Fine (high-pass / correction)
        self.kan_fine = KANLayer(in_dim=2, out_dim=16, num=20, k=3, grid_range=grid_range)
        self.out_fine = nn.Linear(16, 1, bias=False)
        nn.init.constant_(self.out_fine.weight, 0.0)

    def coarse(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_coarse(self.kan_coarse(x))

    def fine(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_fine(self.kan_fine(x))

    def forward(self, x: torch.Tensor):
        uc = self.coarse(x)
        uf = self.fine(x)
        return uc, uf

    def total(self, x: torch.Tensor) -> torch.Tensor:
        uc, uf = self.forward(x)
        return uc + uf

    def freeze_coarse_params(self) -> None:
        for p in self.kan_coarse.parameters():
            p.requires_grad_(False)
        for p in self.out_coarse.parameters():
            p.requires_grad_(False)

    def unfreeze_coarse_params(self) -> None:
        for p in self.kan_coarse.parameters():
            p.requires_grad_(True)
        for p in self.out_coarse.parameters():
            p.requires_grad_(True)


# ==========================================
# 4. PINN helpers
# ==========================================
def laplacian(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute Δu for scalar u(x). x must have requires_grad=True."""
    grads = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=create_graph)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=create_graph)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=create_graph)[0][:, 1:2]
    return u_xx + u_yy


@dataclass
class TrainConfig:
    seed: int = 0

    pretrain_steps: int = 500
    phase1_steps: int = 1000
    phase2_steps: int = 4000

    batch_domain: int = 2000
    batch_boundary: int = 500

    lambda_bc: float = 100.0

    # Fine regularization (start simple; increase later if needed)
    lambda_fine_reg: float = 0.0   # ||u_f||^2
    lambda_fine_bc: float = 1.0    # ||u_f(boundary)||^2

    lr_coarse: float = 1e-2
    lr_fine: float = 1e-3
    fine_lr_gamma: float = 0.9
    fine_lr_step: int = 1000

    err_every: int = 200
    err_samples: int = 2000

    output_dir: str = "output_kan_serial_refactor"


# ==========================================
# 5. Training loop (structured like heat_pinn_pytorch.py)
# ==========================================
def train(cfg: TrainConfig = TrainConfig()):
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    problem = SinusoidalProblem()
    model = HeatKANSerialPINN().to(device)

    optimizer_c = torch.optim.Adam(
        list(model.kan_coarse.parameters()) + list(model.out_coarse.parameters()),
        lr=cfg.lr_coarse,
    )
    optimizer_f = torch.optim.Adam(
        list(model.kan_fine.parameters()) + list(model.out_fine.parameters()),
        lr=cfg.lr_fine,
    )
    scheduler_f = torch.optim.lr_scheduler.ExponentialLR(optimizer_f, gamma=cfg.fine_lr_gamma)

    err_steps, err_l2, err_h1 = [], [], []
    start_time = time.time()

    # ---------------------------------------------
    # Step A: Pretrain (Boundary Anchoring for Coarse)
    # ---------------------------------------------
    print(f">>> Pretrain: Anchoring Dirichlet boundary ({cfg.pretrain_steps} steps)")
    model.unfreeze_coarse_params()
    for i in range(cfg.pretrain_steps + 1):
        x_bd = torch.tensor(SquareSampler.sample_boundary(cfg.batch_boundary), dtype=torch.float32, device=device)
        u_pred = model.coarse(x_bd)
        u_exact = problem.exact_solution(x_bd)

        loss_pre = torch.mean((u_pred - u_exact) ** 2)

        optimizer_c.zero_grad()
        loss_pre.backward()
        optimizer_c.step()

        if i % 100 == 0:
            print(f"Pretrain step {i}: BC loss={loss_pre.item():.6e}")

    # ---------------------------------------------
    # Step B: Phase 1 (Coarse: Physics + Boundary)
    # ---------------------------------------------
    print(f"\n>>> Phase1: COARSE (PDE + BC), steps={cfg.phase1_steps}")
    for step in range(cfg.phase1_steps + 1):
        x_dom = torch.tensor(SquareSampler.sample_domain(cfg.batch_domain), dtype=torch.float32, device=device)
        x_dom.requires_grad_(True)
        x_bd = torch.tensor(SquareSampler.sample_boundary(cfg.batch_boundary), dtype=torch.float32, device=device)

        u_dom = model.coarse(x_dom)
        res = -laplacian(u_dom, x_dom, create_graph=True) - problem.source_term(x_dom)
        loss_pde = torch.mean(res ** 2)

        u_bd = model.coarse(x_bd)
        loss_bc = torch.mean((u_bd - problem.exact_solution(x_bd)) ** 2)

        loss = loss_pde + cfg.lambda_bc * loss_bc

        optimizer_c.zero_grad()
        loss.backward()
        optimizer_c.step()

        if step % cfg.err_every == 0:
            l2, h1 = evaluate(problem, model, device, cfg.err_samples, mode="coarse")
            err_steps.append(step)
            err_l2.append(l2)
            err_h1.append(h1)
            print(f"[COARSE] step {step:5d}/{cfg.phase1_steps}: loss={loss.item():.6f} (pde={loss_pde.item():.2e}, bc={loss_bc.item():.2e}) L2={l2:.2e}")

    # ---------------------------------------------
    # Step C: Phase 2 (Fine: Physics + Boundary on total)
    # ---------------------------------------------
    print(f"\n>>> Phase2: FINE with frozen coarse, steps={cfg.phase2_steps}")
    # ---- Critical fix:
    # Freeze coarse PARAMETERS so optimizer_f won't update them,
    # BUT DO NOT wrap coarse forward in torch.no_grad().
    # This keeps u_c(x) differentiable w.r.t. x, so Δu_total includes Δu_c.
    model.freeze_coarse_params()

    for step in range(cfg.phase2_steps + 1):
        x_dom = torch.tensor(SquareSampler.sample_domain(cfg.batch_domain), dtype=torch.float32, device=device)
        x_dom.requires_grad_(True)
        x_bd = torch.tensor(SquareSampler.sample_boundary(cfg.batch_boundary), dtype=torch.float32, device=device)

        # Physics loss on total
        u_dom_total = model.total(x_dom)
        res = -laplacian(u_dom_total, x_dom, create_graph=True) - problem.source_term(x_dom)
        loss_pde = torch.mean(res ** 2)

        # Boundary loss on total
        u_bd_total = model.total(x_bd)
        loss_bc = torch.mean((u_bd_total - problem.exact_solution(x_bd)) ** 2)

        # Fine regularization (optional)
        u_f_dom = model.fine(x_dom)
        u_f_bd = model.fine(x_bd)
        loss_freg = torch.mean(u_f_dom ** 2)
        loss_fbc = torch.mean(u_f_bd ** 2)

        loss = loss_pde + cfg.lambda_bc * loss_bc
        loss = loss + cfg.lambda_fine_reg * loss_freg + cfg.lambda_fine_bc * loss_fbc

        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()

        if (step > 0) and (step % cfg.fine_lr_step == 0):
            scheduler_f.step()

        if step % cfg.err_every == 0:
            l2, h1 = evaluate(problem, model, device, cfg.err_samples, mode="total")
            err_steps.append(cfg.phase1_steps + step)
            err_l2.append(l2)
            err_h1.append(h1)
            print(f"[FINE  ] step {step:5d}/{cfg.phase2_steps}: loss={loss.item():.6f} (pde={loss_pde.item():.2e}, bc={loss_bc.item():.2e}, fbc={loss_fbc.item():.2e}) L2={l2:.2e}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")

    # ---------------------------------------------
    # Visualization (same interface as your scripts)
    # ---------------------------------------------
    if plot_square_triplet is not None:
        x = np.linspace(0, 1, 200)
        y = np.linspace(0, 1, 200)
        X, Y = np.meshgrid(x, y)
        pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

        with torch.no_grad():
            T_pred = model.total(pts).cpu().numpy().reshape(X.shape)
            T_exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

        plot_square_triplet(
            X, Y, T_pred, T_exact,
            title_prefix="Heat KAN Serial PINN (Refactor)",
            filename=os.path.join(cfg.output_dir, "kan_field.png"),
        )

        if plot_error_history is not None:
            plot_error_history(
                err_steps, err_l2, err_h1,
                filename=os.path.join(cfg.output_dir, "error_history.png"),
                title="Error History (L2 / H1)",
            )
    else:
        print("plot_square_triplet not available; skipping visualization.")

    return {"step": err_steps, "l2": err_l2, "h1": err_h1}


def evaluate(problem: SinusoidalProblem, model: HeatKANSerialPINN, device: torch.device, n: int, mode: str = "total"):
    # L2
    with torch.no_grad():
        pts = torch.tensor(SquareSampler.sample_domain(n), dtype=torch.float32, device=device)
        if mode == "coarse":
            u_pred = model.coarse(pts)
        else:
            u_pred = model.total(pts)
        u_ref = problem.exact_solution(pts)
        l2 = torch.sqrt(torch.mean((u_pred - u_ref) ** 2)).item()

    # H1
    with torch.enable_grad():
        pts_h1 = torch.tensor(SquareSampler.sample_domain(n), dtype=torch.float32, device=device)
        pts_h1.requires_grad_(True)
        if mode == "coarse":
            u_h1 = model.coarse(pts_h1)
        else:
            u_h1 = model.total(pts_h1)
        g_pred = torch.autograd.grad(u_h1, pts_h1, torch.ones_like(u_h1), create_graph=False)[0]
        g_ref = problem.exact_gradient(pts_h1)
        h1 = torch.sqrt(torch.mean(torch.sum((g_pred - g_ref) ** 2, dim=1))).item()

    return l2, h1


if __name__ == "__main__":
    train()
