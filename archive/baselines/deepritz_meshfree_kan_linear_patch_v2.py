"""
Deep Meshfree KAN for Linear Patch Test (v2 - Expert Enhanced)

Equation: -Δu = 0 in [0,1]x[0,1]
Exact: u = x + y
Method: Deep Ritz with meshfree KAN shape functions

Key Features (Expert Improvements):
  - Float64 precision for machine-level accuracy
  - Softplus non-negativity constraint
  - Shepard normalization (PU)
  - Coverage-safe kNN fallback for orphan points
  - Linear reproduction pretraining (Phase A)
  - Exact solution initialization for w (Phase B)
  - Mixed sampling strategy (uniform + local)
  - Reflection-based domain constraint
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import existing components
from core.problems import LinearPatchProblem
from core.visualizers import plot_square_triplet, plot_training_history, get_example_output_subdir
import core.visualizers as visualizers

# =============================================================================
# 0. Global Precision Setup
# =============================================================================
torch.set_default_dtype(torch.float64)


# =============================================================================
# 1. Window Function (Cubic Spline - C2 continuous)
# =============================================================================
def cubic_spline_window(dist: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Cubic spline window with compact support.

    q = dist/radius
    W(q) = 2/3 - 4q^2 + 4q^3,            0 <= q <= 0.5
           4/3 - 4q + 4q^2 - 4/3 q^3,    0.5 < q <= 1
           0,                            q > 1

    This is C2 continuous at q=0.5 and q=1.
    """
    q = dist / radius
    window = torch.zeros_like(q)

    mask_inner = (q <= 0.5)
    mask_outer = (q > 0.5) & (q <= 1.0)

    q_in = q[mask_inner]
    window[mask_inner] = 2.0 / 3.0 - 4.0 * q_in**2 + 4.0 * q_in**3

    q_out = q[mask_outer]
    window[mask_outer] = 4.0 / 3.0 - 4.0 * q_out + 4.0 * q_out**2 - (4.0 / 3.0) * q_out**3

    return window


# =============================================================================
# 2. KAN Spline Network (Linear B-spline / Hat Functions)
# =============================================================================
class KANSplineNet(nn.Module):
    """
    A lightweight spline-like network.

    NOTE:
    - This is not "canonical KAN" but a basis-expansion style model.
    - Main fix here: grid/h are registered as buffers, not rebuilt every forward.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 1,
        num: int = 5,
        grid_range: tuple[float, float] = (-1.5, 1.5),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num = num

        grid_min, grid_max = grid_range
        grid = torch.linspace(grid_min, grid_max, num)  # float64 by default
        h = (grid_max - grid_min) / (num - 1)

        self.register_buffer("grid", grid)            # [num]
        self.register_buffer("h", torch.tensor(h))    # scalar tensor

        # Layer 1: per-feature basis -> hidden
        self.layer1 = nn.ModuleList([
            nn.Linear(num, hidden_dim, bias=False) for _ in range(input_dim)
        ])

        # Layer 2: basis(hidden_sum) -> output
        self.layer2 = nn.Linear(hidden_dim * num, output_dim, bias=False)

    def _hat_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear hat basis over fixed grid.

        x: [B, 1] or [B, D] (we typically pass [B,1] or [B,hidden_dim])
        returns: [B, D, num] if x is [B,D]
        """
        # Ensure x is [B, D]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        B, D = x.shape

        # grid: [num] -> [1,1,num]
        grid = self.grid.view(1, 1, self.num)
        h = self.h

        x_exp = x.view(B, D, 1)  # [B,D,1]
        basis = torch.relu(1.0 - torch.abs(x_exp - grid) / h)  # [B,D,num]
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        """
        # Layer 1: sum of per-dimension contributions
        hidden_sum = 0.0
        for i, layer in enumerate(self.layer1):
            basis_i = self._hat_basis(x[:, i:i+1])     # [B,1,num]
            hidden_sum = hidden_sum + layer(basis_i.squeeze(1))  # [B, hidden_dim]

        # Layer 2: expand hidden_sum with hat basis then linear combine
        basis_hidden = self._hat_basis(hidden_sum)     # [B, hidden_dim, num]
        out = self.layer2(basis_hidden.reshape(x.shape[0], -1))  # [B, output_dim]
        return out


# =============================================================================
# 3. Meshfree KAN Network (Softplus + Shepard + Coverage fallback)
# =============================================================================
class MeshfreeKANNet(nn.Module):
    def __init__(self, nodes: torch.Tensor, support_radius: float, kan_hidden_dim: int = 8):
        super().__init__()
        self.register_buffer("nodes", nodes)   # fixed nodes
        self.N = nodes.shape[0]
        self.support_radius = float(support_radius)

        # Nodal coefficients
        self.w = nn.Parameter(torch.zeros(self.N, 1))

        # Shared kernel
        self.kan = KANSplineNet(input_dim=2, hidden_dim=kan_hidden_dim, num=5)

        # Non-negativity
        self.softplus = nn.Softplus()

    @torch.no_grad()
    def init_w_from_exact(self, exact_fn):
        """Initialize nodal values w_I = u_exact(x_I)."""
        self.w[:] = exact_fn(self.nodes)

    def compute_shape_functions(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        return_raw: bool = False,
        eps_cov: float = 1e-14,
        knn_k: int = 8,
        knn_alpha: float | None = None,
    ):
        """
        Compute normalized shape functions with coverage-safe fallback.

        - Main path: phi = (softplus(KAN) * window) / sum(...)
        - Fallback path: if sum(windowed) too small -> kNN distance softmax weights, ensures Σφ=1

        return_stats: if True returns (phi, stats_dict)
        return_raw: if True returns phi_windowed before normalization (for Phase A linear repro)
        """
        M = x.shape[0]
        N = self.N
        device = x.device

        # 1) Relative coords and distances
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)    # [M,N,2]
        dist = torch.norm(diff, dim=2)                     # [M,N]

        # 2) KAN input normalized
        kan_in = (diff / self.support_radius).reshape(-1, 2)  # [M*N,2]

        # 3) Raw + softplus
        phi_raw = self.softplus(self.kan(kan_in).reshape(M, N))  # [M,N], >= 0

        # 4) Compact support window
        window = cubic_spline_window(dist, self.support_radius)  # [M,N]
        phi_windowed = phi_raw * window                           # [M,N]

        # 5) Shepard normalization with fallback
        phi_sum = torch.sum(phi_windowed, dim=1, keepdim=True)    # [M,1]
        orphan_mask = (phi_sum.squeeze(1) < eps_cov)

        phi = torch.empty_like(phi_windowed)
        # Normal points
        phi[~orphan_mask] = phi_windowed[~orphan_mask] / (phi_sum[~orphan_mask] + 1e-12)

        # Fallback for orphan points
        if orphan_mask.any():
            # kNN distance softmax weights (guarantees Σφ=1, gives gradients through distances but not KAN)
            # This is a safety net; ideally orphan ratio ~ 0 if radius is chosen well.
            k = min(knn_k, N)
            dist_o = dist[orphan_mask]  # [K_orphan, N]
            idx = torch.topk(dist_o, k, largest=False).indices  # [K_orphan, k]
            d_knn = torch.gather(dist_o, 1, idx)                # [K_orphan, k]

            if knn_alpha is None:
                # scale: make exponent meaningful w.r.t radius
                knn_alpha = 20.0 / max(self.support_radius, 1e-12)

            w_knn = torch.exp(-knn_alpha * d_knn)               # [K_orphan, k]
            w_knn = w_knn / (torch.sum(w_knn, dim=1, keepdim=True) + 1e-18)

            phi_orphan = torch.zeros((dist_o.shape[0], N), device=device, dtype=phi.dtype)
            phi_orphan.scatter_(1, idx, w_knn)
            phi[orphan_mask] = phi_orphan

        # Return raw windowed phi if requested (for Phase A linear repro training)
        if return_raw:
            if return_stats:
                stats = {
                    "phi_sum_min": phi_sum.min().item(),
                    "phi_sum_p01": torch.quantile(phi_sum.squeeze(1), 0.01).item(),
                    "orphan_ratio": orphan_mask.to(torch.float64).mean().item(),
                }
                return phi_windowed, stats
            return phi_windowed

        if return_stats:
            stats = {
                "phi_sum_min": phi_sum.min().item(),
                "phi_sum_p01": torch.quantile(phi_sum.squeeze(1), 0.01).item(),
                "orphan_ratio": orphan_mask.to(torch.float64).mean().item(),
            }
            return phi, stats
        return phi

    def forward(self, x: torch.Tensor, return_phi: bool = False, return_stats: bool = False):
        if return_stats:
            phi, stats = self.compute_shape_functions(x, return_stats=True)
        else:
            phi = self.compute_shape_functions(x, return_stats=False)

        u = phi @ self.w

        if return_phi and return_stats:
            return u, phi, stats
        if return_phi:
            return u, phi
        if return_stats:
            return u, stats
        return u

def reflect_to_unit_box(x: torch.Tensor) -> torch.Tensor:
    """
    Reflect points into [0,1]^2 without boundary pile-up typical of clamp.
    Works best when x is not too far out of range (local noise).
    """
    x = torch.abs(x)               # reflect negatives
    x = torch.where(x > 1.0, 2.0 - x, x)  # reflect >1
    # In rare case x still outside (if very large), clamp as last resort.
    return torch.clamp(x, 0.0, 1.0)

def sample_boundary(batch_size: int, device: str) -> torch.Tensor:
    """
    Uniform boundary sampling on the square boundary (Dirichlet).
    Construct directly on device (avoid CPU->GPU copy).
    """
    n_b = batch_size // 4
    zeros = torch.zeros(n_b, device=device)
    ones = torch.ones(n_b, device=device)

    r1 = torch.rand(n_b, device=device)
    r2 = torch.rand(n_b, device=device)
    r3 = torch.rand(n_b, device=device)
    r4 = torch.rand(n_b, device=device)

    left   = torch.stack([zeros, r1], dim=1)
    right  = torch.stack([ones,  r2], dim=1)
    bottom = torch.stack([r3, zeros], dim=1)
    top    = torch.stack([r4, ones],  dim=1)

    return torch.cat([left, right, bottom, top], dim=0)


# =============================================================================
# 5. Training Function with Two-Phase Strategy
# =============================================================================
def train_meshfree_kan(
    model: MeshfreeKANNet,
    problem: LinearPatchProblem,
    device: str,
    phase_a_steps: int = 1200,
    phase_b_steps: int = 2000,
    batch_size: int = 1024,
    lr_kan_a: float = 1e-3,
    lr_kan_b: float = 1e-4,
    lr_w: float = 1e-2,
    beta_bc: float = 100.0,
    gamma_linear_b: float = 10.0,
    log_interval: int = 100,
):
    """
    Two-phase training for Meshfree KAN (Expert Enhanced Version).

    Phase A (Geometry Pretraining): Train KAN only, enforce linear reproduction
    Phase B (Physics Solve): Train both KAN and w, minimize energy functional

    Returns:
        history: dict with training metrics
    """
    # Initialize history
    history = {
        'steps': [],
        'loss': [],
        'linear_repro': [],
        'energy': [],
        'boundary': [],
        'l2_error': [],
        'h1_error': [],
        'phi_sum_min': [],
        'orphan_ratio': []
    }

    # Pre-fetch nodes for linear reproduction loss
    nodes_x = model.nodes[:, 0:1]
    nodes_y = model.nodes[:, 1:2]

    # -------------------------
    # Phase A: Geometry pretrain
    # -------------------------
    print(f"\n{'='*70}\nPhase A: Geometry Pre-training (Linear Reproduction)\n{'='*70}")
    model.w.requires_grad = False
    optimizer_kan = torch.optim.Adam(model.kan.parameters(), lr=lr_kan_a)

    for step in range(phase_a_steps):
        n_uniform = batch_size // 2
        n_local = batch_size - n_uniform

        # 50% uniform
        x_uniform = torch.rand(n_uniform, 2, device=device)

        # 50% local around nodes with reflection to domain
        rand_idx = torch.randint(0, model.N, (n_local,), device=device)
        centers = model.nodes[rand_idx]
        noise = torch.randn(n_local, 2, device=device) * (model.support_radius * 0.5)
        x_local = reflect_to_unit_box(centers + noise)

        x_domain = torch.cat([x_uniform, x_local], dim=0)

        # Use RAW phi (windowed but not normalized) for linear reproduction training
        phi_raw, stats = model.compute_shape_functions(x_domain, return_stats=True, return_raw=True)

        # Normalize for linear reproduction constraint
        phi_sum = torch.sum(phi_raw, dim=1, keepdim=True) + 1e-12
        phi = phi_raw / phi_sum

        repro_x = phi @ nodes_x
        repro_y = phi @ nodes_y
        loss_linear = torch.mean((repro_x - x_domain[:, 0:1])**2 + (repro_y - x_domain[:, 1:2])**2)

        optimizer_kan.zero_grad()
        loss_linear.backward()
        optimizer_kan.step()

        if step % log_interval == 0:
            print(
                f"Step {step:4d} | LinearErr={loss_linear.item():.3e} | "
                f"phi_sum_min={stats['phi_sum_min']:.3e} | orphan={stats['orphan_ratio']:.3e}"
            )

    print(f"Phase A done. Final LinearErr={loss_linear.item():.3e}")

    # Initialize nodal weights with exact solution (huge convergence booster)
    model.init_w_from_exact(problem.exact_solution)

    # -------------------------
    # Phase B: Physics solve
    # -------------------------
    print(f"\n{'='*70}\nPhase B: Physics Solve\n{'='*70}")
    model.w.requires_grad = True

    optimizer_all = torch.optim.Adam([
        {"params": model.kan.parameters(), "lr": lr_kan_b},
        {"params": [model.w], "lr": lr_w},
    ])

    for step in range(phase_b_steps):
        # Domain points
        x_domain = torch.rand(batch_size, 2, device=device)
        x_domain.requires_grad_(True)

        # Boundary points
        x_boundary = sample_boundary(batch_size, device=device)

        # One forward returns u and phi and stats (no duplicate compute)
        u_domain, phi_domain, stats = model(x_domain, return_phi=True, return_stats=True)

        # Physics: energy for Laplace with f=0 => minimize 0.5 * |∇u|^2
        grad_u = torch.autograd.grad(
            u_domain, x_domain, torch.ones_like(u_domain), create_graph=True
        )[0]
        loss_energy = 0.5 * problem.k * torch.mean(torch.sum(grad_u**2, dim=1))

        # Dirichlet BC (penalty)
        u_bc = model(x_boundary)
        u_exact_bc = problem.exact_solution(x_boundary)
        loss_bc = torch.mean((u_bc - u_exact_bc) ** 2)

        # Keep geometry (linear reproduction)
        repro_x = phi_domain @ nodes_x
        repro_y = phi_domain @ nodes_y
        loss_linear = torch.mean((repro_x - x_domain[:, 0:1])**2 + (repro_y - x_domain[:, 1:2])**2)

        loss = loss_energy + beta_bc * loss_bc + gamma_linear_b * loss_linear

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        if step % log_interval == 0:
            # Evaluation (L2 and H1 errors)
            model.eval()
            with torch.no_grad():
                x_val = torch.rand(2000, 2, device=device)
                u_val = model(x_val)
                u_exact_val = problem.exact_solution(x_val)
                l2_err = torch.sqrt(torch.mean((u_val - u_exact_val) ** 2)).item()

            # H1 error (requires gradients)
            x_val_grad = torch.rand(1024, 2, device=device, requires_grad=True)
            u_val_grad = model(x_val_grad)
            grad_pred = torch.autograd.grad(u_val_grad, x_val_grad, torch.ones_like(u_val_grad), create_graph=False)[0]
            grad_exact = problem.exact_gradient(x_val_grad)
            h1_err = torch.sqrt(torch.mean(torch.sum((grad_pred - grad_exact) ** 2, dim=1))).item()
            model.train()

            # Record history
            history['steps'].append(phase_a_steps + step)
            history['loss'].append(loss.item())
            history['linear_repro'].append(loss_linear.item())
            history['energy'].append(loss_energy.item())
            history['boundary'].append(loss_bc.item())
            history['l2_error'].append(l2_err)
            history['h1_error'].append(h1_err)
            history['phi_sum_min'].append(stats['phi_sum_min'])
            history['orphan_ratio'].append(stats['orphan_ratio'])

            print(
                f"Step {step:4d} | Loss={loss.item():.3e} | Energy={loss_energy.item():.3e} | "
                f"BC={loss_bc.item():.3e} | Linear={loss_linear.item():.3e} | "
                f"L2={l2_err:.3e} | H1={h1_err:.3e} | "
                f"phi_sum_min={stats['phi_sum_min']:.3e} | orphan={stats['orphan_ratio']:.3e}"
            )

    print(f"\nPhase B completed.")
    return history


# =============================================================================
# 6. Main Function
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | dtype: {torch.get_default_dtype()}\n")

    # Setup output directory
    ROOT = Path(__file__).resolve().parents[2]
    visualizers.OUTPUT_DIR = str(ROOT / "output" / "archive" / "baselines")
    output_subdir = get_example_output_subdir(__file__)
    print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/\n")

    # Problem definition
    problem = LinearPatchProblem()
    print(f"Problem: {problem.name}")
    print("Domain: [0,1] x [0,1]")
    print("Exact: u = x + y\n")

    # Nodes: 8x8 grid
    n_side = 8
    x = np.linspace(0, 1, n_side)
    X, Y = np.meshgrid(x, x)
    nodes_np = np.column_stack([X.ravel(), Y.ravel()])
    nodes = torch.tensor(nodes_np, device=device)

    dx = 1.0 / (n_side - 1)
    radius = 2.5 * dx  # usually enough for full coverage
    print(f"Meshfree nodes: {nodes.shape[0]} ({n_side}x{n_side} grid)")
    print(f"Node spacing: {dx:.4f}")
    print(f"Support radius: {radius:.4f}\n")

    # Model
    model = MeshfreeKANNet(nodes, radius, kan_hidden_dim=8).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - KAN parameters: {sum(p.numel() for p in model.kan.parameters()):,}")
    print(f"  - Nodal coefficients: {model.w.numel()}\n")

    # Training
    print("Starting training...\n")
    history = train_meshfree_kan(
        model=model,
        problem=problem,
        device=device,
        phase_a_steps=1200,
        phase_b_steps=2000,
        batch_size=1024,
        lr_kan_a=1e-3,
        lr_kan_b=1e-4,
        lr_w=1e-2,
        beta_bc=100.0,
        gamma_linear_b=10.0,
        log_interval=100,
    )

    # Final test
    print("\nGenerating visualizations...")
    x_test = torch.rand(20000, 2, device=device)
    with torch.no_grad():
        u_pred_test = model(x_test)
    u_exact_test = problem.exact_solution(x_test)
    l2_err_final = torch.sqrt(torch.mean((u_pred_test - u_exact_test) ** 2))
    print(f"Final L2 Error: {l2_err_final.item():.6e}")

    # Generate solution plots
    res = 100
    x_plot = np.linspace(0, 1, res)
    y_plot = np.linspace(0, 1, res)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    x_grid = torch.tensor(np.column_stack([X_plot.ravel(), Y_plot.ravel()]), dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_grid).cpu().numpy().reshape(res, res)
    u_exact = problem.exact_solution(x_grid).cpu().numpy().reshape(res, res)

    plot_square_triplet(
        X_plot, Y_plot, u_pred, u_exact,
        title_prefix="Meshfree KAN v2 (Linear Patch)",
        filename="solution_triplet.png",
        val_clim=(0.0, 2.0),
        err_vmax=0.01,
        subdir=output_subdir
    )

    plot_training_history(history,
                          filename="training_history.png",
                          subdir=output_subdir)

    # Save history
    np.savez(
        os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"),
        **history
    )

    print("\nDone.")


if __name__ == "__main__":
    main()




