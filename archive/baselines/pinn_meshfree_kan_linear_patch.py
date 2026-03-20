"""
Example: PINN Meshfree KAN for Linear Patch Test
Equation: -Δu = 0 in [0,1]x[0,1]
Exact: u = x + y
Method: Physics-Informed Neural Networks (strong form)
Target: Machine precision with PU + linear completeness
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
from core.trainers import PINNTrainer  # For _laplacian static method
from core.visualizers import plot_square_triplet, plot_training_history, get_example_output_subdir
import core.visualizers as visualizers


# ============================================================================
# 1. Window Function (Cubic Spline - C2 continuous)
# ============================================================================
def cubic_spline_window(dist, radius):
    """
    Cubic spline window function with compact support.

    W(q) = 1 - 6q^2 + 8q^3 - 3q^4,  0 <= q <= 1
           0,                        q > 1

    where q = dist / radius

    Args:
        dist: [M, N] distance matrix
        radius: support radius

    Returns:
        window: [M, N] window values
    """
    q = dist / radius
    window = torch.zeros_like(q)

    # Apply window function where q <= 1
    mask = (q <= 1.0)
    q_masked = q[mask]
    window[mask] = 1.0 - 6.0 * q_masked**2 + 8.0 * q_masked**3 - 3.0 * q_masked**4

    return window


# ============================================================================
# 2. Shared KAN for Shape Functions
# ============================================================================
class KANSplineNet(nn.Module):
    """
    KAN network using B-spline basis functions.
    Copied from networks.py for standalone use.
    """
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1, num=5, k=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num = num
        self.k = k
        self.grid_range = grid_range

        # Layer 1: input -> hidden
        self.layer1 = nn.ModuleList([
            self._create_spline_layer(1, hidden_dim) for _ in range(input_dim)
        ])

        # Layer 2: hidden -> output
        self.layer2 = self._create_spline_layer(hidden_dim, output_dim)

    def _create_spline_layer(self, in_features, out_features):
        """Create a B-spline layer."""
        # For linear B-splines (hat functions), we have exactly 'num' basis functions
        layer = nn.Linear(in_features * self.num, out_features, bias=False)
        return layer

    def _bspline_basis(self, x):
        """
        Linear B-spline basis functions (Hat Functions).
        Fixed version with non-zero gradients for PDE solving.

        Args:
            x: [batch, features] input tensor

        Returns:
            basis: [batch, features, num] basis values
        """
        batch_size, features = x.shape
        grid_min, grid_max = self.grid_range

        # Generate uniform grid points
        grid = torch.linspace(grid_min, grid_max, self.num, device=x.device)  # [num]
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, num]

        x_expanded = x.unsqueeze(-1)  # [batch, features, 1]

        # Compute grid spacing h
        h = (grid_max - grid_min) / (self.num - 1)

        # Linear B-spline formula: max(0, 1 - |x - c| / h)
        # This is a triangular wave with non-zero derivatives
        basis = torch.relu(1 - torch.abs(x_expanded - grid) / h)

        return basis

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [batch, input_dim] input tensor

        Returns:
            output: [batch, output_dim] output tensor
        """
        # Layer 1: process each input dimension separately
        hidden_outputs = []
        for i, layer in enumerate(self.layer1):
            x_i = x[:, i:i+1]  # [batch, 1]
            basis = self._bspline_basis(x_i)  # [batch, 1, num]
            basis_flat = basis.reshape(x.shape[0], -1)  # [batch, num]
            hidden_outputs.append(layer(basis_flat))  # [batch, hidden_dim]

        # Sum contributions from all input dimensions
        hidden = sum(hidden_outputs)  # [batch, hidden_dim]

        # Layer 2: hidden -> output
        basis_hidden = self._bspline_basis(hidden)  # [batch, hidden_dim, num]
        basis_flat = basis_hidden.reshape(hidden.shape[0], -1)  # [batch, hidden_dim*num]
        output = self.layer2(basis_flat)  # [batch, output_dim]

        return output


# ============================================================================
# 3. Meshfree KAN Network
# ============================================================================
class MeshfreeKANNet(nn.Module):
    """
    Meshfree network using shared KAN for shape functions.

    Architecture:
        u(x) = Σ φ_I(x) * w_I

    where φ_I(x) is learned by a shared KAN with compact support.
    """

    def __init__(self, nodes, support_radius, kan_hidden_dim=8):
        """
        Args:
            nodes: [N, 2] fixed node positions
            support_radius: compact support radius
            kan_hidden_dim: hidden dimension for KAN
        """
        super().__init__()

        self.nodes = nn.Parameter(nodes, requires_grad=False)
        self.N = nodes.shape[0]
        self.support_radius = support_radius

        # Learnable nodal coefficients
        self.w = nn.Parameter(torch.zeros(self.N, 1))

        # Shared KAN for shape functions (input: 2D relative coords, output: 1D)
        self.kan = KANSplineNet(
            input_dim=2,
            hidden_dim=kan_hidden_dim,
            output_dim=1,
            num=5,
            k=3,
            grid_range=(-1.5, 1.5)  # Slightly larger than [-1, 1] for safety
        )

    def compute_shape_functions(self, x, return_unnormalized=False):
        """
        Compute shape functions φ_I(x) for all nodes.

        Args:
            x: [M, 2] sample points
            return_unnormalized: if True, return unnormalized phi_windowed

        Returns:
            phi: [M, N] shape function values (normalized unless return_unnormalized=True)
        """
        M = x.shape[0]
        N = self.N

        # 1. Compute relative coordinates [M, N, 2]
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)  # [M, N, 2]

        # 2. Compute distances [M, N]
        dist = torch.norm(diff, dim=2)

        # 3. Normalize input for KAN (map to [-1, 1] range)
        kan_input = diff / self.support_radius  # [M, N, 2]

        # 4. Evaluate KAN (shared across all nodes)
        # Flatten to [M*N, 2], pass through KAN, reshape to [M, N]
        kan_input_flat = kan_input.reshape(-1, 2)  # [M*N, 2]
        phi_raw = self.kan(kan_input_flat).reshape(M, N)  # [M, N]

        # 5. Apply compact support window
        window = cubic_spline_window(dist, self.support_radius)  # [M, N]
        phi_windowed = phi_raw * window  # [M, N]

        # Return unnormalized if requested (for PU loss computation)
        if return_unnormalized:
            return phi_windowed

        # 6. Normalize to enforce partition of unity: Σφ_I = 1
        phi_sum = torch.sum(phi_windowed, dim=1, keepdim=True) + 1e-10  # [M, 1]
        phi_normalized = phi_windowed / phi_sum  # [M, N]

        return phi_normalized

    def forward(self, x):
        """
        Forward pass: compute u(x) = Σ φ_I(x) * w_I

        Args:
            x: [M, 2] sample points

        Returns:
            u: [M, 1] field values
        """
        phi = self.compute_shape_functions(x)  # [M, N]
        u = torch.matmul(phi, self.w)  # [M, 1]
        return u


# ============================================================================
# 4. Training Function with Two-Phase Strategy
# ============================================================================
def train_meshfree_kan_pinn(
    model,
    problem,
    device,
    phase_a_steps=500,
    phase_b_steps=2000,
    batch_size=1000,
    lr_kan=1e-3,
    lr_w=1e-2,
    beta_bc=100.0,
    log_interval=100
):
    """
    Two-phase training for Meshfree KAN using PINN method.

    Phase A (Geometry Warmup): Train KAN only, enforce PU
    Phase B (Physics Solve): Train both KAN and w, minimize PDE residual
    """
    history = {
        'steps': [],
        'loss': [],
        'pu_loss': [],
        'pde': [],
        'boundary': [],
        'l2_error': [],
        'h1_error': []
    }

    # Phase A: Geometry Warmup (Train KAN only)
    print("=" * 70)
    print(f"Phase A: Geometry Warmup ({phase_a_steps} steps)")
    print("=" * 70)

    model.w.requires_grad = False
    optimizer_kan = torch.optim.Adam(model.kan.parameters(), lr=lr_kan)

    for step in range(phase_a_steps):
        x_domain = torch.rand(batch_size, 2, device=device)

        # Compute UNNORMALIZED shape functions for PU loss
        phi_unnorm = model.compute_shape_functions(x_domain, return_unnormalized=True)
        phi_sum = torch.sum(phi_unnorm, dim=1)
        loss_pu = torch.mean((phi_sum - 1.0) ** 2)

        optimizer_kan.zero_grad()
        loss_pu.backward()
        optimizer_kan.step()

        if step % log_interval == 0:
            print(f"Step {step:4d} | PU Loss={loss_pu.item():.6e}")

    print(f"Phase A completed. Final PU Loss={loss_pu.item():.6e}\n")



    # Phase B: Physics Solve (Train both KAN and w)
    print("=" * 70)
    print(f"Phase B: Physics Solve ({phase_b_steps} steps)")
    print("=" * 70)

    model.w.requires_grad = True
    optimizer_all = torch.optim.Adam([
        {'params': model.kan.parameters(), 'lr': lr_kan},
        {'params': [model.w], 'lr': lr_w}
    ])

    for step in range(phase_b_steps):
        x_domain = torch.rand(batch_size, 2, device=device)
        x_domain.requires_grad_(True)

        # Boundary sampling with proper remainder handling
        n_per_edge = batch_size // 4
        n_remainder = batch_size % 4

        x_boundary = torch.cat([
            torch.stack([torch.zeros(n_per_edge), torch.rand(n_per_edge)], dim=1),  # left
            torch.stack([torch.ones(n_per_edge), torch.rand(n_per_edge)], dim=1),   # right
            torch.stack([torch.rand(n_per_edge), torch.zeros(n_per_edge)], dim=1),  # bottom
            torch.stack([torch.rand(n_per_edge), torch.ones(n_per_edge)], dim=1),   # top
        ], dim=0)

        # Add remainder points to left edge
        if n_remainder > 0:
            x_boundary_extra = torch.stack([torch.zeros(n_remainder), torch.rand(n_remainder)], dim=1)
            x_boundary = torch.cat([x_boundary, x_boundary_extra], dim=0)

        x_boundary = x_boundary.to(device)

        u_domain = model(x_domain)

        # Compute Laplacian for PDE residual
        laplacian = PINNTrainer._laplacian(u_domain, x_domain)
        f = problem.source_term(x_domain)

        # PDE residual: -Δu = f  =>  residual = Δu + f
        pde_residual = laplacian + f
        loss_pde = torch.mean(pde_residual ** 2) * problem.area_domain

        u_boundary = model(x_boundary)
        u_exact_boundary = problem.exact_solution(x_boundary)
        loss_bc = torch.mean((u_boundary - u_exact_boundary) ** 2) * beta_bc

        # Linear reproduction constraint (for machine precision on patch test)
        # Σ φ_I(x) · x_I = x and Σ φ_I(x) · y_I = y
        phi_domain = model.compute_shape_functions(x_domain)  # [M, N]
        nodes_x = model.nodes[:, 0:1]  # [N, 1]
        nodes_y = model.nodes[:, 1:2]  # [N, 1]

        # Compute Σ φ_I(x) · x_I and Σ φ_I(x) · y_I
        reproduced_x = torch.matmul(phi_domain, nodes_x)  # [M, 1]
        reproduced_y = torch.matmul(phi_domain, nodes_y)  # [M, 1]

        # Target: should equal x_domain coordinates
        target_x = x_domain[:, 0:1]  # [M, 1]
        target_y = x_domain[:, 1:2]  # [M, 1]

        # Linear reproduction loss
        loss_linear = torch.mean((reproduced_x - target_x) ** 2 + (reproduced_y - target_y) ** 2)
        gamma_linear = 10.0  # Weight for linear reproduction constraint

        loss = loss_pde + loss_bc + gamma_linear * loss_linear

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        if step % log_interval == 0:
            # === 1. L2 Error (不需要梯度，使用 no_grad 节省显存) ===
            with torch.no_grad():
                x_eval = torch.rand(2000, 2, device=device)
                u_pred = model(x_eval)
                u_exact = problem.exact_solution(x_eval)
                l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()

            # === 2. H1 Error (必须需要梯度，移出 no_grad 块) ===
            # 重新生成一个需要梯度的输入
            x_eval_grad = torch.rand(2000, 2, device=device)
            x_eval_grad.requires_grad_(True)  # 开启梯度追踪

            u_pred_grad = model(x_eval_grad)

            # 此时不在 no_grad 块内，u_pred_grad 带有 grad_fn，可以求导
            grad_pred = torch.autograd.grad(
                outputs=u_pred_grad,
                inputs=x_eval_grad,
                grad_outputs=torch.ones_like(u_pred_grad),
                create_graph=False  # 这里不需要二阶导，False即可
            )[0]

            grad_exact = problem.exact_gradient(x_eval_grad)
            h1_error = torch.sqrt(torch.mean(torch.sum((grad_pred - grad_exact) ** 2, dim=1))).item()

            # === 3. 记录日志 ===
            history['steps'].append(phase_a_steps + step)
            history['loss'].append(loss.item())
            history['pde'].append(loss_pde.item())
            history['boundary'].append(loss_bc.item())
            history['l2_error'].append(l2_error)
            history['h1_error'].append(h1_error)

            print(f"Step {step:4d} | Loss={loss.item():.5e} | "
                  f"PDE={loss_pde.item():.5e} | BC={loss_bc.item():.5e} | "
                  f"L2={l2_error:.2e} | H1={h1_error:.2e}")

    print(f"\nPhase B completed.")
    return history


# ============================================================================
# 5. Main Function
# ============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    ROOT = Path(__file__).resolve().parents[2]
    visualizers.OUTPUT_DIR = str(ROOT / "output" / "archive" / "baselines")
    output_subdir = get_example_output_subdir(__file__)
    print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/\n")

    problem = LinearPatchProblem()
    print(f"Problem: {problem.name}")
    print("Domain: [0,1] x [0,1]")
    print("Exact: u = x + y\n")

    n_nodes_per_dim = 8
    x_nodes = np.linspace(0, 1, n_nodes_per_dim)
    y_nodes = np.linspace(0, 1, n_nodes_per_dim)
    X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes)
    nodes_np = np.column_stack([X_nodes.ravel(), Y_nodes.ravel()])
    nodes = torch.tensor(nodes_np, dtype=torch.float32, device=device)

    print(f"Meshfree nodes: {nodes.shape[0]} ({n_nodes_per_dim}x{n_nodes_per_dim} grid)")

    node_spacing = 1.0 / (n_nodes_per_dim - 1)
    support_radius = 2.5 * node_spacing
    print(f"Node spacing: {node_spacing:.4f}")
    print(f"Support radius: {support_radius:.4f}\n")

    model = MeshfreeKANNet(
        nodes=nodes,
        support_radius=support_radius,
        kan_hidden_dim=8
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - KAN parameters: {sum(p.numel() for p in model.kan.parameters()):,}")
    print(f"  - Nodal coefficients: {model.w.numel()}\n")

    print("Starting training...\n")
    history = train_meshfree_kan_pinn(
        model=model,
        problem=problem,
        device=device,
        phase_a_steps=500,
        phase_b_steps=2000,
        batch_size=1000,
        lr_kan=1e-3,
        lr_w=1e-2,
        beta_bc=100.0,
        log_interval=100
    )

    print("\nGenerating visualizations...")

    res = 100
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y)
    x_test = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy().reshape(res, res)
    u_exact = problem.exact_solution(x_test).cpu().numpy().reshape(res, res)

    plot_square_triplet(
        X, Y, u_pred, u_exact,
        title_prefix="PINN Meshfree KAN (Linear Patch)",
        filename="solution_triplet.png",
        val_clim=(0.0, 2.0),
        err_vmax=0.01,
        subdir=output_subdir
    )

    plot_training_history(history, filename="training_history.png", subdir=output_subdir)

    np.savez(
        os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"),
        **history
    )

    print("\nDone.")


if __name__ == "__main__":
    main()



