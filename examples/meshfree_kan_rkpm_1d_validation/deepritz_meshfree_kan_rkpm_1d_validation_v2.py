"""
1D validation example: KAN shape functions vs analytical RKPM shape functions.

Goal:
- Train KAN-based meshfree shape functions in Phase A (geometry only)
- Build analytical RKPM shape functions in 1D with linear basis
- Compare curves and report L2/Linf errors

Outputs are saved to:
  output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_output/
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from visualizers import get_example_output_subdir
import visualizers
from examples.meshfree_kan_rkpm_1d_validation.common import (
    ensure_legacy_figure_artifacts_1d,
    plot_diagnostic_shape_overlay_1d,
    plot_main_figure_shape_validation_1d,
)


torch.set_default_dtype(torch.float64)


def cubic_spline_window_torch(dist: torch.Tensor, radius: float) -> torch.Tensor:
    """C2 cubic spline window with compact support."""
    q = dist / radius
    out = torch.zeros_like(q)

    m1 = q <= 0.5
    m2 = (q > 0.5) & (q <= 1.0)

    q1 = q[m1]
    out[m1] = 2.0 / 3.0 - 4.0 * q1**2 + 4.0 * q1**3

    q2 = q[m2]
    out[m2] = 4.0 / 3.0 - 4.0 * q2 + 4.0 * q2**2 - (4.0 / 3.0) * q2**3
    return out


def cubic_spline_window_np(dist: np.ndarray, radius: float) -> np.ndarray:
    """Numpy version of cubic spline window."""
    q = dist / radius
    out = np.zeros_like(q, dtype=np.float64)

    m1 = q <= 0.5
    m2 = (q > 0.5) & (q <= 1.0)

    q1 = q[m1]
    out[m1] = 2.0 / 3.0 - 4.0 * q1**2 + 4.0 * q1**3

    q2 = q[m2]
    out[m2] = 4.0 / 3.0 - 4.0 * q2 + 4.0 * q2**2 - (4.0 / 3.0) * q2**3
    return out


class KANSpline1D(nn.Module):
    """Lightweight KAN-like spline network in 1D."""

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
        b, d = x.shape
        grid = self.grid.view(1, 1, self.num_basis)
        x_exp = x.view(b, d, 1)
        return torch.relu(1.0 - torch.abs(x_exp - grid) / self.h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1]
        b1 = self._hat_basis(x).squeeze(1)            # [B, num_basis]
        h = self.layer1(b1)                           # [B, hidden_dim]
        b2 = self._hat_basis(h)                       # [B, hidden_dim, num_basis]
        y = self.layer2(b2.reshape(x.shape[0], -1))   # [B, 1]
        return y


class MeshfreeKAN1D(nn.Module):
    def __init__(self, nodes: torch.Tensor, support_radius: float, hidden_dim: int = 16):
        super().__init__()
        self.register_buffer("nodes", nodes.reshape(-1))
        self.n_nodes = int(nodes.numel())
        self.support_radius = float(support_radius)

        self.kan = KANSpline1D(hidden_dim=hidden_dim)
        self.softplus = nn.Softplus()

    def compute_shape_functions(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
        eps_cov: float = 1e-14,
        knn_k: int = 3,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:, 0]

        m = x.shape[0]
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)  # [M, N]
        dist = torch.abs(diff)

        kan_in = (diff / self.support_radius).reshape(-1, 1)
        phi_raw = self.softplus(self.kan(kan_in).reshape(m, self.n_nodes))
        window = cubic_spline_window_torch(dist, self.support_radius)
        phi_w = phi_raw * window

        if return_raw:
            return phi_w

        phi_sum = torch.sum(phi_w, dim=1, keepdim=True)
        orphan = phi_sum.squeeze(1) < eps_cov

        phi = torch.empty_like(phi_w)
        normal = ~orphan
        phi[normal] = phi_w[normal] / (phi_sum[normal] + 1e-12)

        if orphan.any():
            k = min(knn_k, self.n_nodes)
            dist_o = dist[orphan]
            idx = torch.topk(dist_o, k, largest=False).indices
            d_knn = torch.gather(dist_o, 1, idx)

            alpha = 20.0 / max(self.support_radius, 1e-12)
            w = torch.exp(-alpha * d_knn)
            w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-18)

            phi_o = torch.zeros((dist_o.shape[0], self.n_nodes), device=x.device, dtype=x.dtype)
            phi_o.scatter_(1, idx, w)
            phi[orphan] = phi_o

        return phi


def rkpm_shape_matrix_1d(
    x_eval: np.ndarray,
    nodes: np.ndarray,
    support_radius: float,
    cond_threshold: float = 1e12,
) -> np.ndarray:
    """
    Analytical RKPM shape functions in 1D with linear basis p=[1, r]^T.
    Psi_I(x) = p(0)^T M(x)^{-1} p(r_I) W(r_I)
    """
    p0 = np.array([1.0, 0.0], dtype=np.float64)
    m = x_eval.shape[0]
    n = nodes.shape[0]
    phi = np.zeros((m, n), dtype=np.float64)

    for ix, x in enumerate(x_eval):
        r = x - nodes
        w = cubic_spline_window_np(np.abs(r), support_radius)

        m11 = np.sum(w)
        m12 = np.sum(w * r)
        m22 = np.sum(w * r * r)
        mat = np.array([[m11, m12], [m12, m22]], dtype=np.float64)

        if np.linalg.cond(mat) > cond_threshold:
            mat_inv = np.linalg.pinv(mat, rcond=1e-12)
        else:
            mat_inv = np.linalg.inv(mat)

        for i in range(n):
            p_i = np.array([1.0, r[i]], dtype=np.float64)
            phi[ix, i] = p0 @ mat_inv @ p_i * w[i]

    return phi


def train_phase_a(
    model: MeshfreeKAN1D,
    nodes: torch.Tensor,
    device: str,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_pu: float = 0.1,
    log_interval: int = 100,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nodes_col = nodes.reshape(-1, 1)

    history = {"steps": [], "loss": [], "linear": [], "pu": []}

    for step in range(steps):
        x = torch.rand(batch_size, 1, device=device)

        # Train on raw-then-normalized phi to focus on geometric consistency.
        # IMPORTANT:
        # - Linear reproduction uses normalized phi (strict partition behavior)
        # - PU regularization uses raw phi sum (non-tautological regularization)
        phi_raw = model.compute_shape_functions(x, return_raw=True)
        phi_sum = torch.sum(phi_raw, dim=1, keepdim=True) + 1e-12
        phi = phi_raw / phi_sum

        repro_x = phi @ nodes_col
        loss_linear = torch.mean((repro_x - x) ** 2)
        loss_pu = torch.mean((torch.sum(phi_raw, dim=1, keepdim=True) - 1.0) ** 2)
        loss = loss_linear + lambda_pu * loss_pu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == steps - 1:
            history["steps"].append(step)
            history["loss"].append(float(loss.item()))
            history["linear"].append(float(loss_linear.item()))
            history["pu"].append(float(loss_pu.item()))
            print(
                f"Step {step:4d} | Loss={loss.item():.3e} | "
                f"Linear={loss_linear.item():.3e} | PU={loss_pu.item():.3e}"
            )

    return history


def build_plots(
    x_eval: np.ndarray,
    phi_rkpm: np.ndarray,
    phi_kan: np.ndarray,
    history: dict,
    center_idx: int,
    boundary_idx: int,
    out_dir: Path,
    metrics: dict,
):
    figures_dir, diagnostics_dir = ensure_legacy_figure_artifacts_1d(out_dir)
    plot_main_figure_shape_validation_1d(
        x_eval=x_eval,
        phi_rkpm=phi_rkpm,
        phi_kan=phi_kan,
        history=history,
        center_idx=center_idx,
        boundary_idx=boundary_idx,
        metrics=metrics,
        path=figures_dir / "main_figure.png",
    )
    plot_diagnostic_shape_overlay_1d(
        x_eval=x_eval,
        phi_rkpm=phi_rkpm,
        phi_kan=phi_kan,
        path=diagnostics_dir / "shape_subset_overlay.png",
    )


def main():
    parser = argparse.ArgumentParser(description="1D RKPM vs KAN shape function validation")
    parser.add_argument("--n-nodes", type=int, default=11, help="Number of uniform nodes in [0,1]")
    parser.add_argument("--support-factor", type=float, default=2.0, help="Support radius factor: R = support_factor*h")
    parser.add_argument("--steps", type=int, default=2000, help="Phase A training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-pu", type=float, default=0.1, help="PU penalty weight")
    parser.add_argument("--hidden-dim", type=int, default=16, help="KAN hidden dimension")
    parser.add_argument("--n-eval", type=int, default=2001, help="Number of evaluation points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix for output subdir name, e.g. 'raw_pu_fix'",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(__file__).resolve().parents[2]
    visualizers.OUTPUT_DIR = str(root / "output" / "meshfree_kan_rkpm_1d_validation")
    base_subdir = get_example_output_subdir(__file__)
    tag = args.output_tag.strip()
    subdir = f"{base_subdir}_{tag}" if tag else base_subdir
    out_dir = Path(visualizers.OUTPUT_DIR) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | dtype: {torch.get_default_dtype()}")
    print(f"Output directory: {out_dir}")

    # Nodes and support
    nodes_np = np.linspace(0.0, 1.0, args.n_nodes, dtype=np.float64)
    h = 1.0 / (args.n_nodes - 1)
    support_radius = args.support_factor * h
    nodes_t = torch.tensor(nodes_np, device=device)
    print(f"Nodes: {args.n_nodes}, h={h:.6f}, R={support_radius:.6f}")

    # Train KAN Phase A
    model = MeshfreeKAN1D(nodes_t, support_radius=support_radius, hidden_dim=args.hidden_dim).to(device)
    history = train_phase_a(
        model=model,
        nodes=nodes_t,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_pu=args.lambda_pu,
        log_interval=args.log_interval,
    )

    # Evaluate
    x_eval = np.linspace(0.0, 1.0, args.n_eval, dtype=np.float64)
    x_eval_t = torch.tensor(x_eval, device=device).unsqueeze(1)

    with torch.no_grad():
        phi_kan = model.compute_shape_functions(x_eval_t).cpu().numpy()
        phi_raw_eval = model.compute_shape_functions(x_eval_t, return_raw=True).cpu().numpy()
    phi_rkpm = rkpm_shape_matrix_1d(x_eval, nodes_np, support_radius=support_radius)

    center_idx = int(np.argmin(np.abs(nodes_np - 0.5)))
    boundary_idx = 0

    # Metrics
    diff = phi_kan - phi_rkpm
    l2_global = float(np.sqrt(np.mean(diff**2)))
    linf_global = float(np.max(np.abs(diff)))

    c_diff = phi_kan[:, center_idx] - phi_rkpm[:, center_idx]
    b_diff = phi_kan[:, boundary_idx] - phi_rkpm[:, boundary_idx]
    l2_center = float(np.sqrt(np.mean(c_diff**2)))
    linf_center = float(np.max(np.abs(c_diff)))
    l2_boundary = float(np.sqrt(np.mean(b_diff**2)))
    linf_boundary = float(np.max(np.abs(b_diff)))

    pu_max = float(np.max(np.abs(np.sum(phi_kan, axis=1) - 1.0)))
    pu_raw_rmse = float(np.sqrt(np.mean((np.sum(phi_raw_eval, axis=1) - 1.0) ** 2)))
    lin_repro = float(np.sqrt(np.mean((phi_kan @ nodes_np.reshape(-1, 1) - x_eval.reshape(-1, 1))**2)))

    metrics = {
        "n_nodes": int(args.n_nodes),
        "h": float(h),
        "support_radius": float(support_radius),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "center_node_index": int(center_idx),
        "boundary_node_index": int(boundary_idx),
        "global_l2": l2_global,
        "global_linf": linf_global,
        "center_l2": l2_center,
        "center_linf": linf_center,
        "boundary_l2": l2_boundary,
        "boundary_linf": linf_boundary,
        "pu_max_error": pu_max,
        "pu_raw_sum_rmse": pu_raw_rmse,
        "linear_reproduction_rmse": lin_repro,
    }

    # Save artifacts
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_dir / "curves.npz",
        x_eval=x_eval,
        nodes=nodes_np,
        phi_rkpm=phi_rkpm,
        phi_kan=phi_kan,
        history_steps=np.array(history["steps"], dtype=np.int64),
        history_loss=np.array(history["loss"], dtype=np.float64),
        history_linear=np.array(history["linear"], dtype=np.float64),
        history_pu=np.array(history["pu"], dtype=np.float64),
    )

    build_plots(
        x_eval=x_eval,
        phi_rkpm=phi_rkpm,
        phi_kan=phi_kan,
        history=history,
        center_idx=center_idx,
        boundary_idx=boundary_idx,
        out_dir=out_dir,
        metrics=metrics,
    )

    print("\nValidation summary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()

