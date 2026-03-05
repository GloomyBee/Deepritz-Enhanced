"""
1D validation example (v5): Teacher distillation toward analytical RKPM.

Goal:
- Remove Softplus to test expressive capacity
- Distill KAN shape functions to analytical RKPM targets
- Compare curves and report L2/Linf errors
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from visualizers import get_example_output_subdir
import visualizers


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
        phi_raw = self.kan(kan_in).reshape(m, self.n_nodes)
        window = cubic_spline_window_torch(dist, self.support_radius)
        phi_w = phi_raw * window

        if return_raw:
            return phi_w

        phi_sum = torch.sum(phi_w, dim=1, keepdim=True)
        orphan = torch.abs(phi_sum.squeeze(1)) < eps_cov

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


def rkpm_shape_matrix_1d_torch(
    x_eval: torch.Tensor,
    nodes: torch.Tensor,
    support_radius: float,
    eps_det: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized torch RKPM target in 1D with linear basis p=[1, r]^T.
    """
    if x_eval.dim() == 2:
        x_eval = x_eval[:, 0]
    if nodes.dim() != 1:
        nodes = nodes.reshape(-1)

    r = x_eval.unsqueeze(1) - nodes.unsqueeze(0)  # [M, N]
    w = cubic_spline_window_torch(torch.abs(r), support_radius)

    s0 = torch.sum(w, dim=1, keepdim=True)          # [M,1]
    s1 = torch.sum(w * r, dim=1, keepdim=True)      # [M,1]
    s2 = torch.sum(w * r * r, dim=1, keepdim=True)  # [M,1]

    det = s0 * s2 - s1 * s1
    sign = torch.where(det >= 0.0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = det + sign * eps_det

    phi = ((s2 - s1 * r) / det_safe) * w
    return phi


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
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nodes_col = nodes.reshape(-1, 1)

    history = {
        "steps": [],
        "loss": [],
        "teacher": [],
        "linear": [],
        "pu": [],
        "bd": [],
        "reg": [],
    }
    x_bd = torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float64)

    for step in range(steps):
        x = torch.rand(batch_size, 1, device=device)
        x_all = torch.cat([x, x_bd], dim=0)

        phi = model.compute_shape_functions(x_all)
        with torch.no_grad():
            phi_teacher = rkpm_shape_matrix_1d_torch(
                x_all,
                nodes,
                support_radius=model.support_radius,
            )

        # Distillation loss: directly match analytical RKPM target.
        loss_teacher = torch.mean((phi - phi_teacher) ** 2)

        # Optional endpoint anchoring.
        phi_bd = phi[-2:, :]
        loss_bd_0 = (phi_bd[0, 0] - 1.0) ** 2 + torch.sum(phi_bd[0, 1:] ** 2)
        loss_bd_1 = (phi_bd[1, -1] - 1.0) ** 2 + torch.sum(phi_bd[1, :-1] ** 2)
        loss_bd = loss_bd_0 + loss_bd_1

        # Mild amplitude regularization on raw outputs to suppress cancellation spikes.
        phi_raw_all = model.compute_shape_functions(x_all, return_raw=True)
        loss_reg = torch.mean(phi_raw_all ** 2)

        # Diagnostics (not primary objective).
        repro_x = phi @ nodes_col
        loss_linear = torch.mean((repro_x - x_all) ** 2)
        loss_pu = torch.mean((torch.sum(phi, dim=1, keepdim=True) - 1.0) ** 2)

        loss = lambda_teacher * loss_teacher + lambda_bd * loss_bd + lambda_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == steps - 1:
            history["steps"].append(step)
            history["loss"].append(float(loss.item()))
            history["teacher"].append(float(loss_teacher.item()))
            history["linear"].append(float(loss_linear.item()))
            history["pu"].append(float(loss_pu.item()))
            history["bd"].append(float(loss_bd.item()))
            history["reg"].append(float(loss_reg.item()))
            print(
                f"Step {step:4d} | Loss={loss.item():.3e} | "
                f"Teacher={loss_teacher.item():.3e} | Linear={loss_linear.item():.3e} | "
                f"PU={loss_pu.item():.3e} | BD={loss_bd.item():.3e} | Reg={loss_reg.item():.3e}"
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
):
    # Main comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(x_eval, phi_rkpm[:, center_idx], color="tab:blue", lw=2, label="RKPM Exact")
    ax.plot(x_eval, phi_kan[:, center_idx], color="tab:red", lw=2, ls="--", label="KAN Learned")
    ax.set_title(f"Center Node Shape Function (index={center_idx})")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x)$")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x_eval, phi_rkpm[:, boundary_idx], color="tab:blue", lw=2, label="RKPM Exact")
    ax.plot(x_eval, phi_kan[:, boundary_idx], color="tab:red", lw=2, ls="--", label="KAN Learned")
    ax.set_title(f"Boundary Node Shape Function (index={boundary_idx})")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x)$")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    err_center = np.abs(phi_kan[:, center_idx] - phi_rkpm[:, center_idx])
    err_boundary = np.abs(phi_kan[:, boundary_idx] - phi_rkpm[:, boundary_idx])
    ax = axes[1, 0]
    ax.semilogy(x_eval, np.maximum(err_center, 1e-16), lw=2, label="Center Error")
    ax.semilogy(x_eval, np.maximum(err_boundary, 1e-16), lw=2, label="Boundary Error")
    ax.set_title("Pointwise Absolute Error")
    ax.set_xlabel("x")
    ax.set_ylabel("abs error")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    ax = axes[1, 1]
    steps = np.array(history["steps"], dtype=np.int64)
    for key, label in [
        ("loss", "Total"),
        ("teacher", "Teacher"),
        ("linear", "Linear"),
        ("pu", "PU"),
        ("bd", "BD"),
        ("reg", "Reg"),
    ]:
        if key in history and len(history[key]) > 0:
            ax.semilogy(steps, np.maximum(np.array(history[key]), 1e-16), lw=2, label=label)
    ax.set_title("Phase A Training Curves")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "shape_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Optional all-node heatmap-like overlay (compact check)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    n_nodes = phi_kan.shape[1]
    step = max(1, n_nodes // 5)
    for i in range(0, n_nodes, step):
        ax.plot(x_eval, phi_rkpm[:, i], color="tab:blue", alpha=0.35)
        ax.plot(x_eval, phi_kan[:, i], color="tab:red", alpha=0.35, ls="--")
    ax.set_title("Subset of Node Shape Functions (blue=RKPM, red=KAN)")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi_i(x)$")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "shape_subset_overlay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="1D RKPM vs KAN shape function validation")
    parser.add_argument("--n-nodes", type=int, default=11, help="Number of uniform nodes in [0,1]")
    parser.add_argument("--support-factor", type=float, default=2.0, help="Support radius factor: R = support_factor*h")
    parser.add_argument("--steps", type=int, default=2000, help="Phase A training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-teacher", type=float, default=1.0, help="Teacher loss weight")
    parser.add_argument("--lambda-bd", type=float, default=0.1, help="Boundary anchor penalty weight")
    parser.add_argument("--lambda-reg", type=float, default=1e-4, help="Raw amplitude regularization weight")
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
    history = train_phase_a_distill(
        model=model,
        nodes=nodes_t,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_teacher=args.lambda_teacher,
        lambda_bd=args.lambda_bd,
        lambda_reg=args.lambda_reg,
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
    x0_idx = int(np.argmin(np.abs(x_eval - 0.0)))
    x1_idx = int(np.argmin(np.abs(x_eval - 1.0)))
    phi0_at_0 = float(phi_kan[x0_idx, 0])
    phiN_at_1 = float(phi_kan[x1_idx, -1])
    lin_moment_at_0 = float(np.sum(phi_kan[x0_idx, :] * nodes_np))
    lin_moment_at_1 = float(np.sum(phi_kan[x1_idx, :] * nodes_np))

    metrics = {
        "n_nodes": int(args.n_nodes),
        "h": float(h),
        "support_radius": float(support_radius),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "lambda_teacher": float(args.lambda_teacher),
        "lambda_bd": float(args.lambda_bd),
        "lambda_reg": float(args.lambda_reg),
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
        "phi0_at_x0": phi0_at_0,
        "phiN_at_x1": phiN_at_1,
        "first_moment_at_x0": lin_moment_at_0,
        "first_moment_at_x1": lin_moment_at_1,
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
        history_teacher=np.array(history["teacher"], dtype=np.float64),
        history_linear=np.array(history["linear"], dtype=np.float64),
        history_pu=np.array(history["pu"], dtype=np.float64),
        history_bd=np.array(history["bd"], dtype=np.float64),
        history_reg=np.array(history["reg"], dtype=np.float64),
    )

    build_plots(
        x_eval=x_eval,
        phi_rkpm=phi_rkpm,
        phi_kan=phi_kan,
        history=history,
        center_idx=center_idx,
        boundary_idx=boundary_idx,
        out_dir=out_dir,
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

