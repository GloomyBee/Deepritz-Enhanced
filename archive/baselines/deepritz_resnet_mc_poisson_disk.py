"""
Example 1: Deep Ritz for Poisson on a disk domain.
Equation: -Δu = f in Ω = {x in R^2: |x| < 1}
Exact: u = sin(pi x) sin(pi y)
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
from pathlib import Path
from core.problems import PoissonDisk
from core.networks import RitzNet
from core.integrators import MonteCarloIntegrator
from core.trainers import DeepRitzTrainer
from core.visualizers import plot_training_history, plot_square_triplet, get_example_output_subdir
import core.visualizers as visualizers

# ============================================================================
# 0. 项目初始化（种子，设备，路径）
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Output directory management - use absolute path
ROOT = Path(__file__).resolve().parents[2]
visualizers.OUTPUT_DIR = str(ROOT / "output" / "archive" / "baselines")
output_subdir = get_example_output_subdir(__file__)
print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/")

# ============================================================================
# 1. 定义问题
# ============================================================================
problem = PoissonDisk(radius=1.0)
print(f"Problem: {problem.name}")

# ============================================================================
# 2. 模型参数
# ============================================================================
model = RitzNet(
    input_dim=2,
    hidden_dims=[100, 100, 100],
    output_dim=1
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================================
# 3. 积分参数
# ============================================================================
integrator = MonteCarloIntegrator(
    n_points=8000,
    domain_bounds=(-1, 1, -1, 1)
)


# ============================================================================
# 4. 优化器
# ============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================================================
# 5. 训练器
# ============================================================================
trainer = DeepRitzTrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device=device,
    beta_bc=1000.0
)

print("\nTraining...")
history = trainer.train(n_steps=5000, log_interval=200, eval_interval=200)

# ============================================================================
# 6.可视化
# ============================================================================
print("\nGenerating visualizations...")

# Generate test grid
res = 100
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
X, Y = np.meshgrid(x, y)
x_test = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

# Prediction and exact solution
model.eval()
with torch.no_grad():
    u_pred = model(x_test).cpu().numpy().reshape(res, res)
u_exact = problem.exact_solution(x_test).cpu().numpy().reshape(res, res)

# Mask outside disk
mask = (X**2 + Y**2) > 1.0
u_pred_masked = np.copy(u_pred)
u_exact_masked = np.copy(u_exact)
u_pred_masked[mask] = np.nan
u_exact_masked[mask] = np.nan

# Plot triplet
plot_square_triplet(
    X, Y, u_pred_masked, u_exact_masked,
    title_prefix="Deep Ritz (Poisson Disk)",
    filename="solution_triplet.png",
    val_clim=(0.0, 1.0),
    err_vmax=0.05,
    subdir=output_subdir
)

# Plot training history
plot_training_history(history, filename="training_history.png", subdir=output_subdir)

# Save training history data
trainer.save_history(os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"))

print("\nDone.")



