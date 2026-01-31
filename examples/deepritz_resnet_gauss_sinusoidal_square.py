"""
Example 2: Sinusoidal solution on a square domain.
Equation: -Δu = f in [0,1]x[0,1], u=0 on boundary
Exact: u = sin(pi x) sin(pi y)
"""

import sys
sys.path.append('..')

import os
import torch
import numpy as np
from pathlib import Path
from problems import SinusoidalSquare
from networks import RitzNet
from integrators import GaussIntegrator
from trainers import DeepRitzTrainer
from visualizers import plot_training_history, plot_square_triplet, get_example_output_subdir
import visualizers

# ============================================================================
# 0. 项目初始化（种子，设备，路径）
# ============================================================================
# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Output directory management - use absolute path
ROOT = Path(__file__).resolve().parents[1]
visualizers.OUTPUT_DIR = str(ROOT / "output")
output_subdir = get_example_output_subdir(__file__)
print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/")

# ============================================================================
# 1. 定义问题
# ============================================================================
problem = SinusoidalSquare()
print(f"Problem: {problem.name}")
print("Domain: [0,1] x [0,1]")
print("Exact: u = sin(pi x) sin(pi y)")

# 2) Model
model = RitzNet(
    input_dim=2,
    hidden_dims=[128, 128, 128, 128],
    output_dim=1
).to(device)

# 3) Integrator
integrator = GaussIntegrator(nx=16, ny=16, domain_bounds=(0, 1, 0, 1), order=4)

# 4) Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================================
# 5. 训练
# ============================================================================
trainer = DeepRitzTrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device=device,
    beta_bc=500.0
)

print("\nTraining...")
history = trainer.train(n_steps=8000, log_interval=200, eval_interval=200)

# ============================================================================
# 6. 可视化
# ============================================================================
print("\nGenerating visualizations...")

# 生成测试网格
res = 100
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
X, Y = np.meshgrid(x, y)
x_test = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

# 预测和解析解
model.eval()
with torch.no_grad():
    u_pred = model(x_test).cpu().numpy().reshape(res, res)
u_exact = problem.exact_solution(x_test).cpu().numpy().reshape(res, res)

# 绘制三联图
plot_square_triplet(
    X, Y, u_pred, u_exact,
    title_prefix="Deep Ritz (Sinusoidal)",
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
