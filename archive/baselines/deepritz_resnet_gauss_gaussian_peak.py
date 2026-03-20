"""
Example 4: Gaussian peak (high gradient) on a square domain.
Equation: -Δu = f in [0,1]x[0,1]
Exact: u = exp(-r^2/alpha^2), r^2 = (x-0.5)^2 + (y-0.5)^2
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
from pathlib import Path
from core.problems import GaussianPeak2D
from core.networks import RitzNet
from core.integrators import GaussIntegrator
from core.trainers import DeepRitzTrainer
from core.visualizers import plot_training_history, plot_square_triplet, get_example_output_subdir
import core.visualizers as visualizers

# ============================================================================
# 0. 项目初始化（种子，设备，路径）
# ============================================================================
# Set random seed
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
alpha = 0.1
problem = GaussianPeak2D(alpha=alpha, xc=0.5, yc=0.5)
print(f"Problem: {problem.name}")
print(f"Alpha: {alpha}")
print("Center: (0.5, 0.5)")

# ============================================================================
# 2) Model
# ============================================================================
model = RitzNet(
    input_dim=2,
    hidden_dims=[256, 256, 256, 256, 256, 256],
    output_dim=1
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 3) Integrator
# ============================================================================
integrator = GaussIntegrator(nx=24, ny=24, domain_bounds=(0, 1, 0, 1), order=4)

# ============================================================================
# 4) Optimizer
# ============================================================================
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
print("Note: high gradient problem may take longer...")
history = trainer.train(n_steps=15000, log_interval=500, eval_interval=500)

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

# 绘制三联图（高斯峰值域范围不同）
plot_square_triplet(
    X, Y, u_pred, u_exact,
    title_prefix="Deep Ritz (Gaussian Peak)",
    filename="solution_triplet.png",
    val_clim=(0.0, 1.0),
    err_vmax=0.1,  # 高梯度问题误差可能更大
    subdir=output_subdir
)

# Plot training history
plot_training_history(history, filename="training_history.png", subdir=output_subdir)

# Save training history data
trainer.save_history(os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"))

print("\nDone.")



