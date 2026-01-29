"""
Example 4: Gaussian peak (high gradient) on a square domain.
Equation: -Δu = f in [0,1]x[0,1]
Exact: u = exp(-r^2/alpha^2), r^2 = (x-0.5)^2 + (y-0.5)^2
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from problems import GaussianPeak2D
from networks import RitzNet
from integrators import GaussIntegrator
from trainers import DeepRitzTrainer
from visualizers import plot_training_history

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================================
# 1. 定义问题
# ============================================================================
alpha = 0.1
problem = GaussianPeak2D(alpha=alpha, xc=0.5, yc=0.5)
print(f"Problem: {problem.name}")
print(f"Alpha: {alpha}")
print("Center: (0.5, 0.5)")

# 2) Model
model = RitzNet(
    input_dim=2,
    hidden_dims=[256, 256, 256, 256, 256, 256],
    output_dim=1
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3) Integrator
integrator = GaussIntegrator(nx=24, ny=24, domain_bounds=(0, 1, 0, 1), order=4)

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
print("Note: high gradient problem may take longer...")
history = trainer.train(n_steps=15000, log_interval=500, eval_interval=500)

# ============================================================================
# 6. 可视化
# ============================================================================
print("\nPlotting...")
plot_training_history(history, filename="gaussian_peak_history.png")

print("\nDone.")
