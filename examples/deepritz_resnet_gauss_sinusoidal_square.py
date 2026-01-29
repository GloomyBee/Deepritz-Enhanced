"""
Example 2: Sinusoidal solution on a square domain.
Equation: -Δu = f in [0,1]x[0,1], u=0 on boundary
Exact: u = sin(pi x) sin(pi y)
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from problems import SinusoidalSquare
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
print("\nPlotting...")
plot_training_history(history, filename="sinusoidal_square_history.png")

print("\nDone.")
