"""
Example 1: Deep Ritz for Poisson on a disk domain.
Equation: -Δu = f in Ω = {x in R^2: |x| < 1}
Exact: u = sin(pi x) sin(pi y)
"""

import sys
sys.path.append("..")

import torch
import numpy as np
from problems import PoissonDisk
from networks import RitzNet
from integrators import MonteCarloIntegrator
from trainers import DeepRitzTrainer
from visualizers import plot_training_history

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1) Problem
problem = PoissonDisk(radius=1.0)
print(f"Problem: {problem.name}")

# 2) Model
model = RitzNet(
    input_dim=2,
    hidden_dims=[100, 100, 100],
    output_dim=1
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3) Integrator (sample on bounding square, mask handled in problem.domain_indicator)
integrator = MonteCarloIntegrator(
    n_points=8000,
    domain_bounds=(-1, 1, -1, 1)
)

# 4) Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5) Trainer
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

print("\nPlotting...")
plot_training_history(history, filename="poisson_disk_history.png")

print("\nDone.")
