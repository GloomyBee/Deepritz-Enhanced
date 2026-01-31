"""
Example: Serial KAN PINN for sinusoidal square problem.
Coarse + fine KAN decomposition with two-phase training.
"""

import sys
sys.path.append('..')

import os
import torch
import numpy as np
from pathlib import Path

from problems import SinusoidalSquare
from samplers import SquareSampler
from networks import KANSerialPINN
from trainers import SerialKANTrainer
from visualizers import plot_square_triplet, plot_training_history, get_example_output_subdir
import visualizers


# ============================================================================
# 0. Project initialization (seed, device, paths)
# ============================================================================
torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Output directory management - use absolute path
ROOT = Path(__file__).resolve().parents[1]
visualizers.OUTPUT_DIR = str(ROOT / "output")
output_subdir = get_example_output_subdir(__file__)
print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/")

# ============================================================================
# 1. Problem definition
# ============================================================================
problem = SinusoidalSquare()
print(f"Problem: {problem.name}")
print("Domain: [0,1] x [0,1]")
print("Exact: u = sin(pi x) sin(pi y)")

# ============================================================================
# 2. Sampler
# ============================================================================
sampler = SquareSampler(x_range=(0, 1), y_range=(0, 1))

# ============================================================================
# 3. Model
# ============================================================================
model = KANSerialPINN().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 4. Optimizers
# ============================================================================
optimizer_coarse = torch.optim.Adam(
    list(model.kan_coarse.parameters()) + list(model.out_coarse.parameters()),
    lr=1e-2
)
optimizer_fine = torch.optim.Adam(
    list(model.kan_fine.parameters()) + list(model.out_fine.parameters()),
    lr=1e-3
)
scheduler_fine = torch.optim.lr_scheduler.ExponentialLR(optimizer_fine, gamma=0.9)

# ============================================================================
# 5. Trainer
# ============================================================================
trainer = SerialKANTrainer(
    model=model,
    problem=problem,
    sampler=sampler,
    optimizer_coarse=optimizer_coarse,
    optimizer_fine=optimizer_fine,
    device=device,
    beta_bc=100.0,
    lambda_fine_reg=0.0,
    lambda_fine_bc=1.0
)

print("\nTraining...")
history = trainer.train(
    pretrain_steps=500,
    phase1_steps=1000,
    phase2_steps=4000,
    batch_domain=2000,
    batch_boundary=500,
    scheduler_fine=scheduler_fine,
    scheduler_interval=1000,
    log_interval=100,
    eval_interval=200,
    n_eval_points=2000
)

# ============================================================================
# 6. Visualization
# ============================================================================
print("\nGenerating visualizations...")

# Generate test grid
res = 200
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
X, Y = np.meshgrid(x, y)
pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

# Prediction and exact solution
model.eval()
with torch.no_grad():
    pred = model.total(pts).cpu().numpy().reshape(X.shape)
    exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

# Plot triplet
plot_square_triplet(
    X, Y, pred, exact,
    title_prefix="KAN Serial PINN",
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
