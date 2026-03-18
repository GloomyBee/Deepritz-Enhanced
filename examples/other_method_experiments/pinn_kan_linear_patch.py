"""
Example: KAN PINN patch test (u = x + y) on [0,1]^2.
Strong form PINN with KANSplineNet.
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
import torch
from pathlib import Path

from problems import LinearPatchProblem
from samplers import SquareSampler
from networks import KANSplineNet
from trainers import PINNTrainer
from visualizers import plot_square_triplet, plot_training_history, get_example_output_subdir
import visualizers


# ============================================================================
# 0. Project initialization (seed, device, paths)
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Output directory management - use absolute path
ROOT = Path(__file__).resolve().parents[2]
visualizers.OUTPUT_DIR = str(ROOT / "output" / "other_method_experiments")
output_subdir = get_example_output_subdir(__file__)
print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/")

# ============================================================================
# 1. Problem definition
# ============================================================================
problem = LinearPatchProblem()
print(f"Problem: {problem.name}")
print("Domain: [0,1] x [0,1]")
print("Exact: u = x + y")
print("Target error: < 1e-8")

# ============================================================================
# 2. Sampler
# ============================================================================
sampler = SquareSampler(x_range=(0, 1), y_range=(0, 1))

# ============================================================================
# 3. Model
# ============================================================================
model = KANSplineNet(input_dim=2, hidden_dim=8, output_dim=1, num=5, k=3, grid_range=(-0.2, 1.2)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 4. Optimizer
# ============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ============================================================================
# 5. Trainer
# ============================================================================
trainer = PINNTrainer(
    model=model,
    problem=problem,
    sampler=sampler,
    optimizer=optimizer,
    device=device,
    beta_bc=100.0
)

print("\nTraining...")
history = trainer.train(
    n_steps=2000,
    batch_domain=1000,
    batch_boundary=400,
    log_interval=200,
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
    pred = model(pts).cpu().numpy().reshape(X.shape)
    exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

# Plot triplet
plot_square_triplet(
    X, Y, pred, exact,
    title_prefix="KAN PINN (Patch Test)",
    val_clim=(0.0, 2.0),
    err_vmax=0.01,
    filename="solution_triplet.png",
    subdir=output_subdir
)

# Plot training history
plot_training_history(history, filename="training_history.png", subdir=output_subdir)

# Save training history data
trainer.save_history(os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"))

print("\nDone.")
