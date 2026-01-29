"""
Example: KAN PINN patch test (u = x + y) on [0,1]^2.
Strong form PINN with KANSplineNet.
"""

import sys
sys.path.append("..")

import numpy as np
import torch

from problems import LinearPatchProblem
from samplers import SquareSampler
from networks import KANSplineNet
from trainers import PINNTrainer
from visualizers import plot_square_triplet, plot_error_history


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    problem = LinearPatchProblem()
    sampler = SquareSampler(x_range=(0, 1), y_range=(0, 1))
    model = KANSplineNet(input_dim=2, hidden_dim=8, output_dim=1, num=5, k=3, grid_range=(-0.2, 1.2)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = PINNTrainer(
        model=model,
        problem=problem,
        sampler=sampler,
        optimizer=optimizer,
        device=device,
        beta_bc=100.0
    )

    print("Training...")
    trainer.train(
        n_steps=2000,
        batch_domain=1000,
        batch_boundary=400,
        log_interval=200,
        eval_interval=200,
        n_eval_points=2000
    )

    history = trainer.get_history()

    # Visualization
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(pts).cpu().numpy().reshape(X.shape)
        exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

    plot_square_triplet(
        X, Y, pred, exact,
        title_prefix="KAN PINN (Patch Test)",
        val_clim=(0.0, 2.0),
        err_vmax=0.01,
        filename="kan_pinn_patch_field.png"
    )

    if len(history["steps"]) > 0:
        plot_error_history(
            np.array(history["steps"]),
            np.array(history["l2_error"]),
            np.array(history["h1_error"]),
            title="KAN PINN Patch Test Errors",
            filename="kan_pinn_patch_errors.png",
            ylim=(1e-6, 1.0)
        )


if __name__ == "__main__":
    main()
