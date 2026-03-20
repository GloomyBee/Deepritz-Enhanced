import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.two_d.basis import (
    MeshfreeKAN2D,
    compute_patch_metrics,
    generate_square_nodes,
    grid_points,
)
from experiments.shape_validation.two_d.common import (
    build_case_name,
    ensure_run_artifacts,
    save_run_bundle,
    seed_everything,
)
from experiments.shape_validation.two_d.plotting import (
    plot_heatmap,
    plot_main_figure_patch_test,
    plot_training_curves,
)
from experiments.shape_validation.two_d.summary_plotting import (
    plot_patch_test_summary_figure,
)
from experiments.shape_validation.two_d.training import (
    resolve_variant_config,
    train_phase_a,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D meshfree KAN RKPM patch test")
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
    parser.add_argument("--phase-a-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--grid-resolution", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("current device:", device)
    variant_cfg = resolve_variant_config(args.variant)
    nodes_np, h = generate_square_nodes(n_side=args.n_side)
    support_radius = args.kappa * h
    case_name = build_case_name(
        variant=args.variant,
        n_side=args.n_side,
        kappa=args.kappa,
        seed=args.seed,
        tag=args.output_tag,
    )
    artifacts = ensure_run_artifacts("patch_test", case_name)
    nodes_t = torch.tensor(nodes_np, device=device)
    model = MeshfreeKAN2D(
        nodes=nodes_t,
        support_radius=support_radius,
        hidden_dim=args.hidden_dim,
        use_softplus=variant_cfg["use_softplus"],
        enable_fallback=variant_cfg["enable_fallback"],
    ).to(device)
    history = train_phase_a(
        model=model,
        nodes=nodes_t,
        device=device,
        variant=args.variant,
        steps=args.phase_a_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
    )

    x_grid, y_grid, x_eval = grid_points(args.grid_resolution)
    x_eval_t = torch.tensor(x_eval, device=device)
    with torch.no_grad():
        phi = model.compute_shape_functions(x_eval_t).cpu().numpy()

    patch_metrics = compute_patch_metrics(phi=phi, nodes=nodes_np, x_eval=x_eval)
    pu_field = np.sum(phi, axis=1).reshape(args.grid_resolution, args.grid_resolution) - 1.0
    linear_x_field = (phi @ nodes_np[:, 0] - x_eval[:, 0]).reshape(args.grid_resolution, args.grid_resolution)
    linear_y_field = (phi @ nodes_np[:, 1] - x_eval[:, 1]).reshape(args.grid_resolution, args.grid_resolution)
    lambda_field = np.sum(np.abs(phi), axis=1).reshape(args.grid_resolution, args.grid_resolution)

    plot_main_figure_patch_test(
        x_grid,
        y_grid,
        pu_field,
        linear_x_field,
        linear_y_field,
        lambda_field,
        history,
        artifacts.figures_dir / "main_figure.png",
    )
    plot_patch_test_summary_figure(
        x_grid,
        y_grid,
        linear_x_field,
        linear_y_field,
        nodes_np,
        patch_metrics,
        artifacts.figures_dir / "summary_figure.png",
    )
    plot_training_curves(history, artifacts.diagnostics_dir / "loss_curves.png", "Patch Test Phase A")
    plot_heatmap(x_grid, y_grid, pu_field, artifacts.diagnostics_dir / "pu_heatmap.png", "PU residual", cmap="coolwarm")
    plot_heatmap(x_grid, y_grid, linear_x_field, artifacts.diagnostics_dir / "linear_x_heatmap.png", "Linear reproduction x", cmap="coolwarm")
    plot_heatmap(x_grid, y_grid, linear_y_field, artifacts.diagnostics_dir / "linear_y_heatmap.png", "Linear reproduction y", cmap="coolwarm")
    plot_heatmap(x_grid, y_grid, lambda_field, artifacts.diagnostics_dir / "lambda_h_heatmap.png", "Lambda_h field")

    metrics = {
        "problem_name": "linear_patch_2d",
        "variant": args.variant,
        "n_side": args.n_side,
        "n_nodes": int(nodes_np.shape[0]),
        "h": float(h),
        "kappa": float(args.kappa),
        "support_radius": float(support_radius),
        "seed": args.seed,
        "phase_a_steps": args.phase_a_steps,
        **patch_metrics,
    }
    summary_lines = [f"{key}: {value}" for key, value in metrics.items()]
    save_run_bundle(
        artifacts=artifacts,
        config=vars(args),
        metrics=metrics,
        history=history,
        arrays={
            "x_grid": x_grid,
            "y_grid": y_grid,
            "x_eval": x_eval,
            "nodes": nodes_np,
            "phi": phi,
            "history_steps": np.array(history["steps"], dtype=np.float64),
            "history_loss": np.array(history["loss"], dtype=np.float64),
            "history_linear": np.array(history["linear"], dtype=np.float64),
            "history_pu": np.array(history["pu"], dtype=np.float64),
            "history_bd": np.array(history["bd"], dtype=np.float64),
            "history_teacher": np.array(history["teacher"], dtype=np.float64),
        },
        summary_lines=summary_lines,
    )

    print(f"Saved patch test artifacts to {artifacts.root_dir}")


if __name__ == "__main__":
    main()


