import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.two_d.common import (
    MeshfreeKAN2D,
    build_case_name,
    ensure_run_artifacts,
    generate_square_nodes,
    plot_heatmap,
    plot_training_curves,
    resolve_variant_config,
    save_run_bundle,
    seed_everything,
    train_phase_a,
)
from experiments.shape_validation.two_d.shape_validation import (
    build_shape_validation_summary_lines_2d,
    evaluate_shape_validation_case_2d,
    plot_diagnostic_shape_representatives_2d,
    plot_main_figure_shape_case_2d,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D uniform-node shape validation")
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
    parser.add_argument("--phase-a-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--grid-resolution", type=int, default=81)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    artifacts = ensure_run_artifacts("uniform_nodes", case_name)
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
    case_meta = {
        "dimension": 2,
        "layout": "uniform",
        "variant": args.variant,
        "n_side": args.n_side,
        "n_nodes": int(nodes_np.shape[0]),
        "h": h,
        "kappa": args.kappa,
        "support_radius": support_radius,
        "seed": args.seed,
        "quadrature_order": args.quadrature_order,
    }
    result = evaluate_shape_validation_case_2d(
        model=model,
        nodes=nodes_np,
        support_radius=support_radius,
        grid_resolution=args.grid_resolution,
        quadrature_order=args.quadrature_order,
        device=device,
        case_meta=case_meta,
    )
    arrays = dict(result["arrays"])
    arrays.update(
        {
            "history_steps": np.asarray(history.get("steps", []), dtype=np.float64),
            "history_loss": np.asarray(history.get("loss", []), dtype=np.float64),
            "history_linear": np.asarray(history.get("linear", []), dtype=np.float64),
            "history_pu": np.asarray(history.get("pu", []), dtype=np.float64),
            "history_bd": np.asarray(history.get("bd", []), dtype=np.float64),
            "history_teacher": np.asarray(history.get("teacher", []), dtype=np.float64),
        }
    )
    summary_lines = build_shape_validation_summary_lines_2d(result["metrics"])
    save_run_bundle(
        artifacts=artifacts,
        config=vars(args),
        metrics=result["metrics"],
        history=history,
        arrays=arrays,
        summary_lines=summary_lines,
    )
    plot_main_figure_shape_case_2d(
        x_eval=arrays["x_eval"],
        x_grid=arrays["x_grid"],
        y_grid=arrays["y_grid"],
        nodes=arrays["nodes"],
        phi_learned=arrays["phi_learned"],
        phi_rkpm=arrays["phi_rkpm"],
        history=history,
        metrics_payload=result["metrics"],
        shape_arrays=result["shape_arrays"],
        path=artifacts.figures_dir / "main_figure.png",
    )
    plot_diagnostic_shape_representatives_2d(
        x_grid=arrays["x_grid"],
        y_grid=arrays["y_grid"],
        phi_learned=arrays["phi_learned"],
        phi_rkpm=arrays["phi_rkpm"],
        representative_labels=result["metrics"]["learned"]["shape"]["representative_labels"],
        representative_indices=result["metrics"]["learned"]["shape"]["representative_node_indices"],
        diagnostics_dir=artifacts.diagnostics_dir,
    )
    plot_training_curves(history, artifacts.diagnostics_dir / "loss_curves.png", "Uniform node shape validation")
    plot_heatmap(arrays["x_grid"], arrays["y_grid"], result["shape_arrays"]["pu_field"], artifacts.diagnostics_dir / "pu_residual.png", "PU residual", cmap="coolwarm")
    plot_heatmap(arrays["x_grid"], arrays["y_grid"], result["shape_arrays"]["linear_x_field"], artifacts.diagnostics_dir / "linear_x_residual.png", "Linear reproduction x", cmap="coolwarm")
    plot_heatmap(arrays["x_grid"], arrays["y_grid"], result["shape_arrays"]["linear_y_field"], artifacts.diagnostics_dir / "linear_y_residual.png", "Linear reproduction y", cmap="coolwarm")
    plot_heatmap(arrays["x_grid"], arrays["y_grid"], result["shape_arrays"]["lambda_h_field"], artifacts.diagnostics_dir / "lambda_h_field.png", "Lambda_h field")
    plot_heatmap(arrays["x_grid"], arrays["y_grid"], result["shape_arrays"]["global_abs_error_field"], artifacts.diagnostics_dir / "shape_abs_error_field.png", "Global shape abs error")
    print(f"Saved 2D uniform-node shape validation to {artifacts.root_dir}")


if __name__ == "__main__":
    main()
