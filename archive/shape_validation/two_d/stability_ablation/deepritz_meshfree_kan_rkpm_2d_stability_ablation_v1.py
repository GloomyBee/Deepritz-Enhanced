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
    ROOT_DIR,
    build_case_name,
    compute_patch_metrics,
    ensure_run_artifacts,
    grid_points,
    generate_square_nodes,
    plot_heatmap,
    plot_main_figure_stability,
    plot_stability_summary_2d,
    plot_training_curves,
    resolve_variant_config,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
    train_phase_a,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D meshfree KAN RKPM stability ablation")
    parser.add_argument(
        "--variants",
        type=str,
        default="softplus_raw_pu_bd,no_softplus_raw_pu_bd,no_softplus_teacher,no_softplus_teacher_reg",
    )
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
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
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    nodes_np, h = generate_square_nodes(n_side=args.n_side)
    support_radius = args.kappa * h
    x_grid, y_grid, x_eval = grid_points(args.grid_resolution)
    group_root = ROOT_DIR / "output" / "shape_validation" / "two_d" / "stability_ablation"
    group_root.mkdir(parents=True, exist_ok=True)
    summary_entries: list[dict] = []

    for variant in variants:
        variant_cfg = resolve_variant_config(variant)
        case_name = build_case_name(
            variant=variant,
            n_side=args.n_side,
            kappa=args.kappa,
            seed=args.seed,
            tag=args.output_tag,
        )
        artifacts = ensure_run_artifacts("stability_ablation", case_name)
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
            variant=variant,
            steps=args.phase_a_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            log_interval=args.log_interval,
        )
        with torch.no_grad():
            phi = model.compute_shape_functions(torch.tensor(x_eval, device=device)).cpu().numpy()
        metrics = {
            "problem_name": "stability_ablation_2d",
            "variant": variant,
            "n_side": args.n_side,
            "n_nodes": int(nodes_np.shape[0]),
            "h": float(h),
            "kappa": float(args.kappa),
            "support_radius": float(support_radius),
            "seed": args.seed,
            "phase_a_steps": args.phase_a_steps,
            **compute_patch_metrics(phi, nodes_np, x_eval),
        }
        lambda_field = np.sum(np.abs(phi), axis=1).reshape(args.grid_resolution, args.grid_resolution)
        plot_main_figure_stability(
            x_grid=x_grid,
            y_grid=y_grid,
            lambda_field=lambda_field,
            nodes=nodes_np,
            metrics=metrics,
            history=history,
            path=artifacts.figures_dir / "main_figure.png",
        )
        plot_training_curves(history, artifacts.diagnostics_dir / "loss_curves.png", f"Ablation {variant}")
        plot_heatmap(x_grid, y_grid, lambda_field, artifacts.diagnostics_dir / "lambda_h_heatmap.png", f"Lambda_h {variant}")
        save_run_bundle(
            artifacts=artifacts,
            config=vars(args),
            metrics=metrics,
            history=history,
            arrays={
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
            summary_lines=[f"{key}: {value}" for key, value in metrics.items()],
        )
        summary_entries.append(metrics)

    save_json(group_root / "stability_summary.json", {"entries": summary_entries})
    save_summary(group_root / "stability_summary.txt", [str(item) for item in summary_entries])
    if summary_entries:
        plot_stability_summary_2d(summary_entries, group_root / "stability_summary.png")


if __name__ == "__main__":
    main()



