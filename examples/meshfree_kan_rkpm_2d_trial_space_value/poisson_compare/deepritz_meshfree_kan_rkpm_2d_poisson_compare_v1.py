import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_2d_trial_space_value.common import (
    METHOD_LABELS,
    MeshfreeKAN2D,
    SinusoidalPoissonProblem2D,
    build_case_name,
    ensure_method_artifacts,
    ensure_trial_space_artifacts,
    generate_square_nodes,
    history_to_arrays,
    plot_conditioning_bars,
    plot_main_figure_trial_method,
    plot_metric_bars,
    plot_representative_shape_functions,
    plot_solution_triplet,
    plot_training_curves,
    resolve_variant_config,
    run_trial_method,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
    train_phase_a,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D fixed-basis Poisson trial-space comparison")
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
    parser.add_argument("--methods", type=str, default="classical,frozen_w,joint")
    parser.add_argument("--phase-a-steps", type=int, default=1200)
    parser.add_argument("--phase-b-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr-phase-a", type=float, default=1e-3)
    parser.add_argument("--lr-kan-b", type=float, default=1e-4)
    parser.add_argument("--lr-w", type=float, default=1e-2)
    parser.add_argument("--beta-bc", type=float, default=100.0)
    parser.add_argument("--gamma-linear", type=float, default=10.0)
    parser.add_argument("--warmup-w-steps", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-resolution", type=int, default=41)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--grid-resolution", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = SinusoidalPoissonProblem2D()
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
    artifacts = ensure_trial_space_artifacts("poisson_compare", case_name)
    nodes_t = torch.tensor(nodes_np, device=device)
    phase_a_model = MeshfreeKAN2D(
        nodes=nodes_t,
        support_radius=support_radius,
        hidden_dim=args.hidden_dim,
        use_softplus=variant_cfg["use_softplus"],
        enable_fallback=variant_cfg["enable_fallback"],
    ).to(device)

    history_a = train_phase_a(
        model=phase_a_model,
        nodes=nodes_t,
        device=device,
        variant=args.variant,
        steps=args.phase_a_steps,
        batch_size=args.batch_size,
        lr=args.lr_phase_a,
        log_interval=args.log_interval,
    )
    plot_training_curves(
        history_a,
        artifacts.diagnostics_dir / "phase_a_loss_curves.png",
        f"Phase A basis training ns={args.n_side}",
    )
    plot_representative_shape_functions(
        phase_a_model,
        artifacts.diagnostics_dir / "phase_a_shape_representatives.png",
        resolution=args.grid_resolution,
    )

    entries: list[dict[str, object]] = []
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in methods:
        model, history, metrics, x_grid, y_grid, pred, exact = run_trial_method(
            method=method,
            phase_a_model=phase_a_model,
            problem=problem,
            device=device,
            phase_b_steps=args.phase_b_steps,
            batch_size=args.batch_size,
            lr_kan_b=args.lr_kan_b,
            lr_w=args.lr_w,
            beta_bc=args.beta_bc,
            gamma_linear=args.gamma_linear,
            warmup_w_steps=args.warmup_w_steps,
            eval_interval=args.eval_interval,
            eval_resolution=args.eval_resolution,
            log_interval=args.log_interval,
            domain_order=args.quadrature_order,
            boundary_order=args.quadrature_order,
            grid_resolution=args.grid_resolution,
        )
        method_artifacts = ensure_method_artifacts(artifacts, method)
        plot_main_figure_trial_method(
            x_grid=x_grid,
            y_grid=y_grid,
            pred=pred,
            exact=exact,
            history=history,
            metrics=metrics,
            title_prefix=METHOD_LABELS[method],
            path=method_artifacts.figures_dir / "main_figure.png",
        )
        plot_solution_triplet(
            x_grid,
            y_grid,
            pred,
            exact,
            method_artifacts.diagnostics_dir / "solution_triplet.png",
            METHOD_LABELS[method],
        )
        if history.get("steps"):
            plot_training_curves(
                history,
                method_artifacts.diagnostics_dir / "loss_curves.png",
                METHOD_LABELS[method],
            )
        entry = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "variant": args.variant,
            "n_side": args.n_side,
            "n_nodes": int(nodes_np.shape[0]),
            "h": float(h),
            "kappa": float(args.kappa),
            "support_radius": float(support_radius),
            "seed": args.seed,
            "phase_a_steps": args.phase_a_steps,
            "phase_b_steps": args.phase_b_steps,
            **metrics,
        }
        entries.append(entry)
        summary_lines = [f"{key}: {value}" for key, value in entry.items()]
        save_run_bundle(
            artifacts=method_artifacts,
            config={
                **vars(args),
                "method": method,
                "device": device,
            },
            metrics=entry,
            history=history,
            arrays={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "pred": pred,
                "exact": exact,
                **history_to_arrays(history),
            },
            summary_lines=summary_lines,
        )
        print(f"Saved {method} results to {method_artifacts.root_dir}")

    comparison_payload = {
        "case_name": case_name,
        "variant": args.variant,
        "entries": entries,
    }
    save_json(artifacts.root_dir / "config.json", vars(args))
    save_json(artifacts.root_dir / "comparison_metrics.json", comparison_payload)
    save_summary(
        artifacts.root_dir / "comparison_summary.txt",
        [str(item) for item in entries],
    )
    plot_metric_bars(entries, artifacts.figures_dir / "main_figure.png")
    plot_conditioning_bars(entries, artifacts.figures_dir / "summary_figure.png")


if __name__ == "__main__":
    main()
