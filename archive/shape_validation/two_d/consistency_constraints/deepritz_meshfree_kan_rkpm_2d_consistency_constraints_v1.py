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
    build_consistency_metrics_payload,
    build_consistency_summary_lines,
    evaluate_model_consistency_bundle_2d,
    evaluate_rkpm_consistency_bundle_2d,
    generate_square_nodes,
    history_to_arrays,
)
from experiments.shape_validation.two_d.common import (
    ROOT_DIR,
    build_case_name,
    ensure_run_artifacts,
    parse_float_list,
    parse_int_list,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
)
from experiments.shape_validation.two_d.plotting import (
    plot_derivative_consistency_bars,
    plot_derivative_node_residuals_2d,
    plot_main_figure_consistency_2d,
    plot_training_curves,
    plot_value_consistency_bars,
)
from experiments.shape_validation.two_d.summary_plotting import (
    plot_consistency_summary,
    plot_summary_figure_consistency_2d,
)
from experiments.shape_validation.two_d.training import (
    resolve_variant_config,
    train_phase_a,
)


def prefixed_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": value for key, value in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="2D meshfree KAN RKPM consistency constraints")
    parser.add_argument("--n-sides", type=str, default="15")
    parser.add_argument("--kappas", type=str, default="2.5")
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
    parser.add_argument("--phase-a-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--grid-resolution", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_cfg = resolve_variant_config(args.variant)
    group_root = ROOT_DIR / "output" / "shape_validation" / "two_d" / "consistency_constraints"
    group_root.mkdir(parents=True, exist_ok=True)
    summary_entries: list[dict[str, object]] = []

    for n_side in parse_int_list(args.n_sides):
        for kappa in parse_float_list(args.kappas):
            nodes_np, h = generate_square_nodes(n_side=n_side)
            support_radius = kappa * h
            case_name = build_case_name(
                variant=args.variant,
                n_side=n_side,
                kappa=kappa,
                seed=args.seed,
                tag=args.output_tag,
            )
            artifacts = ensure_run_artifacts("consistency_constraints", case_name)
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
            learned_bundle = evaluate_model_consistency_bundle_2d(
                model=model,
                quadrature_order=args.quadrature_order,
                device=device,
            )
            rkpm_bundle = evaluate_rkpm_consistency_bundle_2d(
                nodes=nodes_np,
                support_radius=support_radius,
                quadrature_order=args.quadrature_order,
                device=device,
            )
            payload = build_consistency_metrics_payload(
                case={
                    "dimension": 2,
                    "variant": args.variant,
                    "n_side": n_side,
                    "n_nodes": int(nodes_np.shape[0]),
                    "h": float(h),
                    "kappa": float(kappa),
                    "support_radius": float(support_radius),
                    "quadrature_order": int(args.quadrature_order),
                    "seed": int(args.seed),
                },
                learned=learned_bundle["metrics"],
                rkpm=rkpm_bundle["metrics"],
            )
            plot_main_figure_consistency_2d(
                payload=payload,
                learned_arrays=learned_bundle["arrays"],
                rkpm_arrays=rkpm_bundle["arrays"],
                nodes=nodes_np,
                history=history,
                path=artifacts.figures_dir / "main_figure.png",
            )
            plot_summary_figure_consistency_2d(
                payload=payload,
                learned_arrays=learned_bundle["arrays"],
                rkpm_arrays=rkpm_bundle["arrays"],
                path=artifacts.figures_dir / "summary_figure.png",
            )
            plot_training_curves(history, artifacts.diagnostics_dir / "loss_curves.png", f"Consistency Phase A ns={n_side}")
            plot_value_consistency_bars(payload, artifacts.diagnostics_dir / "value_consistency.png")
            plot_derivative_consistency_bars(payload, artifacts.diagnostics_dir / "derivative_consistency.png")
            plot_derivative_node_residuals_2d(
                learned_residuals=learned_bundle["arrays"]["boundary_residuals"],
                rkpm_residuals=rkpm_bundle["arrays"]["boundary_residuals"],
                nodes=nodes_np,
                path=artifacts.diagnostics_dir / "derivative_node_residuals.png",
            )
            arrays = {}
            arrays.update(prefixed_arrays("learned", learned_bundle["arrays"]))
            arrays.update(prefixed_arrays("rkpm", rkpm_bundle["arrays"]))
            arrays.update(history_to_arrays(history))
            save_run_bundle(
                artifacts=artifacts,
                config={**vars(args), "device": device},
                metrics=payload,
                history=history,
                arrays=arrays,
                summary_lines=build_consistency_summary_lines(payload),
            )
            summary_entries.append({"case_label": case_name, "payload": payload})
            print(f"Saved 2D consistency case to {artifacts.root_dir}")

    save_json(group_root / "consistency_summary.json", {"entries": summary_entries})
    save_summary(group_root / "consistency_summary.txt", [str(item) for item in summary_entries])
    if summary_entries:
        plot_consistency_summary(summary_entries, group_root / "consistency_summary.png")


if __name__ == "__main__":
    main()



