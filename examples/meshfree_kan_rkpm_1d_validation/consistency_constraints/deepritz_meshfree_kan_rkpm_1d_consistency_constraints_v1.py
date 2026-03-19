import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_1d_validation.common import (
    MeshfreeKAN1D,
    ROOT_DIR,
    build_case_name,
    build_consistency_metrics_payload,
    build_consistency_summary_lines,
    ensure_run_artifacts,
    evaluate_model_consistency_bundle_1d,
    evaluate_rkpm_consistency_bundle_1d,
    generate_interval_nodes,
    history_to_arrays,
    parse_float_list,
    parse_int_list,
    plot_consistency_summary,
    plot_derivative_consistency_bars,
    plot_derivative_node_residuals_1d,
    plot_main_figure_consistency_1d,
    plot_training_curves,
    plot_value_consistency_bars,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
    train_phase_a_distill,
)


def prefixed_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{key}": value for key, value in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="1D meshfree KAN RKPM consistency constraints")
    parser.add_argument("--n-nodes", type=str, default="11")
    parser.add_argument("--support-factors", type=str, default="2.0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument("--lambda-bd", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    group_root = ROOT_DIR / "output" / "meshfree_kan_rkpm_1d_validation" / "consistency_constraints"
    group_root.mkdir(parents=True, exist_ok=True)
    summary_entries: list[dict[str, object]] = []

    for n_nodes in parse_int_list(args.n_nodes):
        for support_factor in parse_float_list(args.support_factors):
            nodes_np, h = generate_interval_nodes(n_nodes=n_nodes)
            support_radius = support_factor * h
            case_name = build_case_name(
                n_nodes=n_nodes,
                support_factor=support_factor,
                seed=args.seed,
                tag=args.output_tag,
            )
            artifacts = ensure_run_artifacts("consistency_constraints", case_name)
            nodes_t = torch.tensor(nodes_np, device=device)
            model = MeshfreeKAN1D(nodes_t, support_radius=support_radius, hidden_dim=args.hidden_dim).to(device)
            history = train_phase_a_distill(
                model=model,
                nodes=nodes_t,
                device=device,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                lambda_teacher=args.lambda_teacher,
                lambda_bd=args.lambda_bd,
                lambda_reg=args.lambda_reg,
                log_interval=args.log_interval,
            )
            learned_bundle = evaluate_model_consistency_bundle_1d(
                model=model,
                quadrature_order=args.quadrature_order,
                device=device,
            )
            rkpm_bundle = evaluate_rkpm_consistency_bundle_1d(
                nodes=nodes_np,
                support_radius=support_radius,
                quadrature_order=args.quadrature_order,
                device=device,
            )
            payload = build_consistency_metrics_payload(
                case={
                    "dimension": 1,
                    "n_nodes": int(n_nodes),
                    "h": float(h),
                    "support_factor": float(support_factor),
                    "support_radius": float(support_radius),
                    "quadrature_order": int(args.quadrature_order),
                    "seed": int(args.seed),
                },
                learned=learned_bundle["metrics"],
                rkpm=rkpm_bundle["metrics"],
            )
            plot_main_figure_consistency_1d(
                payload=payload,
                learned_arrays=learned_bundle["arrays"],
                rkpm_arrays=rkpm_bundle["arrays"],
                nodes=nodes_np,
                history=history,
                path=artifacts.figures_dir / "main_figure.png",
            )
            plot_training_curves(history, artifacts.diagnostics_dir / "loss_curves.png", f"1D Consistency nn={n_nodes}")
            plot_value_consistency_bars(payload, artifacts.diagnostics_dir / "value_consistency.png")
            plot_derivative_consistency_bars(payload, artifacts.diagnostics_dir / "derivative_consistency.png")
            plot_derivative_node_residuals_1d(
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
            print(f"Saved 1D consistency case to {artifacts.root_dir}")

    save_json(group_root / "consistency_summary.json", {"entries": summary_entries})
    save_summary(group_root / "consistency_summary.txt", [str(item) for item in summary_entries])
    if summary_entries:
        plot_consistency_summary(summary_entries, group_root / "consistency_summary.png")


if __name__ == "__main__":
    main()
