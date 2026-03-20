from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
BOOTSTRAP_ROOT = next(parent for parent in THIS_FILE.parents if ((parent / "requirements.txt").is_file() and (parent / "core").is_dir()))
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.append(str(BOOTSTRAP_ROOT))

import torch

from experiments.shape_validation.one_d.common import (
    build_case_name,
    resolve_repo_root,
    save_run_bundle,
    seed_everything,
)
from experiments.trial_space_value.one_d.common import ensure_trial_space_artifacts_1d
from experiments.trial_space_value.one_d.poisson_fixed_basis import (
    build_poisson_summary_lines_1d,
    plot_main_figure_poisson_1d,
    solve_fixed_basis_poisson_1d,
)
from experiments.shape_validation.one_d.basis import (
    evaluate_shape_validation_case_1d,
    generate_interval_nodes,
    history_to_arrays,
)
from experiments.shape_validation.one_d.plotting import (
    plot_diagnostic_shape_consistency_1d,
    plot_diagnostic_shape_overlay_1d,
)
from experiments.shape_validation.one_d.training import (
    build_shape_model_1d,
    train_shape_model_1d,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="1D Poisson validation with fixed learned basis")
    parser.add_argument("--n-nodes", type=int, default=11)
    parser.add_argument("--support-factor", type=float, default=2.0)
    parser.add_argument("--variant", type=str, default="teacher_distill")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=2001)
    parser.add_argument("--eval-resolution", type=int, default=401)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    _ = resolve_repo_root(THIS_FILE)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nodes, h = generate_interval_nodes(args.n_nodes)
    support_radius = args.support_factor * h
    tag = "_".join(item for item in ["poisson_fixed_basis", args.variant, args.output_tag.strip()] if item)
    case_name = build_case_name(
        n_nodes=args.n_nodes,
        support_factor=args.support_factor,
        seed=args.seed,
        tag=tag,
    )
    artifacts = ensure_trial_space_artifacts_1d("poisson_compare", case_name)

    model = build_shape_model_1d(
        nodes=nodes,
        support_radius=support_radius,
        hidden_dim=args.hidden_dim,
        variant_name=args.variant,
        device=device,
    )
    history = train_shape_model_1d(
        model=model,
        nodes=model.nodes,
        variant_name=args.variant,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
    )
    shape_result = evaluate_shape_validation_case_1d(
        model=model,
        nodes=nodes,
        support_radius=support_radius,
        n_eval=args.n_eval,
        quadrature_order=args.quadrature_order,
        device=device,
        case_meta={
            "dimension": 1,
            "layout": "uniform",
            "variant": args.variant,
            "n_nodes": args.n_nodes,
            "h": h,
            "support_factor": args.support_factor,
            "support_radius": support_radius,
            "quadrature_order": args.quadrature_order,
            "seed": args.seed,
        },
    )
    poisson_result = solve_fixed_basis_poisson_1d(
        model=model,
        quadrature_order=args.quadrature_order,
        eval_resolution=args.eval_resolution,
        device=device,
    )
    metrics = {
        "case": {
            "dimension": 1,
            "group": "poisson_compare",
            "variant": args.variant,
            "n_nodes": args.n_nodes,
            "support_factor": args.support_factor,
            "support_radius": support_radius,
            "quadrature_order": args.quadrature_order,
            "seed": args.seed,
        },
        "basis_quality": shape_result["metrics"]["learned"],
        "poisson": poisson_result["metrics"],
    }
    arrays = {
        "shape_x_eval": shape_result["arrays"]["x_eval"],
        "shape_nodes": shape_result["arrays"]["nodes"],
        "shape_phi_learned": shape_result["arrays"]["phi_learned"],
        "shape_phi_rkpm": shape_result["arrays"]["phi_rkpm"],
    }
    arrays.update(poisson_result["arrays"])
    arrays.update(history_to_arrays(history))
    summary_lines = build_poisson_summary_lines_1d(metrics["case"], metrics["poisson"])
    config = {
        "group": "poisson_compare",
        "case_name": case_name,
        "variant": args.variant,
        "n_nodes": args.n_nodes,
        "support_factor": args.support_factor,
        "support_radius": support_radius,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_eval": args.n_eval,
        "eval_resolution": args.eval_resolution,
        "quadrature_order": args.quadrature_order,
        "seed": args.seed,
        "output_tag": args.output_tag,
        "device": device,
    }
    save_run_bundle(
        artifacts=artifacts,
        config=config,
        metrics=metrics,
        history=history,
        arrays=arrays,
        summary_lines=summary_lines,
    )
    plot_main_figure_poisson_1d(
        x_eval=poisson_result["arrays"]["x_eval"],
        u_pred=poisson_result["arrays"]["u_pred"],
        u_exact=poisson_result["arrays"]["u_exact"],
        solver_iterations=poisson_result["arrays"]["solver_iterations"],
        solver_residual_norm=poisson_result["arrays"]["solver_residual_norm"],
        metrics=poisson_result["metrics"],
        path=artifacts.figures_dir / "main_figure.png",
    )
    plot_diagnostic_shape_overlay_1d(
        x_eval=shape_result["arrays"]["x_eval"],
        phi_rkpm=shape_result["arrays"]["phi_rkpm"],
        phi_kan=shape_result["arrays"]["phi_learned"],
        path=artifacts.diagnostics_dir / "shape_subset_overlay.png",
    )
    plot_diagnostic_shape_consistency_1d(
        metrics_payload=shape_result["metrics"],
        learned_consistency_bundle=shape_result["learned_consistency_bundle"],
        rkpm_consistency_bundle=shape_result["rkpm_consistency_bundle"],
        nodes=nodes,
        diagnostics_dir=artifacts.diagnostics_dir,
    )
    print(f"Saved 1D fixed-basis Poisson example to {artifacts.root_dir}")


if __name__ == "__main__":
    main()


