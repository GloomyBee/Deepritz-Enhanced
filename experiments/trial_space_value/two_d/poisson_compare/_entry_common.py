from __future__ import annotations

import argparse

import torch

from experiments.shape_validation.two_d.config import get_shape_training_variant_2d
from experiments.trial_space_value.two_d.basis import MeshfreeKAN2D, SinusoidalPoissonProblem2D, generate_square_nodes
from experiments.trial_space_value.two_d.config import build_trial_space_value_2d_config
from experiments.trial_space_value.two_d.common import (
    METHOD_LABELS,
    build_metrics_payload,
    ensure_trial_space_artifacts,
    save_run_bundle,
    seed_everything,
)
from experiments.trial_space_value.two_d.plotting import (
    plot_main_figure_trial_space_2d,
    plot_representative_shape_functions,
    plot_solution_triplet,
    plot_training_curves,
)
from experiments.trial_space_value.two_d.training import history_to_arrays, train_phase_a
from experiments.trial_space_value.two_d.trial_space import build_trial_space_summary_lines, run_trial_method_case


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
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
    return parser


def _prefix_history_arrays(prefix: str, history: dict[str, list[float]]) -> dict[str, object]:
    arrays = history_to_arrays(history)
    return {f"{prefix}_{key}": value for key, value in arrays.items()}


def run_poisson_compare_method_entry(method: str, description: str) -> None:
    parser = build_parser(description)
    args = parser.parse_args()

    cfg = build_trial_space_value_2d_config(args, method=method)
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem = SinusoidalPoissonProblem2D()
    variant_cfg = get_shape_training_variant_2d(cfg.variant)
    nodes_np, h = generate_square_nodes(n_side=cfg.n_side)
    support_radius = cfg.support_radius(h)
    case_name = cfg.case_name()
    artifacts = ensure_trial_space_artifacts("poisson_compare", case_name)
    nodes_t = torch.tensor(nodes_np, device=device)
    phase_a_model = MeshfreeKAN2D(
        nodes=nodes_t,
        support_radius=support_radius,
        hidden_dim=cfg.hidden_dim,
        use_softplus=variant_cfg.use_softplus,
        enable_fallback=variant_cfg.enable_fallback,
    ).to(device)

    phase_a_history = train_phase_a(
        model=phase_a_model,
        nodes=nodes_t,
        device=device,
        variant=cfg.variant,
        steps=cfg.phase_a_steps,
        batch_size=cfg.batch_size,
        lr=cfg.lr_phase_a,
        log_interval=cfg.log_interval,
    )
    plot_training_curves(
        phase_a_history,
        artifacts.diagnostics_dir / "phase_a_loss_curves.png",
        f"Phase A basis training ns={args.n_side}",
    )
    plot_representative_shape_functions(
        phase_a_model,
        artifacts.diagnostics_dir / "phase_a_shape_representatives.png",
        resolution=args.grid_resolution,
    )

    model, method_history, basis_quality, trial_metrics, arrays = run_trial_method_case(
        method=method,
        phase_a_model=phase_a_model,
        problem=problem,
        device=device,
        phase_b_steps=cfg.phase_b_steps,
        batch_size=cfg.batch_size,
        lr_kan_b=cfg.lr_kan_b,
        lr_w=cfg.lr_w,
        beta_bc=cfg.beta_bc,
        gamma_linear=cfg.gamma_linear,
        warmup_w_steps=cfg.warmup_w_steps,
        eval_interval=cfg.eval_interval,
        eval_resolution=cfg.eval_resolution,
        log_interval=cfg.log_interval,
        domain_order=cfg.quadrature_order,
        boundary_order=cfg.quadrature_order,
        grid_resolution=cfg.grid_resolution,
    )
    metrics = build_metrics_payload(
        case=cfg.build_case_payload(
            h=float(h),
            support_radius=float(support_radius),
            n_nodes=int(nodes_np.shape[0]),
        ),
        basis_quality=basis_quality,
        trial_space=trial_metrics,
        method=cfg.method,
    )
    summary_lines = build_trial_space_summary_lines(metrics)
    config = cfg.to_config_payload(
        device=device,
        case_name=case_name,
        support_radius=support_radius,
    )
    run_arrays = {
        **arrays,
        **_prefix_history_arrays("phase_a", phase_a_history),
    }
    save_run_bundle(
        artifacts=artifacts,
        config=config,
        metrics=metrics,
        history=method_history,
        arrays=run_arrays,
        summary_lines=summary_lines,
    )
    plot_main_figure_trial_space_2d(
        x_grid=arrays["x_grid"],
        y_grid=arrays["y_grid"],
        pred=arrays["pred"],
        exact=arrays["exact"],
        history=method_history,
        metrics=trial_metrics,
        title_prefix=METHOD_LABELS[cfg.method],
        path=artifacts.figures_dir / "main_figure.png",
    )
    plot_solution_triplet(
        arrays["x_grid"],
        arrays["y_grid"],
        arrays["pred"],
        arrays["exact"],
        artifacts.diagnostics_dir / "solution_triplet.png",
        METHOD_LABELS[cfg.method],
    )
    plot_representative_shape_functions(
        model,
        artifacts.diagnostics_dir / "final_shape_representatives.png",
        resolution=args.grid_resolution,
    )
    if method_history.get("steps"):
        plot_training_curves(
            method_history,
            artifacts.diagnostics_dir / "loss_curves.png",
            METHOD_LABELS[cfg.method],
        )
    print(f"Saved {cfg.method} results to {artifacts.root_dir}")
