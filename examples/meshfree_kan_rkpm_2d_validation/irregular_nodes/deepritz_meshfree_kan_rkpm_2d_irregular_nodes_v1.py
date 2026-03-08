import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from examples.meshfree_kan_rkpm_2d_validation.common import (
    MeshfreeKAN2D,
    ROOT_DIR,
    SinusoidalPoissonProblem2D,
    build_case_name,
    compute_patch_metrics,
    ensure_run_artifacts,
    evaluate_solution_metrics,
    generate_square_nodes,
    parse_float_list,
    plot_solution_triplet,
    plot_training_curves,
    resolve_variant_config,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
    train_phase_a,
    train_phase_b_poisson,
    merge_histories,
)


def plot_jitter_curve(entries: list[dict], path: Path) -> None:
    jitters = np.array([item["jitter"] for item in entries], dtype=np.float64)
    l2 = np.array([item["l2_error"] for item in entries], dtype=np.float64)
    lam = np.array([item["lambda_h_max"] for item in entries], dtype=np.float64)
    order = np.argsort(jitters)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(jitters[order], l2[order], "o-")
    axes[0].set_title("Jitter vs L2")
    axes[0].set_xlabel("jitter")
    axes[0].set_ylabel("L2 error")
    axes[1].plot(jitters[order], lam[order], "s-")
    axes[1].set_title("Jitter vs Lambda_h")
    axes[1].set_xlabel("jitter")
    axes[1].set_ylabel("lambda_h_max")
    for axis in axes:
        axis.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D meshfree KAN RKPM irregular node experiment")
    parser.add_argument("--jitters", type=str, default="0.0,0.1,0.2")
    parser.add_argument("--n-side", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--variant", type=str, default="softplus_raw_pu_bd")
    parser.add_argument("--phase-a-steps", type=int, default=1200)
    parser.add_argument("--phase-b-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr-phase-a", type=float, default=1e-3)
    parser.add_argument("--lr-kan-b", type=float, default=1e-4)
    parser.add_argument("--lr-w", type=float, default=1e-2)
    parser.add_argument("--beta-bc", type=float, default=100.0)
    parser.add_argument("--gamma-linear", type=float, default=10.0)
    parser.add_argument("--warmup-w-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-resolution", type=int, default=41)
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
    group_root = ROOT_DIR / "output" / "meshfree_kan_rkpm_2d_validation" / "irregular_nodes"
    group_root.mkdir(parents=True, exist_ok=True)
    summary_entries: list[dict] = []

    for jitter in parse_float_list(args.jitters):
        nodes_np, h = generate_square_nodes(n_side=args.n_side, jitter=jitter, seed=args.seed)
        support_radius = args.kappa * h
        case_name = build_case_name(
            variant=args.variant,
            n_side=args.n_side,
            kappa=args.kappa,
            jitter=jitter,
            seed=args.seed,
            tag=args.output_tag,
        )
        artifacts = ensure_run_artifacts("irregular_nodes", case_name)
        nodes_t = torch.tensor(nodes_np, device=device)
        model = MeshfreeKAN2D(
            nodes=nodes_t,
            support_radius=support_radius,
            hidden_dim=args.hidden_dim,
            use_softplus=variant_cfg["use_softplus"],
            enable_fallback=variant_cfg["enable_fallback"],
        ).to(device)
        history_a = train_phase_a(
            model=model,
            nodes=nodes_t,
            device=device,
            variant=args.variant,
            steps=args.phase_a_steps,
            batch_size=args.batch_size,
            lr=args.lr_phase_a,
            log_interval=args.log_interval,
        )
        history_b = train_phase_b_poisson(
            model=model,
            problem=problem,
            device=device,
            steps=args.phase_b_steps,
            batch_size=args.batch_size,
            lr_kan=args.lr_kan_b,
            lr_w=args.lr_w,
            beta_bc=args.beta_bc,
            gamma_linear=args.gamma_linear,
            warmup_w_steps=args.warmup_w_steps,
            eval_interval=args.eval_interval,
            eval_resolution=args.eval_resolution,
            log_interval=args.log_interval,
        )
        x_eval = np.linspace(0.0, 1.0, args.grid_resolution)
        xx, yy = np.meshgrid(x_eval, x_eval, indexing="xy")
        x_eval_pts = np.column_stack([xx.ravel(), yy.ravel()])
        with torch.no_grad():
            phi_eval = model.compute_shape_functions(torch.tensor(x_eval_pts, device=device)).cpu().numpy()
        patch_metrics = compute_patch_metrics(phi_eval, nodes_np, x_eval_pts)
        solution_metrics, x_grid, y_grid, pred, exact = evaluate_solution_metrics(
            model=model,
            problem=problem,
            device=device,
            resolution=args.grid_resolution,
        )
        metrics = {
            "problem_name": "irregular_poisson_2d",
            "variant": args.variant,
            "n_side": args.n_side,
            "n_nodes": int(nodes_np.shape[0]),
            "h": float(h),
            "kappa": float(args.kappa),
            "jitter": float(jitter),
            "support_radius": float(support_radius),
            "seed": args.seed,
            "phase_a_steps": args.phase_a_steps,
            "phase_b_steps": args.phase_b_steps,
            **patch_metrics,
            **solution_metrics,
        }
        history = merge_histories(history_a, history_b)
        plot_training_curves(
            history,
            artifacts.figures_dir / "loss_curves.png",
            f"Irregular jitter={jitter}",
            phase_split_step=float(args.phase_a_steps),
        )
        plot_solution_triplet(x_grid, y_grid, pred, exact, artifacts.figures_dir / "solution_triplet.png", "Deep-RKPM-KAN")
        save_run_bundle(
            artifacts=artifacts,
            config=vars(args),
            metrics=metrics,
            history=history,
            arrays={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "pred": pred,
                "exact": exact,
                "phi": phi_eval,
                "nodes": nodes_np,
                "history_steps": np.array(history["steps"], dtype=np.float64),
                "history_loss": np.array(history["loss"], dtype=np.float64),
                "history_linear": np.array(history["linear"], dtype=np.float64),
                "history_pu": np.array(history["pu"], dtype=np.float64),
                "history_energy": np.array(history["energy"], dtype=np.float64),
                "history_bc": np.array(history["bc"], dtype=np.float64),
                "history_val_l2": np.array(history["val_l2"], dtype=np.float64),
                "history_val_h1": np.array(history["val_h1"], dtype=np.float64),
            },
            summary_lines=[f"{key}: {value}" for key, value in metrics.items()],
        )
        summary_entries.append(metrics)

    save_json(group_root / "irregular_summary.json", {"entries": summary_entries})
    save_summary(group_root / "irregular_summary.txt", [str(item) for item in summary_entries])
    if summary_entries:
        plot_jitter_curve(summary_entries, group_root / "irregular_summary.png")


if __name__ == "__main__":
    main()
