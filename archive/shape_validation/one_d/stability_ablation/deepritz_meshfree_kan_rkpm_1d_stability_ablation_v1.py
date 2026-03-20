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
    ROOT_DIR as COMMON_ROOT,
    build_case_name,
    ensure_run_artifacts,
    resolve_repo_root,
    save_json,
    save_run_bundle,
    save_summary,
    seed_everything,
)
from experiments.shape_validation.one_d.summary_plotting import (
    build_ablation_summary_lines_1d,
    build_ablation_summary_payload_1d,
    plot_ablation_summary_1d,
)
from experiments.shape_validation.one_d.basis import (
    build_shape_validation_summary_lines_1d,
    evaluate_shape_validation_case_1d,
    generate_interval_nodes,
    history_to_arrays,
)
from experiments.shape_validation.one_d.plotting import (
    plot_diagnostic_shape_consistency_1d,
    plot_diagnostic_shape_overlay_1d,
    plot_main_figure_shape_case_1d,
)
from experiments.shape_validation.one_d.training import (
    build_shape_model_1d,
    train_shape_model_1d,
)


def parse_variant_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="1D ablation sweep for shape validation")
    parser.add_argument(
        "--variants",
        type=str,
        default="raw_pu,raw_pu_fix,boundary_anchor,no_softplus,teacher_distill",
    )
    parser.add_argument("--n-nodes", type=int, default=11)
    parser.add_argument("--support-factor", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=2001)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    _ = resolve_repo_root(THIS_FILE)
    variants = parse_variant_list(args.variants)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nodes, h = generate_interval_nodes(args.n_nodes)
    support_radius = args.support_factor * h
    entries: list[dict[str, object]] = []
    group_root = COMMON_ROOT / "output" / "shape_validation" / "one_d" / "stability_ablation"
    group_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        seed_everything(args.seed)
        model = build_shape_model_1d(
            nodes=nodes,
            support_radius=support_radius,
            hidden_dim=args.hidden_dim,
            variant_name=variant,
            device=device,
        )
        history = train_shape_model_1d(
            model=model,
            nodes=model.nodes,
            variant_name=variant,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            log_interval=args.log_interval,
        )
        case_meta = {
            "dimension": 1,
            "layout": "uniform",
            "variant": variant,
            "n_nodes": args.n_nodes,
            "h": h,
            "support_factor": args.support_factor,
            "support_radius": support_radius,
            "quadrature_order": args.quadrature_order,
            "seed": args.seed,
        }
        result = evaluate_shape_validation_case_1d(
            model=model,
            nodes=nodes,
            support_radius=support_radius,
            n_eval=args.n_eval,
            quadrature_order=args.quadrature_order,
            device=device,
            case_meta=case_meta,
        )
        tag = "_".join(item for item in ["ablation", variant, args.output_tag.strip()] if item)
        case_name = build_case_name(
            n_nodes=args.n_nodes,
            support_factor=args.support_factor,
            seed=args.seed,
            tag=tag,
        )
        artifacts = ensure_run_artifacts("stability_ablation", case_name)
        arrays = dict(result["arrays"])
        arrays.update(history_to_arrays(history))
        summary_lines = build_shape_validation_summary_lines_1d(result["metrics"])
        config = {
            "group": "ablations",
            "case_name": case_name,
            "variant": variant,
            "n_nodes": args.n_nodes,
            "support_factor": args.support_factor,
            "support_radius": support_radius,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "n_eval": args.n_eval,
            "quadrature_order": args.quadrature_order,
            "seed": args.seed,
            "output_tag": args.output_tag,
            "device": device,
        }
        save_run_bundle(
            artifacts=artifacts,
            config=config,
            metrics=result["metrics"],
            history=history,
            arrays=arrays,
            summary_lines=summary_lines,
        )
        plot_main_figure_shape_case_1d(
            x_eval=arrays["x_eval"],
            nodes=arrays["nodes"],
            phi_learned=arrays["phi_learned"],
            phi_rkpm=arrays["phi_rkpm"],
            history=history,
            metrics_payload=result["metrics"],
            path=artifacts.figures_dir / "main_figure.png",
        )
        plot_diagnostic_shape_overlay_1d(
            x_eval=arrays["x_eval"],
            phi_rkpm=arrays["phi_rkpm"],
            phi_kan=arrays["phi_learned"],
            path=artifacts.diagnostics_dir / "shape_subset_overlay.png",
        )
        plot_diagnostic_shape_consistency_1d(
            metrics_payload=result["metrics"],
            learned_consistency_bundle=result["learned_consistency_bundle"],
            rkpm_consistency_bundle=result["rkpm_consistency_bundle"],
            nodes=nodes,
            diagnostics_dir=artifacts.diagnostics_dir,
        )
        entries.append({"variant": variant, "case_name": case_name, "metrics": result["metrics"]})

    plot_ablation_summary_1d(entries, group_root / "summary_figure.png")
    save_json(
        group_root / "ablation_summary.json",
        build_ablation_summary_payload_1d(
            variants=variants,
            n_nodes=args.n_nodes,
            support_factor=args.support_factor,
            seed=args.seed,
            entries=entries,
        ),
    )
    save_summary(
        group_root / "ablation_summary.txt",
        [
            f"variants: {','.join(variants)}",
            f"n_nodes: {args.n_nodes}",
            f"support_factor: {args.support_factor}",
            f"seed: {args.seed}",
        ]
        + build_ablation_summary_lines_1d(entries),
    )
    print(f"Saved 1D ablations to {group_root}")


if __name__ == "__main__":
    main()



