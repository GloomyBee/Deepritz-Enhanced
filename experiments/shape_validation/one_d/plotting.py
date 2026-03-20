from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.shape_validation.one_d.common import (
    AUX_COLORS,
    DIAGNOSTIC_FIGURE_SIZE,
    FIGURE_DPI,
    LEARNED_COLOR,
    LINESTYLE_LEARNED,
    LINESTYLE_RKPM,
    MAIN_FIGURE_SIZE,
    RKPM_COLOR,
    SUMMARY_FIGURE_SIZE,
)
from experiments.shape_validation.one_d.basis import build_consistency_metrics_payload


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_line_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, log_scale: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def _plot_history_series(ax: plt.Axes, history: dict[str, list[float]]) -> None:
    steps = np.array(history.get("steps", []), dtype=np.float64)
    if steps.size == 0:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        _style_line_axis(ax, "Training History", "step", "loss")
        return
    for key, color in zip(["loss", "teacher", "linear", "pu", "bd", "reg"], AUX_COLORS + ["#666666"]):
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0 or np.allclose(values, 0.0):
            continue
        count = min(values.size, steps.size)
        ax.semilogy(steps[:count], np.maximum(values[:count], 1e-16), lw=1.8, color=color, label=key)
    _style_line_axis(ax, "Training History", "step", "loss")
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)


def _plot_metric_point_panel(
    ax: plt.Axes,
    labels: list[str],
    learned: np.ndarray,
    rkpm: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    x_pos = np.arange(len(labels), dtype=np.float64)
    ax.plot(
        x_pos,
        np.maximum(learned, 1e-16),
        f"o{LINESTYLE_LEARNED}",
        lw=1.8,
        ms=6,
        color=LEARNED_COLOR,
        label="learned",
    )
    ax.plot(
        x_pos,
        np.maximum(rkpm, 1e-16),
        f"s{LINESTYLE_RKPM}",
        lw=1.8,
        ms=6,
        color=RKPM_COLOR,
        label="rkpm",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    _style_line_axis(ax, title, "metric", ylabel, log_scale=True)
    ax.legend(fontsize=8)


def plot_value_consistency_bars(payload: dict[str, Any], path: Path) -> None:
    metric_keys = ["mass_sum_residual", "moment_x_residual", "moment_matrix_residual_fro"]
    labels = ["mass", "mx", "moment_fro"]
    learned = np.array([payload["learned"][key] for key in metric_keys], dtype=np.float64)
    rkpm = np.array([payload["rkpm"][key] for key in metric_keys], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
    _plot_metric_point_panel(ax, labels, learned, rkpm, "Value consistency", "residual")
    _save_figure(fig, path)


def plot_derivative_consistency_bars(payload: dict[str, Any], path: Path) -> None:
    metric_keys = [
        "derivative_boundary_residual_max",
        "derivative_boundary_residual_fro",
        "derivative_matrix_residual_max",
        "derivative_matrix_residual_fro",
    ]
    labels = ["bd-max", "bd-fro", "mat-max", "mat-fro"]
    learned = np.array([payload["learned"][key] for key in metric_keys], dtype=np.float64)
    rkpm = np.array([payload["rkpm"][key] for key in metric_keys], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
    _plot_metric_point_panel(ax, labels, learned, rkpm, "Derivative consistency", "residual")
    _save_figure(fig, path)


def plot_derivative_node_residuals_1d(
    learned_residuals: np.ndarray,
    rkpm_residuals: np.ndarray,
    nodes: np.ndarray,
    path: Path,
) -> None:
    learned_node = np.max(np.abs(learned_residuals), axis=(1, 2))
    rkpm_node = np.max(np.abs(rkpm_residuals), axis=(1, 2))
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.5))
    ax.semilogy(nodes, np.maximum(learned_node, 1e-16), f"o{LINESTYLE_LEARNED}", lw=1.8, color=LEARNED_COLOR, label="learned")
    ax.semilogy(nodes, np.maximum(rkpm_node, 1e-16), f"s{LINESTYLE_RKPM}", lw=1.8, color=RKPM_COLOR, label="rkpm")
    _style_line_axis(ax, "Nodewise derivative boundary residual", "node x", "residual")
    ax.legend(fontsize=8)
    _save_figure(fig, path)


def plot_main_figure_consistency_1d(
    payload: dict[str, Any],
    learned_arrays: dict[str, np.ndarray],
    rkpm_arrays: dict[str, np.ndarray],
    nodes: np.ndarray,
    history: dict[str, list[float]],
    path: Path,
) -> None:
    x_eval = np.asarray(learned_arrays["domain_points"], dtype=np.float64).reshape(-1)
    order = np.argsort(x_eval)
    x_eval = x_eval[order]
    learned_phi = np.asarray(learned_arrays["phi"], dtype=np.float64)[order]
    rkpm_phi = np.asarray(rkpm_arrays["phi"], dtype=np.float64)[order]
    diff_phi = np.abs(learned_phi - rkpm_phi)
    rep_indices = sorted(set([0, len(nodes) // 2, len(nodes) - 1]))

    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    for index in rep_indices:
        axes[0, 0].plot(x_eval, rkpm_phi[:, index], lw=1.4, ls=LINESTYLE_RKPM, color=RKPM_COLOR, alpha=0.85)
        axes[0, 0].plot(x_eval, learned_phi[:, index], lw=1.8, ls=LINESTYLE_LEARNED, color=LEARNED_COLOR, alpha=0.85)
    axes[0, 0].scatter(nodes[rep_indices], np.zeros(len(rep_indices)), s=28, color="black", zorder=3)
    _style_line_axis(axes[0, 0], "Representative shape functions", "x", "phi")
    axes[0, 0].text(0.02, 0.03, "solid: learned, dashed: RKPM", transform=axes[0, 0].transAxes, fontsize=9)

    max_error = np.max(diff_phi, axis=1)
    mean_error = np.mean(diff_phi, axis=1)
    axes[0, 1].semilogy(x_eval, np.maximum(max_error, 1e-16), color=LEARNED_COLOR, lw=1.8, label="max |phi_l-r|")
    axes[0, 1].semilogy(x_eval, np.maximum(mean_error, 1e-16), color=AUX_COLORS[2], lw=1.8, label="mean |phi_l-r|")
    _style_line_axis(axes[0, 1], "Shape error curves", "x", "error")
    axes[0, 1].legend(fontsize=8)

    value_keys = ["mass_sum_residual", "moment_x_residual", "moment_matrix_residual_fro"]
    value_labels = ["mass", "mx", "moment_fro"]
    _plot_metric_point_panel(
        axes[1, 0],
        value_labels,
        np.array([payload["learned"][key] for key in value_keys], dtype=np.float64),
        np.array([payload["rkpm"][key] for key in value_keys], dtype=np.float64),
        "Consistency indicators",
        "residual",
    )
    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)


def ensure_legacy_figure_artifacts_1d(out_dir: Path) -> tuple[Path, Path]:
    figures_dir = Path(out_dir) / "figures"
    diagnostics_dir = figures_dir / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, diagnostics_dir


def _select_shape_validation_metric_items(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    metric_items: list[tuple[str, float]] = [
        ("global_l2", float(metrics["global_l2"])),
        ("center_l2", float(metrics["center_l2"])),
        ("boundary_l2", float(metrics["boundary_l2"])),
    ]
    if "pu_windowed_sum_rmse" in metrics:
        metric_items.append(("pu_windowed_sum_rmse", float(metrics["pu_windowed_sum_rmse"])))
    elif "pu_max_error" in metrics:
        metric_items.append(("pu_max_error", float(metrics["pu_max_error"])))
    if "linear_reproduction_rmse" in metrics:
        metric_items.append(("linear_reproduction_rmse", float(metrics["linear_reproduction_rmse"])))
    return metric_items


def plot_main_figure_shape_validation_1d(
    x_eval: np.ndarray,
    phi_rkpm: np.ndarray,
    phi_kan: np.ndarray,
    history: dict[str, list[float]],
    center_idx: int,
    boundary_idx: int,
    metrics: dict[str, Any],
    path: Path,
    history_keys: list[str] | None = None,
) -> None:
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    phi_rkpm = np.asarray(phi_rkpm, dtype=np.float64)
    phi_kan = np.asarray(phi_kan, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)

    ax = axes[0, 0]
    ax.plot(x_eval, phi_rkpm[:, center_idx], color=RKPM_COLOR, lw=1.5, ls=LINESTYLE_RKPM, label=f"RKPM center ({center_idx})")
    ax.plot(x_eval, phi_kan[:, center_idx], color=LEARNED_COLOR, lw=1.8, ls=LINESTYLE_LEARNED, label=f"Learned center ({center_idx})")
    ax.plot(x_eval, phi_rkpm[:, boundary_idx], color=AUX_COLORS[3], lw=1.5, ls=LINESTYLE_RKPM, label=f"RKPM boundary ({boundary_idx})")
    ax.plot(x_eval, phi_kan[:, boundary_idx], color=AUX_COLORS[2], lw=1.8, ls=LINESTYLE_LEARNED, label=f"Learned boundary ({boundary_idx})")
    _style_line_axis(ax, "Representative shape functions", "x", "phi")
    ax.legend(fontsize=8)

    err_center = np.abs(phi_kan[:, center_idx] - phi_rkpm[:, center_idx])
    err_boundary = np.abs(phi_kan[:, boundary_idx] - phi_rkpm[:, boundary_idx])
    ax = axes[0, 1]
    ax.semilogy(x_eval, np.maximum(err_center, 1e-16), color=LEARNED_COLOR, lw=1.8, label="center error")
    ax.semilogy(x_eval, np.maximum(err_boundary, 1e-16), color=RKPM_COLOR, lw=1.8, label="boundary error")
    _style_line_axis(ax, "Pointwise shape error", "x", "abs error")
    ax.legend(fontsize=8)

    metric_items = _select_shape_validation_metric_items(metrics)
    metric_labels = [label for label, _ in metric_items]
    metric_values = np.array([value for _, value in metric_items], dtype=np.float64)
    ax = axes[1, 0]
    x_pos = np.arange(len(metric_items), dtype=np.float64)
    ax.plot(x_pos, np.maximum(metric_values, 1e-16), "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels, rotation=18, ha="right")
    _style_line_axis(ax, "Validation indicators", "metric", "value", log_scale=True)

    ax = axes[1, 1]
    steps = np.array(history.get("steps", []), dtype=np.float64)
    keys = history_keys or ["loss", "teacher", "linear", "pu", "bd", "reg"]
    plotted = False
    for key in keys:
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0 or steps.size == 0 or np.allclose(values, 0.0):
            continue
        count = min(values.size, steps.size)
        ax.semilogy(steps[:count], np.maximum(values[:count], 1e-16), lw=1.8, label=key)
        plotted = True
    _style_line_axis(ax, "Training history", "step", "loss")
    if plotted:
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "No history available", ha="center", va="center", transform=ax.transAxes)

    _save_figure(fig, path)


def plot_diagnostic_shape_overlay_1d(
    x_eval: np.ndarray,
    phi_rkpm: np.ndarray,
    phi_kan: np.ndarray,
    path: Path,
    stride: int | None = None,
) -> None:
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    phi_rkpm = np.asarray(phi_rkpm, dtype=np.float64)
    phi_kan = np.asarray(phi_kan, dtype=np.float64)
    n_nodes = phi_kan.shape[1]
    step = stride or max(1, n_nodes // 5)
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 6.0))
    for index in range(0, n_nodes, step):
        ax.plot(x_eval, phi_rkpm[:, index], color=RKPM_COLOR, alpha=0.35, lw=1.2, ls=LINESTYLE_RKPM)
        ax.plot(x_eval, phi_kan[:, index], color=LEARNED_COLOR, alpha=0.35, ls=LINESTYLE_LEARNED, lw=1.2)
    _style_line_axis(ax, "Subset overlay of node shape functions", "x", "phi_i(x)")
    ax.text(0.02, 0.04, "solid: learned, dashed: RKPM", transform=ax.transAxes, fontsize=9)
    _save_figure(fig, path)


def plot_consistency_summary(entries: list[dict[str, Any]], path: Path) -> None:
    case_labels = [item["case_label"] for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    value_learned = np.maximum(np.array([item["payload"]["learned"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    value_rkpm = np.maximum(np.array([item["payload"]["rkpm"]["moment_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    derivative_learned = np.maximum(np.array([item["payload"]["learned"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    derivative_rkpm = np.maximum(np.array([item["payload"]["rkpm"]["derivative_matrix_residual_fro"] for item in entries], dtype=np.float64), 1e-16)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, learned_values, rkpm_values, title in zip(
        axes,
        [value_learned, derivative_learned],
        [value_rkpm, derivative_rkpm],
        ["Value consistency summary", "Derivative consistency summary"],
    ):
        ax.plot(x_pos, learned_values, f"o{LINESTYLE_LEARNED}", lw=1.8, ms=6, color=LEARNED_COLOR, label="learned")
        ax.plot(x_pos, rkpm_values, f"s{LINESTYLE_RKPM}", lw=1.8, ms=6, color=RKPM_COLOR, label="rkpm")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        _style_line_axis(ax, title, "case", "residual", log_scale=True)
        ax.legend(fontsize=8)
    _save_figure(fig, path)


def plot_main_figure_shape_case_1d(
    *,
    x_eval: np.ndarray,
    nodes: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
    history: dict[str, list[float]],
    metrics_payload: dict[str, Any],
    path: Path,
) -> None:
    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    phi_learned = np.asarray(phi_learned, dtype=np.float64)
    phi_rkpm = np.asarray(phi_rkpm, dtype=np.float64)
    learned_shape = metrics_payload["learned"]["shape"]
    representative_indices = sorted(set([0, int(learned_shape["center_node_index"]), len(nodes) - 1]))
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)

    ax = axes[0, 0]
    node_colors = AUX_COLORS[: len(representative_indices)]
    for color, index in zip(node_colors, representative_indices):
        ax.plot(x_eval, phi_rkpm[:, index], ls=LINESTYLE_RKPM, lw=1.5, color=color, alpha=0.9)
        ax.plot(x_eval, phi_learned[:, index], lw=1.8, ls=LINESTYLE_LEARNED, color=color, alpha=0.9)
    ax.scatter(nodes[representative_indices], np.zeros(len(representative_indices)), s=32, c=node_colors, edgecolors="black", zorder=3)
    _style_line_axis(ax, "Representative shape functions", "x", "phi")
    method_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.8, ls=LINESTYLE_LEARNED, label="learned"),
            Line2D([0], [0], color="black", lw=1.5, ls=LINESTYLE_RKPM, label="RKPM"),
        ],
        loc="upper left",
        fontsize=8,
        title="method",
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=2.0, label=f"node {index}")
            for color, index in zip(node_colors, representative_indices)
        ],
        loc="upper right",
        fontsize=8,
        title="representative",
    )

    diff = phi_learned - phi_rkpm
    center_idx = int(learned_shape["center_node_index"])
    boundary_idx = int(learned_shape["boundary_node_index"])
    rel_curve = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(phi_rkpm, axis=1), 1.0e-16)
    ax = axes[0, 1]
    ax.semilogy(x_eval, np.maximum(rel_curve, 1.0e-16), lw=1.8, color=LEARNED_COLOR, label="relative field error")
    ax.semilogy(x_eval, np.maximum(np.abs(diff[:, center_idx]), 1.0e-16), lw=1.5, color=AUX_COLORS[2], label="center abs error")
    ax.semilogy(x_eval, np.maximum(np.abs(diff[:, boundary_idx]), 1.0e-16), lw=1.5, color=AUX_COLORS[3], label="boundary abs error")
    _style_line_axis(ax, "Shape error curves", "x", "error")
    ax.legend(fontsize=8)

    labels = ["relL2", "gL2", "cL2", "bL2", "PU", "linear"]
    values = np.array(
        [
            learned_shape["shape_relative_l2"],
            learned_shape["global_l2"],
            learned_shape["center_l2"],
            learned_shape["boundary_l2"],
            learned_shape["pu_max_error"],
            learned_shape["linear_reproduction_rmse"],
        ],
        dtype=np.float64,
    )
    ax = axes[1, 0]
    x_pos = np.arange(values.size, dtype=np.float64)
    safe_values = np.maximum(values, 1.0e-16)
    ax.bar(x_pos, safe_values, width=0.62, color=LEARNED_COLOR, alpha=0.82, edgecolor="black", linewidth=0.6)
    ax.scatter(x_pos, safe_values, s=22, color="black", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    _style_line_axis(ax, "Learned shape indicators", "metric", "value", log_scale=True)

    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)


def plot_summary_figure_shape_seeds_1d(entries: list[dict[str, Any]], path: Path) -> None:
    seeds = [int(item["metrics"]["case"]["seed"]) for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGURE_SIZE)
    panels = [
        ("shape_relative_l2", "Relative shape error"),
        ("global_l2", "Global shape error"),
        ("moment_matrix_residual_fro", "Moment matrix residual"),
        ("derivative_matrix_residual_fro", "Derivative matrix residual"),
    ]
    for ax, (metric_key, title) in zip(axes.reshape(-1), panels):
        if metric_key in ("shape_relative_l2", "global_l2"):
            values = np.array([item["metrics"]["learned"]["shape"][metric_key] for item in entries], dtype=np.float64)
        else:
            values = np.array([item["metrics"]["learned"]["consistency"][metric_key] for item in entries], dtype=np.float64)
        ax.plot(x_pos, np.maximum(values, 1.0e-16), "o-", lw=1.8, ms=6, color=LEARNED_COLOR)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(seed) for seed in seeds])
        _style_line_axis(ax, title, "seed", "value", log_scale=True)
    _save_figure(fig, path)


def plot_diagnostic_shape_consistency_1d(
    *,
    metrics_payload: dict[str, Any],
    learned_consistency_bundle: dict[str, Any],
    rkpm_consistency_bundle: dict[str, Any],
    nodes: np.ndarray,
    diagnostics_dir: Path,
) -> None:
    consistency_payload = build_consistency_metrics_payload(
        case=metrics_payload["case"],
        learned=metrics_payload["learned"]["consistency"],
        rkpm=metrics_payload["rkpm"]["consistency"],
    )
    plot_value_consistency_bars(consistency_payload, diagnostics_dir / "value_consistency.png")
    plot_derivative_consistency_bars(consistency_payload, diagnostics_dir / "derivative_consistency.png")
    plot_derivative_node_residuals_1d(
        learned_residuals=learned_consistency_bundle["arrays"]["boundary_residuals"],
        rkpm_residuals=rkpm_consistency_bundle["arrays"]["boundary_residuals"],
        nodes=np.asarray(nodes, dtype=np.float64),
        path=diagnostics_dir / "derivative_node_residuals.png",
    )

