from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.shape_validation.two_d.basis import (
    AUX_COLORS,
    LEARNED_COLOR,
    MAIN_FIGURE_SIZE,
    STRUCTURE_CMAP,
    _plot_history_series,
    _save_figure,
    _style_line_axis,
    plot_derivative_consistency_bars,
    plot_derivative_node_residuals_2d,
    plot_heatmap,
    plot_main_figure_consistency_2d,
    plot_main_figure_irregular_nodes,
    plot_main_figure_patch_test,
    plot_main_figure_poisson,
    plot_main_figure_stability,
    plot_solution_triplet,
    plot_training_curves,
)


def plot_main_figure_shape_case_2d(
    *,
    x_eval: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    nodes: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
    history: dict[str, list[float]],
    metrics_payload: dict[str, Any],
    shape_arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    del x_eval, x_grid, y_grid, phi_learned, phi_rkpm
    learned_shape = metrics_payload["learned"]["shape"]
    labels = learned_shape["representative_labels"]
    indices = learned_shape["representative_node_indices"]
    node_colors = AUX_COLORS[: len(labels)]
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)

    ax = axes[0, 0]
    for row, (label, index, color) in enumerate(zip(labels, indices, node_colors)):
        cut_x = shape_arrays["cut_x"][row]
        ax.plot(cut_x, shape_arrays["cut_rkpm"][row], ls="--", lw=1.5, color=color, alpha=0.9)
        ax.plot(cut_x, shape_arrays["cut_learned"][row], ls="-", lw=1.8, color=color, alpha=0.9)
        ax.scatter([float(nodes[index, 0])], [0.0], s=28, color=color, edgecolors="black", zorder=3)
    _style_line_axis(ax, "Representative shape functions", "x on horizontal cut", "phi")
    method_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.8, ls="-", label="learned"),
            Line2D([0], [0], color="black", lw=1.5, ls="--", label="RKPM"),
        ],
        loc="upper left",
        fontsize=8,
        title="method",
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=2.0, label=f"{label} (node {index})")
            for label, index, color in zip(labels, indices, node_colors)
        ],
        loc="upper right",
        fontsize=8,
        title="representative",
    )

    ax = axes[0, 1]
    for row, (label, color) in enumerate(zip(labels, node_colors)):
        ax.semilogy(
            shape_arrays["cut_x"][row],
            np.maximum(shape_arrays["cut_abs_error"][row], 1.0e-16),
            lw=1.8,
            color=color,
            label=label,
        )
    _style_line_axis(ax, "Shape error on representative cuts", "x on horizontal cut", "abs error")
    ax.legend(fontsize=8)

    indicator_labels = ["relL2", "gL2", "corner", "edge", "center", "PU", "linear", "Lh"]
    indicator_values = np.array(
        [
            learned_shape["shape_relative_l2"],
            learned_shape["global_l2"],
            learned_shape["corner_l2"],
            learned_shape["edge_mid_l2"],
            learned_shape["center_l2"],
            learned_shape["pu_max_error"],
            max(learned_shape["linear_x_rmse"], learned_shape["linear_y_rmse"]),
            learned_shape["lambda_h_max"],
        ],
        dtype=np.float64,
    )
    ax = axes[1, 0]
    x_pos = np.arange(indicator_values.size, dtype=np.float64)
    safe_values = np.maximum(indicator_values, 1.0e-16)
    ax.bar(x_pos, safe_values, width=0.62, color=LEARNED_COLOR, alpha=0.82, edgecolor="black", linewidth=0.6)
    ax.scatter(x_pos, safe_values, s=20, color="black", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(indicator_labels, rotation=16, ha="right")
    _style_line_axis(ax, "Learned shape indicators", "metric", "value", log_scale=True)

    _plot_history_series(axes[1, 1], history)
    _save_figure(fig, path)


def plot_diagnostic_shape_representatives_2d(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
    representative_labels: list[str],
    representative_indices: list[int],
    diagnostics_dir: Path,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for label, index in zip(representative_labels, representative_indices):
        learned_field = phi_learned[:, index].reshape(x_grid.shape)
        rkpm_field = phi_rkpm[:, index].reshape(x_grid.shape)
        plot_heatmap(x_grid, y_grid, learned_field, diagnostics_dir / f"representative_{label}_learned.png", f"{label} learned shape")
        plot_heatmap(x_grid, y_grid, rkpm_field, diagnostics_dir / f"representative_{label}_rkpm.png", f"{label} RKPM shape")
        plot_heatmap(
            x_grid,
            y_grid,
            np.abs(learned_field - rkpm_field),
            diagnostics_dir / f"representative_{label}_abs_error.png",
            f"{label} abs error",
            cmap=STRUCTURE_CMAP,
        )
