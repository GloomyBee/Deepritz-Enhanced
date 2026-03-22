from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.trial_space_value.two_d.basis import MeshfreeKAN2D, grid_points
from experiments.trial_space_value.two_d.common import (
    AUX_COLORS,
    DIAGNOSTIC_FIGURE_SIZE,
    ERROR_CMAP,
    FIGURE_DPI,
    LEARNED_COLOR,
    MAIN_FIGURE_SIZE,
    SOLUTION_CMAP,
    STRUCTURE_CMAP,
)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
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
        ax.text(0.5, 0.56, "No iterative training", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.42, "Classical solve / direct assembly", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        _style_line_axis(ax, "Solver history", "step", "loss / metric")
        return
    series = [
        ("loss", LEARNED_COLOR),
        ("linear", AUX_COLORS[2]),
        ("energy", AUX_COLORS[1]),
        ("bc", AUX_COLORS[3]),
        ("val_l2", AUX_COLORS[0]),
        ("val_h1", "#666666"),
    ]
    for key, color in series:
        values = np.array(history.get(key, []), dtype=np.float64)
        if values.size == 0 or np.allclose(values, 0.0):
            continue
        count = min(values.size, steps.size)
        ax.semilogy(steps[:count], np.maximum(values[:count], 1e-16), lw=1.8, color=color, label=key)
    _style_line_axis(ax, "Training History", "step", "loss / metric")
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)


def _plot_field_on_axis(
    fig: plt.Figure,
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
) -> None:
    contour = ax.contourf(x_grid, y_grid, values, levels=100, cmap=cmap)
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def plot_training_curves(
    history: dict[str, list[float]],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    _plot_history_series(ax, history)
    ax.set_title(title)
    _save_figure(fig, path)


def plot_solution_triplet(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    path: Path,
    title_prefix: str,
) -> None:
    error = np.abs(pred - exact)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, values, title, cmap in [
        (axes[0], pred, f"{title_prefix} prediction", SOLUTION_CMAP),
        (axes[1], exact, "Exact field", SOLUTION_CMAP),
        (axes[2], error, "Absolute error", ERROR_CMAP),
    ]:
        _plot_field_on_axis(fig, axis, x_grid, y_grid, values, title, cmap)
    _save_figure(fig, path)


def plot_main_figure_trial_space_2d(
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    title_prefix: str,
    path: Path,
) -> None:
    error = np.abs(pred - exact)
    fig, axes = plt.subplots(2, 2, figsize=MAIN_FIGURE_SIZE)
    _plot_field_on_axis(fig, axes[0, 0], x_grid, y_grid, pred, f"{title_prefix} prediction", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[0, 1], x_grid, y_grid, exact, "Exact field", SOLUTION_CMAP)
    _plot_field_on_axis(fig, axes[1, 0], x_grid, y_grid, error, "Absolute error", ERROR_CMAP)
    if history.get("steps"):
        _plot_history_series(axes[1, 1], history)
    else:
        axes[1, 1].axis("off")
        lines = [
            "Solver information",
            "",
            f"solver: {metrics.get('solver_name', 'N/A')}",
            f"cond_k: {metrics.get('cond_k', float('nan')):.3e}",
            f"lambda_min_k: {metrics.get('lambda_min_k', float('nan')):.3e}",
            f"solver_residual: {metrics.get('solver_residual', float('nan')):.3e}",
            f"solver_shift: {metrics.get('solver_shift', float('nan')):.3e}",
        ]
        axes[1, 1].text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    _save_figure(fig, path)


def plot_representative_shape_functions(
    model: MeshfreeKAN2D,
    path: Path,
    resolution: int = 81,
) -> None:
    x_grid, y_grid, points = grid_points(resolution)
    points_t = torch.tensor(points, device=model.nodes.device, dtype=model.nodes.dtype)
    with torch.no_grad():
        phi = model.compute_shape_functions(points_t).cpu().numpy()
    nodes = model.nodes.detach().cpu().numpy()
    targets = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, target in zip(axes.ravel(), targets):
        node_index = int(np.argmin(np.sum((nodes - target) ** 2, axis=1)))
        field = phi[:, node_index].reshape(resolution, resolution)
        image = ax.contourf(x_grid, y_grid, field, levels=50, cmap=STRUCTURE_CMAP)
        ax.scatter(nodes[node_index, 0], nodes[node_index, 1], color="white", s=25, edgecolors="black")
        ax.set_title(f"node {node_index} @ ({nodes[node_index, 0]:.2f}, {nodes[node_index, 1]:.2f})")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)
