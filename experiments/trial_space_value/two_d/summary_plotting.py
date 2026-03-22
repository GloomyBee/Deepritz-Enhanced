from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.trial_space_value.two_d.common import LEARNED_COLOR, RKPM_COLOR, SUMMARY_FIGURE_SIZE


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_line_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, log_scale: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def plot_metric_bars(entries: list[dict[str, Any]], path: Path) -> None:
    methods = [str(item["method"]) for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    l2 = np.maximum(np.array([item["trial_space"]["l2_error"] for item in entries], dtype=np.float64), 1e-16)
    h1 = np.maximum(np.array([item["trial_space"]["h1_semi_error"] for item in entries], dtype=np.float64), 1e-16)
    boundary = np.maximum(np.array([item["trial_space"]["boundary_l2"] for item in entries], dtype=np.float64), 1e-16)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, values, title, color, marker in zip(
        axes,
        [l2, h1, boundary],
        ["L2 error", "H1 semi error", "Boundary L2"],
        [LEARNED_COLOR, RKPM_COLOR, "#2f7d32"],
        ["o", "s", "^"],
    ):
        ax.plot(x_pos, values, marker=marker, lw=1.8, ms=6, color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        _style_line_axis(ax, title, "method", "error", log_scale=True)
    _save_figure(fig, path)


def plot_conditioning_bars(entries: list[dict[str, Any]], path: Path) -> None:
    methods = [str(item["method"]) for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    cond_values = np.maximum(np.array([item["trial_space"]["cond_k"] for item in entries], dtype=np.float64), 1e-16)
    lambda_min = np.array([item["trial_space"]["lambda_min_k"] for item in entries], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].plot(x_pos, cond_values, "o-", lw=1.8, ms=6, color=LEARNED_COLOR)
    axes[1].plot(x_pos, lambda_min, "s-", lw=1.8, ms=6, color=RKPM_COLOR)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    _style_line_axis(axes[0], "Condition number", "method", "cond_k", log_scale=True)
    _style_line_axis(axes[1], "Smallest eigenvalue", "method", "lambda_min_k")
    _save_figure(fig, path)
