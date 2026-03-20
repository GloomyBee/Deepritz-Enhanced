from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.shape_validation.one_d.common import (
    FIGURE_DPI,
    LEARNED_COLOR,
    SUMMARY_FIGURE_SIZE,
)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)


def plot_ablation_summary_1d(entries: list[dict[str, Any]], path: Path) -> None:
    variants = [str(item["variant"]) for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGURE_SIZE)
    panels = [
        ("shape", "shape_relative_l2", "Relative shape error"),
        ("shape", "linear_reproduction_rmse", "Linear reproduction"),
        ("consistency", "moment_matrix_residual_fro", "Moment matrix residual"),
        ("consistency", "derivative_matrix_residual_fro", "Derivative matrix residual"),
    ]

    for ax, (group, key, title) in zip(axes.reshape(-1), panels):
        values = np.array([item["metrics"]["learned"][group][key] for item in entries], dtype=np.float64)
        ax.plot(x_pos, np.maximum(values, 1.0e-16), "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=18, ha="right")
        _style_axis(ax, title, "variant", "value")

    _save_figure(fig, path)


def build_ablation_summary_payload_1d(
    *,
    variants: list[str],
    n_nodes: int,
    support_factor: float,
    seed: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "variants": list(variants),
        "n_nodes": int(n_nodes),
        "support_factor": float(support_factor),
        "seed": int(seed),
        "entries": list(entries),
    }


def build_ablation_summary_lines_1d(entries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in entries:
        shape = item["metrics"]["learned"]["shape"]
        consistency = item["metrics"]["learned"]["consistency"]
        lines.append(
            f"{item['variant']} "
            f"shape_relative_l2={shape['shape_relative_l2']:.6e} "
            f"linear_reproduction_rmse={shape['linear_reproduction_rmse']:.6e} "
            f"moment_fro={consistency['moment_matrix_residual_fro']:.6e} "
            f"derivative_fro={consistency['derivative_matrix_residual_fro']:.6e}"
        )
    return lines

