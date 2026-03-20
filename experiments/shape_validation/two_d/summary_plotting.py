from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.shape_validation.two_d.basis import (
    LEARNED_COLOR,
    SUMMARY_FIGURE_SIZE,
    _save_figure,
    _style_line_axis,
    plot_consistency_summary,
    plot_irregular_summary_2d,
    plot_patch_test_summary_figure,
    plot_poisson_convergence_summary_2d,
    plot_stability_summary_2d,
    plot_summary_figure_consistency_2d,
)


def plot_summary_figure_shape_jitters_2d(entries: list[dict[str, Any]], path: Path) -> None:
    jitters = np.array([float(item["jitter"]) for item in entries], dtype=np.float64)
    order = np.argsort(jitters)
    jitters = jitters[order]
    rel_l2 = np.maximum(
        np.array([float(entries[idx]["metrics"]["learned"]["shape"]["shape_relative_l2"]) for idx in order], dtype=np.float64),
        1.0e-16,
    )
    center_l2 = np.maximum(
        np.array([float(entries[idx]["metrics"]["learned"]["shape"]["center_l2"]) for idx in order], dtype=np.float64),
        1.0e-16,
    )
    linear_rmse = np.maximum(
        np.array(
            [
                max(
                    float(entries[idx]["metrics"]["learned"]["shape"]["linear_x_rmse"]),
                    float(entries[idx]["metrics"]["learned"]["shape"]["linear_y_rmse"]),
                )
                for idx in order
            ],
            dtype=np.float64,
        ),
        1.0e-16,
    )
    lambda_max = np.maximum(
        np.array([float(entries[idx]["metrics"]["learned"]["shape"]["lambda_h_max"]) for idx in order], dtype=np.float64),
        1.0e-16,
    )
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_FIGURE_SIZE)
    panels = [
        (axes[0, 0], rel_l2, "Jitter vs relative shape error", "shape_relative_l2"),
        (axes[0, 1], center_l2, "Jitter vs center shape error", "center_l2"),
        (axes[1, 0], linear_rmse, "Jitter vs linear reproduction", "max linear rmse"),
        (axes[1, 1], lambda_max, "Jitter vs Lambda_h", "lambda_h_max"),
    ]
    for ax, values, title, ylabel in panels:
        ax.plot(jitters, values, "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
        _style_line_axis(ax, title, "jitter", ylabel, log_scale=True)
    _save_figure(fig, path)
