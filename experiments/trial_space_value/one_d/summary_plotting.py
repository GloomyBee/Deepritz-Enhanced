from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.shape_validation.one_d.common import FIGURE_DPI, LEARNED_COLOR, SUMMARY_FIGURE_SIZE


def plot_summary_figure_trial_space_1d(entries: list[dict[str, Any]], path: Path) -> None:
    if not entries:
        return
    labels = [str(item["method"]) for item in entries]
    x_pos = np.arange(len(entries), dtype=np.float64)
    l2 = np.maximum(np.array([float(item["trial_space"]["l2_error"]) for item in entries], dtype=np.float64), 1.0e-16)
    h1 = np.maximum(np.array([float(item["trial_space"]["h1_semi_error"]) for item in entries], dtype=np.float64), 1.0e-16)

    fig, axes = plt.subplots(1, 2, figsize=SUMMARY_FIGURE_SIZE)
    for ax, values, title in [
        (axes[0], l2, "L2 error"),
        (axes[1], h1, "H1 semi error"),
    ]:
        ax.plot(x_pos, values, "o-", color=LEARNED_COLOR, lw=1.8, ms=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_xlabel("method")
        ax.set_ylabel("error")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
