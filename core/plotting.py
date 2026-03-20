from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

