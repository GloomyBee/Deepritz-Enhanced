from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.io_or_artifacts import RunArtifacts, ensure_run_artifacts as ensure_run_artifacts_base, save_json, save_npz, save_summary
from core.utils import ensure_repo_root_on_path, parse_float_list, parse_int_list, safe_token, seed_everything


ROOT_DIR = ensure_repo_root_on_path(Path(__file__).resolve())


FIGURE_DPI = 180
MAIN_FIGURE_SIZE = (12.0, 9.0)
SUMMARY_FIGURE_SIZE = (12.0, 8.2)
DIAGNOSTIC_FIGURE_SIZE = (6.0, 5.0)
LEARNED_COLOR = "#1f4e79"
RKPM_COLOR = "#b03a2e"
AUX_COLORS = ["#1f4e79", "#b03a2e", "#2f7d32", "#7a3e9d", "#8c5a2b"]
SOLUTION_CMAP = "inferno"
ERROR_CMAP = "magma"
STRUCTURE_CMAP = "viridis"

METHOD_LABELS = {
    "classical": "Fixed basis + classical solve",
    "frozen_w": "Fixed basis + optimize w",
    "joint": "Joint Phase B retraining",
}


def build_case_name(
    *,
    variant: str | None = None,
    method: str | None = None,
    n_side: int | None = None,
    kappa: float | None = None,
    seed: int | None = None,
    jitter: float | None = None,
    tag: str = "",
) -> str:
    parts: list[str] = []
    if variant:
        parts.append(f"variant_{safe_token(variant)}")
    if method:
        parts.append(f"method_{safe_token(method)}")
    if n_side is not None:
        parts.append(f"ns{n_side}")
    if kappa is not None:
        parts.append(f"k{safe_token(kappa)}")
    if jitter is not None:
        parts.append(f"jit{safe_token(jitter)}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(safe_token(tag))
    return "_".join(parts)


def ensure_trial_space_artifacts(group: str, case_name: str) -> RunArtifacts:
    return ensure_run_artifacts_base(ROOT_DIR, "trial_space_value", "two_d", group, case_name)


def build_metrics_payload(
    *,
    case: dict[str, Any],
    basis_quality: dict[str, Any],
    trial_space: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    return {
        "case": case,
        "method": method,
        "basis_quality": basis_quality,
        "trial_space": trial_space,
    }


def save_run_bundle(
    artifacts: RunArtifacts,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    arrays: dict[str, np.ndarray],
    summary_lines: list[str],
) -> None:
    del history
    save_json(artifacts.root_dir / "config.json", config)
    save_json(artifacts.root_dir / "metrics.json", metrics)
    save_summary(artifacts.root_dir / "summary.txt", summary_lines)
    save_npz(artifacts.root_dir / "curves.npz", arrays)
