from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.io_or_artifacts import RunArtifacts, ensure_run_artifacts as ensure_run_artifacts_base, save_json, save_npz, save_summary
from core.utils import ensure_repo_root_on_path, parse_float_list, parse_int_list, resolve_repo_root, safe_token, seed_everything


ROOT_DIR = ensure_repo_root_on_path(Path(__file__).resolve())

torch.set_default_dtype(torch.float64)


FIGURE_DPI = 180
MAIN_FIGURE_SIZE = (12.0, 9.0)
SUMMARY_FIGURE_SIZE = (12.0, 8.0)
DIAGNOSTIC_FIGURE_SIZE = (6.0, 4.6)
LEARNED_COLOR = "#1f4e79"
RKPM_COLOR = "#b03a2e"
AUX_COLORS = ["#1f4e79", "#b03a2e", "#2f7d32", "#7a3e9d", "#8c5a2b"]
LINESTYLE_LEARNED = "-"
LINESTYLE_RKPM = "--"


def build_case_name(
    *,
    n_nodes: int | None = None,
    support_factor: float | None = None,
    seed: int | None = None,
    tag: str = "",
) -> str:
    parts: list[str] = []
    if n_nodes is not None:
        parts.append(f"nn{n_nodes}")
    if support_factor is not None:
        parts.append(f"sf{safe_token(support_factor)}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(safe_token(tag))
    return "_".join(parts)


def ensure_run_artifacts(group: str, case_name: str) -> RunArtifacts:
    return ensure_run_artifacts_base(ROOT_DIR, "shape_validation", "one_d", group, case_name)


def save_run_bundle(
    artifacts: RunArtifacts,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    arrays: dict[str, np.ndarray],
    summary_lines: list[str],
) -> None:
    save_json(artifacts.root_dir / "config.json", config)
    save_json(artifacts.root_dir / "metrics.json", metrics)
    save_summary(artifacts.root_dir / "summary.txt", summary_lines)
    save_npz(artifacts.root_dir / "curves.npz", arrays)
