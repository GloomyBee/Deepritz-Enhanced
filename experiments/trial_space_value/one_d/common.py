from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.io_or_artifacts import RunArtifacts, ensure_run_artifacts as ensure_run_artifacts_base, save_json, save_npz, save_summary
from core.utils import ensure_repo_root_on_path, parse_float_list, parse_int_list, resolve_repo_root, safe_token, seed_everything


ROOT_DIR = ensure_repo_root_on_path(Path(__file__).resolve())


def build_case_name(
    *,
    n_nodes: int | None = None,
    support_factor: float | None = None,
    method: str | None = None,
    seed: int | None = None,
    tag: str = "",
) -> str:
    parts: list[str] = []
    if n_nodes is not None:
        parts.append(f"nn{n_nodes}")
    if support_factor is not None:
        parts.append(f"sf{safe_token(support_factor)}")
    if method:
        parts.append(f"method_{safe_token(method)}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(safe_token(tag))
    return "_".join(parts)


def ensure_trial_space_artifacts_1d(group: str, case_name: str) -> RunArtifacts:
    return ensure_run_artifacts_base(ROOT_DIR, "trial_space_value", "one_d", group, case_name)


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
