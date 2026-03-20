from __future__ import annotations

from pathlib import Path

from core.io_or_artifacts import TrialSpaceArtifacts, ensure_trial_space_artifacts
from core.utils import ensure_repo_root_on_path


ROOT_DIR = ensure_repo_root_on_path(Path(__file__).resolve())


def ensure_trial_space_artifacts_1d(group: str, case_name: str) -> TrialSpaceArtifacts:
    return ensure_trial_space_artifacts(ROOT_DIR, "trial_space_value", "one_d", group, case_name)
