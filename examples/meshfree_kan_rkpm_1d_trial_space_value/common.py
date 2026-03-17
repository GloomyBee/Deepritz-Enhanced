from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


@dataclass
class TrialSpaceArtifacts1D:
    root_dir: Path
    figures_dir: Path
    methods_dir: Path


def ensure_trial_space_artifacts_1d(group: str, case_name: str) -> TrialSpaceArtifacts1D:
    root_dir = ROOT_DIR / "output" / "meshfree_kan_rkpm_1d_trial_space_value" / group / case_name
    figures_dir = root_dir / "figures"
    methods_dir = root_dir / "methods"
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    return TrialSpaceArtifacts1D(root_dir=root_dir, figures_dir=figures_dir, methods_dir=methods_dir)
