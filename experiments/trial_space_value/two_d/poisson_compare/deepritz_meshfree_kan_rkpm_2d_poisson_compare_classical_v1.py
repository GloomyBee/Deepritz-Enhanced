from __future__ import annotations

import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
BOOTSTRAP_ROOT = next(parent for parent in THIS_FILE.parents if ((parent / "requirements.txt").is_file() and (parent / "core").is_dir()))
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.append(str(BOOTSTRAP_ROOT))

from experiments.trial_space_value.two_d.poisson_compare._entry_common import run_poisson_compare_method_entry


if __name__ == "__main__":
    run_poisson_compare_method_entry("classical", "2D fixed-basis Poisson solve with classical coefficient assembly")
