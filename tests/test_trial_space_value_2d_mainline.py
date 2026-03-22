import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


class TrialSpace2DMainlineTests(unittest.TestCase):
    def test_active_entrypoints_are_split_by_method(self):
        base = ROOT_DIR / "experiments" / "trial_space_value" / "two_d" / "poisson_compare"
        classical_entry = base / "deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py"
        frozen_entry = base / "deepritz_meshfree_kan_rkpm_2d_poisson_compare_frozen_w_v1.py"
        joint_entry = base / "deepritz_meshfree_kan_rkpm_2d_poisson_compare_joint_v1.py"
        retired_entry = base / "deepritz_meshfree_kan_rkpm_2d_poisson_compare_v1.py"
        archived_entry = (
            ROOT_DIR
            / "archive"
            / "trial_space_value"
            / "two_d"
            / "poisson_compare"
            / "deepritz_meshfree_kan_rkpm_2d_poisson_compare_v1.py"
        )
        self.assertTrue(classical_entry.is_file())
        self.assertTrue(frozen_entry.is_file())
        self.assertTrue(joint_entry.is_file())
        self.assertFalse(retired_entry.exists())
        self.assertTrue(archived_entry.is_file())

    def test_trial_space_two_d_root_has_split_modules(self):
        base = ROOT_DIR / "experiments" / "trial_space_value" / "two_d"
        expected = [
            "common.py",
            "basis.py",
            "training.py",
            "trial_space.py",
            "plotting.py",
            "summary_plotting.py",
        ]
        for name in expected:
            self.assertTrue((base / name).is_file(), msg=name)


if __name__ == "__main__":
    unittest.main()
