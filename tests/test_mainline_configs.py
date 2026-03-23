import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.shape_validation.one_d.config import (
    build_nonuniform_shape_validation_1d_sweep_config,
    build_uniform_shape_validation_1d_config,
    get_shape_training_variant_1d,
)
from experiments.shape_validation.two_d.config import (
    build_irregular_shape_validation_2d_sweep_config,
    build_uniform_shape_validation_2d_config,
    get_shape_training_variant_2d,
)
from experiments.trial_space_value.one_d.config import build_trial_space_value_1d_config
from experiments.trial_space_value.two_d.config import build_trial_space_value_2d_config


class MainlineConfigTests(unittest.TestCase):
    def test_shape_validation_1d_uniform_config_builds_case_and_payload(self):
        cfg = build_uniform_shape_validation_1d_config(
            Namespace(
                n_nodes=11,
                support_factor=2.0,
                variant="teacher_distill",
                steps=2000,
                batch_size=512,
                lr=1.0e-3,
                hidden_dim=16,
                n_eval=2001,
                quadrature_order=12,
                seed=42,
                log_interval=100,
                output_tag="pilot",
            )
        )
        self.assertEqual(cfg.case_name(), "nn11_sf2p0_seed42_uniform_teacher_distill_pilot")
        self.assertAlmostEqual(cfg.support_radius(0.1), 0.2)
        self.assertEqual(cfg.build_case_meta(h=0.1, support_radius=0.2)["layout"], "uniform")
        payload = cfg.to_config_payload(device="cpu", case_name=cfg.case_name(), support_radius=0.2)
        self.assertEqual(payload["group"], "uniform_nodes")
        self.assertEqual(payload["device"], "cpu")

    def test_shape_validation_1d_nonuniform_config_requires_representative_seed(self):
        with self.assertRaises(ValueError):
            build_nonuniform_shape_validation_1d_sweep_config(
                Namespace(
                    n_nodes=11,
                    support_factor=2.0,
                    jitter_factor=0.1,
                    variant="teacher_distill",
                    steps=2000,
                    batch_size=512,
                    lr=1.0e-3,
                    hidden_dim=16,
                    n_eval=2001,
                    quadrature_order=12,
                    seeds="42,43,44",
                    representative_seed=99,
                    log_interval=100,
                    output_tag="pilot",
                )
            )

    def test_shape_validation_1d_variant_is_typed(self):
        variant = get_shape_training_variant_1d("teacher_distill")
        self.assertEqual(variant.name, "teacher_distill")
        self.assertEqual(variant.objective, "distill")
        self.assertFalse(variant.use_softplus)

    def test_shape_validation_2d_variant_and_uniform_config_are_typed(self):
        variant = get_shape_training_variant_2d("no_softplus_raw_pu_bd_no_fallback")
        self.assertFalse(variant.enable_fallback)
        self.assertFalse(variant.use_softplus)

        cfg = build_uniform_shape_validation_2d_config(
            Namespace(
                n_side=15,
                kappa=2.5,
                variant="softplus_raw_pu_bd",
                phase_a_steps=1500,
                batch_size=1024,
                lr=1.0e-3,
                hidden_dim=16,
                grid_resolution=81,
                quadrature_order=4,
                seed=42,
                log_interval=100,
                output_tag="pilot",
            )
        )
        self.assertEqual(cfg.case_name(), "variant_softplus_raw_pu_bd_ns15_k2p5_seed42_pilot")
        payload = cfg.to_config_payload(device="cpu", case_name=cfg.case_name(), support_radius=0.25)
        self.assertEqual(payload["group"], "uniform_nodes")
        self.assertEqual(payload["support_radius"], 0.25)

    def test_shape_validation_2d_irregular_sweep_parses_jitters(self):
        cfg = build_irregular_shape_validation_2d_sweep_config(
            Namespace(
                jitters="0.0,0.1,0.2",
                n_side=9,
                kappa=2.5,
                variant="softplus_raw_pu_bd",
                phase_a_steps=1200,
                batch_size=1024,
                lr=1.0e-3,
                hidden_dim=16,
                grid_resolution=81,
                quadrature_order=4,
                seed=42,
                log_interval=100,
                output_tag="pilot",
            )
        )
        self.assertEqual(cfg.jitters, (0.0, 0.1, 0.2))
        self.assertIn("jit0p1", cfg.case_name(jitter=0.1))

    def test_trial_space_value_1d_config_builds_method_specific_case_name(self):
        cfg = build_trial_space_value_1d_config(
            Namespace(
                n_nodes=11,
                support_factor=2.0,
                variant="teacher_distill",
                steps=2000,
                batch_size=512,
                lr=1.0e-3,
                hidden_dim=16,
                n_eval=2001,
                eval_resolution=401,
                quadrature_order=12,
                seed=42,
                log_interval=100,
                output_tag="pilot",
            )
        )
        self.assertEqual(cfg.case_name(), "nn11_sf2p0_method_fixed_basis_seed42_teacher_distill_pilot")
        case_payload = cfg.build_case_payload(support_radius=0.2)
        self.assertEqual(case_payload["method"], "fixed_basis")

    def test_trial_space_value_2d_config_builds_method_payload(self):
        cfg = build_trial_space_value_2d_config(
            Namespace(
                n_side=7,
                kappa=2.5,
                variant="softplus_raw_pu_bd",
                phase_a_steps=1200,
                phase_b_steps=1500,
                batch_size=1024,
                lr_phase_a=1.0e-3,
                lr_kan_b=1.0e-4,
                lr_w=1.0e-2,
                beta_bc=100.0,
                gamma_linear=10.0,
                warmup_w_steps=400,
                eval_interval=100,
                eval_resolution=41,
                quadrature_order=8,
                hidden_dim=16,
                grid_resolution=81,
                seed=42,
                log_interval=100,
                output_tag="pilot",
            ),
            method="classical",
        )
        self.assertEqual(cfg.case_name(), "variant_softplus_raw_pu_bd_method_classical_ns7_k2p5_seed42_pilot")
        payload = cfg.build_case_payload(h=1.0 / 6.0, support_radius=2.5 / 6.0, n_nodes=49)
        self.assertEqual(payload["method"], "classical")
        cfg_payload = cfg.to_config_payload(device="cpu", case_name=cfg.case_name(), support_radius=2.5 / 6.0)
        self.assertEqual(cfg_payload["method"], "classical")
        self.assertEqual(cfg_payload["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
