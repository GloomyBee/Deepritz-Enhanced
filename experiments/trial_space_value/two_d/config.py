from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.utils import safe_token
from experiments.shape_validation.two_d.config import get_shape_training_variant_2d


@dataclass(frozen=True)
class TrialSpaceValue2DConfig:
    method: str
    n_side: int
    kappa: float
    variant: str
    phase_a_steps: int
    phase_b_steps: int
    batch_size: int
    lr_phase_a: float
    lr_kan_b: float
    lr_w: float
    beta_bc: float
    gamma_linear: float
    warmup_w_steps: int
    eval_interval: int
    eval_resolution: int
    quadrature_order: int
    hidden_dim: int
    grid_resolution: int
    seed: int
    log_interval: int
    output_tag: str = ""
    group: str = "poisson_compare"

    def support_radius(self, h: float) -> float:
        return self.kappa * h

    def case_name(self) -> str:
        parts = [
            f"variant_{safe_token(self.variant)}",
            f"method_{safe_token(self.method)}",
            f"ns{self.n_side}",
            f"k{safe_token(self.kappa)}",
            f"seed{self.seed}",
        ]
        if self.output_tag.strip():
            parts.append(safe_token(self.output_tag.strip()))
        return "_".join(parts)

    def variant_preset(self):
        return get_shape_training_variant_2d(self.variant)

    def build_case_payload(self, *, h: float, support_radius: float, n_nodes: int) -> dict[str, Any]:
        return {
            "dimension": 2,
            "group": self.group,
            "variant": self.variant,
            "method": self.method,
            "n_side": self.n_side,
            "n_nodes": n_nodes,
            "h": h,
            "kappa": self.kappa,
            "support_radius": support_radius,
            "seed": self.seed,
            "phase_a_steps": self.phase_a_steps,
            "phase_b_steps": self.phase_b_steps,
        }

    def to_config_payload(self, *, device: str, case_name: str, support_radius: float) -> dict[str, Any]:
        return {
            "method": self.method,
            "device": device,
            "case_name": case_name,
            "group": self.group,
            "n_side": self.n_side,
            "kappa": self.kappa,
            "variant": self.variant,
            "support_radius": support_radius,
            "phase_a_steps": self.phase_a_steps,
            "phase_b_steps": self.phase_b_steps,
            "batch_size": self.batch_size,
            "lr_phase_a": self.lr_phase_a,
            "lr_kan_b": self.lr_kan_b,
            "lr_w": self.lr_w,
            "beta_bc": self.beta_bc,
            "gamma_linear": self.gamma_linear,
            "warmup_w_steps": self.warmup_w_steps,
            "eval_interval": self.eval_interval,
            "eval_resolution": self.eval_resolution,
            "quadrature_order": self.quadrature_order,
            "hidden_dim": self.hidden_dim,
            "grid_resolution": self.grid_resolution,
            "seed": self.seed,
            "log_interval": self.log_interval,
            "output_tag": self.output_tag,
        }


def build_trial_space_value_2d_config(args: Any, *, method: str) -> TrialSpaceValue2DConfig:
    return TrialSpaceValue2DConfig(
        method=method,
        n_side=args.n_side,
        kappa=args.kappa,
        variant=args.variant,
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        batch_size=args.batch_size,
        lr_phase_a=args.lr_phase_a,
        lr_kan_b=args.lr_kan_b,
        lr_w=args.lr_w,
        beta_bc=args.beta_bc,
        gamma_linear=args.gamma_linear,
        warmup_w_steps=args.warmup_w_steps,
        eval_interval=args.eval_interval,
        eval_resolution=args.eval_resolution,
        quadrature_order=args.quadrature_order,
        hidden_dim=args.hidden_dim,
        grid_resolution=args.grid_resolution,
        seed=args.seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )
