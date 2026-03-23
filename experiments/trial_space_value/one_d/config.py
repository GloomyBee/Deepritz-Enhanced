from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.utils import safe_token


@dataclass(frozen=True)
class TrialSpaceValue1DConfig:
    n_nodes: int
    support_factor: float
    variant: str
    steps: int
    batch_size: int
    lr: float
    hidden_dim: int
    n_eval: int
    eval_resolution: int
    quadrature_order: int
    seed: int
    log_interval: int
    output_tag: str = ""
    method: str = "fixed_basis"
    group: str = "poisson_compare"

    def support_radius(self, h: float) -> float:
        return self.support_factor * h

    def case_name(self) -> str:
        parts = [
            f"nn{self.n_nodes}",
            f"sf{safe_token(self.support_factor)}",
            f"method_{safe_token(self.method)}",
            f"seed{self.seed}",
            safe_token(self.variant),
        ]
        if self.output_tag.strip():
            parts.append(safe_token(self.output_tag.strip()))
        return "_".join(parts)

    def build_shape_case_meta(self, *, h: float, support_radius: float) -> dict[str, Any]:
        return {
            "dimension": 1,
            "layout": "uniform",
            "variant": self.variant,
            "n_nodes": self.n_nodes,
            "h": h,
            "support_factor": self.support_factor,
            "support_radius": support_radius,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed,
        }

    def build_case_payload(self, *, support_radius: float) -> dict[str, Any]:
        return {
            "dimension": 1,
            "group": self.group,
            "variant": self.variant,
            "method": self.method,
            "n_nodes": self.n_nodes,
            "support_factor": self.support_factor,
            "support_radius": support_radius,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed,
        }

    def to_config_payload(self, *, device: str, case_name: str, support_radius: float) -> dict[str, Any]:
        return {
            "group": self.group,
            "case_name": case_name,
            "method": self.method,
            "variant": self.variant,
            "n_nodes": self.n_nodes,
            "support_factor": self.support_factor,
            "support_radius": support_radius,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "n_eval": self.n_eval,
            "eval_resolution": self.eval_resolution,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed,
            "log_interval": self.log_interval,
            "output_tag": self.output_tag,
            "device": device,
        }


def build_trial_space_value_1d_config(args: Any) -> TrialSpaceValue1DConfig:
    return TrialSpaceValue1DConfig(
        n_nodes=args.n_nodes,
        support_factor=args.support_factor,
        variant=args.variant,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_eval=args.n_eval,
        eval_resolution=args.eval_resolution,
        quadrature_order=args.quadrature_order,
        seed=args.seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )
