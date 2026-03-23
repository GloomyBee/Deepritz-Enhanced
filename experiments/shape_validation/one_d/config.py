from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.utils import parse_int_list, safe_token


@dataclass(frozen=True)
class ShapeTrainingVariant1D:
    name: str
    objective: str
    use_softplus: bool
    lambda_pu: float = 0.0
    lambda_bd: float = 0.0
    lambda_teacher: float = 0.0
    lambda_reg: float = 0.0
    pu_on_raw: bool = False


SHAPE_TRAINING_VARIANTS_1D: dict[str, ShapeTrainingVariant1D] = {
    "raw_pu": ShapeTrainingVariant1D(
        name="raw_pu",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=0.0,
        pu_on_raw=False,
    ),
    "raw_pu_fix": ShapeTrainingVariant1D(
        name="raw_pu_fix",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=0.0,
        pu_on_raw=True,
    ),
    "boundary_anchor": ShapeTrainingVariant1D(
        name="boundary_anchor",
        objective="raw",
        use_softplus=True,
        lambda_pu=0.1,
        lambda_bd=1.0,
        pu_on_raw=True,
    ),
    "no_softplus": ShapeTrainingVariant1D(
        name="no_softplus",
        objective="raw",
        use_softplus=False,
        lambda_pu=0.1,
        lambda_bd=1.0,
        pu_on_raw=True,
    ),
    "teacher_distill": ShapeTrainingVariant1D(
        name="teacher_distill",
        objective="distill",
        use_softplus=False,
        lambda_teacher=1.0,
        lambda_bd=0.1,
        lambda_reg=1.0e-4,
    ),
}


@dataclass(frozen=True)
class ShapeValidation1DRunConfig:
    group: str
    layout: str
    n_nodes: int
    support_factor: float
    variant: str
    steps: int
    batch_size: int
    lr: float
    hidden_dim: int
    n_eval: int
    quadrature_order: int
    seed: int
    log_interval: int
    output_tag: str = ""
    jitter_factor: float | None = None

    def support_radius(self, h: float) -> float:
        return self.support_factor * h

    def _tag_parts(self) -> list[str]:
        parts = [self.layout]
        if self.jitter_factor is not None:
            parts.append(f"jf{safe_token(self.jitter_factor)}")
        parts.append(self.variant)
        if self.output_tag.strip():
            parts.append(self.output_tag.strip())
        return [part for part in parts if part]

    def case_name(self, *, seed: int | None = None) -> str:
        seed_value = self.seed if seed is None else seed
        parts = [f"nn{self.n_nodes}", f"sf{safe_token(self.support_factor)}", f"seed{seed_value}"]
        parts.extend(safe_token(part) for part in self._tag_parts())
        return "_".join(parts)

    def build_case_meta(self, *, h: float, support_radius: float, seed: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dimension": 1,
            "layout": self.layout,
            "variant": self.variant,
            "n_nodes": self.n_nodes,
            "h": h,
            "support_factor": self.support_factor,
            "support_radius": support_radius,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed if seed is None else seed,
        }
        if self.jitter_factor is not None:
            payload["jitter_factor"] = self.jitter_factor
        return payload

    def to_config_payload(self, *, device: str, case_name: str, support_radius: float, seed: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group": self.group,
            "case_name": case_name,
            "n_nodes": self.n_nodes,
            "support_factor": self.support_factor,
            "support_radius": support_radius,
            "variant": self.variant,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "n_eval": self.n_eval,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed if seed is None else seed,
            "output_tag": self.output_tag,
            "device": device,
        }
        if self.jitter_factor is not None:
            payload["jitter_factor"] = self.jitter_factor
        return payload


@dataclass(frozen=True)
class ShapeValidation1DNonuniformSweepConfig:
    run_config: ShapeValidation1DRunConfig
    seeds: tuple[int, ...]
    representative_seed: int

    def case_config_for_seed(self, seed: int) -> ShapeValidation1DRunConfig:
        return replace(self.run_config, seed=seed)


def get_shape_training_variant_1d(name: str) -> ShapeTrainingVariant1D:
    key = name.strip().lower()
    if key not in SHAPE_TRAINING_VARIANTS_1D:
        raise KeyError(f"Unknown 1D training variant: {name}")
    return SHAPE_TRAINING_VARIANTS_1D[key]


def build_uniform_shape_validation_1d_config(args: Any) -> ShapeValidation1DRunConfig:
    return ShapeValidation1DRunConfig(
        group="uniform_nodes",
        layout="uniform",
        n_nodes=args.n_nodes,
        support_factor=args.support_factor,
        variant=args.variant,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_eval=args.n_eval,
        quadrature_order=args.quadrature_order,
        seed=args.seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )


def build_nonuniform_shape_validation_1d_sweep_config(args: Any) -> ShapeValidation1DNonuniformSweepConfig:
    seeds = tuple(parse_int_list(args.seeds))
    if args.representative_seed not in seeds:
        raise ValueError("representative seed must be included in --seeds")
    run_config = ShapeValidation1DRunConfig(
        group="nonuniform_nodes",
        layout="nonuniform",
        n_nodes=args.n_nodes,
        support_factor=args.support_factor,
        variant=args.variant,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_eval=args.n_eval,
        quadrature_order=args.quadrature_order,
        seed=args.representative_seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
        jitter_factor=args.jitter_factor,
    )
    return ShapeValidation1DNonuniformSweepConfig(
        run_config=run_config,
        seeds=seeds,
        representative_seed=args.representative_seed,
    )
