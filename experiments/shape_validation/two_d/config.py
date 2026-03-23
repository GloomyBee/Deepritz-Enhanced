from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.utils import parse_float_list, safe_token


@dataclass(frozen=True)
class ShapeTrainingVariant2D:
    name: str
    use_softplus: bool
    enable_fallback: bool
    use_linear_loss: bool
    use_pu_loss: bool
    use_bd_loss: bool
    use_teacher_loss: bool
    lambda_pu: float = 0.0
    lambda_bd: float = 0.0
    lambda_teacher: float = 0.0
    lambda_reg: float = 0.0


SHAPE_TRAINING_VARIANTS_2D: dict[str, ShapeTrainingVariant2D] = {
    "softplus_raw_pu_bd": ShapeTrainingVariant2D(
        name="softplus_raw_pu_bd",
        use_softplus=True,
        enable_fallback=True,
        use_linear_loss=True,
        use_pu_loss=True,
        use_bd_loss=True,
        use_teacher_loss=False,
        lambda_pu=0.1,
        lambda_bd=1.0,
    ),
    "no_softplus_raw_pu_bd": ShapeTrainingVariant2D(
        name="no_softplus_raw_pu_bd",
        use_softplus=False,
        enable_fallback=True,
        use_linear_loss=True,
        use_pu_loss=True,
        use_bd_loss=True,
        use_teacher_loss=False,
        lambda_pu=0.1,
        lambda_bd=1.0,
    ),
    "no_softplus_teacher": ShapeTrainingVariant2D(
        name="no_softplus_teacher",
        use_softplus=False,
        enable_fallback=True,
        use_linear_loss=False,
        use_pu_loss=False,
        use_bd_loss=True,
        use_teacher_loss=True,
        lambda_bd=0.1,
        lambda_teacher=1.0,
        lambda_reg=1e-4,
    ),
    "no_softplus_teacher_reg": ShapeTrainingVariant2D(
        name="no_softplus_teacher_reg",
        use_softplus=False,
        enable_fallback=True,
        use_linear_loss=False,
        use_pu_loss=False,
        use_bd_loss=True,
        use_teacher_loss=True,
        lambda_bd=0.1,
        lambda_teacher=1.0,
        lambda_reg=5e-4,
    ),
    "no_softplus_raw_pu_bd_no_fallback": ShapeTrainingVariant2D(
        name="no_softplus_raw_pu_bd_no_fallback",
        use_softplus=False,
        enable_fallback=False,
        use_linear_loss=True,
        use_pu_loss=True,
        use_bd_loss=True,
        use_teacher_loss=False,
        lambda_pu=0.1,
        lambda_bd=1.0,
    ),
}


@dataclass(frozen=True)
class ShapeValidation2DRunConfig:
    group: str
    layout: str
    n_side: int
    kappa: float
    variant: str
    phase_a_steps: int
    batch_size: int
    lr: float
    hidden_dim: int
    grid_resolution: int
    quadrature_order: int
    seed: int
    log_interval: int
    output_tag: str = ""
    jitter: float | None = None

    def support_radius(self, h: float) -> float:
        return self.kappa * h

    def case_name(self, *, jitter: float | None = None) -> str:
        jitter_value = self.jitter if jitter is None else jitter
        parts = [f"variant_{safe_token(self.variant)}", f"ns{self.n_side}", f"k{safe_token(self.kappa)}"]
        if jitter_value is not None:
            parts.append(f"jit{safe_token(jitter_value)}")
        parts.append(f"seed{self.seed}")
        if self.output_tag.strip():
            parts.append(safe_token(self.output_tag.strip()))
        return "_".join(parts)

    def build_case_meta(
        self,
        *,
        h: float,
        support_radius: float,
        n_nodes: int,
        jitter: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dimension": 2,
            "layout": self.layout,
            "variant": self.variant,
            "n_side": self.n_side,
            "n_nodes": n_nodes,
            "h": h,
            "kappa": self.kappa,
            "support_radius": support_radius,
            "seed": self.seed,
            "quadrature_order": self.quadrature_order,
        }
        jitter_value = self.jitter if jitter is None else jitter
        if jitter_value is not None:
            payload["jitter"] = jitter_value
        return payload

    def to_config_payload(self, *, device: str, case_name: str, support_radius: float, jitter: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group": self.group,
            "case_name": case_name,
            "layout": self.layout,
            "n_side": self.n_side,
            "kappa": self.kappa,
            "support_radius": support_radius,
            "variant": self.variant,
            "phase_a_steps": self.phase_a_steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "grid_resolution": self.grid_resolution,
            "quadrature_order": self.quadrature_order,
            "seed": self.seed,
            "log_interval": self.log_interval,
            "output_tag": self.output_tag,
            "device": device,
        }
        jitter_value = self.jitter if jitter is None else jitter
        if jitter_value is not None:
            payload["jitter"] = jitter_value
        return payload


@dataclass(frozen=True)
class ShapeValidation2DIrregularSweepConfig:
    run_config: ShapeValidation2DRunConfig
    jitters: tuple[float, ...]

    def case_config_for_jitter(self, jitter: float) -> ShapeValidation2DRunConfig:
        return replace(self.run_config, jitter=jitter)

    def case_name(self, *, jitter: float) -> str:
        return self.case_config_for_jitter(jitter).case_name(jitter=jitter)


def get_shape_training_variant_2d(name: str) -> ShapeTrainingVariant2D:
    key = name.strip().lower()
    if key not in SHAPE_TRAINING_VARIANTS_2D:
        raise ValueError(f"Unknown variant: {name}")
    return SHAPE_TRAINING_VARIANTS_2D[key]


def resolve_variant_config(name: str) -> dict[str, Any]:
    variant = get_shape_training_variant_2d(name)
    return {
        "use_softplus": variant.use_softplus,
        "enable_fallback": variant.enable_fallback,
        "use_linear_loss": variant.use_linear_loss,
        "use_pu_loss": variant.use_pu_loss,
        "use_bd_loss": variant.use_bd_loss,
        "use_teacher_loss": variant.use_teacher_loss,
        "lambda_pu": variant.lambda_pu,
        "lambda_bd": variant.lambda_bd,
        "lambda_teacher": variant.lambda_teacher,
        "lambda_reg": variant.lambda_reg,
    }


def build_uniform_shape_validation_2d_config(args: Any) -> ShapeValidation2DRunConfig:
    return ShapeValidation2DRunConfig(
        group="uniform_nodes",
        layout="uniform",
        n_side=args.n_side,
        kappa=args.kappa,
        variant=args.variant,
        phase_a_steps=args.phase_a_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        grid_resolution=args.grid_resolution,
        quadrature_order=args.quadrature_order,
        seed=args.seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )


def build_irregular_shape_validation_2d_sweep_config(args: Any) -> ShapeValidation2DIrregularSweepConfig:
    run_config = ShapeValidation2DRunConfig(
        group="irregular_nodes",
        layout="irregular",
        n_side=args.n_side,
        kappa=args.kappa,
        variant=args.variant,
        phase_a_steps=args.phase_a_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        grid_resolution=args.grid_resolution,
        quadrature_order=args.quadrature_order,
        seed=args.seed,
        log_interval=args.log_interval,
        output_tag=args.output_tag,
    )
    return ShapeValidation2DIrregularSweepConfig(
        run_config=run_config,
        jitters=tuple(parse_float_list(args.jitters)),
    )
