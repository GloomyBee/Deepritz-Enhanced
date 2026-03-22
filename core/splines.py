from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


ArrayLike = np.ndarray | torch.Tensor


def build_open_uniform_knot_vector(
    num_basis: int,
    degree: int = 3,
    grid_range: tuple[float, float] = (-1.5, 1.5),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_basis <= degree:
        raise ValueError(f"num_basis={num_basis} must exceed degree={degree}")
    lower, upper = grid_range
    if upper <= lower:
        raise ValueError(f"grid_range must be increasing, got {grid_range}")

    interior_count = num_basis - degree - 1
    knots: list[float] = [lower] * (degree + 1)
    if interior_count > 0:
        interior = np.linspace(lower, upper, interior_count + 2, dtype=np.float64)[1:-1]
        knots.extend(float(value) for value in interior)
    knots.extend([upper] * (degree + 1))
    return torch.tensor(knots, dtype=dtype, device=device)


def _as_torch_input(x: ArrayLike) -> tuple[torch.Tensor, bool]:
    if isinstance(x, torch.Tensor):
        return x, False
    return torch.as_tensor(x, dtype=torch.float64), True


def evaluate_open_uniform_bspline_basis(
    x: ArrayLike,
    num_basis: int,
    degree: int = 3,
    grid_range: tuple[float, float] = (-1.5, 1.5),
) -> ArrayLike:
    x_tensor, return_numpy = _as_torch_input(x)
    original_shape = x_tensor.shape
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(1)
    if x_tensor.dim() != 2:
        raise ValueError(f"Expected x to have shape [batch, width], got {tuple(original_shape)}")

    knots = build_open_uniform_knot_vector(
        num_basis=num_basis,
        degree=degree,
        grid_range=grid_range,
        dtype=x_tensor.dtype,
        device=x_tensor.device,
    )

    upper = knots[-1]
    eps = max(float(torch.finfo(x_tensor.dtype).eps) * max(abs(float(upper)), 1.0) * 32.0, 1.0e-12)
    x_eval = torch.where(
        torch.isclose(x_tensor, upper, atol=eps, rtol=0.0),
        x_tensor.new_full((), float(upper) - eps),
        x_tensor,
    )

    x_expanded = x_eval.unsqueeze(-1)
    basis = ((x_expanded >= knots[:-1]) & (x_expanded < knots[1:])).to(dtype=x_tensor.dtype)

    for current_degree in range(1, degree + 1):
        next_count = knots.numel() - current_degree - 1
        next_basis = x_tensor.new_zeros((*x_tensor.shape, next_count))
        for index in range(next_count):
            left_denom = float(knots[index + current_degree] - knots[index])
            if left_denom > 0.0:
                next_basis[..., index] = next_basis[..., index] + (
                    (x_eval - knots[index]) / left_denom
                ) * basis[..., index]

            right_denom = float(knots[index + current_degree + 1] - knots[index + 1])
            if right_denom > 0.0:
                next_basis[..., index] = next_basis[..., index] + (
                    (knots[index + current_degree + 1] - x_eval) / right_denom
                ) * basis[..., index + 1]
        basis = next_basis

    if return_numpy:
        return basis.detach().cpu().numpy()
    return basis
