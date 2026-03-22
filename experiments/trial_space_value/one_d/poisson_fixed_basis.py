from __future__ import annotations

from experiments.trial_space_value.one_d.plotting import (
    plot_diagnostic_shape_consistency_1d,
    plot_diagnostic_shape_overlay_1d,
    plot_main_figure_poisson_1d,
)
from experiments.trial_space_value.one_d.trial_space import (
    build_poisson_summary_lines_1d,
    poisson_exact_gradient_1d,
    poisson_exact_solution_1d,
    poisson_source_term_1d,
    solve_fixed_basis_poisson_1d,
)

__all__ = [
    "build_poisson_summary_lines_1d",
    "plot_diagnostic_shape_consistency_1d",
    "plot_diagnostic_shape_overlay_1d",
    "plot_main_figure_poisson_1d",
    "poisson_exact_gradient_1d",
    "poisson_exact_solution_1d",
    "poisson_source_term_1d",
    "solve_fixed_basis_poisson_1d",
]
