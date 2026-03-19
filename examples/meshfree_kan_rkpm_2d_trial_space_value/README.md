# meshfree_kan_rkpm_2d_trial_space_value

This package evaluates the standalone numerical value of learned shape functions in 2D.

Current experiment groups:

- `poisson_compare/`: compare fixed-basis classical solve, frozen-`w` optimization, and joint Phase B on the same sinusoidal Poisson problem.

Outputs are written to `output/meshfree_kan_rkpm_2d_trial_space_value/`.

Case-level figure layout now follows the mainline paper-style convention:

- root case comparison: `figures/main_figure.png` and optional `figures/summary_figure.png`
- root diagnostics: `figures/diagnostics/`
- each method case: `methods/<method>/figures/main_figure.png`
- each method diagnostics: `methods/<method>/figures/diagnostics/`

Each run still stores:

- `config.json`
- `comparison_metrics.json`
- `comparison_summary.txt`
- `methods/<method>/metrics.json`
- `methods/<method>/curves.npz`
- `methods/<method>/summary.txt`
