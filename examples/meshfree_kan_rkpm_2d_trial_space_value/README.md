# meshfree_kan_rkpm_2d_trial_space_value

This package evaluates the standalone numerical value of learned shape
functions in 2D.

Current experiment groups:

- `poisson_compare/`: compare fixed-basis classical solve, frozen-`w`
  optimization, and joint Phase B on the same sinusoidal Poisson problem.

Outputs are written to:

- `output/meshfree_kan_rkpm_2d_trial_space_value/`

Each run stores:

- `config.json`
- `comparison_metrics.json`
- `comparison_summary.txt`
- `figures/*.png`
- `methods/<method>/metrics.json`
- `methods/<method>/curves.npz`
- `methods/<method>/summary.txt`
- `methods/<method>/figures/*.png`
