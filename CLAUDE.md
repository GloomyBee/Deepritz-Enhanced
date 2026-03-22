# CLAUDE.md

This file provides guidance to Claude Code style agents working in this repository.

## Repository Identity

This is a lightweight research repository centered on two active lines:

- `shape_validation`
- `trial_space_value`

The repository should serve the paper workflow first. Avoid turning it into a large generic framework.

## Real Structure

```text
core/
experiments/
  shape_validation/
    one_d/
    two_d/
  trial_space_value/
    one_d/
    two_d/
archive/
docs/
output/
tests/
```

### `experiments/`

All active work belongs here. Organize by research line, then dimension, then leaf experiment group.

### `core/`

`core/` is the thin shared layer. Prefer adding code here only when it is already stable across multiple experiment families.

Current preferred shared boundary:

- artifacts and run-bundle helpers
- plotting helpers
- spline basis helpers
- numerical quadrature
- seeds and parse helpers

Historical modules remain in `core/`, but they are not the preferred extension point for new mainline logic:

- `problems.py`
- `samplers.py`
- `integrators.py`
- `networks.py`
- `trainers.py`
- `visualizers.py`

### `archive/`

`archive/` is reference-only. Keep old or low-activity material there and do not re-promote it without an explicit reason.

## Active Entrypoints

### Shape Validation

- `experiments/shape_validation/one_d/uniform_nodes/deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py`
- `experiments/shape_validation/one_d/nonuniform_nodes/deepritz_meshfree_kan_rkpm_1d_nonuniform_nodes_v1.py`
- `experiments/shape_validation/two_d/uniform_nodes/deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py`
- `experiments/shape_validation/two_d/irregular_nodes/deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py`

### Trial Space Value

- `experiments/trial_space_value/one_d/poisson_compare/deepritz_meshfree_kan_rkpm_1d_poisson_compare_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_frozen_w_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_joint_v1.py`

The old all-in-one 2D compare entry is archived and should stay out of the default workflow.

## Output Protocol

Each active case should emit:

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/main_figure.png`
- `figures/diagnostics/*`

Canonical output locations:

- `output/shape_validation/one_d/<group>/<case_name>/`
- `output/shape_validation/two_d/<group>/<case_name>/`
- `output/trial_space_value/one_d/<group>/<case_name>/`
- `output/trial_space_value/two_d/<group>/<case_name>/`

## Semantic Guardrails

### Shape Validation

Focus on shape-function properties:

- partition of unity residual
- linear reproduction residual
- `Lambda_h`
- learned vs RKPM discrepancy

Do not let 2D validation drift back into a PDE-solve-first narrative.

### Trial Space Value

Keep method semantics explicit:

- `classical`
- `frozen_w`
- `joint`

Do not merge them back into a single mixed runtime entry.

## Validation Workflow

Prefer this minimum verification after substantial refactors:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Then run at least:

- one 1D active smoke
- one 2D active smoke

## Documentation Sync

When commands, entrypoints, or structure change, update:

- `README.md`
- `AGENTS.md`
- `docs/experiment-matrix.md`
- `docs/result-snapshot.md`

The documentation should always make it obvious how a new reader can:

- install dependencies
- run a 1D smoke
- run a 2D smoke
- inspect `summary.txt` and `figures/main_figure.png`
- continue to irregular sweeps or Poisson compare runs
