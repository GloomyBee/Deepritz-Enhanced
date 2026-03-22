# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Scope

This repository is no longer a general PDE playground. The active research scope is limited to two mainlines:

- `shape_validation`: validate the structural quality of learned meshfree shape functions.
- `trial_space_value`: evaluate the numerical value of learned basis functions as trial spaces for Poisson problems.

Treat this as a lightweight paper-oriented research codebase, not a heavy reusable framework.

## Canonical Layout

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

This is the only active development surface. New experiment entrypoints should be added here and organized by:

- research line
- dimension
- leaf experiment group

### `core/`

`core/` is intentionally thin. Stable shared helpers belong here only after they are clearly reused by multiple mainlines.

Current active boundaries:

- `core/io_or_artifacts.py`
- `core/plotting.py`
- `core/splines.py`
- `core/numerics/quadrature.py`
- `core/utils.py`

Historical root-style modules still exist:

- `core/problems.py`
- `core/samplers.py`
- `core/integrators.py`
- `core/networks.py`
- `core/trainers.py`
- `core/visualizers.py`

Do not treat those as the preferred place for new mainline logic. Prefer keeping research-specific logic inside the corresponding experiment tree unless it has a proven stable shared boundary.

### `archive/`

`archive/` means reference-only, non-mainline, and not the default place to extend. Old validation branches, old mixed compare scripts, and baselines live there.

Do not restore archived scripts to the main README or default smoke workflow unless explicitly requested.

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

The old mixed 2D compare entry no longer belongs to the active mainline. It lives under `archive/trial_space_value/two_d/poisson_compare/`.

## Output Contract

Every active case should write to:

- `output/shape_validation/one_d/<group>/<case_name>/`
- `output/shape_validation/two_d/<group>/<case_name>/`
- `output/trial_space_value/one_d/<group>/<case_name>/`
- `output/trial_space_value/two_d/<group>/<case_name>/`

Each case directory must keep this protocol stable:

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/main_figure.png`
- `figures/diagnostics/*`

For `trial_space_value`, the canonical metrics payload should keep:

- top-level `case`
- top-level `method`
- top-level `basis_quality`
- top-level `trial_space`

For `shape_validation`, keep the learned vs RKPM comparison semantics stable. Avoid ad-hoc flat metric renames that break downstream plotting.

## Research Semantics

### `shape_validation`

This mainline is about basis/shape-function quality, not PDE solve quality.

Keep the 1D/2D main figure protocol aligned:

- representative shape functions
- shape or consistency error
- main indicators
- training history

2D validation should not drift back into using Poisson solution plots as the main figure narrative.

### `trial_space_value`

This mainline is about the value of the learned basis as a trial space.

Method semantics must stay explicit:

- `classical`: learned basis fixed, coefficients solved by classical assembly
- `frozen_w`: learned basis fixed, optimize only `w`
- `joint`: Phase B jointly updates basis and coefficients

Do not blur these three into one generic “Poisson compare” narrative.

## Recommended Verification

Before closing substantial changes, prefer:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

And at least one 1D plus one 2D smoke from the active mainline.

## Documentation Expectations

When updating structure, commands, or entrypoints, also update:

- `README.md`
- `CLAUDE.md`
- `docs/experiment-matrix.md`
- `docs/result-snapshot.md`

The root documentation should always answer these questions for a newcomer:

- What are the two active research lines?
- Which script should I run first?
- Where do outputs appear?
- Which archived directories are no longer primary?
