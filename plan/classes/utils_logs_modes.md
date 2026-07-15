# Utils, Logs, And Modes

Status: review draft.
Last updated: 2026-07-15.

## Utils

`utils/` holds tools that are useful but are not optimizers.

Important first tools:

```text
verify_derivatives
geometry_probe
repair_newton
project_gradient
control_spectrum
finite_difference_gradient
finite_difference_jacobian
fourier_tools
```

## Diagnostics

Diagnostics measure. They should not move controls unless the function name clearly
says repair/project.

Examples:

```python
diag = opt.geometry_probe(sys, controls)
check = opt.verify_derivatives(sys, controls)
spec = opt.control_spectrum(sys, controls)
```

## Repairs

Repairs move controls to restore feasibility:

```python
fixed = opt.repair_newton(sys, controls, residuals="hard")
```

Repairs are not ordinary descent optimizers.

## Logs

Logs are in `logs/` and should support:

```text
trace rows
checkpoint metadata
technical reports
CSV/JSON export
rollback lookup
```

## Modes

`modes/` remains a placeholder for now.

Later possible modes:

```text
manual
benchmark
curriculum
auto
pipeline
```

Do not build modes before the lower layers are stable.

