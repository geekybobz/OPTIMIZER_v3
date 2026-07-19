---
title: UTIL_COMMON_API
type: common_api
module: optimizer/utils
tags:
  - optimizer
  - utils
  - common_api
---

# Util Common API

The utils package has several categories, but most system-aware calls share the same
small argument vocabulary.

## Core Inputs

```text
system
controls
```

`system` provides metrics, gradients, residuals, or Jacobians.

`controls` is the `Controls` object being inspected, verified, projected, or
repaired.

## Residual Selection

Residual-aware methods use:

```text
residuals="hard"
```

The value names the residual hook requested from the system.

## Finite-Difference Scale

Derivative and Jacobian fallback methods use:

```text
eps=1.0e-6
```

The scale must be finite and positive.

## Finite-Difference Method

Full coordinate finite differences support:

```text
central
forward
```

`central` costs more evaluations but is usually more accurate.

## Directional Checks

Gradient and Jacobian verification use:

```text
directions=8
seed=12345
rtol=1.0e-4
atol=1.0e-7
```

`directions` can be an integer number of random directions or an explicit direction
matrix in flattened control space.

## Jacobian Fallback

Residual geometry, projection, and repair can use:

```text
fallback=True
```

When `fallback=True`, a missing analytical `jacobian(...)` hook is replaced by
`finite_difference_jacobian(...)`.

When `fallback=False`, missing analytical Jacobians are surfaced as errors.

## Rank and Damping

Residual geometry and projection use:

```text
rcond
damping
```

`rcond` controls numerical rank decisions.

`damping` regularizes residual-space solves such as `J J.T`.

## Spectrum Inputs

Spectrum diagnostics use:

```text
dt
high_frequency_cutoff
```

If `dt is None`, `controls.spec.dt` is used when available. Otherwise the sample
spacing defaults to `1.0`.

## Return Styles

Most reports return dictionaries:

```text
metric_report
diagnostic_report
geometry_probe
verify_gradient
verify_jacobian
control_spectrum
smoothness_report
```

Shape-preserving tools return `Controls`:

```text
finite_difference_gradient
project_gradient
```

Linear algebra tools return arrays:

```text
finite_difference_jacobian
nullspace_basis
```

Repair returns:

```text
RepairResult
```

## Related Notes

- [Util Contract](UTIL_CONTRACT.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md)
- [Repair](UTIL_REPAIR.md)
