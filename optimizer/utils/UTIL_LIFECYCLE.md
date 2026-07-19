---
title: UTIL_LIFECYCLE
type: lifecycle
module: optimizer/utils
tags:
  - optimizer
  - utils
  - lifecycle
---

# Util Lifecycle

Utilities are most useful as a debugging and repair workflow around optimizer runs.

## Normal Debug Order

```text
controls
  -> metric_report
  -> diagnostic_report
  -> verify_gradient
  -> verify_jacobian or finite_difference_jacobian
  -> geometry_probe
  -> project_gradient or repair_newton
  -> control_spectrum and smoothness_report
```

## Step 1: Metrics

Use `metric_report` when the only question is:

```text
what metrics does this system return at these controls?
```

Use `diagnostic_report` when the question is broader:

```text
are controls finite?
is the gradient finite?
are residuals available?
how large are all norms?
```

## Step 2: Gradient Trust

Run `verify_gradient` before blaming an optimizer.

This catches:

```text
sign mistakes
missing dt factors
wrong channel ordering
wrong objective metric name
scale errors in analytical gradients
```

## Step 3: Residual Jacobian Trust

Use `verify_jacobian` when a system exposes analytical Jacobians.

Use `finite_difference_jacobian` on small systems when the analytical hook is missing
or suspect.

## Step 4: Local Geometry

Use `geometry_probe` to inspect:

```text
residual norm
Jacobian rank
nullspace dimension
condition number
local repair feasibility
```

## Step 5: Projection or Repair

Use `project_gradient` when the goal is to keep a descent direction from changing
hard residuals to first order.

Use `repair_newton` when the current controls already violate residuals and the goal
is to reduce those residuals directly.

## Step 6: Shape Diagnostics

Use `control_spectrum` and `smoothness_report` when controls are numerically good but
physically ugly:

```text
high-frequency artifacts
large jumps
excessive roughness
channel imbalance
```

## Related Notes

- [Diagnostics](UTIL_DIAGNOSTICS.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md)
- [Repair](UTIL_REPAIR.md)
- [Spectrum and Smoothness](UTIL_SPECTRUM_SMOOTHNESS.md)
