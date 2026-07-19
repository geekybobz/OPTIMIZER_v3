---
title: UTIL_CONTRACT
type: contract
module: optimizer/utils
tags:
  - optimizer
  - utils
  - contract
---

# Util Contract

Utilities share the same `Controls` and system contract as optimizers, but they do
not run objective-improvement loops.

## Required Controls Contract

Every system-aware utility expects `controls` to match the target system:

```text
controls.spec.keys
controls.spec.control_dim
controls.spec.size
controls.spec.dtype
```

Most utilities call `validate_controls_for_system(system, controls)` before doing
work.

## Required System Hooks

Metric utilities require:

```text
evaluate(controls) -> metrics dict
```

Gradient verification requires:

```text
gradient(controls) -> Controls
```

Residual geometry and repair require:

```text
residuals(controls, name=...) -> np.ndarray
```

Analytical Jacobian support is optional when finite-difference fallback is allowed:

```text
jacobian(controls, name=...) -> np.ndarray
```

## Scalar Metric Rule

Finite-difference gradient checks target one scalar metric:

```text
metric = "J"
```

The selected metric must be present, scalar, finite, and convertible to `float`.

## Residual Jacobian Shape

Residual Jacobians use this shape:

```text
(n_residuals, controls.spec.size)
```

Rows correspond to residual equations. Columns correspond to flattened control
values.

## Public Return Contract

Utilities return one of:

```text
dict
Controls
np.ndarray
RepairResult
```

Report dictionaries are intended to be JSON-friendly for notebooks, logs, and
Obsidian snippets.

## Failure Contract

Utilities should fail clearly when:

```text
controls do not match the system
eps is nonpositive or nonfinite
metric is missing, vector-valued, or nonfinite
residual hooks are missing and fallback is disabled
jacobian shapes are incompatible
projection gradient layout differs from controls layout
```

## Related Notes

- [Utils Hub](UTILS_HUB.md)
- [Util Common API](UTIL_COMMON_API.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md)
- [System Contract](../system_olgs/README.md)
