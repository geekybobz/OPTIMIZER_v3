---
title: UTILS_HUB
type: module_hub
module: optimizer/utils
tags:
  - optimizer
  - utils
  - api
  - hub
---

# Utils Documentation Hub

This folder contains diagnostics, derivative checks, residual geometry, repair, and
control-shape inspection tools for OPTIMIZER v3.

The goal is compact, traversable documentation: start here, then open the category
note needed for the current debugging or analysis task.

## Design Scope

Utility methods inspect, verify, repair, or measure `Controls`.

They do not own:

```text
optimizer update rules
initial-control generation
physical objective derivation
system propagation
step-size schedules
deep mathematical theory notes
```

## Documentation Map

Read in this order:

```text
UTILS_HUB.md
  Entry point and graph map.

UTIL_CONTRACT.md
  Public utility API contract and required system hooks.

UTIL_COMMON_API.md
  Shared arguments, finite-difference options, and return styles.

UTIL_METHODS.md
  Compact table of implemented utility categories and methods.

UTIL_LIFECYCLE.md
  Practical debug and repair workflow.

UTIL_BOUNDARY.md
  What belongs in utils, core, systems, optimizers, guesses, and theory.
```

Category references:

```text
UTIL_DIAGNOSTICS.md
UTIL_DERIVATIVE_CHECKS.md
UTIL_RESIDUAL_GEOMETRY.md
UTIL_REPAIR.md
UTIL_SPECTRUM_SMOOTHNESS.md
```

## Public Entry Points

Namespace style:

```python
report = opt.utils.diagnostic_report(system, controls)
```

Root shortcut style:

```python
check = opt.verify_gradient(system, controls, eps=1.0e-6, directions=8)
```

Alias namespace:

```python
fixed = opt.util.repair_newton(system, controls, residuals="hard")
```

Bound context style:

```python
ctx = opt.context(system)
geometry = ctx.geometry_probe(controls, eps=1.0e-6)
```

## Implementation Files

Current source files:

```text
optimizer/utils/
  __init__.py
  diagnostics.py
  derivatives.py
  geometry.py
  repairs.py
  spectrum.py
```

## Theory Map

Long mathematical notes live outside this runtime package:

```text
Theory/utils/
  UTILS_THEORY_HUB.md
  THEORY_DIAGNOSTIC_REPORTS.md
  THEORY_FINITE_DIFFERENCE_CHECKS.md
  THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md
  THEORY_NULLSPACE_PROJECTION.md
  THEORY_NEWTON_REPAIR.md
  THEORY_SPECTRUM_SMOOTHNESS.md
```

Each category reference links to its corresponding theory note, and each theory note
links back to the API category reference.

## Related Notes

- [Util Contract](UTIL_CONTRACT.md)
- [Util Common API](UTIL_COMMON_API.md)
- [Util Methods](UTIL_METHODS.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Utils Theory Hub](../../Theory/utils/UTILS_THEORY_HUB.md)
