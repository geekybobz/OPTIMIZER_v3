---
title: UTILS_THEORY_HUB
type: theory_hub
module: Theory/utils
tags:
  - optimizer
  - utils
  - theory
  - hub
---

# Utils Theory Hub

This folder contains mathematical and research-facing notes for utility methods.

Runtime API documentation lives in:

```text
optimizer/utils/
```

Theory notes live here so equations, geometric meaning, and compact illustrations do
not make the API notes too large.

## Theory Map

```text
THEORY_DIAGNOSTIC_REPORTS.md
  Norms, finite checks, metric summaries, and report interpretation.

THEORY_FINITE_DIFFERENCE_CHECKS.md
  Coordinate and directional finite-difference derivative checks.

THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md
  Local residual linearization, rank, singular values, and conditioning.

THEORY_NULLSPACE_PROJECTION.md
  Row-space removal and first-order residual-preserving directions.

THEORY_NEWTON_REPAIR.md
  Newton, least-squares, LM/damped residual repair, and line search.

THEORY_SPECTRUM_SMOOTHNESS.md
  FFT power, high-frequency fraction, total variation, and roughness.
```

## Reading Rule

Use API docs when coding:

- [Util Methods](../../optimizer/utils/UTIL_METHODS.md)
- [Util Common API](../../optimizer/utils/UTIL_COMMON_API.md)

Use theory notes when reasoning about:

```text
finite-difference accuracy
Jacobian rank
nullspace directions
repair feasibility
conditioning
frequency content
roughness
```

## Graph Convention

Each theory note links back to the API category reference.

Each API category reference links forward to the relevant theory note.

This gives two-way traversal:

```text
optimizer/utils/UTIL_REPAIR.md
  -> Theory/utils/THEORY_NEWTON_REPAIR.md
      -> optimizer/utils/UTIL_REPAIR.md
```

## Related Notes

- [Utils Documentation Hub](../../optimizer/utils/UTILS_HUB.md)
- [Util Lifecycle](../../optimizer/utils/UTIL_LIFECYCLE.md)
- [Util Boundary](../../optimizer/utils/UTIL_BOUNDARY.md)
