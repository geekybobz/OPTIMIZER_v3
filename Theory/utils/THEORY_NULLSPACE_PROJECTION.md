---
title: THEORY_NULLSPACE_PROJECTION
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md
tags:
  - optimizer
  - utils
  - theory
  - projection
  - nullspace
  - residual
---

# Nullspace Projection Theory

This note gives the mathematical context for:

- [Residual Geometry API](../../optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md)

## Constraint-Preserving Directions

For residuals `r(u)`, first-order preservation means:

```text
r(u + delta) approx r(u)
```

Using the local linearization:

```text
r(u + delta) approx r(u) + J delta
```

so first-order residual-preserving directions satisfy:

```text
J delta = 0
```

These directions live in the nullspace of `J`.

## Gradient Decomposition

A gradient-like vector `g` can be decomposed into:

```text
g = g_null + g_row
```

where:

```text
J g_null approx 0
g_row lies in the row space of J
```

`project_gradient` removes `g_row` and returns `g_null`.

## Residual-Space Solve

The projection implemented in utils solves:

```text
(J J^T + damping I) lambda = J g
```

then computes:

```text
g_row = J^T lambda
g_projected = g - g_row
```

This avoids solving directly in high-dimensional control space.

## Why Use `J J^T`

Control problems often have:

```text
n_control_values >> n_residuals
```

Solving in residual space is smaller and naturally handles many-control,
few-residual systems.

## Illustration

```text
raw gradient g
   |
   | split by local residual geometry
   v
g_row changes residuals        J g_row != 0
g_null preserves residuals     J g_null approx 0

project_gradient returns g_null
```

## Practical Meaning

If a later optimizer steps along:

```text
-g_projected
```

the first-order residual change should be smaller than stepping along the raw
gradient.

Projection does not choose a step size and does not guarantee nonlinear feasibility
after a finite step.

## API Reference

- [Residual Geometry](../../optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md)
- [Residual Jacobian Geometry Theory](THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md)
