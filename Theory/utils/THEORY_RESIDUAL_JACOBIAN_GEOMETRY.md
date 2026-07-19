---
title: THEORY_RESIDUAL_JACOBIAN_GEOMETRY
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md
tags:
  - optimizer
  - utils
  - theory
  - residual
  - jacobian
  - geometry
---

# Residual Jacobian Geometry Theory

This note gives the mathematical context for:

- [Residual Geometry API](../../optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md)

## Local Residual Linearization

Hard residuals define a vector function:

```text
r(u) in R^p
```

Near the current controls:

```text
r(u + delta) approx r(u) + J delta
```

where:

```text
J = partial r / partial u
shape(J) = (n_residuals, n_control_values)
```

## Rank

The numerical rank tells how many independent residual directions the Jacobian can
control locally:

```text
rank(J) <= min(n_residuals, n_control_values)
```

If rank is low, some residual combinations may be locally uncontrollable.

## Singular Values

The singular value decomposition is:

```text
J = U Sigma V^T
```

Large singular values indicate residual directions that respond strongly to control
changes.

Small singular values indicate weakly controlled or nearly redundant directions.

## Condition Number

For nonzero singular values:

```text
condition(J) = sigma_max / sigma_min
```

Large condition number means the residual solve can be sensitive to noise,
finite-difference error, or small modeling mistakes.

## Nullspace Dimension

The local nullspace dimension is:

```text
n_control_values - rank(J)
```

Large nullspace dimension means there are many first-order directions that can move
controls without changing residuals.

## Illustration

```text
control space
    |
    | delta
    v
residual space

J maps control movements to first-order residual changes:

delta  ->  J delta
```

## Practical Reading

Use `geometry_probe` before repair or projection to answer:

```text
are residuals large?
is the Jacobian analytical or finite-difference?
is the local rank enough to repair?
is the solve ill-conditioned?
how many residual-preserving directions remain?
```

## API Reference

- [Residual Geometry](../../optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md)
- [Newton Repair Theory](THEORY_NEWTON_REPAIR.md)
- [Nullspace Projection Theory](THEORY_NULLSPACE_PROJECTION.md)
