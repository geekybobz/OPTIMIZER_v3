---
title: THEORY_NEWTON_REPAIR
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_REPAIR.md
tags:
  - optimizer
  - utils
  - theory
  - repair
  - newton
  - lm
---

# Newton Repair Theory

This note gives the mathematical context for:

- [Repair API](../../optimizer/utils/UTIL_REPAIR.md)

## Repair Objective

Repair tries to reduce a named residual vector:

```text
r(u)
```

It is not trying to minimize the main objective `J(u)` directly.

## Local Linear Model

Near current controls:

```text
r(u + delta) approx r(u) + J delta
```

To reduce residuals, solve:

```text
J delta approx -r(u)
```

then update:

```text
u_new = u + alpha delta
```

where `alpha` is chosen by line search.

## Newton / Least-Squares Step

The direct mode solves:

```text
min_delta ||J delta + r||_2
```

When there are more control variables than residual equations, the least-squares
solution can choose a small-norm repair step among many possible steps.

## LM / Damped Step

The damped residual-space form solves:

```text
(J J^T + lambda I) y = -r
delta = J^T y
```

Here `lambda` is the damping value.

Damping reduces sensitivity when `J J^T` is ill-conditioned.

## Line Search

The repair line search tests:

```text
u + alpha delta
```

with shrinking values:

```text
alpha = 1, shrink, shrink^2, ...
```

A step is accepted only when:

```text
||r(u + alpha delta)||_2 < ||r(u)||_2
```

## Stop Reasons

Typical stop reasons:

```text
converged
maxiter
line_search_failed
```

`line_search_failed` means the local linear repair direction did not produce a lower
residual norm for the attempted step sizes.

## Illustration

```text
current u
  -> residual r(u)
  -> Jacobian J
  -> solve J delta approx -r
  -> try u + alpha delta
  -> accept only if residual norm decreases
```

## Practical Use

Use repair after:

```text
an optimizer improves J but violates hard residuals
a candidate is close to feasible
geometry_probe shows useful Jacobian rank
verify_jacobian has passed on the system
```

## API Reference

- [Repair](../../optimizer/utils/UTIL_REPAIR.md)
- [Residual Geometry](../../optimizer/utils/UTIL_RESIDUAL_GEOMETRY.md)
- [Residual Jacobian Geometry Theory](THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md)
