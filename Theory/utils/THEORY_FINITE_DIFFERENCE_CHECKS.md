---
title: THEORY_FINITE_DIFFERENCE_CHECKS
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_DERIVATIVE_CHECKS.md
tags:
  - optimizer
  - utils
  - theory
  - finite_difference
  - gradient
  - jacobian
---

# Finite Difference Checks Theory

This note gives the mathematical context for:

- [Derivative Checks API](../../optimizer/utils/UTIL_DERIVATIVE_CHECKS.md)

## Scalar Gradient Check

For a scalar metric:

```text
f(u) = J(u)
```

the analytical gradient should satisfy:

```text
D f(u)[d] = grad f(u) dot d
```

for any direction `d` in flattened control space.

## Central Directional Difference

The central finite-difference directional derivative is:

```text
f(u + eps d) - f(u - eps d)
--------------------------------  approx  grad f(u) dot d
             2 eps
```

This is what `verify_gradient` compares against the analytical gradient.

## Coordinate Gradient

The full finite-difference gradient checks each coordinate direction `e_k`:

```text
partial f / partial u_k
  approx
f(u + eps e_k) - f(u - eps e_k)
--------------------------------
             2 eps
```

This costs roughly:

```text
2 * controls.spec.size
```

system evaluations for the central method.

## Residual Jacobian Check

For residuals:

```text
r(u) in R^p
J_r(u) in R^(p x n)
```

the directional derivative is:

```text
r(u + eps d) - r(u - eps d)
--------------------------------  approx  J_r(u) d
             2 eps
```

`verify_jacobian` compares this finite-difference vector with the analytical
Jacobian-vector product.

## Error Interpretation

Absolute error:

```text
||analytical - finite_difference||
```

Relative error:

```text
||analytical - finite_difference||
----------------------------------
max(||analytical||, ||finite_difference||, atol)
```

Use both because near-zero derivatives can make relative error unstable.

## Choosing `eps`

Too small:

```text
roundoff dominates
```

Too large:

```text
nonlinear curvature hides local derivative errors
```

For many smooth double-precision problems, `1.0e-6` is a practical starting point.

## Illustration

```text
u - eps d      u       u + eps d
    |----------|----------|
       evaluate both sides
       compare slope with analytical derivative
```

## API Reference

- [Derivative Checks](../../optimizer/utils/UTIL_DERIVATIVE_CHECKS.md)
- [Diagnostics](../../optimizer/utils/UTIL_DIAGNOSTICS.md)
