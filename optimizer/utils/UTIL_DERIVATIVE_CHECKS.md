---
role: topic
parent: UTILS_HUB.md
summary: "Gradient and Jacobian checks against finite differences."
read_when: "Read before trusting an optimizer on a new system."
status: active
reviewed: 2026-10-12
---
# Derivative Checks

Derivative utilities compare analytical hooks against finite-difference behavior.

Use them before trusting optimizer behavior on a new system.

## Public Calls

```python
check = opt.utils.verify_gradient(system, controls, eps=1.0e-6, directions=8)
jcheck = opt.utils.verify_jacobian(system, controls, eps=1.0e-6, directions=8)
```

For small systems:

```python
fd_grad = opt.utils.finite_difference_gradient(system, controls, metric="J")
jac = opt.utils.finite_difference_jacobian(system, controls, residuals="hard")
```

## Gradient Tools

`verify_gradient` performs directional checks:

```text
analytical directional derivative
finite-difference directional derivative
absolute error
relative error
passed
```

`finite_difference_gradient` computes every coordinate derivative and returns a
`Controls` object in the same layout.

## Jacobian Tools

`verify_jacobian` checks residual directional derivatives.

`finite_difference_jacobian` returns:

```text
shape = (n_residuals, controls.spec.size)
```

## Best For

```text
new system validation
debugging optimizer failures
finding sign errors
finding dt scaling errors
checking residual Jacobian hooks before repair
```

## Watch Out

```text
full coordinate finite differences are expensive
directional checks are cheaper but not exhaustive
eps that is too small can amplify roundoff
eps that is too large can hide local derivative errors
```

## Related Theory

- [Finite Difference Checks Theory](../../Theory/utils/THEORY_FINITE_DIFFERENCE_CHECKS.md)

## Related Notes

- [Diagnostics](UTIL_DIAGNOSTICS.md)
- [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md)
- [Repair](UTIL_REPAIR.md)
- [Core Evaluation](../core/CORE_EVALUATION.md)
- [Source](./derivatives.py)

---

[⌂ Home](UTILS_HUB.md)
