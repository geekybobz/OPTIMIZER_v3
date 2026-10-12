---
role: topic
parent: SYSTEM_OLGS_HUB.md
summary: "The required analytical gradient, optional derivative hooks, and the numerical-fallback policy."
read_when: "Read when implementing derivatives for an OLGS system."
status: active
reviewed: 2026-10-12
---
# OLGS Derivatives

Gradient is required. Other derivative tools are optional.

## Required Gradient

Every OLGS system must define:

```text
gradient(controls) -> Controls
```

This is expected to be analytical.

The mandatory `system_spec()` must declare the gradient target, analytical source,
`dJ/du` convention, and an optional derivation reference. Runtime checks validate
shape and finiteness; explicit numerical comparisons test the analytical formula.

The optimizer should not silently replace missing gradient logic with numerical
finite differences during normal optimization.

## Optional Derivatives

Systems may define:

```text
jacobian(controls, name="hard")
hessian(controls)
hvp(controls, vector)
second_derivative(controls, ...)
```

These are useful for:

```text
repair tools
projected gradients
Newton-like methods
trust-region methods
diagnostics
local geometry checks
```

`newton_cg` and `trust_region` require an analytical `hvp` capability. They do
not silently select `finite_difference_hvp`, because one finite-difference HVP
costs multiple gradient calls and an inner CG solve may request many HVPs.

`projected_gradient` likewise requires an analytical `jacobian` capability in
normal optimization. Explicit finite-difference tools remain available for
small-system checks and diagnostics.

## Numerical Fallback Policy

If an optional derivative is missing, OLGS may provide numerical approximations in:

```text
optimizer/system_olgs/derivatives.py
```

Possible tools:

```text
finite_difference_jacobian
finite_difference_hessian
finite_difference_hvp
directional_derivative
complex_step_derivative where mathematically valid
gradient_check
jacobian_check
```

Fallbacks should be explicit.

Good:

```python
J = olgs_derivatives.finite_difference_jacobian(system, controls)
```

Avoid:

```text
optimizer silently replacing missing analytical gradient
```

## Shape Rules

Jacobian:

```text
shape = (n_residuals, controls.spec.size)
```

Hessian:

```text
shape = (controls.spec.size, controls.spec.size)
```

HVP:

```text
input vector shape = controls shape or flat controls size
output shape = controls shape or flat controls size, defined by API
```

The exact HVP return convention should be documented by the system that exposes it.

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [API Contract](CONTRACT.md)
- [Computation Lifecycle](LIFECYCLE.md)
- [Logging Boundary](LOGGING_BOUNDARY.md)

---

[⌂ Home](SYSTEM_OLGS_HUB.md)
