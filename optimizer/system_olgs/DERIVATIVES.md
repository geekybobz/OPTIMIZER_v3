---
title: OLGS Derivatives
type: derivative_policy
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - derivatives
depends_on:
  - README
  - CONTRACT
  - LIFECYCLE
connects_to:
  - LOGGING_BOUNDARY
  - OLGS_SYSTEM_BUILD_SKILL
---

# OLGS Derivatives

Gradient is required. Other derivative tools are optional.

## Required Gradient

Every OLGS system must define:

```text
gradient(controls) -> Controls
```

This is expected to be analytical.

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

- [[README|system_olgs]]
- [[OLGS_SYSTEM_BUILD_SKILL|OLGS System Build Skill]]
- [[CONTRACT|API Contract]]
- [[LIFECYCLE|Computation Lifecycle]]
- [[LOGGING_BOUNDARY|Logging Boundary]]
