---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Contract and use cases for equality- and bound-constrained projected gradient."
read_when: "Optimizing an OLGS system with hard equality constraints or box bounds."
status: active
reviewed: 2026-10-12
---
# Projected Gradient

`opt.projected_gradient` alternates feasibility restoration with a gradient step
projected into the constraint tangent space. It stops using feasibility and
projected-KKT stationarity tests.

## Requires

`evaluate`, `gradient`, `residuals`, and `jacobian`; `control_bounds` is optional.
The method never invents a finite-difference Jacobian. General nonlinear inequality
constraints are outside the current contract.

## Main Controls

`step_size`, `residual_name`, `feasibility_tolerance`,
`stationarity_tolerance`, `damping`, `restoration_maxiter`, `max_backtracks`,
`shrink`, and `max_step_norm`.

```python
result = opt.projected_gradient(system, controls, maxiter=50)
```

For exact defaults and return fields, see the [generated API reference](../../API_REFERENCE.md).

## Connections

- Related: [Optimizer Hub](OPTIMIZERS_HUB.md)
- Related: [Numerical capabilities](NUMERICAL_CAPABILITIES.md), [Repair tools](../utils/UTIL_REPAIR.md)

---

[⌂ Home](OPTIMIZERS_HUB.md)
