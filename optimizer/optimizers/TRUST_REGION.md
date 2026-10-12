---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Contract and use cases for Steihaug trust-region minimization."
read_when: "A guarded second-order method is needed for an OLGS minimization problem."
status: active
reviewed: 2026-10-12
---
# Trust Region

`opt.trust_region` uses Steihaug conjugate gradients inside a radius and accepts a
trial from the ratio of actual to predicted improvement. It requires `evaluate`,
`gradient`, and analytical `hvp` and currently supports minimization only.

Main controls are `initial_radius`, `max_radius`, `gradient_tolerance`, inner-CG
tolerances, model-ratio thresholds, and radius shrink/grow factors.

```python
result = opt.trust_region(system, controls, initial_radius=1.0, maxiter=20)
```

Exact defaults are in the [generated API reference](../../API_REFERENCE.md).

## Connections

- Related: [Optimizer Hub](OPTIMIZERS_HUB.md)
- Related: [Newton-CG](NEWTON_CG.md), [Numerical capabilities](NUMERICAL_CAPABILITIES.md)

---

[⌂ Home](OPTIMIZERS_HUB.md)
