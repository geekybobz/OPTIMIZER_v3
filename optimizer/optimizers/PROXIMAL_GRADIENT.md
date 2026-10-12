---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Contract and use cases for composite-objective proximal-gradient optimization."
read_when: "An OLGS system has a smooth term plus a system-owned proximal operator."
status: active
reviewed: 2026-10-12
---
# Proximal Gradient

`opt.proximal_gradient` combines `smooth_gradient(controls)` with
`prox(controls, step_size)`. The system's `evaluate` result must report the total
composite objective `J`, not only its smooth component.

Main controls are `step_size`, `mapping_tolerance`, `max_step_norm`, and
`reject_shrink`.

```python
result = opt.proximal_gradient(system, controls, step_size=0.1, maxiter=50)
```

This method fits L1, total-variation, and other penalties only when the project can
provide the corresponding proximal operation. Exact defaults are in the
[generated API reference](../../API_REFERENCE.md).

## Connections

- Related: [Optimizer Hub](OPTIMIZERS_HUB.md)
- Related: [Numerical capabilities](NUMERICAL_CAPABILITIES.md), [Optimizer contract](OPTIMIZER_CONTRACT.md)

---

[⌂ Home](OPTIMIZERS_HUB.md)
