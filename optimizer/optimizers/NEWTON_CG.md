---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Contract and use cases for matrix-free Newton-CG optimization."
read_when: "An OLGS system supplies analytical Hessian-vector products."
status: active
reviewed: 2026-10-12
---
# Newton-CG

`opt.newton_cg` approximately solves the Newton system with conjugate gradients and
an analytical `hvp(controls, vector)` hook. It is suited to smooth deterministic
objectives whose full Hessian would be too large to form.

Main controls are `gradient_tolerance`, `cg_tolerance`, `cg_maxiter`, `damping`,
`curvature_tolerance`, `max_step_norm`, and `reject_shrink`.

```python
result = opt.newton_cg(system, controls, maxiter=20)
```

No finite-difference HVP is selected implicitly. Use `opt.curvature_probe` when
deciding whether a second-order method is appropriate. Exact defaults are in the
[generated API reference](../../API_REFERENCE.md).

## Connections

- Related: [Optimizer Hub](OPTIMIZERS_HUB.md)
- Related: [Trust Region](TRUST_REGION.md), [Numerical capabilities](NUMERICAL_CAPABILITIES.md)

---

[⌂ Home](OPTIMIZERS_HUB.md)
