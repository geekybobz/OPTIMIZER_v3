---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Optional system capabilities that unlock projected, second-order, and proximal methods."
read_when: "Read when a method needs a hook beyond the five required OLGS calls."
status: active
reviewed: 2026-10-12
---
# Conditional Numerical Capabilities

The five required OLGS hooks remain:

```text
system_spec, control_spec, evaluate, gradient, with_secondary
```

Advanced methods validate only the additional structural capability they consume.
`opt.system_info(system)["capabilities"]` reports compatibility without numerical
work.

| Public call | Conditional system hooks | Stationarity evidence |
|---|---|---|
| `opt.value_and_gradient` | `value_and_gradient` | ordinary gradient norm |
| `opt.projected_gradient` | `residuals+jacobian`, `control_bounds`, or both | feasibility plus projected-KKT norm |
| `opt.newton_cg` | `hvp` | gradient and inner-CG residual norms |
| `opt.trust_region` | `hvp` | gradient norm plus actual/predicted reduction ratio |
| `opt.proximal_gradient` | `smooth_gradient`, `prox` | proximal-gradient mapping norm |
| `opt.continuation` | existing `with_secondary`; explicit transfer if layout/dt changes | declared per-stage gate |

## Projected Flow

For equality residuals `r(u)=0` with Jacobian `A=dr/du`, the tangent gradient is

```text
g_t = g - A.T solve(A A.T + damping I, A g).
```

An infeasible input first enters restoration. A feasible objective trial is clipped
to declared bounds, repaired against the equality residual, then accepted only when
the objective rule and feasibility tolerance both pass. Pure box constraints do not
need artificial zero residuals.

## Matrix-Free Second Order

`newton_cg` approximately solves `H p = -g` using only `system.hvp(u, v)`.
`trust_region` uses Steihaug CG inside `||p|| <= radius` and updates the radius from

```text
rho = actual objective reduction / predicted quadratic reduction.
```

Neither method silently activates finite-difference HVPs.

## Proximal Flow

For `J(u)=f(u)+R(u)`, the system returns total `J` from `evaluate`, `grad f` from
`smooth_gradient`, and applies

```text
prox_{alpha R}(u - alpha grad f)
```

through `prox`. Convergence uses the proximal-gradient mapping, not the ordinary
smooth-gradient norm.

## Continuation Boundary

Each stage builds a new system with `with_secondary` or an explicit system factory,
copies or explicitly transfers controls, discards optimizer-private memory,
re-evaluates the changed objective, runs one bounded optimizer segment, and checks
the declared stage gate. `ContinuationResult.to_dict()` preserves the plan id,
stage results, transfer name, gate evidence, and parent checkpoint ids.

## Evidence Boundary

All residual, Jacobian, HVP, smooth-gradient, prox, and bound calls go through
`SystemEvaluator`. Its compact counter snapshot retains the original evaluation and
gradient keys and adds only nonzero capability counters. Observers receive these
already-computed facts; tracking performs no extra numerical calls.

---

[⌂ Home](OPTIMIZERS_HUB.md)
