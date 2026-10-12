---
role: topic
parent: SYSTEM_OLGS_HUB.md
summary: "Normative OLGS request, response, invariant, and optional capability contract."
read_when: "Implementing a system class or integrating an optimizer with OLGS."
status: active
reviewed: 2026-10-12
---
# OLGS Contract

Any object implementing the five required calls satisfies OLGS. Subclassing
`optimizer.system_olgs.OLGS` is optional; `require_system` validates the structural
contract and the declared specification.

## Required Transactions

| Call | Request | Response | Invariants |
|---|---|---|---|
| `system_spec()` | none | validated `optimizer.system-spec/v1` mapping | static, complete, no propagation |
| `control_spec()` | none | `ControlSpec` | canonical shape is `(n_controls, control_dim)` |
| `evaluate(controls)` | compatible `Controls` | metrics mapping with finite scalar `J` | every returned metric is declared in `system_spec` |
| `gradient(controls)` | compatible `Controls` | analytical gradient as `Controls` | same keys and shape; finite values |
| `with_secondary(**updates)` | declared secondary values | new compatible system | primary model identity and control layout remain compatible |

`system_spec()` is the canonical discovery source. `opt.system_info(system)` combines
that declaration with detected hooks. `describe()` is only a convenience view.

## Optional Capability Bundles

| Capability | Hooks | Library consumer |
|---|---|---|
| Combined evaluation | `value_and_gradient` | ordinary gradient methods |
| Equality constraints | `residuals`, `jacobian` | `projected_gradient`, geometry and repair tools |
| Box bounds | `control_bounds` | shared engine and projected methods |
| Matrix-free curvature | `hvp` | `newton_cg`, `trust_region`, curvature diagnostics |
| Dense curvature | `hessian` | explicit advanced analysis only |
| Composite objective | `smooth_gradient`, `prox` | `proximal_gradient` |
| Visualization data | `plot_data` | explicit plotting tools |
| Cache control | `cache_reset`, `cache_status` | optional system-local reuse |

Missing optional hooks are reported before a dependent method begins. The library
does not silently substitute finite differences for the required analytical
gradient or for an undeclared advanced capability.

## Recommended Lifecycle Calls

`forward_prop`, `back_prop`, `metrics`, and `describe` make project systems easier to
inspect, but optimizers do not require them. `metrics()` must never silently report
values for controls different from the latest valid evaluation.

## Standard Containers and Validators

The package provides `ForwardResult`, `BackwardResult`, `SystemEvaluation`,
`GradientResult`, and `SimulationResult`, plus:

```text
probe_system              structural capability report without evaluation
require_system            required-call and system-spec validation
get_system_spec           validated static declaration
validate_controls_for_system
validate_metrics
basic_system_spec, metric_spec
describe_system, contract_info
```

See the exact runtime signatures in the [generated API reference](../../API_REFERENCE.md).
A complete minimal system is in
[quadratic_system.py](../../tests/fixtures/quadratic_system.py).

## Transaction Rules

OLGS uses in-process typed objects. CLI and MCP adapters should serialize the same
meaning through the shared [transaction model](../TRANSACTIONS.md); they should not
invent a second optimizer contract.

## Connections

- Related: [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- Related: [Parameters](PARAMS.md), [Lifecycle](LIFECYCLE.md), [Derivative capabilities](DERIVATIVES.md), [Transactions](../TRANSACTIONS.md)

---

[← Previous](SYSTEM_OLGS_HUB.md) · [⌂ Home](SYSTEM_OLGS_HUB.md) · [Next →](PARAMS.md)
