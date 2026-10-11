---
title: OLGS Contract
type: api_contract
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - api
---

# OLGS Contract

This file defines the intended method surface for an OLGS system.

OLGS should be strict about the optimizer-facing API and flexible about
system-specific physics.

## Required Methods

Every OLGS system must provide:

```text
system_spec()
control_spec()
evaluate(controls)
gradient(controls)
with_secondary(**updates)
```

### `system_spec()`

Returns the complete static `optimizer.system-spec/v1` declaration. It exposes
system identity, controls, parameters, objective direction, every optimizer-facing
metric, analytical derivative provenance, constraints, and declared explicit data
products. It must not propagate dynamics or evaluate controls.

`opt.system_info(system)` combines this semantic declaration with structurally
detected hook availability, so users and tools need one discovery call.

### `control_spec()`

Returns an `optimizer.controls.ControlSpec`.

It defines:

```text
control channel names
control_dim
dt when known
dtype
metadata
```

Canonical control shape:

```text
(n_controls, control_dim)
```

### `evaluate(controls)`

Fast optimizer-facing objective call.

Returns a metrics dict with mandatory finite scalar:

```python
{"J": ...}
```

Other metric names are system-defined.

Every returned key must also be declared under `system_spec()["metrics"]`, including
its label, meaning, role, dtype, shape, unit, and required/optional status.

### `gradient(controls)`

Returns analytical gradient as `Controls`.

Rules:

```text
same control keys as input
same shape as input
finite values
analytical system implementation, not normal numerical fallback
```

### `with_secondary(**updates)`

Returns a new equivalent system with updated secondary parameters.

This is the curriculum hook:

```python
system2 = system.with_secondary(residual_weight=0.5, energy_weight=0.01)
```

## Recommended Methods

OLGS systems should also provide:

```text
forward_prop(controls)
back_prop(controls)
metrics()
describe()
```

### `forward_prop(controls)`

Runs physical forward propagation and stores or returns forward quantities.

Typical outputs:

```text
state trajectory
terminal state
objective ingredients
trajectory diagnostics
plotting quantities
```

### `back_prop(controls)`

Runs adjoint or costate propagation.

It usually depends on a matching forward pass.

### `metrics()`

Returns latest valid metrics.

It should not silently return stale metrics from unrelated controls.

### `describe()`

Returns compact system information:

```text
system name
primary params
secondary params
control spec
objective terms
available optional hooks
```

This is a legacy convenience view. `system_spec()` is the required canonical
machine-readable source.

## Optional Methods

The required five-method core does not grow when advanced numerical methods are
added. Advanced systems opt into structural capability bundles:

```text
value_and_gradient(controls)
residuals(controls, name="hard")
jacobian(controls, name="hard")
control_bounds()
hessian(controls)
hvp(controls, vector)
smooth_gradient(controls)
prox(controls, step_size)
second_derivative(controls, ...)
plot_data(controls, kind=..., **options)
cache_reset()
cache_status()
```

Optional methods should be discoverable and validated by OLGS helpers.
Metric and constraint semantics belong in the mandatory `system_spec()` rather
than separate schema hooks.

| Capability | Required hooks | Consumer |
|---|---|---|
| Combined value/gradient | `value_and_gradient` | All ordinary gradient methods may reuse it at the initial controls |
| Equality constraints | `residuals`, `jacobian` | `projected_gradient` |
| Bounds | `control_bounds` | `projected_gradient` |
| Second order | `hvp`; dense `hessian` remains optional | `newton_cg`, `trust_region` |
| Composite/proximal | `smooth_gradient`, `prox` | `proximal_gradient` |
| Explicit visualization data | `plot_data` | `opt.plot_data`; never automatic tracking |

`probe_system()` and `system_info()` report these bundles. A method rejects a
missing bundle before beginning numerical work. Simple Adam systems do not need
constraint, HVP, or proximal stubs.

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [System Configuration](PARAMS.md)
- [Computation Lifecycle](LIFECYCLE.md)
- [Derivative Hooks](DERIVATIVES.md)
- [Logging Boundary](LOGGING_BOUNDARY.md)

---

[⌂ Home](SYSTEM_OLGS_HUB.md)
