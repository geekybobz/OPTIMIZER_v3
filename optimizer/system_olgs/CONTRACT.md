---
title: OLGS Contract
type: api_contract
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - api
depends_on:
  - README
connects_to:
  - PARAMS
  - LIFECYCLE
  - DERIVATIVES
  - LOGGING_BOUNDARY
  - OLGS_SYSTEM_BUILD_SKILL
---

# OLGS Contract

This file defines the intended method surface for an OLGS system.

OLGS should be strict about the optimizer-facing API and flexible about
system-specific physics.

## Required Methods

Every OLGS system must provide:

```text
control_spec()
evaluate(controls)
gradient(controls)
with_secondary(**updates)
```

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

## Optional Methods

Advanced systems may provide:

```text
residuals(controls, name="hard")
jacobian(controls, name="hard")
hessian(controls)
hvp(controls, vector)
second_derivative(controls, ...)
residual_schema()
metric_schema()
cache_reset()
cache_status()
```

Optional methods should be discoverable and validated by OLGS helpers.

## Related Notes

- [[README|system_olgs]]
- [[OLGS_SYSTEM_BUILD_SKILL|OLGS System Build Skill]]
- [[PARAMS|System Configuration]]
- [[LIFECYCLE|Computation Lifecycle]]
- [[DERIVATIVES|Derivative Hooks]]
- [[LOGGING_BOUNDARY|Logging Boundary]]
