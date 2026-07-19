---
title: OLGS Computation Lifecycle
type: computation_flow
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - lifecycle
depends_on:
  - README
  - CONTRACT
connects_to:
  - PARAMS
  - DERIVATIVES
  - LOGGING_BOUNDARY
  - OLGS_SYSTEM_BUILD_SKILL
---

# OLGS Computation Lifecycle

This file defines how an OLGS system should move from controls to metrics and
gradients.

## Main Flow

```text
Controls
  -> forward_prop
  -> metrics/evaluate
  -> back_prop
  -> gradient
```

The common analytical optimization path is:

```python
metrics = system.evaluate(controls)
grad = system.gradient(controls)
```

Evaluator code may use a combined call if available:

```text
value_and_gradient(controls)
```

but this is optional.

## `forward_prop`

Responsible for physical forward evolution.

Can compute:

```text
Hamiltonians
propagators
state trajectories
terminal overlaps
objective ingredients
Bloch coordinates
projection values
normalization checks
```

## `evaluate`

Responsible for fast optimizer metrics.

Should compute only what optimizer iterations need.

Must return:

```python
{"J": finite_scalar}
```

May also return:

```text
fidelity
infidelity
energy
constraint norms
objective component terms
```

## `back_prop`

Responsible for adjoint or costate propagation.

Allowed behavior:

```text
call forward_prop if the matching forward state is missing
or raise a clear error telling the user to run forward_prop first
```

## `gradient`

Responsible for analytical `dJ/du`.

Should usually:

```text
ensure forward quantities exist
ensure backward quantities exist
build gradient in control row order
return Controls
```

## `simulate`

`simulate` is optional and richer than `evaluate`.

It may run:

```text
full trajectory export
ensemble checks
finite perturbation robustness checks
plotting data
extra diagnostics
```

Optimizers should not depend on expensive simulation behavior.

## `metrics`

`metrics()` means latest valid metrics.

It should be clear whether metrics are available. If no valid computation has run,
raise a clear error.

## Related Notes

- [[README|system_olgs]]
- [[OLGS_SYSTEM_BUILD_SKILL|OLGS System Build Skill]]
- [[CONTRACT|API Contract]]
- [[PARAMS|System Configuration]]
- [[DERIVATIVES|Derivative Hooks]]
- [[LOGGING_BOUNDARY|Logging Boundary]]
