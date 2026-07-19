---
title: OLGS Parameters
type: configuration_contract
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - parameters
---

# OLGS Parameters

OLGS uses two parameter groups:

```text
primary
secondary
```

No runtime parameter group is included for now. Runtime choices remain internal
choices of the system author until a real need appears.

The public construction shape should be dict-first:

```python
system = System(
    primary={"tau": 1.0, "N": 1001},
    secondary={"residual_weight": 0.1, "energy_weight": 0.0},
)
```

Individual systems may normalize or validate these dictionaries internally, but
the notebook/API-facing surface should remain simple mappings.

## Primary Params

Primary params define the physical system and numerical grid.

They usually stay fixed during an optimization run.

Examples:

```text
tau or T
N
dt
state_dim
control_dim
control_channels
initial state
target state
coupling constants
Hamiltonian constants
ensemble definition if it is part of the physics
```

Recommended convention:

```text
If tau and N are given, derive dt.
If tau, N, and dt are all given, validate consistency.
```

Endpoint-sampled convention:

```text
N = endpoint-including samples
dt = tau / (N - 1)
control_dim = N
propagation intervals = N - 1
```

## Secondary Params

Secondary params define tunable objective and curriculum values.

They are expected to change between optimization stages.

Examples:

```text
infidelity_weight
residual_weight
energy_weight
smoothness_weight
constraint weights
curriculum weights
```

Use:

```python
system2 = system.with_secondary(residual_weight=0.2, energy_weight=1.0e-3)
```

## Dict-First Construction

Systems should accept dictionaries directly:

```python
system = System(
    primary={"tau": 1.0, "N": 1001},
    secondary={"residual_weight": 0.1, "energy_weight": 0.0},
)
```

Validation should reject unknown keys clearly and keep defaults explicit.

```text
None means use system defaults.
dict means override named defaults.
unknown keys should raise a clear error.
```

Parameter classes are not part of the public OLGS contract.

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [OLGS System Build Skill](templates/OLGS_SYSTEM_BUILD_SKILL.md)
- [API Contract](CONTRACT.md)
- [Computation Lifecycle](LIFECYCLE.md)
