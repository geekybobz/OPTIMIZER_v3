---
role: front-door
parent: ../../README.md
summary: "Entry point for implementing, validating, and using Open Loop Shooting Systems."
read_when: "Start here when a project is formulated through shooting variables and boundary residuals."
status: active
reviewed: 2026-10-12
---
# OLSS System Hub

OLSS means **Open Loop Shooting System**. It is the project contract for
problems solved by finding shooting variables whose complete boundary residual
vector vanishes.

## In brief

```text
project owns
  shooting variables, residual equations, residual Jacobian,
  metrics, independent verification, and optional reconstruction

library owns
  validation, root solving, continuation, multistart, seed tools,
  diagnostics, replay, and structured evidence
```

OLSS is separate from OLGS because a residual Jacobian is not an objective
gradient, and shooting variables are not necessarily waveform controls.

## Start here

1. Read the [OLSS Contract](CONTRACT.md).
2. Define the [OLSS Data Model](DATA_MODEL.md) for the problem.
3. Define independent acceptance evidence using
   [OLSS Verification](VERIFICATION.md).
4. Select numerical operations from the [Shooting Hub](../shooting/SHOOTING_HUB.md).
5. Use the [System Transaction](../TRANSACTIONS.md) boundary only when a generic
   consumer needs request-response execution.

## Details

## Required surface

```text
system_spec()
shooting_spec()
residuals(variables, options=...)
jacobian(variables, options=...)
evaluate(variables, options=...)
verify(variables, options=...)
with_secondary(**updates)
```

All numerical calls receive complete immutable `ShootingVariables`. A call must
not depend on a hidden preparatory call or mutate the candidate.

## Scaffold

```bash
optimizer-system my_problem/system.py --family olss
```

The scaffold creates `system.py` and an initial contract test. It uses the
standard `System(primary=..., secondary=...)` construction convention and
leaves physics-specific residual and Jacobian definitions as explicit TODOs.

## Implementation ownership

| Area | Source |
|---|---|
| Contract discovery and optional hooks | `contract.py` |
| Variable layouts and values | `variables.py` |
| Static system declarations | `specs.py` |
| Return validation | `validation.py` |
| Evidence and result objects | `results.py` |
| Optional base implementation | `olss.py` |
| Project templates | `templates/` |

## Limitations

- Dense Jacobians are consumed by current solvers; sparse and operator hooks are
  declarations for later numerical methods.
- Variable scales guide seed and root-comparison tools but do not yet
  precondition every solver step.
- The direct constrained-cost method is first-order rather than SQP.
- Qualification is evidence of a verified root, not a local- or global-minimum
  certificate.

## Connections

- Related: [V4 entry](../../README.md)
- Prerequisite: [System Transactions](../TRANSACTIONS.md)
- Related: [OLSS Contract](CONTRACT.md)
- Related: [OLSS Data Model](DATA_MODEL.md)
- Related: [OLSS Verification](VERIFICATION.md)
- Related: [Shooting Hub](../shooting/SHOOTING_HUB.md)
- Related: [OLGS System Hub](../system_olgs/SYSTEM_OLGS_HUB.md)

---

[⌂ Home](../../README.md)
