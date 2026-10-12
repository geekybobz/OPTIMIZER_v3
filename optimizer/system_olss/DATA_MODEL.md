---
role: topic
parent: SYSTEM_OLSS_HUB.md
summary: "OLSS system declaration, shooting-variable layout, and structured result objects."
read_when: "Read when defining variable blocks, metric declarations, serialization, or result handling."
status: active
reviewed: 2026-10-12
---
# OLSS Data Model

OLSS keeps physical meaning in named blocks while exposing one deterministic
flat vector to numerical methods.

## System declaration

`system_spec()` uses schema `optimizer.olss-system-spec/v1` and contains:

| Field | Meaning |
|---|---|
| `family` | Always `olss` |
| `identity` | Stable ID, name, system type, and description |
| `variables` | Serialized `ShootingSpec` |
| `parameters` | Normally `primary` and `secondary` mappings |
| `residuals` | Size, named groups, and mandatory unscaled declaration |
| `metrics` | Required and optional metric declarations |
| `derivatives` | Jacobian target, hook, source, and fallback policy |
| `selection` | Optional metric and minimize/maximize direction |
| `capabilities` | Optional operations implemented by the project |
| `reconstruction` | Project-owned output reconstruction information |
| `verification` | Optional distinct default verification options and default verification tolerance |

`basic_shooting_system_spec(...)` constructs and validates this mapping.

## Variable hierarchy

### `ShootingBlock`

One logical part of the vector:

| Field | Meaning |
|---|---|
| `name` | Unique block name |
| `shape` | Physical array shape; scalar when empty |
| `meaning` | Project-defined interpretation |
| `unit` | Optional unit |
| `scale` | Positive characteristic scale used by seeds and root comparison |
| `role` | General role such as `variable` or `continuation_parameter` |
| `metadata` | Additional project-owned declarations |

### `ShootingSpec`

An ordered tuple of blocks. It owns block names, flat offsets, total size,
dtype, and one scale per flat coordinate. Numerical code must use this order
rather than reconstructing it from dictionaries.

### `ShootingVariables`

Finite immutable values attached to one `ShootingSpec`. Construct them with:

```text
ShootingVariables.zeros(spec)
ShootingVariables.from_flat(spec, values)
ShootingVariables.from_blocks(spec, mapping)
```

Use `block`, `unpack`, and `flatten` to read values; use `with_flat` to produce a
new candidate.

## Metric declarations

Every metric declares its label, description, role, dtype, shape, unit,
required status, and non-finite policy. `residual_norm` and
`residual_max_abs` are always required feasibility metrics. A selection metric
is used only to rank qualified roots.

## Evaluation and verification objects

| Type | Holds |
|---|---|
| `ShootingEvaluation` | Candidate, residual vector, declared metrics, and working options |
| `VerificationCheck` | Name, pass status, value, tolerance, and message |
| `ShootingVerification` | Independent residuals, metrics, options, numerical success, independence source, and checks |

## Solver and coordinator results

| Type | Holds |
|---|---|
| `ShootingResult` | Candidate, residuals, metrics, convergence, verification, qualification, stop reason, error text, verification tolerance and its source, counters, and history |
| `MultiStartResult` | All results, qualified roots, unique roots, ranking, selection rule, and whether ranking used verified or working values |
| `ContinuationPoint` | Candidate, residual norm, tangent, corrector iterations, and verification status |
| `ShootingContinuationResult` | Branch points, completion status, final step, step history, and verification tolerance with its source |
| `ParameterContinuationPoint` | Secondary parameter, attempted step, and shooting result |
| `ParameterContinuationResult` | Parameter path, target, points, stop reason, target status, and one step-history row per attempt, including failed attempts and their reasons |

All public objects provide `to_dict()` for JSON-safe representation.

## Construction convention

Project systems use:

```python
System(primary={...}, secondary={...})
```

Primary values define the physical model or fixed discretization. Secondary
values define staged targets, weights, accuracy-independent curriculum values,
or other settings that `with_secondary` may replace immutably.

## Connections

- Related: [OLSS System Hub](SYSTEM_OLSS_HUB.md)
- Prerequisite: [OLSS Contract](CONTRACT.md)
- Related: [OLSS Verification](VERIFICATION.md)
- Related: [Shooting Methods](../shooting/SHOOTING_METHODS.md)
- Related: [Generated API Reference](../../API_REFERENCE.md)

---

[⌂ Home](SYSTEM_OLSS_HUB.md)
