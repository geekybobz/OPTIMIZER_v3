---
role: topic
parent: SYSTEM_OLSS_HUB.md
summary: "Required and optional project-system operations for the OLSS contract."
read_when: "Read when implementing an OLSS system or validating a shooting-system call."
status: active
reviewed: 2026-10-12
---
# OLSS Contract

OLSS is strict about optimizer-facing operations while leaving the physical
shooting formulation inside the project system.

## Required calls

| Call | Input | Output | Core invariant |
|---|---|---|---|
| `system_spec()` | none | static mapping | Complete `optimizer.olss-system-spec/v1`; no propagation |
| `shooting_spec()` | none | `ShootingSpec` | One ordered, real-valued variable layout |
| `residuals(z, options=...)` | complete `ShootingVariables` | finite array `(m,)` | Complete, unscaled physical residual vector |
| `jacobian(z, options=...)` | complete `ShootingVariables` | finite matrix `(m,n)` | Derivative of exactly the declared residual vector |
| `evaluate(z, options=...)` | complete `ShootingVariables` | metrics mapping | Includes every required declared metric |
| `verify(z, options=...)` | complete `ShootingVariables` | `ShootingVerification` or equivalent mapping | Independent numerical and project acceptance evidence |
| `with_secondary(**updates)` | named secondary updates | new OLSS system | Immutable, validated staged reconfiguration |

Subclassing `OLSS` is optional. The base class can supply `evaluate` and
`verify` when the only metrics are `residual_norm` and
`residual_max_abs`.

## Call rules

### `system_spec()`

Declares identity, variable layout, parameter groups, residual groups, metrics,
Jacobian provenance, optional selection metric, capabilities, reconstruction,
default verification options, and an optional default verification tolerance.
It must be static and must not run the model.

The complete schema is owned by [OLSS Data Model](DATA_MODEL.md).

### `shooting_spec()`

Returns the same ordered layout declared by `system_spec()["variables"]`.
Every candidate, bound, Jacobian column, seed, and continuation tangent uses
this order.

### `residuals(z, options=...)`

Returns the complete physical residual vector. Residual components may represent
terminal mismatch, transversality, integral constraints, junction matching, or
other project equations. They remain unscaled; solver scaling is numerical
policy.

### `jacobian(z, options=...)`

Returns `d residuals / d z`. The declared source is one of `analytical`,
`variational`, `automatic`, or `numerical-declared`. No silent fallback changes
the declared derivative during a solve.

### `evaluate(z, options=...)`

Returns named metrics declared in `system_spec()`. The mandatory feasibility
metrics are `residual_norm` and `residual_max_abs`. A project may add energy,
fidelity, stationarity, or other metrics.

### `verify(z, options=...)`

Recomputes the candidate at distinct accuracy or through another declared
independent route. A solver-converged point cannot be qualified without
independent verification. See [OLSS Verification](VERIFICATION.md).

### `with_secondary(**updates)`

Returns a new system preserving primary configuration and replacing only named
secondary values. It is the system boundary used by natural continuation.

## Shape and mutation invariants

- `n = shooting_spec().size`.
- `m = system_spec()["residuals"]["size"]`.
- `residuals` has shape `(m,)`.
- `jacobian` has shape `(m, n)`.
- Numerical arrays must be finite unless a metric explicitly permits otherwise.
- Calls must not mutate `ShootingVariables` or depend on a previous call.
- Complex physical quantities use a project-documented real/imaginary packing.
- Per-call `options` select accuracy or numerical realization; they do not
  mutate construction parameters.

## Optional capabilities

| Hook | Input and output | Consumer |
|---|---|---|
| `residuals_and_jacobian` | variables/options → residual vector and Jacobian | Reuse one propagation and sensitivity calculation |
| `shooting_bounds` | none → lower and upper flat arrays | Root-step clipping and seed bounds |
| `cost_gradient` | variables/options → array `(n,)` | Direct constrained-cost shooting |
| `recover_controls` | variables and requested grid → `Controls` or values mapping | Cross-formulation replay and export |
| `roots_equivalent` | two candidates and tolerance → bool | Project symmetry or gauge deduplication |
| `sparse_jacobian` | project-defined sparse matrix | Future sparse solver |
| `jacobian_operator` | project-defined operator | Future matrix-free solver |
| `forward_prop`, `simulate`, `plot_data` | project-defined inspection products | Explicit inspection and export |

## Transactions

The generic adapter maps `residuals`, `jacobian`, `evaluate`, and `verify` to
`payload["variables"]` plus the request `options`. `system_spec` and
`shooting_spec` accept neither payload nor options. See
[System Transactions](../TRANSACTIONS.md) for the envelope and error rules.

## Evidence boundary

The contract establishes a correctly shaped, declared, and independently
verified shooting root. It does not establish that the root is a local or
global optimum. Selection metrics rank only the finite set of qualified roots
provided to a campaign.

## Connections

- Related: [OLSS System Hub](SYSTEM_OLSS_HUB.md)
- Related: [OLSS Data Model](DATA_MODEL.md)
- Related: [OLSS Verification](VERIFICATION.md)
- Related: [Shooting Methods](../shooting/SHOOTING_METHODS.md)
- Related: [System Transactions](../TRANSACTIONS.md)

---

[⌂ Home](SYSTEM_OLSS_HUB.md)
