---
title: OLSS Contract
type: api_contract
module: optimizer/system_olss
tags:
  - optimizer
  - system
  - olss
  - shooting
---

# OLSS Contract

OLSS is strict about optimizer-facing transactions and leaves the physical shooting
formulation entirely inside the project system.

## Required calls

| Call | Return | Requirement |
|---|---|---|
| `system_spec()` | Static mapping | Family, identity, variables, residuals, metrics, derivatives, capabilities, and parameters; no propagation |
| `shooting_spec()` | `ShootingSpec` | One ordered real-valued variable layout |
| `residuals(z, options=...)` | Array `(m,)` | Complete finite unscaled residual vector |
| `jacobian(z, options=...)` | Matrix `(m,n)` | Derivative of exactly the declared residual vector |
| `evaluate(z, options=...)` | Metrics mapping | All required declared metrics |
| `verify(z, options=...)` | `ShootingVerification` or equivalent mapping | Independently evaluated residuals, metrics, options, and numerical status |
| `with_secondary(**updates)` | New OLSS system | Immutable validated staged reconfiguration |

Subclassing `OLSS` is optional. Its inherited `evaluate` and `verify` methods are
sufficient when the only metrics are `residual_norm` and `residual_max_abs`.

All numerical calls take complete `ShootingVariables`. They must not mutate the
candidate or depend on an earlier preparatory call.

## Variable layout

`ShootingSpec` contains ordered `ShootingBlock` objects. Each block declares:

```text
unique name
shape
meaning
optional unit
positive characteristic scale
role
optional metadata
```

The role `continuation_parameter` marks a parameter included directly in the
shooting vector. Its Jacobian column then supports natural or pseudo-arclength
continuation without rebuilding the system at every corrector step.

Complex physical quantities require an explicitly documented real/imaginary
packing convention.

## Residual and derivative rules

- Residuals exposed to the solver are unscaled physical residuals.
- `residual_norm` is their Euclidean 2-norm.
- Solver scaling and tolerances belong to the numerical method.
- The Jacobian may be analytical, variational, automatic, or explicitly declared
  numerical differentiation; there is no silent fallback.
- `verify_shooting_jacobian` is an explicit diagnostic and does not replace the
  system's declared derivative during a run.
- A dense Jacobian is the current solver representation. Sparse and operator hooks
  are discoverable extension points but are not yet consumed by the dense solvers.

## Accuracy and verification

Evaluation options are per-call mappings. They may select integration tolerances,
step counts, grids, or derivative accuracy without reconstructing the system.

The solver first checks its working residual tolerance. A candidate is qualified
only when `verify` also completes successfully and its residual norm satisfies the
verification tolerance. A system with additional required metrics includes them in
both `evaluate` and `verify`.

Independent verification is mandatory for qualification: explicit
`verification_options` or a distinct system-declared default must differ from the
working options. `ShootingVerification.checks` may additionally contain named
`VerificationCheck` records. Every supplied check must pass before the candidate is
qualified; the names and physics remain project-owned.

`simulate` is optional export functionality and has no formal acceptance meaning.

## Optional capabilities

| Hook | Consumer |
|---|---|
| `residuals_and_jacobian` | Root solvers reuse shared propagation and sensitivities |
| `shooting_bounds` | Root-step clipping |
| `cost_gradient` | Direct cost optimization on the feasible residual set |
| `recover_controls` | Project export and cross-formulation checks |
| `sparse_jacobian` | Future large multiple-shooting solver |
| `jacobian_operator` | Future matrix-free solver |
| `forward_prop`, `simulate`, `plot_data` | Explicit inspection and export |

A declared selection metric is enough to rank a finite collection of qualified
roots. Direct shooting additionally requires `cost_gradient` in shooting-variable
coordinates.

## Parallel reconstruction

Serial execution is the default. Parallel multistart requires a caller-supplied
factory so every thread or process obtains a separate system instance. Process
factories must be importable and pickle-compatible with the host's process model.

## Evidence boundary

`ShootingResult` separately records:

```text
solver_converged
verified
qualified
stop_reason
residual vector and norms
declared metrics
Jacobian provenance
verification evidence
evaluation counters
iteration history
```

`solve_shooting` can append to a previous result history and emits optional
start/iteration/terminal event dictionaries. These are the execution boundary for
shared core, CLI, and MCP consumers; they do not add logging responsibility to the
OLSS system.

None of these fields is a local- or global-optimality certificate.

---

[⌂ Home](SYSTEM_OLSS_HUB.md)
