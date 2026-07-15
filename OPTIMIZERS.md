# OPTIMIZER v3 Optimizers Catalog

Status: current review draft.
Last updated: 2026-07-15.

This file lists methods that move controls. Tools that generate guesses, repair
constraints, diagnose runs, log traces, or manage checkpoints live in
[`AUXILIARY.md`](AUXILIARY.md).

## Core Rule

Optimizers minimize the current system objective. They do not build the physical cost.

The system owns:

```text
J
metrics
analytical gradient
cost prefactors
residuals and Jacobians when needed
```

The optimizer asks for:

```python
evaluation = system.evaluate(controls)
gradient = system.gradient(controls)
```

Then it proposes a control update.

## Public Naming

User-facing calls should be direct:

```python
opt.adam(...)
opt.momentum(...)
opt.line_search(...)
opt.lbfgs(...)
opt.projected_descent(...)
opt.levenberg_marquardt(...)
```

Internal modules can still live under `optimizers/`.

## First Implementation Wave

Build these first because they match current downstream usage and prove the new engine:

| public call | internal module | role | needs |
|---|---|---|---|
| `opt.adam` | `optimizers/adam.py` | robust coarse first-order search | `gradient` |
| `opt.momentum` | `optimizers/momentum.py` | smooth first-order refinement | `gradient` |
| `opt.line_search` | `optimizers/line_search.py` | deterministic polish and step repair | `gradient`, trial evaluation |

These methods should share the same engine, result object, trace, checkpoint, and
warmstart logic.

## First-Order Extensions

Add only after the base methods are stable:

| public call | idea | priority | notes |
|---|---|---|---|
| `opt.nag` | Nesterov accelerated gradient | medium | cheap extension of momentum |
| `opt.amsgrad` | Adam with max second moment | medium | safer Adam variant |
| `opt.adabelief` | Adam-like variance around momentum | explore | possibly useful, not first wave |

These are optional quality improvements. They should not delay the core engine.

## Deterministic Classical Methods

Useful after the basic line-search infrastructure is reliable:

| public call | idea | priority | notes |
|---|---|---|---|
| `opt.lbfgs` | limited-memory quasi-Newton | high after first wave | strong deterministic full-gradient method |
| `opt.cg_pr` | nonlinear conjugate gradient, Polak-Ribiere+ | medium | low-memory alternative to L-BFGS |

`wolfe_line_search` should likely be schedule/step infrastructure, not necessarily a
main public optimizer. L-BFGS and CG can use it internally.

## Constraint And Second-Order Methods

These absorb logic that currently lives in advanced project scripts:

| public call | idea | phase | needs |
|---|---|---|---|
| `opt.repair_newton` | min-norm Newton repair of hard residuals | utility/repair | `residuals`, `jacobian` |
| `opt.projected_descent` | descend in tangent space of hard residuals | late-stage optimizer | `gradient`, `residuals`, `jacobian` |
| `opt.levenberg_marquardt` | damped Gauss-Newton root solving | late-stage optimizer | `residuals`, `jacobian` |
| `opt.gauss_newton` | undamped Gauss-Newton variant | late-stage variant | `residuals`, `jacobian` |
| `opt.nullspace_descent` | secondary descent inside `ker J` | late-stage polish | `gradient`, `jacobian`, secondary gradient |

These should not be first-wave methods. They require stable diagnostics, residuals,
Jacobians, trace, and checkpoint rollback.

## Capability Expectations

Normal first-order optimization expects analytical gradients:

```text
system.gradient(controls)
```

Numerical gradients are useful for:

```text
derivative verification
debugging
fallback experiments
small smoke tests
```

They should not be treated as the main path for expensive quantum systems.

For constrained/second-order methods, the preferred path is analytical residuals and
Jacobians:

```text
system.residuals(controls, name="hard")
system.jacobian(controls, name="hard")
```

Finite-difference Jacobians may exist as diagnostics, but should be explicit because
they are expensive.

## Warmstart Rules

Warmstart should work across optimizer changes:

```python
r1 = opt.adam(sys, controls, maxiter=100)
r2 = opt.line_search(sys, r1, maxiter=20, warmstart=True)
```

Safe to transfer across all optimizers:

```text
controls
latest metrics
step estimate
trace/checkpoint context
```

Transfer only when compatible:

```text
Adam moments
momentum velocity
L-BFGS curvature pairs
CG previous direction
LM damping/trust state
```

If incompatible, start the new optimizer from controls and metrics only.

## Rejected Or Deferred

| candidate | decision |
|---|---|
| `adamw` | defer; energy and regularization weights belong in the system cost |
| `radam`, `nadam` | defer; marginal over Adam/AMSGrad/NAG for this use case |
| `lion` | defer; sign-momentum method is less aligned with deterministic smooth objectives |
| full dense `bfgs` | reject for large control dimensions; use L-BFGS |
| `lamb`, `lars` | reject; large-batch neural-network optimizers |
| schedule-free wrappers | defer; consider later under schedules or modes |

## Build Priority

Recommended order:

```text
1. adam
2. momentum
3. line_search
4. warmstart handoff among these three
5. lbfgs and cg_pr
6. repair_newton
7. projected_descent
8. levenberg_marquardt / gauss_newton
9. nullspace_descent
```

The first three are enough to validate the new architecture against current downstream
use before advanced methods are added.

