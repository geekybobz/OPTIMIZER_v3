# Curriculum And Checkpoints

Status: review draft.
Last updated: 2026-07-15.

## Meaning Of Curriculum Here

Curriculum optimization means structured sequential tuning of system cost parameters.

It is not a separate external objective wrapper.

The system owns the multi-objective cost:

```text
J = wF * infidelity
  + w2 * F_norm2
  + w4 * C_sym_norm2
  + wE * energy
```

The optimizer only minimizes the current `J` using the current analytical gradient.

## Why This Fits The Current Research Code

The inspected systems already put cost prefactors inside the system:

```text
control_weight
infidelity_weight
lambda2
lambda4
energy_weight
```

These are part of the derivation and costate equations. Therefore the optimizer
should not rebuild `J` from outside. It should receive a system with current params.

## Basic Manual Curriculum

```python
ctx = opt.context(sys, trace="run_001")

ctx1 = ctx.with_params(wF=1e5, w2=0.0, w4=0.0, wE=0.0)
r1 = ctx1.adam(controls, maxiter=10)

ctx2 = ctx.with_params(wF=1e5, w2=1e6, w4=0.0, wE=0.0)
r2 = ctx2.adam(r1.controls, maxiter=10, warmstart=True)

ctx3 = ctx.with_params(wF=1e5, w2=1e6, w4=10.0, wE=0.0)
r3 = ctx3.adam(r2.controls, maxiter=10, warmstart=True)

ctx4 = ctx.with_params(wF=1e5, w2=1e6, w4=10.0, wE=1e-3)
r4 = ctx4.line_search(r3.controls, maxiter=10, warmstart=True)
```

The user or later `modes/` can decide the next weights by reading logs.

## Soft Guards

Early and middle optimization should not be over-constrained. Use soft guards:

```text
run short chunk
check metrics
accept if important metrics survive
rollback if a chunk breaks them badly
```

Example:

```text
while increasing lambda4:
  allow C_sym_norm2 to improve
  require infidelity below guard
  require F_norm2 below guard
  reject if energy explodes
```

This is less rigid than projected descent. The optimizer is free inside the chunk.
The guard only decides whether the chunk was useful.

## Hard Repairs

Hard tools should exist but should not control every early update:

```python
fixed = opt.repair_newton(sys, controls, residuals="hard")
projected = opt.projected_descent(sys, controls, residuals="hard")
```

Use these near the end, when a good solution needs precise feasibility repair.

## Logs Versus Checkpoints

Logs are lightweight and frequent.

Checkpoints are heavier and restorable.

### Log Row

Each iteration or chunk should record:

```text
run_id
chunk
iter
global_iter
optimizer
stage
system_params
metrics
grad_norm
step_norm
control_norm
max_abs_control
accepted
reason
```

The metrics are whatever the system returns:

```text
J
fidelity
infidelity
F_norm2
C_sym_norm2
energy
```

### Checkpoint

A checkpoint must save enough to resume or rollback:

```text
controls
optimizer_state
system_params
metrics
iteration
stage label
random state
previous checkpoint id
```

Important checkpoint labels:

```text
latest
stage_start
accepted
best_J
best_safe
```

`stage_start` is critical. If changing `lambda4` or `energy_weight` damages the
solution, restore the state before the parameter change.

## Chunk With Rollback

```python
trace = opt.Trace("run_001")
controls = controls0
state = None

for stage in stages:
    sys_stage = sys.with_params(**stage.weights)
    trace.checkpoint("stage_start", sys_stage, controls, state)

    result = opt.adam(
        sys_stage,
        controls,
        maxiter=10,
        warmstart=state,
        trace=trace,
    )

    if stage.accept(result.metrics):
        controls = result.controls
        state = result.state
        trace.checkpoint("accepted", sys_stage, controls, state)
    else:
        controls, state = trace.restore("stage_start")
        trace.event("rollback", reason="guard_failed")
```
