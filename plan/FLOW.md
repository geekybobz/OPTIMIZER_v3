# Execution Flow

Status: review draft.
Last updated: 2026-07-15.

## Normal Direct Flow

The intended usage is:

```python
import optimizer as opt

sys = system(params)
controls = opt.fourier_guess(sys, n_terms=6, amplitude=0.03)

r1 = opt.adam(sys, controls, maxiter=50)
r2 = opt.line_search(sys, r1, maxiter=10, warmstart=True)

fixed = opt.repair_newton(sys, r2.controls, residuals="hard")
diag = opt.geometry_probe(sys, fixed.controls)
```

## What Happens Inside `opt.adam`

```text
1. Resolve system and controls.
2. Read `system.control_spec()`.
3. Build or resume `RunState`.
4. Evaluate current controls through `system.evaluate(controls)`.
5. Ask system for analytical gradient through `system.gradient(controls)`.
6. Optimizer proposes an update.
7. Schedule chooses step size.
8. Constraints apply bounds/projections if configured.
9. System evaluates trial controls.
10. Engine accepts/rejects according to the optimizer policy.
11. Logs receive iteration record.
12. Checkpoint saves if cadence or label requires it.
13. Stop rules decide whether to continue.
14. Return `OptimizerResult`.
```

The optimizer never interprets physics-specific metrics. It only logs whatever the
system returns.

## Evaluation Flow

Preferred system API:

```python
eval = system.evaluate(controls)
J = eval.metrics["J"]
F = eval.metrics["fidelity"]
E = eval.metrics["energy"]
```

`system.metric(controls, "energy")` can exist as convenience, but it should reuse a
cached evaluation when possible. It should not rerun expensive propagation for each
metric name.

## Warmstart Flow

Warmstart should support optimizer switching:

```python
r1 = opt.adam(sys, controls, maxiter=100)
r2 = opt.line_search(sys, r1, maxiter=20, warmstart=True)
```

Always transferable:

```text
controls
latest metrics
step-size estimate
iteration/log context
checkpoint context
```

Transfer only when compatible:

```text
Adam moments
momentum buffers
trust-region radius
LM damping
L-BFGS history
```

If incompatible, the next optimizer should start from controls and metrics only.

## Chunked Flow

Curriculum and manual closed-loop work should use short chunks:

```text
run 5 to 20 iterations
inspect metrics/logs
change system params if needed
continue from result controls
```

This keeps the optimizer simple while giving the user or future modes full control.

