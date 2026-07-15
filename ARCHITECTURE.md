# OPTIMIZER v3 Architecture

Status: current review draft.
Last updated: 2026-07-15.

This file is the top-level architecture source for the v3 redesign. The detailed
layer-by-layer notes live under [`plan/`](plan/README.md), especially the class notes
in [`plan/classes/`](plan/classes/README.md).

## One-Paragraph Summary

OPTIMIZER v3 is a direct-call optimization toolbox. A project `system.py` owns the
physics, control layout, objective components, cost prefactors, analytical gradient,
residuals, Jacobians, and higher derivative hooks. The optimizer library supplies the
shared engineering around that system: vectorized controls, optimizer loops,
step schedules, guesses, constraints, diagnostics, repairs, logs, checkpoints, and
later modes. The common user-facing style should be direct:

```python
import optimizer as opt

sys = system(params)
controls = opt.fourier_guess(sys, n_terms=6, amplitude=0.03)

r1 = opt.adam(sys, controls, maxiter=50)
r2 = opt.line_search(sys, r1, maxiter=10, warmstart=True)

fixed = opt.repair_newton(sys, r2.controls, residuals="hard")
diag = opt.geometry_probe(sys, fixed.controls)
```

## Design Rules

1. The system owns the objective. The optimizer does not assemble `J` from outside.
2. Gradients are expected to be analytical for normal use. Numerical derivatives are
   diagnostics or explicit fallbacks, not the main design path.
3. Public calls should be direct: `opt.adam(...)`, `opt.line_search(...)`,
   `opt.fourier_guess(...)`, `opt.repair_newton(...)`.
4. Controls are vectorized and channel-aware from the start.
5. The engine owns the iteration loop. Individual optimizers do not duplicate driver
   logic, logging, checkpointing, or stopping.
6. Curriculum is short-chunk parameter tuning: run a few iterations, inspect logs,
   change system cost prefactors, continue or rollback.
7. Modes are later. The first library should be a reliable toolbox of low-level
   components.

## Why Restructure

The old optimizer works, but it has three structural limits:

1. Each optimizer owns a duplicated loop. Adam, momentum, and line search repeat
   scheduling, history, checkpoint, stopping, and result handling.
2. The old model contract is too thin for advanced workflows. Current research scripts
   need hard residuals, residual Jacobians, seeded-adjoint Jacobian rows, projected
   gradients, Newton repair, and detailed diagnostics.
3. Closed-loop workflows are script-level. Curriculum, rollback, trace analysis, and
   optimizer switching should be library-level patterns.

The v3 architecture keeps the useful old habits, especially `Controls` and direct
optimizer use, but separates responsibilities cleanly.

## Target Package Layout

Preferred package layout:

```text
optimizer/
  __init__.py
  library.py

  system.py
  controls.py
  result.py
  state.py
  stage.py

  core/
    engine.py
    evaluate.py
    stopping.py
    parallel.py
    registry.py

  optimizers/
    adam.py
    momentum.py
    line_search.py
    newton.py
    projected.py
    lbfgs.py
    cg.py

  schedules/
    constant.py
    adaptive.py
    line_search.py
    trust_region.py

  guesses/
    zero.py
    constant.py
    sine.py
    sinc.py
    fourier.py
    random.py
    mix.py

  constraints/
    bounds.py
    projections.py
    norm.py

  utils/
    diagnostics.py
    derivatives.py
    repairs.py
    geometry.py
    fourier_tools.py

  logs/
    trace.py
    records.py
    checkpoint.py
    reports.py

  modes/
    __init__.py
```

## Why No `problem.py` In The First Skeleton

`problem.py` is not needed initially. For this project, the active optimization task is:

```text
system + controls + current system params
```

The system already knows the objective components and how the gradient is derived.
A separate `Problem` object risks becoming a wrapper around that. If a later
implementation proves a task object removes real complexity, it can be added then.

For staged optimization, use `stage.py`: it describes temporary system params, guard
checks, accept/reject rules, and checkpoint labels. It does not own physics.

## System Contract

The core system interface should look like:

```python
class OptimizerSystem:
    def control_spec(self) -> ControlSpec:
        ...

    def evaluate(self, controls: Controls) -> Evaluation:
        ...

    def gradient(self, controls: Controls) -> Controls:
        ...

    def with_params(self, **updates) -> "OptimizerSystem":
        ...
```

Advanced optional hooks:

```python
    def residuals(self, controls: Controls, name: str = "hard") -> np.ndarray:
        ...

    def jacobian(self, controls: Controls, name: str = "hard") -> np.ndarray:
        ...

    def hessian(self, controls: Controls):
        ...

    def hvp(self, controls: Controls, vector: np.ndarray) -> np.ndarray:
        ...

    def simulate(self, controls: Controls, **kwargs) -> dict:
        ...
```

Old names such as `get_control_spec()`, `forwardprop()`, `corrections()`, and
`metrics()` can be supported through adapters or aliases, but the v3 template should
use the direct names above.

## System-Owned Multi-Objective Cost

The system may have many objective components:

```text
J = wF * infidelity
  + w2 * F_norm2
  + w4 * C_sym_norm2
  + wE * energy
```

These weights are system params:

```python
sys2 = sys.with_params(
    infidelity_weight=1e5,
    lambda2=1e6,
    lambda4=10.0,
    energy_weight=0.0,
)
```

The optimizer only sees:

```python
evaluation = sys2.evaluate(controls)
gradient = sys2.gradient(controls)
```

This matches the inspected downstream systems, where `control_weight`, `lambda2`,
`lambda4`, and `energy_weight` are already embedded in the derived costate and
gradient logic.

## Soft And Hard Quantities

Soft objective terms go into `J`:

```text
infidelity
F_norm2
C_sym_norm2
energy
smoothness penalties
```

Hard equality-style quantities go into `residuals()`:

```text
terminal leakage
exact robustness conditions
F_k(T) constraints
other equality constraints
```

Early optimization should mostly use soft weights and guards. Hard projected methods
and Newton repair are late-stage tools for precise feasibility.

## Core Engine

The engine owns the loop:

```text
initialize RunState
evaluate current controls
request gradient/residual/Jacobian as needed
ask optimizer for an update
apply schedule
apply constraints
evaluate trial controls
accept or reject
record logs
save checkpoints
test stopping rules
return OptimizerResult
```

The native unit is a chunk:

```python
engine.run_chunk(system, controls, optimizer, maxiter=10, state=None)
```

Full optimization, manual chaining, and later modes all use the same chunk path.

## Results And Warmstart

`OptimizerResult` should expose:

```python
result.controls
result.metrics
result.state
result.trace
result.stop_reason
result.iterations
result.to_dict()
```

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
trace/checkpoint context
```

Only transfer when compatible:

```text
Adam moments
momentum buffers
L-BFGS history
trust-region radius
LM damping
```

## Logs And Checkpoints

Logs and checkpoints are separate.

Logs are lightweight and frequent:

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
accepted
reason
```

Checkpoints are restorable:

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

Important labels:

```text
latest
stage_start
accepted
best_J
best_safe
```

`stage_start` is the important curriculum rollback anchor.

## Curriculum Flow

Curriculum means structured sequential tuning of system params:

```python
trace = opt.Trace("run_001")

sys1 = sys.with_params(infidelity_weight=1e5, lambda2=0.0, lambda4=0.0)
trace.checkpoint("stage_start", sys1, controls)
r1 = opt.adam(sys1, controls, maxiter=10, trace=trace)

sys2 = sys.with_params(infidelity_weight=1e5, lambda2=1e6, lambda4=0.0)
trace.checkpoint("stage_start", sys2, r1.controls, r1.state)
r2 = opt.adam(sys2, r1, maxiter=10, warmstart=True, trace=trace)

if not stage_accepts(r2.metrics):
    controls, state = trace.restore("stage_start")
```

The optimizer remains simple. The intelligence is in short chunks, logs, guards,
system-param changes, and rollback.

## Public API

Preferred explicit script style:

```python
import optimizer as opt

sys = system(params)
controls = opt.fourier_guess(sys, n_terms=8, amplitude=0.03)

r = opt.adam(sys, controls, maxiter=1000)
r = opt.line_search(sys, r, maxiter=10, warmstart=True)

fixed = opt.repair_newton(sys, r.controls)
diag = opt.geometry_probe(sys, fixed.controls)
```

Bound-system notebook/curriculum convenience:

```python
ctx = opt.context(sys)
r = ctx.adam(controls, maxiter=1000)

ctx = ctx.with_params(lambda2=1e6, lambda4=10.0)
r = ctx.adam(r.controls, maxiter=10, warmstart=True)
```

The explicit style should remain the core implementation path because it is clearer
for scripts, parallelism, and reproducibility. The context style should be explicit
object state, not a mutable package global.

## Build Order

Recommended implementation order:

```text
1. controls.py
2. system.py contract
3. result.py and state.py
4. logs/trace and checkpoint
5. core engine
6. adam, momentum, line_search
7. guesses
8. diagnostics and derivative checks
9. repairs/projected/Newton tools
10. L-BFGS, CG, LM, nullspace methods
11. modes placeholder
```

This keeps the redesign grounded: first make the base contract and closed-loop
optimizer reliable, then add advanced modes later.

## Compatibility

The old optimizer should remain available until migration is explicit. v3 should
support old system naming through adapters where practical:

```text
get_control_spec() -> control_spec()
forwardprop() + metrics() -> evaluate()
corrections() -> gradient()
```

Downstream migration should be mechanical and opt-in.
