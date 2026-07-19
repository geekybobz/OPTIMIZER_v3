---
title: CORE_HUB
type: module_hub
module: optimizer/core
tags:
  - optimizer
  - core
  - engine
  - hub
---

# Core Documentation Hub

This folder contains the shared chunk engine: the loop every gradient-based optimizer
runs, the validated boundary to the system, the acceptance policy, and the stopping
rules.

A concrete optimizer supplies only a proposal function.  Core owns everything around
it, so Adam, momentum, line search, L-BFGS, and nonlinear CG stay small.

The goal is compact, traversable documentation: start here, then open the note that
owns the part you are changing.

## Design Scope

Core owns the mechanics that must behave identically for every method:

```text
evaluate current controls
request an analytical gradient
hand the method a StepContext
validate and evaluate the proposed trial
accept or reject the trial
update run state and optimizer-private state
record iterations, chunks, and checkpoints
stop for a consistent, named reason
```

Core does not own:

```text
method mathematics (Adam moments, Armijo bracketing, L-BFGS history)
physics, objectives, or analytical gradients
initial-control generation
step-size schedules
map-style parallel execution
```

## Documentation Map

Read in this order:

```text
CORE_HUB.md
  Entry point and graph map.

CORE_CONTRACT.md
  The step and accept callables, and the objects passed across that boundary.

CORE_LIFECYCLE.md
  Exact order of one chunk, and what each failure branch leaves behind.

CORE_ENGINE.md
  run_chunk argument surface, start-state precedence, recording and checkpoints.

CORE_EVALUATION.md
  SystemEvaluator, the evaluation cache, and structured failure outcomes.

CORE_ACCEPTANCE_GUARDS.md
  Default acceptance and multi-metric guards.

CORE_STOPPING.md
  StoppingConfig, StopTracker, rule order, and stop reasons.

CORE_BOUNDARY.md
  What belongs in core, and what belongs in utils, optimizers, logs, and systems.
```

## Two Namespaces, Two Meanings

`core` and `base` are distinct on purpose:

```text
opt.core    this package: run_chunk, StepContext, StepProposal,
            AcceptanceDecision, SystemEvaluator, StoppingConfig, metric_guard

opt.base    top-level spine objects: Controls, ControlSpec, OptimizerResult,
            WarmStartState, Trace, BlackBoxRun, evaluate, gradient, context
```

Both are catalogued, so either can be listed at runtime:

```python
opt.info(h=True)
opt.core.list()
opt.base.list()
opt.core.info("run_chunk")
```

## Public Entry Points

Namespace style:

```python
guard = opt.core.metric_guard(improve="J", require={"fidelity": (">=", 0.99)})
```

Root shortcut style:

```python
result = opt.run_chunk(
    system,
    controls,
    step=my_step,
    optimizer_name="custom",
    maxiter=25,
)
```

Most users never call core directly.  They call a method in
[Optimizers](../optimizers/OPTIMIZERS_HUB.md), which calls `run_chunk` for them.

## Implementation Files

Current source files:

```text
optimizer/core/
  __init__.py     curated public surface
  engine.py       run_chunk and the proposal/acceptance contract
  evaluate.py     SystemEvaluator and structured outcomes
  guards.py       MetricGuard and metric_guard
  stopping.py     StoppingConfig, StopTracker, StopDecision
```

## Reading Path

Writing a custom optimizer:

```text
CORE_HUB.md
CORE_CONTRACT.md
CORE_LIFECYCLE.md
CORE_ENGINE.md
```

Tuning acceptance or stopping:

```text
CORE_ACCEPTANCE_GUARDS.md
CORE_STOPPING.md
```

Changing core itself:

```text
CORE_BOUNDARY.md
CORE_EVALUATION.md
CORE_LIFECYCLE.md
```

## Related Notes

- [Root Entry](../../README.md)
- [Core Contract](CORE_CONTRACT.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Boundary](CORE_BOUNDARY.md)
- [Optimizer Lifecycle](../optimizers/OPTIMIZER_LIFECYCLE.md)
- [OLGS System Hub](../system_olgs/SYSTEM_OLGS_HUB.md)
