---
title: OPTIMIZER_LIFECYCLE
type: computation_flow
module: optimizer/optimizers
tags:
  - optimizer
  - lifecycle
  - engine
---

# Optimizer Lifecycle

Most optimizer methods use `optimizer.core.engine.run_chunk`.

The method file supplies a proposal function. The shared engine owns the outer
iteration mechanics.

## Engine-Based Flow

```text
start controls
  -> validate controls against system.control_spec()
  -> evaluate current controls
  -> check initial stopping rules
  -> compute analytical gradient
  -> method builds StepProposal
  -> validate proposed controls
  -> evaluate proposed controls
  -> accept or reject trial
  -> update RunState and optimizer_state
  -> record trace/checkpoints when enabled
  -> check stopping rules
  -> return OptimizerResult
```

## Step Context

The method-specific proposal function receives:

```text
StepContext
  system
  evaluator
  state
  evaluation
  gradient
  iteration
  stage
```

This gives methods read access to the current controls, metrics, gradient, and
validated evaluator.

## Step Proposal

The method returns:

```text
StepProposal
  controls
  step_size
  optimizer_state_on_accept
  optimizer_state_on_reject
  step_size_on_accept
  step_size_on_reject
  technical
  reason
```

The branch-specific state fields matter. Adam moments, momentum velocity, L-BFGS
history, and nonlinear-CG directions should advance only on the correct branch.

## Acceptance

Default acceptance compares scalar metrics:

```text
accept_metric = "J"
accept_mode = "min"
accept_tolerance = 0.0
```

For minimization, the trial is accepted when the selected metric does not get worse
beyond tolerance.

`line_search` has a custom accept wrapper so backtracking and Armijo details are
recorded inside the normal engine trace.

## Stopping

The engine checks:

```text
maxiter
target_value
non-finite metrics
stall_patience
stall_tolerance
```

`maxiter` means attempted engine iterations, not only accepted iterations.

## CMA-ES Flow

`cma_es` does not use `run_chunk`.

Its flow is:

```text
start controls as population mean
  -> evaluate initial controls
  -> sample population around mean
  -> evaluate candidates
  -> rank candidates by accept_metric
  -> compute elite mean and spread
  -> accept generation if best candidate improves
  -> adapt sigma (accepted: blend toward elite spread; rejected: shrink by 0.9)
  -> record standard trace records where practical
  -> return OptimizerResult
```

`cma_es` still returns the standard result shape, but it does not use gradient
proposals or warmstart state in the current implementation.

## Related Notes

- [Optimizer Contract](OPTIMIZER_CONTRACT.md)
- [Acceptance and Stopping](OPTIMIZER_ACCEPTANCE.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [Core Engine Source](../core/engine.py)
