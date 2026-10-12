---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "How a method's proposal function runs inside the shared engine, and the CMA-ES exception."
read_when: "Read when writing a step function or tracing one iteration."
status: active
reviewed: 2026-10-12
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
  -> update compact numerical aggregates
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

`line_search` uses a custom accept wrapper for backtracking and Armijo checks.
Its numerical method memory retains accepted/rejected counts and step information.

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
  -> finalize compact numerical aggregates
  -> return OptimizerResult
```

`cma_es` still returns the standard result shape, without gradient proposals. Native physical starts and compatible warmstart
fields remain available; exact RNG restoration is not promised.

## Related Notes

- [Optimizer Contract](OPTIMIZER_CONTRACT.md)
- [Acceptance and Stopping](OPTIMIZER_ACCEPTANCE.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [Core Hub](../core/CORE_HUB.md)
- [Core Lifecycle](../core/CORE_LIFECYCLE.md)
- [Core Contract](../core/CORE_CONTRACT.md)
- [Core Engine Source](../core/engine.py)

---

[⌂ Home](OPTIMIZERS_HUB.md)
