---
title: Core Contract
type: api_contract
module: optimizer/core
tags:
  - optimizer
  - core
  - engine
  - api
  - contract
---

# Core Contract

This file defines the boundary between the shared engine and a concrete optimizer.

The engine is strict about the shape of that boundary and indifferent to the method
behind it.  Anything that satisfies the two callables below can be driven by
`run_chunk`.

## The Two Callables

```text
step    StepContext -> StepProposal | Controls        required
accept  (current, trial, proposal, state) -> AcceptanceDecision | bool   optional
```

`step` is the method.  `accept` is the policy that decides whether the method's trial
survives.  When `accept` is omitted the engine uses
[default acceptance](CORE_ACCEPTANCE_GUARDS.md).

Minimal complete optimizer:

```python
def my_step(ctx):
    return ctx.state.controls - 0.01 * ctx.gradient

result = opt.run_chunk(
    system,
    controls,
    step=my_step,
    optimizer_name="steepest",
    maxiter=25,
)
```

Returning bare `Controls` is allowed; the engine wraps it in a `StepProposal`.

## `StepContext`

Read-oriented bundle handed to `step` once per iteration:

```text
system       validated optimizer-facing system
evaluator    SystemEvaluator, for extra evaluations inside the proposal
state        RunState: controls, metrics, step_size, optimizer_state
evaluation   Evaluation snapshot of the current controls
gradient     analytical gradient as Controls
iteration    chunk iteration index
stage        optional curriculum label
```

Rules:

```text
frozen; treat every field as read-only
gradient is analytical, supplied by the system, never a finite-difference fallback
extra evaluations should go through evaluator so they are cached and counted
```

`evaluator` is what lets `line_search` evaluate candidate steps *inside* the proposal
before the engine ever sees a trial.

## `StepProposal`

What `step` returns:

```text
controls                   proposed trial controls, required
step_size                  step size to retain on the chunk state
optimizer_state            state update applied after the proposal
optimizer_state_on_accept  state update applied only when accepted
optimizer_state_on_reject  state update applied only when rejected
step_size_on_accept        step size retained only when accepted
step_size_on_reject        step size retained only when rejected
technical                  free-form payload recorded on the iteration log
reason                     optional label for the proposal
```

Branch resolution:

```text
branch-specific field set     -> that value wins
branch-specific field unset   -> fall back to the plain field
neither set                   -> no update
```

This is the field set that matters most for correctness.  Adam moments, momentum
velocity, L-BFGS history, and nonlinear-CG directions should advance only on the
branch they belong to; using the plain `optimizer_state` field advances them on both.

## `AcceptanceDecision`

What a custom `accept` returns:

```text
accepted    bool, whether the trial replaces current controls
reason      short label recorded on the iteration log
technical   structured detail recorded alongside the reason
```

A bare `bool` is also accepted and is normalized to
`reason="accepted"` / `reason="rejected"` with empty `technical`.

Any other return type raises `TypeError`.

## Validation Guarantees

The engine validates on the method's behalf:

```text
starting controls are validated against system.control_spec()
proposed controls are validated before they are evaluated
a proposal that raises is caught and ends the chunk as proposal_failed
```

A method therefore does not need defensive control-shape checks in its proposal
function.

The same protection does not extend to `accept`.  An exception raised by an accept
function propagates out of `run_chunk` instead of becoming a stop reason, so custom
acceptance should validate its own metrics.  See
[Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md#one-sharp-edge).

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Engine](CORE_ENGINE.md)
- [Core Acceptance and Guards](CORE_ACCEPTANCE_GUARDS.md)
- [Core Evaluation](CORE_EVALUATION.md)
- [Optimizer Contract](../optimizers/OPTIMIZER_CONTRACT.md)
- [Core Engine Source](engine.py)
