---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Common inputs, arguments, run recording, and the return object of every optimizer call."
read_when: "Read when calling any optimizer or writing a new one."
status: active
reviewed: 2026-10-12
---
# Optimizer Contract

This file defines the public contract for optimizer method calls.

The reference style is:

```python
result = opt.optimizers.adam(system, controls, maxiter=100)
```

## Required Inputs

Most optimizer methods require:

```text
system
controls or state or warmstart
```

`system` must satisfy the OLGS optimizer-facing contract:

```text
control_spec()
evaluate(controls)
gradient(controls) for gradient-based methods
```

`controls` must be an `optimizer.controls.Controls` object compatible with
`system.control_spec()`.

## Common Arguments

Engine-based optimizers generally accept:

```text
maxiter
step_size
state
warmstart
accept_metric
accept_mode
accept_tolerance
stage
use_cache
target_value
target_metric
stall_patience
stall_tolerance
```

Not every optimizer exposes every method-specific argument. Read the method file for
the exact surface.

## Numerical context

| Argument | Accepts | Effect |
|---|---|---|
| `stage` | `str` or `None` | Supply a phase label to StepContext |
| `use_cache` | `bool`, default `True` | Reuse the cached evaluation for identical controls |

Numerical calls create no automatic logs or files. The frontend
[command recorder](../history/HISTORY.md) owns interaction history; explicit
[checkpoints](../persistence/PERSISTENCE.md) own durable numerical state.

## Return Object

Every public optimizer returns:

```text
OptimizerResult
```

Important fields:

```text
controls
metrics
J
stop_reason
iterations
optimizer
state
diagnostics
```

## Gradient-Based Methods

These methods require `system.gradient(controls)`:

```text
line_search
momentum
adam
adagrad
rmsprop
lbfgs
nonlinear_cg
ncg
newton_cg
trust_region
projected_gradient
```

The optimizer should not silently replace a missing analytical gradient with finite
differences.

Direction convention:

```text
momentum, adam, adagrad, rmsprop, lbfgs, and nonlinear_cg always step along
  -grad J; for them accept_mode changes only the acceptance test
line_search additionally flips its step direction when accept_mode="max"
```

For the fixed-direction methods, a maximization goal belongs in the objective and
gradient themselves; use `accept_metric`/`accept_mode` for guarded acceptance.

## Conditional-Capability Methods

```text
projected_gradient  -> residuals, jacobian, optional control_bounds
newton_cg           -> hvp
trust_region        -> hvp; minimization only
proximal_gradient   -> smooth_gradient and prox; evaluate returns total J
```

These hooks are conditional. They do not become mandatory on the base OLGS
protocol. `continuation` is an outer coordinator returning `ContinuationResult`,
not an optimizer method pretending to be one numerical iteration.

## Derivative-Free Method

`cma_es` requires `system.evaluate(controls)` but does not call
`system.gradient(...)` during its population iterations.

Important distinction:

```text
engine-based optimizers can start from controls, state, or warmstart
cma_es uses the same controls, state, or warmstart start-state convention
```

## Compatibility Rules

`warmstart` can transfer current controls and metrics across optimizers.

Optimizer-private state transfers only when the source optimizer and target
optimizer match:

```python
warm = result.warmstart(target_optimizer="adam")
second = opt.optimizers.adam(system, warmstart=warm, maxiter=20)
```

If the target optimizer differs, the new method starts from the warmstarted controls
but not from incompatible private memory.

## Related Notes

- [Optimizer Methods](OPTIMIZER_METHODS.md)
- [Optimizer Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [OLGS Contract](../system_olgs/CONTRACT.md)

---

[⌂ Home](OPTIMIZERS_HUB.md)
