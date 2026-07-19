---
title: Line Search Optimizer
type: method_reference
module: optimizer/optimizers
method: line_search
source: optimizer/optimizers/line_search.py
tags:
  - optimizer
  - line_search
  - gradient
  - first_order
---

# Line Search

`line_search` is the auditable gradient-descent family.

It computes an analytical gradient, builds a descent direction, tries one or more
step sizes, and returns a standard `OptimizerResult`.

## Public Call

```python
result = opt.optimizers.line_search(
    system,
    controls,
    variant="backtracking",
    step_size=1.0,
    maxiter=10,
)
```

## Requires

```text
system.evaluate(controls)
system.gradient(controls)
Controls
```

## Variants

```text
fixed
  One raw gradient step.

normalized
  One gradient step using a unit-norm direction.

backtracking
  Shrinks step size until the metric is acceptable.

armijo
  Uses Armijo sufficient-decrease threshold.
```

## Important Arguments

```text
step_size
min_step
max_step
shrink
grow
max_backtracks
normalize
armijo_c1
accept_metric
accept_mode
accept_tolerance
accept
state
warmstart
```

## Optimizer State

Typical keys:

```text
variant
step_size
accept_count
reject_count
last_gradient_norm
last_direction_norm
last_accepted_step_size
```

Iteration technical records include attempted step sizes for backtracking and
Armijo variants.

## Watch Out

```text
fixed steps can reject if too large
zero gradient returns a no-op rejected proposal
Armijo depends on directional derivative sign and accept mode
```

## Related Theory

- [Gradient Descent and Line Search Theory](../../Theory/optimizers/THEORY_LINE_SEARCH.md)

## Related Notes

- [Methods](OPTIMIZER_METHODS.md)
- [Acceptance and Stopping](OPTIMIZER_ACCEPTANCE.md)
- [Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [Source](./line_search.py)
