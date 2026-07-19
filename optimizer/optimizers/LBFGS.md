---
title: L-BFGS Optimizer
type: method_reference
module: optimizer/optimizers
method: lbfgs
source: optimizer/optimizers/lbfgs.py
tags:
  - optimizer
  - lbfgs
  - quasi_newton
  - polish
---

# L-BFGS

`lbfgs` is a limited-memory quasi-Newton polish method.

It uses accepted control differences and gradient differences to approximate inverse
Hessian action without storing a full Hessian.

## Public Call

```python
result = opt.optimizers.lbfgs(
    system,
    controls,
    history_size=10,
    step_size=0.1,
    maxiter=20,
)
```

## Requires

```text
system.evaluate(controls)
system.gradient(controls)
Controls
```

## Important Arguments

```text
history_size
step_size
curvature_eps
max_step_norm
state
warmstart
accept_metric
accept_mode
accept_tolerance
```

## Optimizer State

Typical keys:

```text
history_size
s_history
y_history
step_size
accept_count
reject_count
last_gradient
last_gradient_norm
last_curvature
last_pair_accepted
last_step_clipped
```

Curvature pairs are stored only when:

```text
s dot y > curvature_eps
```

History length never exceeds `history_size`.

## Watch Out

```text
first iteration falls back to steepest descent
nonpositive curvature pairs are skipped
this implementation does not include a strong-Wolfe line search
use after rough optimization rather than as global search
```

## Related Theory

- [L-BFGS Theory](../../Theory/optimizers/lbfgs.md)

## Related Notes

- [Methods](METHODS.md)
- [State and Warmstart](STATE_AND_WARMSTART.md)
- [Source](./lbfgs.py)
