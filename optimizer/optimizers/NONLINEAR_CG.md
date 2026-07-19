---
title: Nonlinear Conjugate Gradient
type: method_reference
module: optimizer/optimizers
method: nonlinear_cg
source: optimizer/optimizers/nonlinear_cg.py
tags:
  - optimizer
  - nonlinear_cg
  - conjugate_gradient
  - low_memory
---

# Nonlinear Conjugate Gradient

`nonlinear_cg` is a low-memory deterministic optimizer for large control vectors.

It combines the current negative gradient with the previous accepted direction.

## Public Call

```python
result = opt.optimizers.nonlinear_cg(
    system,
    controls,
    variant="polak_ribiere_plus",
    step_size=0.05,
    maxiter=25,
)
```

Short alias:

```python
result = opt.optimizers.ncg(system, controls, maxiter=25)
```

## Requires

```text
system.evaluate(controls)
system.gradient(controls)
Controls
```

## Variants

```text
fletcher_reeves
polak_ribiere
polak_ribiere_plus
hestenes_stiefel
```

## Important Arguments

```text
variant
step_size
max_step_norm
restart_on_nondescent
state
warmstart
accept_metric
accept_mode
accept_tolerance
```

## Optimizer State

Typical keys:

```text
variant
previous_gradient
direction
step_size
beta
restarted
accept_count
reject_count
last_gradient_norm
last_direction_norm
last_raw_step_norm
last_step_clipped
```

## Watch Out

```text
non-descent directions restart to steepest descent when enabled
this implementation does not perform a full Wolfe line search
may restart often when gradients are noisy or poorly scaled
```

## Related Theory

- [Nonlinear CG Theory](../../Theory/optimizers/nonlinear_cg.md)

## Related Notes

- [Methods](METHODS.md)
- [L-BFGS](LBFGS.md)
- [Line Search](LINE_SEARCH.md)
- [Source](./nonlinear_cg.py)
