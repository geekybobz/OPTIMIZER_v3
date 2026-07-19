---
title: Nonlinear CG Theory
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/NONLINEAR_CG.md
tags:
  - optimizer
  - theory
  - nonlinear_cg
  - conjugate_gradient
  - low_memory
---

# Nonlinear Conjugate Gradient Theory

This note gives the mathematical context for:

- [Nonlinear CG API](../../optimizer/optimizers/NONLINEAR_CG.md)

## Basic Direction

Nonlinear conjugate gradient builds:

```text
p_t = -g_t + beta_t * p_{t-1}
u_next = u + alpha * p_t
```

The first iteration uses steepest descent:

```text
p_0 = -g_0
```

## Fletcher-Reeves

```text
beta = (g_t^T g_t) / (g_{t-1}^T g_{t-1})
```

This uses the ratio of current and previous gradient energy.

## Polak-Ribiere

Let:

```text
y = g_t - g_{t-1}
```

Then:

```text
beta = (g_t^T y) / (g_{t-1}^T g_{t-1})
```

`polak_ribiere_plus` clips beta below at zero:

```text
beta = max(0, beta)
```

## Hestenes-Stiefel

```text
beta = (g_t^T y) / (p_{t-1}^T y)
```

If the denominator is too small, this implementation returns beta zero.

## Restart Logic

The direction should be descent for minimization:

```text
g_t^T p_t < 0
```

If `restart_on_nondescent=True` and the direction is not descent, the method resets:

```text
p_t = -g_t
beta = 0
```

## Practical Use

Use nonlinear CG when:

```text
control vectors are large
memory should stay low
you want a middle ground between gradient descent and L-BFGS
```

Watch for:

```text
missing strong-Wolfe line search
frequent restarts with noisy gradients
sensitivity to step_size
```

## API Reference

- [Nonlinear CG](../../optimizer/optimizers/NONLINEAR_CG.md)
- [Line Search](../../optimizer/optimizers/LINE_SEARCH.md)
- [L-BFGS](../../optimizer/optimizers/LBFGS.md)
