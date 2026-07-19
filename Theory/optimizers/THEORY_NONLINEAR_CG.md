---
title: THEORY_NONLINEAR_CG
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

## Worked Example

Numbers from an ill-conditioned quadratic after one accepted step:

```text
g_prev = (1, 10)         p_prev = (-1, -10)
g      = (0.9, -0.5)     (the steep coordinate overshot and flipped sign)

y = g - g_prev = (-0.1, -10.5)

fletcher_reeves:    beta = (0.81 + 0.25) / 101         = 0.0105
polak_ribiere(+):   beta = (g^T y) / 101 = 5.16 / 101  = 0.0511
hestenes_stiefel:   beta = 5.16 / (p_prev^T y = 105.1) = 0.0491
```

With the Polak-Ribiere-plus beta:

```text
p = -g + 0.0511 * p_prev = (-0.951, -0.011)
g^T p = -0.850 < 0   -> descent direction
```

The previous-direction term almost cancels the oscillating `u2` component.
Conjugacy is exactly the mechanism that removes the zigzag left over from
steepest descent.

## Convergence Notes

```text
linear CG with exact line search terminates in at most n steps on an
  n-dimensional quadratic
nonlinear variants inherit that behavior only approximately and rely on
  (strong) Wolfe line searches for the classic guarantees
polak_ribiere_plus with restarts is the standard globally convergent choice
with this implementation's fixed engine step, conjugacy is approximate;
  expect more restarts than a Wolfe-based implementation
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

## References

```text
Fletcher & Reeves (1964); Polak & Ribiere (1969)   original beta formulas
Hager & Zhang (2006), A survey of nonlinear conjugate gradient methods
Nocedal & Wright, Numerical Optimization, 2nd ed., ch. 5
```

## API Reference

- [Nonlinear CG](../../optimizer/optimizers/NONLINEAR_CG.md)
- [Line Search](../../optimizer/optimizers/LINE_SEARCH.md)
- [L-BFGS](../../optimizer/optimizers/LBFGS.md)
