---
title: THEORY_LBFGS
type: theory_reference
module: Theory/optimizers
related_method: optimizer/optimizers/LBFGS.md
tags:
  - optimizer
  - theory
  - lbfgs
  - quasi_newton
  - curvature
---

# L-BFGS Theory

This note gives the mathematical context for:

- [L-BFGS API](../../optimizer/optimizers/LBFGS.md)

## Quasi-Newton Idea

Newton's method would use:

```text
p = -H^{-1} g
```

where `H` is the Hessian of the objective.

L-BFGS approximates inverse-Hessian action without forming the full Hessian.

## Curvature Pairs

After an accepted move:

```text
s_t = u_{t+1} - u_t
y_t = g_{t+1} - g_t
```

The pair is useful only when it has positive curvature:

```text
s_t^T y_t > 0
```

This implementation stores a pair only when:

```text
s_t^T y_t > curvature_eps
```

## Limited Memory

Only the newest `history_size` pairs are kept:

```text
s_history = [newest accepted control differences]
y_history = [newest accepted gradient differences]
```

This makes the method practical for large flattened control vectors.

## Two-Loop Recursion

The two-loop recursion applies the approximate inverse Hessian to the gradient using
stored curvature pairs.

Output direction:

```text
p approx -B^{-1} g
```

where `B^{-1}` is the implicit inverse-Hessian approximation.

## Restart To Steepest Descent

If the computed direction is not descent:

```text
g^T p >= 0
```

the implementation restarts to:

```text
p = -g
```

## Worked Example

One accepted move on the 1-D quadratic `J(u) = 2 u^2` (true curvature `H = 4`):

```text
u: 0.5 -> 0.4        s = -0.1
g: 2.0 -> 1.6        y = -0.4

s^T y = 0.04 > curvature_eps   -> pair stored
gamma = (s^T y) / (y^T y) = 0.04 / 0.16 = 0.25 = 1/H
```

With that single pair, the two-loop recursion reproduces the exact Newton
direction at `u = 0.4`:

```text
p = -(1/H) * g = -0.25 * 1.6 = -0.4
u + 1.0 * p = 0.4 - 0.4 = 0.0   (the exact minimizer)
```

`step_size` multiplies `p`, so the default `0.1` applies a damped version of
this ideal step; on a true quadratic `step_size = 1` would land exactly.

## Convergence Notes

```text
BFGS with Wolfe line searches converges superlinearly near a smooth minimum
L-BFGS gives up superlinear speed for O(history_size * n) memory and work;
  close to a solution it still clearly beats first-order methods
stale or noisy curvature pairs degrade the direction; the curvature guard
  and steepest-descent restart bound the damage
```

## Practical Use

Use L-BFGS when:

```text
the objective is smooth and deterministic
Adam or line search has already found a reasonable region
curvature information should improve polish
```

Watch for:

```text
bad curvature pairs being skipped
step_size still needing care
no strong-Wolfe line search in this implementation
poor behavior on noisy or discontinuous objectives
```

## References

```text
Nocedal (1980), Updating quasi-Newton matrices with limited storage
Liu & Nocedal (1989), On the limited memory BFGS method for large scale
  optimization
Nocedal & Wright, Numerical Optimization, 2nd ed., ch. 7
```

## API Reference

- [L-BFGS](../../optimizer/optimizers/LBFGS.md)
- [State and Warmstart](../../optimizer/optimizers/OPTIMIZER_STATE_WARMSTART.md)
- [Optimizer Methods](../../optimizer/optimizers/OPTIMIZER_METHODS.md)
