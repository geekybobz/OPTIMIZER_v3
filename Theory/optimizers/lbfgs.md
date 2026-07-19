---
title: L-BFGS Theory
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

## API Reference

- [L-BFGS](../../optimizer/optimizers/LBFGS.md)
- [State and Warmstart](../../optimizer/optimizers/STATE_AND_WARMSTART.md)
- [Optimizer Methods](../../optimizer/optimizers/METHODS.md)
