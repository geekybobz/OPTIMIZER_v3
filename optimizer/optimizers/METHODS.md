---
title: Optimizer Methods
type: method_index
module: optimizer/optimizers
tags:
  - optimizer
  - methods
  - api
---

# Optimizer Methods

This is the compact lookup table for implemented optimizer methods.

## Method Table

| Method | Source | Requires | Variants | Best For |
|---|---|---|---|---|
| `line_search` | [line_search.py](line_search.py) | `evaluate`, `gradient` | `fixed`, `backtracking`, `normalized`, `armijo` | auditable descent, step-size debugging |
| `momentum` | [momentum.py](momentum.py) | `evaluate`, `gradient` | `heavy_ball`, `nesterov`, `restart`, `clipped` | simple first-order runs, smoothing zigzag motion |
| `adam` | [adam.py](adam.py) | `evaluate`, `gradient` | `adam`, `amsgrad`, `adamw`, `radam`, `adabelief` | rough optimization, uneven gradient scales |
| `adagrad` | [adaptive.py](adaptive.py) | `evaluate`, `gradient` | none | cumulative adaptive scaling |
| `rmsprop` | [adaptive.py](adaptive.py) | `evaluate`, `gradient` | none | moving-window adaptive scaling |
| `lbfgs` | [lbfgs.py](lbfgs.py) | `evaluate`, `gradient` | none | smooth deterministic polishing |
| `nonlinear_cg` | [nonlinear_cg.py](nonlinear_cg.py) | `evaluate`, `gradient` | `fletcher_reeves`, `polak_ribiere`, `polak_ribiere_plus`, `hestenes_stiefel` | low-memory deterministic runs |
| `ncg` | [nonlinear_cg.py](nonlinear_cg.py) | `evaluate`, `gradient` | alias | short alias for `nonlinear_cg` |
| `cma_es` | [cma_es.py](cma_es.py) | `evaluate` | `diagonal`, `isotropic` | derivative-free rough global search |

## Suggested Workflow

Common research workflow:

```text
guess controls
  -> verify gradient when needed
  -> adam or cma_es for rough search
  -> line_search, nonlinear_cg, or lbfgs for deterministic improvement
  -> inspect result.metrics and result.trace
```

Typical chain:

```python
r1 = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)
r2 = opt.optimizers.lbfgs(
    system,
    r1.controls,
    warmstart=r1.warmstart(target_optimizer="lbfgs"),
    maxiter=20,
)
```

In this example, `lbfgs` receives the controls and metrics context, but it does not
reuse Adam moment vectors because Adam private state is not L-BFGS private state.

## Method References

- [Line Search](LINE_SEARCH.md)
- [Momentum](MOMENTUM.md)
- [Adam](ADAM.md)
- [AdaGrad and RMSProp](ADAGRAD_RMSPROP.md)
- [L-BFGS](LBFGS.md)
- [Nonlinear CG](NONLINEAR_CG.md)
- [CMA-ES](CMA_ES.md)

## Related Notes

- [Optimizer Contract](CONTRACT.md)
- [Optimizer Lifecycle](LIFECYCLE.md)
- [Theory Hub](../../Theory/optimizers/README.md)
