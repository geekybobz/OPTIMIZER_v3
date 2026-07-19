---
title: CMA-ES Optimizer
type: method_reference
module: optimizer/optimizers
method: cma_es
source: optimizer/optimizers/cma_es.py
tags:
  - optimizer
  - cma_es
  - derivative_free
  - population
  - global_search
---

# CMA-ES

`cma_es` is a compact derivative-free population search method.

It samples candidate controls around a mean, ranks them by a scalar metric, keeps
elite candidates, and adapts a sampling radius.

This implementation is CMA-ES style, not a full research-grade CMA-ES with evolution
paths and full covariance adaptation.

## Public Call

```python
result = opt.optimizers.cma_es(
    system,
    controls,
    variant="diagonal",
    population_size=24,
    sigma=0.15,
    seed=7,
    maxiter=6,
)
```

## Requires

```text
system.evaluate(controls)
Controls
```

It does not use `system.gradient(...)` during population iterations.

## Variants

```text
diagonal
  One sigma per flattened control coordinate.

isotropic
  One global sigma shared across coordinates.
```

## Important Arguments

```text
variant
maxiter
population_size
elite_fraction
sigma
min_sigma
covariance_lr
seed
accept_metric
accept_mode
accept_tolerance
trace
create_trace
blackbox
blackbox_policy
stage
use_cache
```

Current distinction:

```text
cma_es requires explicit controls
cma_es does not expose state or warmstart arguments
```

## Optimizer State

Typical keys:

```text
variant
mean
diagonal_sigma
population_size
elite_count
generation
seed
best_population_value
best_population_index
```

## Watch Out

```text
uses many system evaluations per generation
best-so-far improves only through accepted generations
rejected generations shrink the sampling radius by a factor of 0.9 toward
  min_sigma
poor sigma or large dimension can be expensive
not intended as a fast local polish method
```

## Related Theory

- [CMA-ES Theory](../../Theory/optimizers/THEORY_CMA_ES.md)

## Related Notes

- [Methods](OPTIMIZER_METHODS.md)
- [Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [Source](./cma_es.py)
