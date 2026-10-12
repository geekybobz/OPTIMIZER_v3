---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Derivative-free population search: variants, arguments, and sampling state."
read_when: "Read when calling cma_es."
status: active
reviewed: 2026-10-12
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
state
warmstart
stopping
target_value / target_metric / target_mode
stall_patience / stall_tolerance / stall_metric / stall_mode
trace
create_trace
stage
use_cache
```

Current distinction:

```text
cma_es accepts controls, state, or warmstart through the standard start-state convention
cma_es exposes standard target and stall stopping controls
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

---

[⌂ Home](OPTIMIZERS_HUB.md)
