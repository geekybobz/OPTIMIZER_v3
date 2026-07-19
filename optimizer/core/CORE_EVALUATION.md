---
title: Core Evaluation
type: helper_reference
module: optimizer/core
tags:
  - optimizer
  - core
  - evaluation
  - cache
---

# Core Evaluation

This file documents `SystemEvaluator`, the only path the engine uses to reach a
system's `evaluate` and `gradient`.

Systems own the physics.  Every optimizer still needs the same surrounding work:
validate that controls match the system, normalize metrics, reject non-finite
results, and avoid repeating an expensive propagation for controls that have not
changed.  `SystemEvaluator` is where that work lives, once.

## Construction

```python
evaluator = opt.SystemEvaluator(system, use_cache=True)
```

```text
system      validated on construction via require_system
use_cache   reuse evaluations for identical control content, default True
```

## Two Call Styles

```text
evaluate(controls)      -> Evaluation, raises on failure
gradient(controls)      -> Controls, raises on failure
try_evaluate(controls)  -> EvaluationOutcome, never raises
try_gradient(controls)  -> GradientOutcome, never raises
```

The engine uses the `try_` forms so a system failure becomes a named stop reason
instead of a traceback.  Direct callers, including diagnostics and a method's own
probing inside a proposal, usually want the raising forms.

Outcome shape:

```text
EvaluationOutcome   evaluation, ok, reason, error
GradientOutcome     gradient, ok, reason, error
```

`error` is formatted as `"TypeName: message"`, which is what appears in the technical
payload of a failed iteration record.

## The Evaluation Cache

The cache key is the exact numeric content of the controls, not their identity:

```text
control keys
control_dim
dtype
shape
raw matrix bytes
```

Two separately built `Controls` objects with identical layout and identical numbers
hit the same cache entry.  A single changed element misses.

Consequences worth knowing:

```text
re-evaluating the current controls in a proposal is free
line search that revisits a candidate step pays once
the cache is per-evaluator, so it dies with the chunk
```

Gradients are **not** cached.  Every `gradient` call reaches the system.  Repeated
gradient requests at the same controls therefore cost full price.

## Counters

```text
evaluation_count   system evaluations actually performed
gradient_count     gradient calls performed
```

Cache hits do not increment `evaluation_count`, so the counters measure real system
work rather than call-site traffic.  Both are copied into the `technical` payload of
each iteration record.

`clear_cache()` drops cached evaluations without resetting the counters or replacing
the system.

## What This Layer Refuses To Do

```text
no finite-difference gradient fallback
no acceptance or stopping decisions
no logging
no control updates
```

Gradients come from the system, analytically.  Finite-difference checking and
fallbacks are diagnostics and live in
[Util Derivative Checks](../utils/UTIL_DERIVATIVE_CHECKS.md).

## Related Notes

- [Core Hub](CORE_HUB.md)
- [Core Lifecycle](CORE_LIFECYCLE.md)
- [Core Contract](CORE_CONTRACT.md)
- [Core Boundary](CORE_BOUNDARY.md)
- [OLGS Contract](../system_olgs/CONTRACT.md)
- [OLGS Computation Lifecycle](../system_olgs/LIFECYCLE.md)
- [Core Evaluation Source](evaluate.py)
