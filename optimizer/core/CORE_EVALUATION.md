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
value_and_gradient(controls) -> (Evaluation, Controls), optional combined capability
residuals/jacobian      -> validated equality-constraint facts
hvp                     -> validated Hessian-vector product
smooth_gradient/prox    -> validated composite-objective operations
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
evaluation_count           successfully validated system evaluations
gradient_count             successfully validated gradient calls
evaluation_attempt_count   cache-miss system evaluation attempts, including failures
gradient_attempt_count     gradient attempts, including failures
value_gradient_count       successful combined value/gradient calls
residual_count             successful hard-residual calls
jacobian_count             successful constraint-Jacobian calls
hvp_count                  successful Hessian-vector products
smooth_gradient_count      successful smooth-component gradients
prox_count                 successful proximal operations
```

Cache hits do not increment either evaluation counter.  The attempted counters make
failed/non-finite trials visible as real work without changing the established
successful-call counters.  Both views are copied into the iteration evidence.

`clear_cache()` drops cached evaluations without resetting the counters or replacing
the system.

`counters(compact=True)` retains the original evaluation/gradient counter keys and
adds only nonzero conditional-capability counters. This keeps ordinary tracker
records compact while preserving advanced-method work evidence.

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

---

[⌂ Home](CORE_HUB.md)
