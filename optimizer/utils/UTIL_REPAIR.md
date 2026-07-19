---
title: UTIL_REPAIR
type: category_reference
module: optimizer/utils
source: optimizer/utils/repairs.py
methods:
  - repair_newton
  - RepairResult
tags:
  - optimizer
  - utils
  - residual
  - repair
  - newton
---

# Repair

Repair utilities move controls to reduce named residual violations.

They do not optimize `J`, fidelity, or energy directly.

## Public Call

```python
fixed = opt.utils.repair_newton(
    system,
    controls,
    residuals="hard",
    method="lm",
    maxiter=10,
    tolerance=1.0e-10,
)
```

## Methods

Supported solve modes:

```text
newton
lm
damped
```

`newton` solves the least-squares residual equation directly.

`lm` and `damped` solve through a damped residual-space Gram system.

## Line Search

By default, repair uses a shrink line search:

```text
candidate residual norm must be lower than current residual norm
```

If no trial improves the residual norm, the stop reason becomes:

```text
line_search_failed
```

## `RepairResult`

Returned fields:

```text
controls
residuals
residual_norm
residual_max_abs
converged
iterations
method
residual_name
jacobian_source
history
stop_reason
```

Use:

```python
payload = fixed.to_dict(include_controls=False)
```

for compact logs.

## Best For

```text
post-optimizer residual cleanup
local feasibility restoration
testing whether residual Jacobian information is useful
residual-polish stages
```

## Watch Out

```text
repair is local
finite-difference fallback can be expensive
damping changes the repair direction
line search can fail when the local model is poor
```

## Related Theory

- [Newton Repair Theory](../../Theory/utils/THEORY_NEWTON_REPAIR.md)

## Related Notes

- [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Source](./repairs.py)
