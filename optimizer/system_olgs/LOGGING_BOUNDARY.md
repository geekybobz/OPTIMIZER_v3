---
title: OLGS Logging Boundary
type: integration_boundary
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - logging
---

# OLGS Logging Boundary

OLGS should not fully own cache, logs, checkpoints, or future tracker records.

Those parts will be redesigned separately.

## Current Boundary

OLGS may expose lightweight lifecycle hooks:

```text
cache_reset()
cache_status()
latest metrics placeholder
latest state placeholder
```

Persistent run history will belong to the tracker layer.

## Future Connection Point

The future tracker can connect to OLGS through:

```text
control identity
forward result metadata
evaluation metrics
gradient metadata
residual/Jacobian metadata
system describe payload
```

OLGS should keep enough structure for logging to observe computations, but not decide
the final storage model.

## Design Rule

Keep OLGS computation-focused:

```text
run physics
return metrics
return gradients
expose derivative hooks
describe itself
```

Keep tracker history-focused:

```text
record evaluations
record optimizer steps
record checkpoints
store replayable run data
manage persistent cache
```

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [API Contract](CONTRACT.md)
- [Computation Lifecycle](LIFECYCLE.md)
- [Derivative Hooks](DERIVATIVES.md)

---

[⌂ Home](SYSTEM_OLGS_HUB.md)
