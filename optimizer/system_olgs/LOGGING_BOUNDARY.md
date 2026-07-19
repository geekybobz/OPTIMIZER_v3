---
title: OLGS Logging Boundary
type: integration_boundary
module: optimizer/system_olgs
tags:
  - optimizer
  - system
  - olgs
  - logging
  - blackbox
---

# OLGS Logging Boundary

OLGS should not fully own cache, logs, checkpoints, or blackbox records yet.

Those parts will be redesigned separately.

## Current Boundary

OLGS may expose lightweight lifecycle hooks:

```text
cache_reset()
cache_status()
latest metrics placeholder
latest state placeholder
```

But persistent run history should belong to the logging/blackbox layer.

## Future Connection Point

The future blackbox/logging layer can connect to OLGS through:

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

Keep logs/blackbox history-focused:

```text
record evaluations
record optimizer steps
record checkpoints
store replayable run data
manage persistent cache
```

## Related Notes

- [OLGS System Hub](SYSTEM_OLGS_HUB.md)
- [OLGS System Build Skill](templates/OLGS_SYSTEM_BUILD_SKILL.md)
- [API Contract](CONTRACT.md)
- [Computation Lifecycle](LIFECYCLE.md)
- [Derivative Hooks](DERIVATIVES.md)
