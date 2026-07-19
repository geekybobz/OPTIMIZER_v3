---
title: SCHEDULES_HUB
type: placeholder_hub
module: optimizer/schedules
tags:
  - optimizer
  - schedules
  - placeholder
  - hub
---

# Schedules Placeholder Hub

This folder contains step-size schedule helpers used by optimizer workflows.

This note is a placeholder. Keep it connected to the graph until schedules get a
full API and theory structure.

## Current Scope

Schedules currently own:

```text
constant step-size policy
adaptive step-size policy
accepted/rejected step updates
small reusable schedule objects
```

## Current Files

```text
optimizer/schedules/
  __init__.py
  step_size.py
```

## Future Documentation Plan

Later structure should likely include:

```text
SCHEDULE_CONTRACT.md
SCHEDULE_METHODS.md
SCHEDULE_CONSTANT.md
SCHEDULE_ADAPTIVE.md
```

Theory can connect to optimizer line-search and adaptive-scaling notes when needed.

## Related Notes

- [Root Entry](../../README.md)
- [Optimizer Acceptance](../optimizers/OPTIMIZER_ACCEPTANCE.md)
- [Line Search](../optimizers/LINE_SEARCH.md)
- [Adaptive Scaling Theory](../../Theory/optimizers/THEORY_ADAPTIVE_SCALING.md)
