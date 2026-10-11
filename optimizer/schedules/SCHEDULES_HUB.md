---
title: SCHEDULES_HUB
type: module_hub
module: optimizer/schedules
tags:
  - optimizer
  - schedules
  - hub
---

# Schedules Hub

Small step-size policy objects for optimizer and repair workflows.

## API

| Schedule | Parameters | Behaviour |
|---|---|---|
| `ConstantSchedule` | `value` | Always returns the same step size |
| `AdaptiveStepSchedule` | `initial_step=1e-3`, `shrink=0.5`, `grow=1.0`, `min_step=1e-12`, `max_step=None` | Shrinks the step after a rejected step and grows it after an accepted one, clamped to `[min_step, max_step]` |

Both offer `initial()` for the first step size and
`update(current, accepted=...)` for the next one. The code lives in
[step_size.py](step_size.py).

## Connections

- Parent: [the V4 entry](../../README.md)
- Related: [Optimizer Acceptance](../optimizers/OPTIMIZER_ACCEPTANCE.md), [Line Search](../optimizers/LINE_SEARCH.md), [Adaptive Scaling Theory](../../Theory/optimizers/THEORY_ADAPTIVE_SCALING.md)

---

[⌂ Home](../../README.md)
