---
role: front-door
parent: ../../README.md
summary: "Step-size schedules for optimizers and repair: constant, adaptive, exponential decay, warm-up, and cosine decay."
read_when: "Read when giving a first-order method a step-size schedule."
status: active
reviewed: 2026-10-12
---
# Schedules Hub

Small step-size policy objects for optimizer and repair workflows.

## API

| Schedule | Parameters | Behaviour |
|---|---|---|
| `ConstantSchedule` | `value` | Always returns the same step size |
| `AdaptiveStepSchedule` | `initial_step=1e-3`, `shrink=0.5`, `grow=1.0`, `min_step=1e-12`, `max_step=None` | Shrinks the step after a rejected step and grows it after an accepted one, clamped to `[min_step, max_step]` |
| `ExponentialDecaySchedule` | `initial_step`, `decay=0.99`, `min_step=1e-12` | Decays from the initial step by accepted-iteration index |
| `LinearWarmupSchedule` | `initial_step`, `target_step`, `warmup_steps` | Increases linearly to a target step; rejection holds the current step |
| `CosineDecaySchedule` | `initial_step`, `total_steps`, `min_step=0` | Cosine decay over a fixed horizon; rejection holds the current step |

All schedules offer `initial()` for the first step size and
`update(current, accepted=...)` for the next one. The code lives in
[step_size.py](step_size.py).

First-order methods accept either a positive scalar or a schedule where documented.
The engine passes the current acceptance result and iteration index; a schedule owns
only scalar policy, never direction, objective evaluation, or acceptance.

## Connections

- Related: [the V4 entry](../../README.md)
- Related: [Optimizer Acceptance](../optimizers/OPTIMIZER_ACCEPTANCE.md), [Line Search](../optimizers/LINE_SEARCH.md), [Adaptive Scaling Theory](../../Theory/optimizers/THEORY_ADAPTIVE_SCALING.md)

---

[⌂ Home](../../README.md)
