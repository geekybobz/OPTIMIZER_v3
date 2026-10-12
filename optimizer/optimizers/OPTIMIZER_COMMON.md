---
role: topic
parent: OPTIMIZERS_HUB.md
summary: "Shared plumbing in _common.py: shapes, clipping, warm starts, schedule resolution, and standard acceptance."
read_when: "Read when writing an optimizer method or changing a shared helper."
status: active
reviewed: 2026-10-12
---
# Common Helpers

`_common.py` contains shared plumbing for optimizer methods.

It does not choose directions, inspect physics, or define schedule classes. It
does hold the standard acceptance function and the helpers that resolve and
advance a step-size schedule, so every first-order method treats them alike.

## Helper Map

```text
accept_finite_trial
  Standard acceptance: accept every trial that evaluates successfully. The
  engine recognises it and may evaluate such a trial with value_and_gradient.

resolve_step_size
  Turn a float or schedule object into the initial step and the schedule.

next_scheduled_step
  Advance a schedule on the accepted or rejected branch of an iteration.

require_variant
  Normalize and validate method variant names.

require_finite
  Validate scalar hyperparameters.

require_probability_like
  Validate coefficients such as beta or momentum in [0, 1).

flat_state_array
  Read a flat vector from optimizer_state with shape and finite checks.

controls_from_flat_like
  Rebuild Controls from a flat vector using a reference Controls layout.

clip_vector
  Clip an update by L2 norm without changing direction.

coerce_warmstart
  Convert result/state-like inputs into WarmStartState.
```

## Shape Rule

Optimizers generally flatten controls for vector math:

```text
Controls shape = (n_controls, control_dim)
flat shape = (controls.spec.size,)
```

Any method state vector must match the flat control size.

## Clipping Rule

`clip_vector` returns:

```text
clipped_vector
original_norm
was_clipped
```

Clipping preserves direction and only changes vector length.

## Warmstart Rule

Optimizer functions call:

```python
coerce_warmstart(warmstart, target_optimizer="adam")
```

This ensures the target optimizer is explicit when private state compatibility is
checked.

## Related Notes

- [State and Warmstart](OPTIMIZER_STATE_WARMSTART.md)
- [Optimizer Lifecycle](OPTIMIZER_LIFECYCLE.md)
- [Source](./_common.py)

---

[⌂ Home](OPTIMIZERS_HUB.md)
