---
title: Optimizer Common Helpers
type: helper_reference
module: optimizer/optimizers
source: optimizer/optimizers/_common.py
tags:
  - optimizer
  - common
  - helpers
---

# Common Helpers

`_common.py` contains shared plumbing for optimizer methods.

It does not choose directions, inspect physics, accept trials, or implement
schedules.

## Helper Map

```text
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

- [State and Warmstart](STATE_AND_WARMSTART.md)
- [Optimizer Lifecycle](LIFECYCLE.md)
- [Source](./_common.py)
