---
title: GUESS_CONTRACT
type: api_contract
module: optimizer/guesses
tags:
  - optimizer
  - guesses
  - contract
---

# Guess Contract

This file defines the public contract for initial-control guess generators.

## Required Input

Fresh guess functions accept a target:

```text
target = ControlSpec
target = system with control_spec()
```

Composite helpers accept existing `Controls`.

## Return Object

Every public guess function returns:

```text
Controls
```

The returned object has:

```text
spec
u matrix
name
meta
```

The canonical shape is:

```text
(n_controls, control_dim)
```

## Public Families

```text
constant_guess
ramp_guess
sine_guess
cosine_guess
gaussian_guess
sinc_guess
fourier_guess
random_guess
random_smooth_guess
random_fourier_guess
scale_guess
mix_guess
perturb_guess
```

## Common Promise

Generated controls should be:

```text
shape-valid
finite
named
metadata-tagged by guess family
compatible with the target ControlSpec
```

Unselected channels remain at their neutral baseline unless a composite helper mixes
existing controls.

## Public Access

All guess methods are exposed through:

```text
opt.guesses.<method>
opt.<method>
ctx.<method>
```

where `ctx` is an `OptimizerContext` bound to one system.

## Related Notes

- [Guess Common API](GUESS_COMMON_API.md)
- [Guess Methods](GUESS_METHODS.md)
- [Guess Lifecycle](GUESS_LIFECYCLE.md)
- [Controls Source](../controls.py)
