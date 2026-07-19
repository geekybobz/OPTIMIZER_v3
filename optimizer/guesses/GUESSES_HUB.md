---
title: GUESSES_HUB
type: module_hub
module: optimizer/guesses
tags:
  - optimizer
  - guesses
  - hub
---

# Guess Documentation Hub

This folder contains initial-control guess generators for OPTIMIZER v3.

The goal is compact, traversable documentation: start here, then open only the
method family or common API note needed for the current task.

## Design Scope

Guess generators create `Controls` from:

```text
system.control_spec()
or
ControlSpec
```

They do not evaluate objectives, compute gradients, run optimizers, or inspect
physical dynamics.

## Documentation Map

Read in this order:

```text
GUESSES_HUB.md
  Entry point and graph map.

GUESS_CONTRACT.md
  Public API expectations and return shape.

GUESS_COMMON_API.md
  Shared target, channel, amplitude, envelope, endpoint, and scale behavior.

GUESS_METHODS.md
  Compact table of implemented guess families.

GUESS_LIFECYCLE.md
  How a raw template becomes validated Controls.

GUESS_BOUNDARY.md
  What belongs in guesses, systems, optimizers, utils, and theory.
```

Method-family references:

```text
GUESS_SIMPLE.md
GUESS_HARMONIC.md
GUESS_LOCALIZED.md
GUESS_FOURIER.md
GUESS_RANDOM.md
GUESS_RANDOM_SMOOTH.md
GUESS_RANDOM_FOURIER.md
GUESS_COMPOSITE.md
```

## Public Entry Points

Namespace style:

```python
controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)
```

Root shortcut style:

```python
controls = opt.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)
```

Bound context style:

```python
ctx = opt.context(system)
controls = ctx.random_fourier_guess(amplitude=0.1, modes=5, seed=1)
```

## Implementation Files

Current source files:

```text
optimizer/guesses/
  __init__.py
  base.py
  simple.py
  harmonic.py
  random.py
  composite.py
```

## Theory Map

Long mathematical notes live outside this runtime package:

```text
Theory/guess/
  GUESS_THEORY_HUB.md
  THEORY_CONSTANT_RAMP.md
  THEORY_HARMONIC_WAVES.md
  THEORY_LOCALIZED_PULSES.md
  THEORY_FOURIER_SERIES.md
  THEORY_RANDOM_CONTROLS.md
  THEORY_SMOOTH_RANDOM.md
  THEORY_RANDOM_FOURIER.md
  THEORY_COMPOSITE_RESTARTS.md
```

Each method-family reference links to its corresponding theory note, and each theory
note links back to the API reference.

## Related Notes

- [Guess Contract](GUESS_CONTRACT.md)
- [Guess Common API](GUESS_COMMON_API.md)
- [Guess Methods](GUESS_METHODS.md)
- [Guess Lifecycle](GUESS_LIFECYCLE.md)
- [Guess Theory Hub](../../Theory/guess/GUESS_THEORY_HUB.md)
