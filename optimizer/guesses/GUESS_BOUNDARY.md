---
title: GUESS_BOUNDARY
type: module_boundary
module: optimizer/guesses
tags:
  - optimizer
  - guesses
  - boundary
---

# Guess Boundary

This file defines what belongs in `optimizer/guesses`.

## Guesses Own

```text
initial Controls construction
waveform templates
random starting controls
amplitude and envelope shaping
endpoint policy
composite restart helpers
compact API docs
links to theory docs
```

## Guesses Do Not Own

```text
system physics
objective metrics
analytical gradients
optimizer loops
accept/reject decisions
repair tools
long mathematical derivations
```

## Systems Own

Systems expose the control layout through:

```text
control_spec()
```

The guess layer reads this layout but does not evaluate the system.

## Optimizers Own

Optimizers consume guesses as starting controls:

```python
controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5)
result = opt.optimizers.adam(system, controls, maxiter=100)
```

## Theory Owns

Theory notes live in `Theory/guess` and hold:

```text
waveform mathematics
frequency-domain intuition
random process intuition
smoothing effects
restart strategy notes
```

## Related Notes

- [Guess Contract](GUESS_CONTRACT.md)
- [Guess Theory Hub](../../Theory/guess/GUESS_THEORY_HUB.md)
- [Optimizer Methods](../optimizers/OPTIMIZER_METHODS.md)
