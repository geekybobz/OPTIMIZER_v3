---
title: UTIL_BOUNDARY
type: boundary
module: optimizer/utils
tags:
  - optimizer
  - utils
  - boundary
---

# Util Boundary

This note keeps the utils module from absorbing unrelated behavior.

## Belongs in Utils

```text
metric and diagnostic summaries
finite-difference derivative checks
residual Jacobian geometry
gradient projection helpers
residual repair tools
control spectrum and smoothness diagnostics
```

## Belongs in Optimizers

```text
objective-improving control updates
optimizer-specific state
trial proposal rules
accept/reject integration with the engine
warmstart-compatible optimizer memory
```

See [Optimizer Documentation Hub](../optimizers/OPTIMIZERS_HUB.md).

## Belongs in Guesses

```text
initial control construction
waveform templates
random starts
multi-start mixtures
amplitude and envelope shaping before optimization
```

See [Guess Documentation Hub](../guesses/GUESSES_HUB.md).

## Belongs in Systems

```text
physics
objective metric definitions
analytical gradient derivations
hard residual definitions
analytical Jacobians
parameter weights
```

## Belongs in Core

```text
engine state
acceptance decisions
metric guards
run chunks
trace and checkpoint plumbing
```

`metric_guard` currently belongs here in implementation:

- [Metric Guard Source](../core/guards.py)

The catalog groups it with utils conceptually, but `optimizer.utils` does not export
`metric_guard` as a runtime function.

## Belongs in Theory

```text
finite-difference equations
Jacobian geometry
nullspace projection interpretation
Newton and LM repair derivation
FFT and roughness interpretation
research-facing explanation
```

See [Utils Theory Hub](../../Theory/utils/UTILS_THEORY_HUB.md).

## Related Notes

- [Utils Hub](UTILS_HUB.md)
- [Util Contract](UTIL_CONTRACT.md)
- [Util Methods](UTIL_METHODS.md)
