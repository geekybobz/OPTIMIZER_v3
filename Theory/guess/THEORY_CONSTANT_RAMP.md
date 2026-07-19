---
title: THEORY_CONSTANT_RAMP
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_SIMPLE.md
tags:
  - optimizer
  - guesses
  - theory
  - deterministic
  - ramp
---

# Constant and Ramp Theory

This note gives the mathematical context for:

- [Simple Guess API](../../optimizer/guesses/GUESS_SIMPLE.md)

## Constant Controls

A constant channel is:

```text
u_i(t_j) = c_i
```

where `i` is the control channel and `j` is the grid sample.

Constants are useful for checking sign conventions, channel mapping, and simple
offset behavior.

## Linear Ramp

A linear ramp between start `a` and stop `b` is:

```text
u(t) = a + (b - a) * t
```

with normalized time:

```text
t in [0, 1]
```

## Quadratic Ramp

The quadratic profile uses:

```text
u(t) = a + (b - a) * t^2
```

This starts more slowly and changes faster near the end.

## Smoothstep Ramp

The smoothstep profile uses:

```text
s(t) = 3t^2 - 2t^3
u(t) = a + (b - a) * s(t)
```

It has flat first derivative at both ends:

```text
s'(0) = 0
s'(1) = 0
```

## Practical Use

Use constants and ramps when:

```text
you need deterministic starting controls
you are checking channel semantics
you want a simple turn-on profile
you want a baseline before harmonic or random starts
```

## API Reference

- [Simple Guesses](../../optimizer/guesses/GUESS_SIMPLE.md)
- [Guess Common API](../../optimizer/guesses/GUESS_COMMON_API.md)
- [Guess Methods](../../optimizer/guesses/GUESS_METHODS.md)
