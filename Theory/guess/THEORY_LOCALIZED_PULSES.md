---
title: THEORY_LOCALIZED_PULSES
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_LOCALIZED.md
tags:
  - optimizer
  - guesses
  - theory
  - localized
  - pulses
---

# Localized Pulse Theory

This note gives the mathematical context for:

- [Localized Guess API](../../optimizer/guesses/GUESS_LOCALIZED.md)

## Gaussian Pulse

The Gaussian template is:

```text
u(t) = exp(-0.5 * ((t - center) / width)^2)
```

`center` sets the pulse location. `width` sets the normalized spread.

Small width gives a narrow pulse. Large width gives a broad pulse.

## Sinc Pulse

The sinc template is:

```text
u(t) = sinc(width * (t - center))
```

The main lobe is centered at `center`. Larger width increases side-lobe density.

## Localization Tradeoff

Localization in time affects frequency content:

```text
narrow time pulse -> broader frequency content
broad time pulse  -> narrower frequency content
```

This matters when a physical system is sensitive to bandwidth or sharp control
features.

## Envelope Interaction

An envelope multiplies the pulse after raw construction and before scaling.

This can:

```text
reduce endpoint values
soften pulse edges
change effective width
change spectral leakage
```

## Practical Use

Use localized pulses when:

```text
the control action should be concentrated in a time window
you want a smooth pulse near a guessed interaction time
you need a deterministic alternative to random starts
```

## API Reference

- [Localized Guesses](../../optimizer/guesses/GUESS_LOCALIZED.md)
- [Guess Common API](../../optimizer/guesses/GUESS_COMMON_API.md)
- [Guess Methods](../../optimizer/guesses/GUESS_METHODS.md)
