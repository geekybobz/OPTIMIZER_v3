---
title: THEORY_SMOOTH_RANDOM
type: theory_reference
module: Theory/guess
related_method: optimizer/guesses/GUESS_RANDOM_SMOOTH.md
tags:
  - optimizer
  - guesses
  - theory
  - random
  - smooth
---

# Smooth Random Theory

This note gives the mathematical context for:

- [Random Smooth Guess API](../../optimizer/guesses/GUESS_RANDOM_SMOOTH.md)

## Smoothing Model

`random_smooth_guess` samples raw random rows and applies a Gaussian convolution:

```text
u_smooth = gaussian_kernel * u_raw
```

Each channel is smoothed independently.

## Correlation Length

If:

```text
0 < correlation <= 1
```

then the implementation interprets it as a fraction of `control_dim`.

Larger correlation gives smoother controls and fewer rapid sample-to-sample changes.

## Edge Padding

Rows are edge-padded before convolution. This avoids shortening the signal and keeps
the output shape unchanged:

```text
(n_controls, control_dim)
```

## Envelope and Scaling

After smoothing, the shared finalization step applies:

```text
envelope
amplitude scaling
offset
endpoint policy
```

The default envelope is `hann`, which further reduces endpoint activity.

## Practical Use

Use smooth random controls when:

```text
you want diverse multi-start guesses
raw random controls are too rough
you want smoothness without explicitly choosing Fourier modes
```

## API Reference

- [Random Smooth Guess](../../optimizer/guesses/GUESS_RANDOM_SMOOTH.md)
- [Random Guess](../../optimizer/guesses/GUESS_RANDOM.md)
- [Random Fourier Guess](../../optimizer/guesses/GUESS_RANDOM_FOURIER.md)
