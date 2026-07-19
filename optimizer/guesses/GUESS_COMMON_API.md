---
title: GUESS_COMMON_API
type: common_api
module: optimizer/guesses
source: optimizer/guesses/base.py
tags:
  - optimizer
  - guesses
  - common_api
---

# Guess Common API

`base.py` owns shared behavior used by simple, harmonic, random, and composite guess
families.

## Target Resolution

Fresh guesses accept:

```text
ControlSpec
system with control_spec()
```

`resolve_spec(target)` returns the `ControlSpec`.

## Time Grid

`time_grid(spec)` returns endpoint-inclusive normalized samples:

```text
t in [0, 1]
length = spec.control_dim
```

Waveform frequencies, centers, and widths are interpreted on this normalized grid.

## Channel Selection

`channels` may be:

```text
None
single channel name
iterable of channel names
```

When `channels is None`, all channels are selected.

## Channel Values

Parameters such as `amplitude`, `offset`, `start`, `stop`, `frequency`, `phase`,
`center`, and `width` can be:

```text
scalar
mapping from channel name to scalar
sequence matching all channels
sequence matching selected channels
```

## Envelope Options

Supported envelopes:

```text
none
flat
free
hann
sin2
gaussian
smoothstep
```

Envelopes multiply the raw waveform before scaling.

## Endpoint Policy

Supported endpoint policies:

```text
free
hold
zero
```

`free` and `hold` preserve endpoints. `zero` sets selected first and last samples to
0 after waveform construction.

## Scale Modes

Supported `scale` modes:

```text
max_abs
l2
energy
none
```

`max_abs` makes the selected channel's maximum absolute value match `amplitude`.

`l2` scales by Euclidean norm.

`energy` scales by:

```text
sqrt(dt * sum(u^2))
```

`none` multiplies raw waveform values directly by `amplitude`.

## Related Notes

- [Guess Lifecycle](GUESS_LIFECYCLE.md)
- [Guess Contract](GUESS_CONTRACT.md)
- [Common Source](./base.py)
