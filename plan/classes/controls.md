# `controls.py`: ControlSpec And Controls

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Controls are the shared vectorized representation used by systems, optimizers,
guesses, constraints, logs, and checkpoints.

## ControlSpec

`ControlSpec` describes the layout:

```python
ControlSpec(
    keys=("ux", "uy", "uz"),
    control_dim=1001,
    dtype=float,
    dt=0.001,
    meta={...},
)
```

It should expose:

```text
keys
n_controls
control_dim
shape
dtype
dt
meta
```

## Controls

`Controls` stores the values:

```python
controls = Controls.zeros(spec)
matrix = controls.as_matrix()
ux = controls.channel("ux")
```

Required behavior:

```text
shape validation
named channel access
matrix conversion
copy
rename
set channel
from dict
from matrix
zeros
constant
arithmetic operations
norms
flatten/unflatten
```

## Vectorization

The internal representation should be a matrix:

```text
(n_controls, control_dim)
```

This allows direct vectorized arithmetic:

```python
new_u = old_u + step * direction
```

## Why This Matters

Downstream systems have 1, 2, 3, and 6 control channels. Guess generation and optimizer
updates should work for all of them without special-case code.

