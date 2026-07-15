# `optimizers/`: Optimizer Methods

Status: review draft.
Last updated: 2026-07-15.

## Purpose

Optimizers move controls. They do not own physical objectives.

## Base Interface

```python
class Optimizer:
    name: str

    def init_state(self, system, controls):
        ...

    def propose(self, system, run_state):
        ...

    def accept(self, before, trial):
        ...
```

The exact names can change, but the responsibility should stay small.

## First Methods

Implement first:

```text
adam
momentum
line_search
```

Then:

```text
lbfgs
cg_pr
levenberg_marquardt
gauss_newton
projected_descent
nullspace_descent
```

## Adam

Needs:

```text
gradient
step schedule
Adam moments
warmstart-compatible state
```

Does not need to know what `fidelity` means.

## Momentum

Needs:

```text
gradient
velocity state
step schedule
```

## Line Search

Needs:

```text
gradient
trial evaluation
accept/reject rule
step shrink/grow policy
```

## Projected And Newton Methods

Need:

```text
residuals
jacobian
gradient for secondary descent
repair utility
```

These should be added only after diagnostics and repairs are stable.

