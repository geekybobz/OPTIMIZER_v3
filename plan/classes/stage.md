# `stage.py`: Stage And Curriculum Metadata

Status: review draft.
Last updated: 2026-07-15.

## Purpose

`Stage` describes a short optimization phase. It does not own physics and does not
assemble the objective externally.

It only says:

```text
which system params to use
which metrics to guard
how to accept or reject a chunk
what checkpoint labels to save
```

## Proposed Shape

```python
Stage(
    name="fourth_order_ramp",
    params={
        "infidelity_weight": 1e5,
        "lambda2": 1e6,
        "lambda4": 10.0,
        "energy_weight": 0.0,
    },
    guards={
        "infidelity": ("<=", 1e-8),
        "F_norm2": ("<=", 1e-6),
    },
    target="C_sym_norm2",
    accept="target_improved_with_guards",
)
```

## Manual First

Stage should first support manual use:

```python
sys_stage = sys.with_params(**stage.params)
result = opt.adam(sys_stage, controls, maxiter=10)
accepted = stage.accept(result.metrics, before_metrics)
```

Only later should `modes.curriculum` automate this.

## Acceptance Should Be Soft

A stage should not over-constrain each update. It should usually inspect the result of
a chunk.

Good:

```text
run 10 iterations, then check if fidelity survived
```

Too restrictive for early search:

```text
project every single step exactly onto all constraints
```

