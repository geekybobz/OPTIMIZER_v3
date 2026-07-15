# Build Order

Status: review draft.
Last updated: 2026-07-15.

The implementation should be built layer by layer. Do not build modes first.

## Layer 1: Controls

Files:

```text
optimizer/controls.py
```

Build:

```text
ControlSpec
Controls
channel access
matrix access
copying
arithmetic
shape validation
```

Reason:

Everything else depends on a reliable vectorized control container.

## Layer 2: System Contract

Files:

```text
optimizer/system.py
```

Build:

```text
OptimizerSystem
control_spec()
evaluate()
gradient()
residuals()
jacobian()
with_params()
```

Reason:

The system contract decides how every downstream model talks to the library.

## Layer 3: Evaluation, Result, State

Files:

```text
optimizer/result.py
optimizer/state.py
```

Build:

```text
Evaluation
OptimizerResult
RunState
WarmStartState
```

Reason:

Optimizers, logs, and warmstart need a stable result/state shape.

## Layer 4: Logs And Checkpoints

Files:

```text
optimizer/logs/trace.py
optimizer/logs/records.py
optimizer/logs/checkpoint.py
```

Build:

```text
Trace
IterationRecord
ChunkRecord
Checkpoint
restore labels
```

Reason:

Curriculum needs rollback and diagnosis from the beginning.

## Layer 5: Core Engine

Files:

```text
optimizer/core/engine.py
optimizer/core/evaluate.py
optimizer/core/stopping.py
optimizer/core/parallel.py
```

Build:

```text
single iteration loop
chunk execution
evaluate/cache handling
accept/reject policy
stopping
trace integration
checkpoint integration
```

Reason:

Avoid v2-style duplicated optimizer loops.

## Layer 6: First Optimizers

Files:

```text
optimizer/optimizers/adam.py
optimizer/optimizers/momentum.py
optimizer/optimizers/line_search.py
```

Build:

```text
Adam
Momentum
LineSearch
warmstart transfer
basic schedules
```

Reason:

These match existing downstream use and prove the new contract.

## Layer 7: Guesses

Files:

```text
optimizer/guesses/
```

Build:

```text
zero
constant
sine
sinc
fourier
random
mix
```

Reason:

Downstream workflows need easy starts based on `system.control_spec()`.

## Layer 8: Diagnostics And Repairs

Files:

```text
optimizer/utils/diagnostics.py
optimizer/utils/derivatives.py
optimizer/utils/repairs.py
optimizer/utils/geometry.py
```

Build:

```text
finite-difference derivative check
geometry probe
Newton repair
projected gradient helpers
control spectrum
```

Reason:

These absorb logic currently living in project-specific scripts.

## Layer 9: Advanced Optimizers

Files:

```text
optimizer/optimizers/newton.py
optimizer/optimizers/projected.py
optimizer/optimizers/lbfgs.py
```

Build:

```text
projected descent
Levenberg-Marquardt
Gauss-Newton
L-BFGS
nonlinear CG
```

Reason:

These require the lower layers to be stable first.

## Layer 10: Modes Placeholder

Files:

```text
optimizer/modes/
```

Build only placeholders first.

Later:

```text
manual
benchmark
curriculum
auto
pipeline
```

