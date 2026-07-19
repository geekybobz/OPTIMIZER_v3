# OPTIMIZER v3 Architecture

## Core Model

OPTIMIZER v3 separates project physics from optimizer engineering.

The project system owns:

```text
control layout
dynamics or simulation
objective value J
metric names
objective weights / system parameters
analytical gradient
optional residuals and Jacobians
```

The library owns:

```text
control containers
optimizer loops
step proposals and acceptance
warmstart state
guess generation
diagnostics and repairs
trace records and checkpoints
standard result objects
```

This keeps optimizer calls general. A different project can use different metric names
without changing optimizer code.

## Public Shape

The preferred import style is:

```python
import optimizer as opt
```

Role namespaces are the stable public surface:

```text
opt.optimizers.*
opt.guesses.*
opt.utils.*
opt.schedules.*
```

Direct shortcuts remain available for compact interactive work, but new examples
should prefer the role namespaces.

## Main Flow

```text
system.control_spec()
        ↓
initial Controls from opt.guesses.*
        ↓
optimizer chunk from opt.optimizers.*
        ↓
OptimizerResult with controls, metrics, state, trace
        ↓
diagnostics/repair from opt.utils.*
        ↓
optional warmstart into another optimizer
```

The native unit of optimization is a short chunk. Longer workflows should chain chunks
and carry `WarmStartState` or `OptimizerResult.warmstart()` when moving between
methods.

## Controls

`ControlSpec` defines the channel layout, control dimension, time step, bounds, and
metadata. `Controls` stores the numeric values as a vectorized channel-by-time matrix.

All optimizers and tools operate on `Controls`, not ad hoc tuples. This makes
warmstart, diagnostics, checkpointing, and serialization consistent.

## System Contract

Required:

```python
control_spec()
evaluate(controls)
gradient(controls)
```

Recommended for constrained or staged workflows:

```python
residuals(controls, name="...")
jacobian(controls, name="...")
with_secondary(...)
metric_schema()
```

`evaluate` must return a dictionary containing finite scalar `J`. Other metrics are
project-defined.

## Optimizers

Optimizers move controls to reduce the current system `J`. They do not know the
physical meaning of that objective.

Implemented families:

```text
Adam
momentum
line search
AdaGrad
RMSProp
L-BFGS
nonlinear conjugate gradient
CMA-ES
```

Method-specific memory lives in `RunState.optimizer_state`, so a result can warmstart
another method without exposing private internals as public API.

## Diagnostics And Repairs

Diagnostics measure without changing controls. Repairs deliberately move controls to
restore feasibility.

Important tool groups:

```text
finite-difference checks
gradient/Jacobian verification
geometry and rank probes
projected gradients
Newton-style residual repair
control spectrum and smoothness summaries
```

These tools use the same `system`, `Controls`, and metric dictionaries as optimizers,
so they can be inserted between chunks without adapter code.

## Logs And Checkpoints

The current log layer is intentionally lightweight:

```text
IterationRecord
ChunkRecord
EventRecord
Trace
Checkpoint
```

`Trace` is in-memory and records technical optimizer data. `Checkpoint` stores
restorable controls plus optional run state. A later persistence layer can write a
compact mathematical ledger for full project runs, but the low-level objects should
remain small and serialization-friendly.

## Installation Model

The repository is installable as package `optimizer-v3` with import name `optimizer`.
The standard install is editable inside conda environment `optimizer_v3`.

`install.sh` records installation state in `~/.optimizer_v3/install.json`.
`update.sh` fetches Git updates, fast-forwards only when safe, then reinstalls.
`uninstall.sh` reads the state file and removes the package or the whole environment.

## Privacy Rule

Generated controls, project-specific reference artifacts, run outputs, checkpoints,
and local result data should not be tracked unless deliberately reviewed. Numeric
artifacts are ignored by default, and `systems/*/reference/` is private/local.
