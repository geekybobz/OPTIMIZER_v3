# Architecture

Status: review draft.
Last updated: 2026-07-15.

## One Sentence

OPTIMIZER v3 is a direct-call optimization toolbox where `system.py` owns the model
and analytical cost structure, while the library supplies vectorized controls,
optimizers, schedules, guesses, diagnostics, repairs, logs, and later modes.

## Main Package Layout

Preferred target layout:

```text
optimizer/
  __init__.py
  library.py

  system.py
  controls.py
  result.py
  state.py
  stage.py

  core/
    engine.py
    evaluate.py
    stopping.py
    parallel.py
    registry.py

  optimizers/
    adam.py
    momentum.py
    line_search.py
    newton.py
    projected.py

  schedules/
    constant.py
    adaptive.py
    trust_region.py

  guesses/
    zero.py
    constant.py
    sine.py
    sinc.py
    fourier.py
    random.py
    mix.py

  constraints/
    bounds.py
    projections.py
    norm.py

  utils/
    diagnostics.py
    derivatives.py
    repairs.py
    geometry.py
    fourier_tools.py

  logs/
    trace.py
    records.py
    checkpoint.py
    reports.py

  modes/
    __init__.py
```

## Why No `problem.py` For Now

`problem.py` risks becoming a wrapper around information that already exists:

```text
system + controls + current system params
```

For this project, the system already knows the objective components and how the
gradient is derived. A separate `Problem` object can be added later only if it removes
real complexity. For the first skeleton, avoid it.

If staged optimization is needed, use `stage.py`. A stage describes temporary cost
prefactors, guard checks, and accept/rollback rules. It does not own physics.

## Responsibility Split

### `system.py`

Owns:

- control dimension and control names
- time grid and `dt`
- physical dynamics
- cost prefactors as system params
- objective value `J`
- metrics dictionary
- analytical gradient
- optional residuals, Jacobian, Hessian, Hessian-vector products
- simulation and diagnostics that depend on the physical model

### `controls.py`

Owns:

- vectorized control container
- shape checking
- named channels
- matrix conversion
- arithmetic needed by optimizers
- metadata needed by guesses and logs

### `optimizers/`

Owns:

- methods that move controls
- Adam, momentum, line search, L-BFGS, projected descent, Newton/LM variants
- optimizer-specific state, such as Adam moments

Optimizers do not own physical cost definitions.

### `schedules/`

Owns:

- learning-rate or step-size policy
- trust-region radius
- LM damping updates
- adaptive shrink/grow logic

### `guesses/`

Owns:

- zero, constant, sine, sinc, Fourier, random, mixed starts
- automatic use of `system.control_spec()`
- amplitude-aware generation

### `constraints/`

Owns:

- bounds
- projections
- norm caps
- feasibility transforms

### `utils/`

Owns:

- derivative checks
- geometry probes
- repairs
- Fourier tools
- diagnostic helpers

### `logs/`

Owns:

- lightweight trace rows
- restorable checkpoints
- reports
- rollback lookup

### `modes/`

Placeholder for later:

- auto mode
- pipeline mode
- benchmark mode
- curriculum mode

Until the lower layers are stable, modes should not drive design.

## Main Object Model

Core classes:

```text
OptimizerSystem
ControlSpec
Controls
Evaluation
OptimizerResult
RunState
WarmStartState
Stage
Trace
Checkpoint
Engine
Optimizer
Schedule
```

These are detailed in `classes/`.

## Multi-Objective Cost Model

The system may have many objective components:

```text
J = wF * infidelity
  + w2 * F_norm2
  + w4 * C_sym_norm2
  + wE * energy
```

The optimizer should not assemble this formula. The system computes `J` and its
analytical gradient using the current system params.

This matches the inspected downstream pattern: current systems already place
`control_weight`, `lambda2`, `lambda4`, and `energy_weight` inside the model-derived
costate and gradient logic.

## Hard And Soft Quantities

Soft quantities go into `J`:

```text
infidelity, F_norm2, C_sym_norm2, energy, smoothness, penalties
```

Hard quantities go into `residuals()`:

```text
terminal leakage, exact robustness constraints, equality constraints
```

Early optimization should mostly use soft weights and guards. Hard projected methods
and Newton repairs should be late-stage tools.

