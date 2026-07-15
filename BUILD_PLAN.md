# OPTIMIZER v3 Build Plan

Status: Phase 3 complete for review.
Last updated: 2026-07-15.

This file defines the phase-by-phase build process. Each phase should be completed,
tested, summarized, and reviewed before moving to the next phase.

## Review Protocol

For every phase:

1. Explain what will be done.
2. Do only that phase.
3. Run the focused checks for that phase.
4. Summarize what changed and what remains.
5. Wait for review.
6. Move to the next phase only after explicit approval.

## Phase 0: Project Setup And Build Context

Goal: make this folder ready for tracked implementation work.

Files:

```text
BUILD_PLAN.md
CONTEXT.md
.gitignore
```

Actions:

```text
create build plan
create context/rules file
create git ignore rules
initialize git repository
```

Acceptance:

```text
BUILD_PLAN.md describes all phases
CONTEXT.md captures current architecture decisions and rules
.gitignore excludes generated/runtime artifacts
git repository exists
no package code skeleton yet
```

## Phase 1: Controls

Goal: build the vectorized control foundation.

Files:

```text
optimizer/controls.py
tests/test_controls.py
```

Build:

```text
ControlSpec
Controls
from_matrix
from_dict
zeros
copy
channel access
set_channel
as_matrix
flatten/unflatten
norms
shape validation
basic vector arithmetic
```

Acceptance:

```text
1, 2, 3, and 6 channel specs work
controls are stored as (n_controls, control_dim)
channel names and matrix order are stable
arithmetic is vectorized
focused tests pass
```

## Phase 2: System Contract

Goal: define the optimizer-facing system template.

Files:

```text
optimizer/system.py
templates/system_template.py
tests/test_system_contract.py
```

Build:

```text
OptimizerSystem protocol/base
control_spec()
evaluate()
gradient()
with_params()
optional residuals()
optional jacobian()
optional hessian()
optional hvp()
optional simulate()
```

Acceptance:

```text
template explains required and optional hooks
analytical evaluate/gradient are the normal path
old-style naming can be adapted later
dummy systems pass contract tests
```

## Phase 3: Evaluation, Result, And State

Goal: define public outputs and internal run/warmstart containers.

Files:

```text
optimizer/result.py
optimizer/state.py
tests/test_result_state.py
```

Build:

```text
Evaluation
OptimizerResult
RunState
WarmStartState
to_dict()
warmstart extraction from result
compatible/incompatible optimizer-state rules
```

Acceptance:

```text
result.controls and result.metrics work
RunState tracks current and best values
WarmStartState transfers safe fields
optimizer-specific state remains isolated
focused tests pass
```

## Phase 4: Logs And Checkpoints

Goal: make rollback and trace inspection available before serious optimization.

Files:

```text
optimizer/logs/trace.py
optimizer/logs/records.py
optimizer/logs/checkpoint.py
tests/test_trace_checkpoint.py
```

Build:

```text
Trace
IterationRecord
ChunkRecord
Checkpoint
checkpoint labels: latest, stage_start, accepted, best_J, best_safe
restore(label)
basic JSON/NPZ-safe persistence decisions
```

Acceptance:

```text
trace records system params and metrics
checkpoint can restore controls/state
stage_start rollback works
generated artifacts go to ignored paths
focused tests pass
```

## Phase 5: Core Engine

Goal: implement the single shared optimizer loop.

Files:

```text
optimizer/core/engine.py
optimizer/core/evaluate.py
optimizer/core/stopping.py
optimizer/core/parallel.py
tests/test_engine.py
```

Build:

```text
run_chunk()
evaluation/cache handling
accept/reject flow
stopping rules
trace/checkpoint integration
basic parallel helper interfaces
```

Acceptance:

```text
no optimizer owns a duplicated driver loop
dummy gradient system runs through engine
maxiter, nonfinite, target, and stall stops work
trace/checkpoint hooks are called
focused tests pass
```

## Phase 6: Public Facade

Goal: expose the direct API.

Files:

```text
optimizer/__init__.py
optimizer/library.py
tests/test_public_api.py
```

Build public calls:

```python
opt.adam(...)
opt.momentum(...)
opt.line_search(...)
opt.fourier_guess(...)
opt.repair_newton(...)
opt.geometry_probe(...)
```

Acceptance:

```text
import optimizer as opt works
explicit style opt.adam(system, controls) is the reference
optional bound-system style can wait
focused tests pass
```

## Phase 7: First Optimizers

Goal: implement the current downstream workhorse methods.

Files:

```text
optimizer/optimizers/adam.py
optimizer/optimizers/momentum.py
optimizer/optimizers/line_search.py
tests/test_optimizers_basic.py
```

Build:

```text
Adam
Momentum
LineSearch
shared engine use
warmstart among first-wave optimizers
```

Acceptance:

```text
simple quadratic system converges
optimizer switching works
trace/checkpoint works during optimization
focused tests pass
```

## Phase 8: Guesses

Goal: generate initial controls from the system control spec.

Files:

```text
optimizer/guesses/
tests/test_guesses.py
```

Build:

```text
zero_guess
constant_guess
sine_guess
sinc_guess
fourier_guess
random_guess
mix_controls
```

Acceptance:

```text
all guesses use system.control_spec()
amplitude and seed options work
1, 2, 3, and 6 channel specs pass
focused tests pass
```

## Phase 9: Diagnostics

Goal: add measurement tools that do not move controls.

Files:

```text
optimizer/utils/derivatives.py
optimizer/utils/diagnostics.py
optimizer/utils/geometry.py
tests/test_diagnostics.py
```

Build:

```text
verify_derivatives
finite_difference_gradient
finite_difference_jacobian
geometry_probe
control_spectrum
stationarity_check
```

Acceptance:

```text
derivative checks catch wrong gradients
geometry probe works when residuals/jacobian exist
diagnostics do not mutate controls
focused tests pass
```

## Phase 10: Repairs And Projected Tools

Goal: absorb advanced hard-residual repair logic.

Files:

```text
optimizer/utils/repairs.py
optimizer/optimizers/projected.py
optimizer/optimizers/newton.py
tests/test_repairs_projected.py
```

Build:

```text
repair_newton
repair_damped_newton
project_gradient
manifold_retract
projected_descent
levenberg_marquardt
gauss_newton
```

Acceptance:

```text
Newton repair reduces hard residual norm
projected direction has low linearized hard leakage
LM handles ill-conditioned residuals better than plain Newton
focused tests pass
```

## Phase 11: Advanced Optimizers

Goal: add deterministic classical methods.

Files:

```text
optimizer/optimizers/lbfgs.py
optimizer/optimizers/cg.py
optimizer/schedules/line_search.py
tests/test_advanced_optimizers.py
```

Build:

```text
strong Wolfe/backtracking infrastructure
L-BFGS
nonlinear CG PR+
restart policies
```

Acceptance:

```text
smooth deterministic test problems converge
first-wave optimizer API remains unchanged
focused tests pass
```

## Phase 12: Downstream Smoke Tests

Goal: verify the design against real systems without forcing migration.

Targets:

```text
../phd/HO_purification
../phd/CODES/three_level_system
../phd/CODES/non_unitary
../phd/CODES/universal_robust_control_4th
```

Acceptance:

```text
at least one simple downstream system runs through v3
old optimizer remains untouched
migration notes are written
```

## Phase 13: Modes Placeholder

Goal: reserve the mode layer without overbuilding it.

Files:

```text
optimizer/modes/__init__.py
```

Later possible modes:

```text
manual
benchmark
curriculum
auto
pipeline
```

Acceptance:

```text
modes are placeholders only
no mode logic precedes stable low-level tools
```
