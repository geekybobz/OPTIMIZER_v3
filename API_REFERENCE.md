---
role: generated
parent: README.md
summary: Generated exact signatures and catalog semantics for every public OPTIMIZER tool.
read_when: Looking up a callable signature, default, requirement, return, or runtime identifier.
status: active
reviewed: 2026-10-12
---

<!-- Generated file. Do not edit directly. Source: optimizer/catalog.py and runtime signatures. Regenerate with: python tools/generate_api_reference.py -->

# API Reference

This file is generated from `optimizer.catalog` and live Python signatures.
Contract pages explain meaning and invariants; this reference owns exact callable
surfaces and defaults.

## Groups

| Group | Entries | Purpose |
|---|---:|---|
| [`jobs`](#jobs) | 15 | Bounded client-lifetime process jobs and explicit candidate retrieval. |
| [`operations`](#operations) | 5 | Foreground session and native operation dispatch for future terminal/MCP adapters. |
| [`optimizers`](#optimizers) | 13 | Methods that move Controls to improve system metrics. |
| [`guesses`](#guesses) | 13 | Initial Controls generators and restart helpers. |
| [`utils`](#utils) | 32 | Diagnostics, derivative checks, residual geometry, repair, guards, and spectrum tools. |
| [`schedules`](#schedules) | 5 | Small step-size policy objects for optimizer or repair workflows. |
| [`base`](#base) | 41 | Base data objects and top-level entry points: Controls, results, warmstart state, diagnostics, and bound context utilities. |
| [`core`](#core) | 19 | Shared chunk engine: the proposal contract, validated system evaluation, acceptance guards, and stopping rules. |
| [`campaigns`](#campaigns) | 9 | Durable multi-stage coordination using objective and provenance declared by system_spec(). |
| [`visualization`](#visualization) | 15 | Render explicit system-owned plot data; visualization never invents a trajectory. |
| [`system_olgs`](#system-olgs) | 52 | Mandatory static system semantics and capability-gated numerical hooks owned by each problem. |
| [`system_olss`](#system-olss) | 36 | Construction types, validation, and discovery for project-owned OLSS contracts. |
| [`shooting`](#shooting) | 12 | Root solving, verification, multistart, continuation, and direct shooting for OLSS systems. |

## jobs

Bounded client-lifetime process jobs and explicit candidate retrieval.

### `jobs.JobHandle`

Retained runtime job identity and frozen problem provenance.

Family: `olgs`, `olss`.

```python
JobHandle(job_id: 'str', runtime_id: 'str', operation: 'str', context: 'str', configuration: 'int', fingerprint: 'str', label: 'str | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | JobHandle |

Source: `optimizer/jobs/contracts.py`.

### `jobs.JobsConfig`

Conservative CPU, memory, queue and scratch reservations.

Family: `olgs`, `olss`.

```python
JobsConfig(max_workers: 'int' = 2, threads_per_worker: 'int' = 1, memory_bytes: 'int' = 1073741824, coordinator_bytes: 'int' = 268435456, max_queue: 'int' = 16, max_jobs: 'int' = 64, input_bytes: 'int' = 8388608, scratch_bytes: 'int' = 134217728, shutdown_seconds: 'float' = 2.0) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | JobsConfig |

Source: `optimizer/jobs/contracts.py`.

### `jobs.configure`

Set scheduler resource reservations before retaining jobs.

Family: `olgs`, `olss`.

```python
configure(*, service=None, **limits)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.submit`

Freeze a source-backed system and initial state, then queue a solver.

Family: `olgs`, `olss`.

```python
submit(operation, *, controls=None, variables=None, options=None, label=None, worker_bytes=268435456, result_bytes=4194304, context=None, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | JobHandle |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.status`

Inspect jobs and resource reservations.

Family: `olgs`, `olss`.

```python
status(job=None, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.wait`

Wait at most thirty seconds for a terminal job status.

Family: `olgs`, `olss`.

```python
wait(job, *, timeout=10.0, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.result`

Retain a native physical candidate from a completed job.

Family: `olgs`, `olss`.

```python
result(job, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult or ShootingResult |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.cancel`

Request safe project-call boundary cancellation.

Family: `olgs`, `olss`.

```python
cancel(job, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.compare`

Rank compatible completed candidates using an explicit metric.

Family: `olgs`, `olss`.

```python
compare(jobs, *, metric='J', direction='minimize', service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.release`

Release a terminal job's frozen inputs and runtime result.

Family: `olgs`, `olss`.

```python
release(job, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.shutdown`

Cancel and reap the service's owned workers.

Family: `olgs`, `olss`.

```python
shutdown(*, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.pause`

Pause a supported method at a numerical iteration boundary.

Family: `olgs`, `olss`.

```python
pause(job, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.checkpoint`

Retrieve an acknowledged physical boundary snapshot.

Family: `olgs`, `olss`.

```python
checkpoint(job, *, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.resume`

Continue a live paused worker or explicitly fork a physical restart.

Family: `olgs`, `olss`.

```python
resume(job, *, mode='exact', maxiter=None, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict or JobHandle |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

### `jobs.detach`

Detach a persistent terminal without stopping jobs.

Family: `olgs`, `olss`.

```python
detach(*, service=None)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |

#### Watch out

- Persistent lifetime requires an explicitly started coordinator; results never autoactivate.

Source: `optimizer/jobs/api.py`.

## operations

Foreground session and native operation dispatch for future terminal/MCP adapters.

### `operations.ObjectRef`

Reference to a retained object version or declared data field.

Family: `olgs`, `olss`. Root shortcut: `opt.ObjectRef`.

```python
ObjectRef(object_id: 'str', path: 'tuple[str | int, ...]' = ()) -> None
```

Source: `optimizer/operations/contracts.py`.

### `operations.OperationRequest`

Native arguments with separate context, binding and activation options.

Family: `olgs`, `olss`. Root shortcut: `opt.OperationRequest`.

```python
OperationRequest(operation: 'str', arguments: 'Mapping[str, Any]' = <factory>, context: 'str | None' = None, bind: 'str | None' = None, activate: 'bool | None' = None, request_id: 'str' = <factory>) -> None
```

Source: `optimizer/operations/contracts.py`.

### `operations.OperationResponse`

Typed result and compact transport metadata with structured failures.

Family: `olgs`, `olss`. Root shortcut: `opt.OperationResponse`.

```python
OperationResponse(request_id: 'str', operation: 'str | None', status: 'str', result: 'Any' = None, object_refs: 'tuple[ObjectRef, ...]' = (), state_before: 'Mapping[str, Any]' = <factory>, state_after: 'Mapping[str, Any]' = <factory>, error: 'Mapping[str, Any] | None' = None, metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/operations/contracts.py`.

### `operations.OperationService`

Validated native dispatch and atomic candidate publication; no automatic history.

Family: `olgs`, `olss`. Root shortcut: `opt.OperationService`.

```python
OperationService(session: 'Session | None' = None) -> 'None'
```

Source: `optimizer/operations/service.py`.

### `operations.Session`

Independent named contexts and bounded owned object versions.

Family: `olgs`, `olss`. Root shortcut: `opt.Session`.

```python
Session(*, max_bytes: 'int' = 67108864, max_objects: 'int' = 256) -> 'None'
```

Source: `optimizer/session.py`.

## optimizers

Methods that move Controls to improve system metrics.

### `optimizers.adam`

Adaptive first-order optimizer with first and second moment state.

Family: `olgs`. Root shortcut: `opt.adam`.

```python
adam(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'adam', maxiter: 'int' = 100, step_size: 'StepSize | None' = None, beta1: 'float' = 0.9, beta2: 'float' = 0.999, eps: 'float' = 1e-08, weight_decay: 'float' = 0.0, max_step_norm: 'float | None' = None, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `controls` | Starting Controls; optional when state or warmstart supplies controls. |
| `variant` | adam, amsgrad, adamw, radam, or adabelief. |
| `maxiter` | Number of chunk iterations. |
| `step_size` | Learning rate as a scalar or serializable schedule object. |
| `beta1/beta2/eps` | Moment and denominator stabilization parameters. |
| `weight_decay` | Decoupled decay for adamw or explicit decay. |
| `max_step_norm` | Optional L2 clipping of the applied update. |
| `warmstart` | Carries Adam moment state only when target_optimizer is adam. |
| `accept` | Optional guard; by default every finite Adam-family trial is accepted. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Variants

- adam
- amsgrad
- adamw
- radam
- adabelief

#### Best for

- rough first optimization
- uneven gradient scales
- warmstarting before polishing

#### Not for

- systems without analytical gradient
- hard residual repair by itself

#### Watch out

- step_size controls the real behavior
- custom guards can reject and preserve moment state

#### Related

- utils.verify_gradient
- optimizers.lbfgs
- guesses.random_fourier_guess

#### Example

```python
result = opt.optimizers.adam(system, controls, step_size=0.05, maxiter=50)
```

Source: `optimizer/optimizers/adam.py`.

### `optimizers.momentum`

Low-memory gradient optimizer with a velocity buffer.

Family: `olgs`. Root shortcut: `opt.momentum`.

```python
momentum(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'heavy_ball', maxiter: 'int' = 100, step_size: 'StepSize | None' = None, momentum: 'float' = 0.9, max_step_norm: 'float | None' = None, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `controls` | Starting Controls; optional with state or warmstart. |
| `variant` | heavy_ball, nesterov, restart, or clipped. |
| `momentum` | Velocity carry coefficient in [0, 1). |
| `step_size` | Gradient step scale as a scalar or serializable schedule object. |
| `max_step_norm` | Optional update clipping; default active for clipped variant. |
| `accept` | Optional guard; by default every finite momentum trial is accepted. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Variants

- heavy_ball
- nesterov
- restart
- clipped

#### Best for

- simple gradient baselines
- smoothing zigzag motion
- debuggable first-order behavior

#### Watch out

- large finite steps are standard behavior unless a custom guard is supplied
- restart variant resets velocity after rejection

#### Related

- optimizers.line_search
- optimizers.adam
- utils.diagnostic_report

#### Example

```python
result = opt.optimizers.momentum(system, controls, variant='nesterov', step_size=0.03, maxiter=20)
```

Source: `optimizer/optimizers/momentum.py`.

### `optimizers.line_search`

Gradient descent family with fixed, normalized, backtracking, and Armijo steps.

Family: `olgs`. Root shortcut: `opt.line_search`.

```python
line_search(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'backtracking', maxiter: 'int' = 100, step_size: 'float | None' = None, min_step: 'float' = 1e-12, max_step: 'float | None' = None, shrink: 'float' = 0.5, grow: 'float' = 1.0, max_backtracks: 'int' = 12, normalize: 'bool | None' = None, armijo_c1: 'float' = 0.0001, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, accept: 'Any | None' = None, gradient_tolerance: 'float | None' = 1e-08, state: 'RunState | None' = None, warmstart: 'Any | None' = None, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `controls` | Starting Controls; optional with state or warmstart. |
| `variant` | fixed, normalized, backtracking, or armijo. |
| `step_size` | Initial trial step. |
| `shrink/grow` | Backtracking and next-step adaptation factors. |
| `max_backtracks` | Trial attempts per iteration for backtracking/Armijo. |
| `accept` | Optional custom accept function such as opt.metric_guard(...). |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Variants

- fixed
- backtracking
- normalized
- armijo

#### Best for

- auditable descent baseline
- step-size debugging
- guarded metric acceptance

#### Watch out

- fixed steps can reject if too large
- Armijo uses directional derivative thresholds

#### Related

- core.metric_guard
- optimizers.momentum
- utils.verify_gradient

#### Example

```python
result = opt.optimizers.line_search(system, controls, variant='backtracking', step_size=1.0, maxiter=10)
```

Source: `optimizer/optimizers/line_search.py`.

### `optimizers.adagrad`

Adaptive first-order method with cumulative squared-gradient scaling.

Family: `olgs`. Root shortcut: `opt.adagrad`.

```python
adagrad(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, step_size: 'StepSize | None' = None, eps: 'float' = 1e-08, initial_accumulator: 'float' = 0.0, max_step_norm: 'float | None' = None, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `step_size` | Global learning-rate multiplier. |
| `eps` | Denominator stabilizer. |
| `initial_accumulator` | Initial per-coordinate squared-gradient history. |
| `max_step_norm` | Optional update clipping. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Best for

- sparse or uneven gradient coordinates
- simple adaptive scaling without momentum

#### Watch out

- a
- c
- c
- u
- m
- u
- l
- a
- t
- o
- r
- 
- o
- n
- l
- y
- 
- g
- r
- o
- w
- s
- ,
- 
- s
- o
- 
- l
- a
- t
- e
- r
- 
- s
- t
- e
- p
- s
- 
- c
- a
- n
- 
- b
- e
- c
- o
- m
- e
- 
- v
- e
- r
- y
- 
- s
- m
- a
- l
- l

#### Related

- optimizers.rmsprop
- optimizers.adam

#### Example

```python
result = opt.optimizers.adagrad(system, controls, step_size=0.05, maxiter=30)
```

Source: `optimizer/optimizers/adaptive.py`.

### `optimizers.rmsprop`

Adaptive first-order method with moving-average squared-gradient scaling.

Family: `olgs`. Root shortcut: `opt.rmsprop`.

```python
rmsprop(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, step_size: 'StepSize | None' = None, decay: 'float' = 0.9, eps: 'float' = 1e-08, initial_accumulator: 'float' = 0.0, max_step_norm: 'float | None' = None, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `step_size` | Global learning-rate multiplier. |
| `decay` | Moving-average memory coefficient in [0, 1). |
| `eps` | Denominator stabilizer. |
| `max_step_norm` | Optional update clipping. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Best for

- nonstationary gradient scales
- adaptive scaling with less memory than Adam

#### Watch out

- d
- e
- c
- a
- y
- 
- c
- o
- n
- t
- r
- o
- l
- s
- 
- h
- o
- w
- 
- q
- u
- i
- c
- k
- l
- y
- 
- t
- h
- e
- 
- s
- c
- a
- l
- e
- 
- e
- s
- t
- i
- m
- a
- t
- e
- 
- r
- e
- a
- c
- t
- s

#### Related

- optimizers.adagrad
- optimizers.adam

#### Example

```python
result = opt.optimizers.rmsprop(system, controls, step_size=0.02, decay=0.8, maxiter=30)
```

Source: `optimizer/optimizers/adaptive.py`.

### `optimizers.lbfgs`

Limited-memory quasi-Newton polish method using accepted curvature history.

Family: `olgs`. Root shortcut: `opt.lbfgs`.

```python
lbfgs(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, history_size: 'int' = 10, step_size: 'float | None' = None, curvature_eps: 'float' = 1e-12, max_step_norm: 'float | None' = None, reject_shrink: 'float' = 0.5, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `controls` | Usually a reasonably good starting Controls object. |
| `history_size` | Number of accepted curvature pairs to keep. |
| `step_size` | Scalar multiplier for the L-BFGS direction. |
| `curvature_eps` | Minimum positive s dot y required to store a pair. |
| `max_step_norm` | Optional step clipping. |
| `reject_shrink` | Step reduction after a rejected proposal. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Best for

- polishing after Adam or line search
- smooth deterministic objectives
- low-memory curvature use

#### Not for

- very rough global exploration
- systems without gradient

#### Watch out

- bad curvature pairs are skipped
- too-large step_size can still reject

#### Related

- optimizers.adam
- optimizers.nonlinear_cg
- utils.diagnostic_report

#### Example

```python
result = opt.optimizers.lbfgs(system, r1.controls, warmstart=r1.warmstart('lbfgs'), maxiter=20)
```

Source: `optimizer/optimizers/lbfgs.py`.

### `optimizers.nonlinear_cg`

Low-memory nonlinear conjugate-gradient optimizer.

Family: `olgs`. Root shortcut: `opt.nonlinear_cg`.

```python
nonlinear_cg(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'polak_ribiere_plus', maxiter: 'int' = 100, step_size: 'float | None' = None, max_step_norm: 'float | None' = None, restart_on_nondescent: 'bool' = True, reject_shrink: 'float' = 0.5, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `variant` | fletcher_reeves, polak_ribiere, polak_ribiere_plus, or hestenes_stiefel. |
| `step_size` | Scalar update multiplier. |
| `restart_on_nondescent` | Restart to steepest descent when conjugate direction is not descent. |
| `max_step_norm` | Optional update clipping. |
| `reject_shrink` | Step reduction and direction restart after rejection. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Variants

- fletcher_reeves
- polak_ribiere
- polak_ribiere_plus
- hestenes_stiefel

#### Best for

- large control vectors
- middle ground between gradient descent and L-BFGS
- low-memory deterministic runs

#### Watch out

- does not do a full Wolfe line search
- may restart often near noisy gradients

#### Related

- optimizers.lbfgs
- optimizers.line_search

#### Example

```python
result = opt.optimizers.nonlinear_cg(system, controls, step_size=0.05, maxiter=25)
```

Source: `optimizer/optimizers/nonlinear_cg.py`.

### `optimizers.nonlinear_cg`

Low-memory nonlinear conjugate-gradient optimizer.

Family: `olgs`. Root shortcut: `opt.nonlinear_cg`.

```python
nonlinear_cg(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'polak_ribiere_plus', maxiter: 'int' = 100, step_size: 'float | None' = None, max_step_norm: 'float | None' = None, restart_on_nondescent: 'bool' = True, reject_shrink: 'float' = 0.5, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, gradient_tolerance: 'float | None' = 1e-08, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `variant` | fletcher_reeves, polak_ribiere, polak_ribiere_plus, or hestenes_stiefel. |
| `step_size` | Scalar update multiplier. |
| `restart_on_nondescent` | Restart to steepest descent when conjugate direction is not descent. |
| `max_step_norm` | Optional update clipping. |
| `reject_shrink` | Step reduction and direction restart after rejection. |
| `gradient_tolerance` | Gradient-norm stationarity stop; default 1e-8. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Variants

- fletcher_reeves
- polak_ribiere
- polak_ribiere_plus
- hestenes_stiefel

#### Best for

- large control vectors
- middle ground between gradient descent and L-BFGS
- low-memory deterministic runs

#### Watch out

- does not do a full Wolfe line search
- may restart often near noisy gradients

#### Related

- optimizers.lbfgs
- optimizers.line_search

#### Example

```python
result = opt.optimizers.nonlinear_cg(system, controls, step_size=0.05, maxiter=25)
```

Source: `optimizer/optimizers/nonlinear_cg.py`.

### `optimizers.cma_es`

Derivative-free population search with isotropic or diagonal sampling radius.

Family: `olgs`. Root shortcut: `opt.cma_es`.

```python
cma_es(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'diagonal', maxiter: 'int' = 100, population_size: 'int | None' = None, elite_fraction: 'float' = 0.5, sigma: 'float' = 0.1, min_sigma: 'float' = 1e-08, covariance_lr: 'float' = 0.2, seed: 'int | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, state: 'RunState | None' = None, warmstart: 'Any | None' = None, stopping: 'StoppingConfig | None' = None, target_value: 'float | None' = None, target_metric: 'str | None' = None, target_mode: 'str | None' = None, stall_patience: 'int | None' = None, stall_tolerance: 'float' = 0.0, stall_metric: 'str | None' = None, stall_mode: 'str | None' = None, system_params: 'Mapping[str, Any] | None' = None, stage: 'str | None' = None, use_cache: 'bool' = True, parallel: 'Any | None' = None, system_factory: 'Any | None' = None) -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Needs evaluate(...). It does not use gradient during iterations. |
| `controls` | Starting mean Controls; optional when state or warmstart supplies it. |
| `variant` | diagonal or isotropic. |
| `population_size` | Candidates sampled per generation; default scales with dimension. |
| `elite_fraction` | Fraction retained to update the mean. |
| `sigma` | Initial sampling radius. |
| `seed` | Random seed for reproducibility. |
| `state/warmstart` | Compatible continuation inputs using the standard optimizer call shape. |
| `stopping/target/stall` | Standard structured stopping configuration or explicit target/stall options. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- Controls, state, or warmstart

#### Variants

- diagonal
- isotropic

#### Best for

- rough global search
- systems with unreliable gradients
- escaping poor starts

#### Not for

- fast polishing near a good deterministic solution
- very high-dimensional expensive evaluations without budget

#### Watch out

- uses many evaluations per iteration
- implementation is compact CMA-ES style, not full research CMA-ES

#### Related

- guesses.random_fourier_guess
- optimizers.adam
- utils.metric_report

#### Example

```python
result = opt.optimizers.cma_es(system, controls, population_size=24, sigma=0.15, seed=7, maxiter=6)
```

Source: `optimizer/optimizers/cma_es.py`.

### `optimizers.projected_gradient`

Equality- and bound-constrained descent with restoration and projected-KKT stopping.

Family: `olgs`. Root shortcut: `opt.projected_gradient`.

```python
projected_gradient(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, step_size: 'float' = 0.1, residual_name: 'str' = 'hard', feasibility_tolerance: 'float' = 1e-08, stationarity_tolerance: 'float' = 1e-08, damping: 'float' = 1e-10, restoration_maxiter: 'int' = 5, max_backtracks: 'int' = 8, shrink: 'float' = 0.5, max_step_norm: 'float | None' = None, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Base OLGS plus residuals(...), jacobian(...), and optional control_bounds(). |
| `residual_name` | Named hard residual set, default hard. |
| `feasibility_tolerance` | Scaled residual norm required in the feasible phase. |
| `stationarity_tolerance` | Projected-KKT norm stopping threshold. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- system.residuals
- system.jacobian
- Controls

#### Best for

- hard equality constraints
- box-bounded feasible polishing

#### Watch out

- general nonlinear inequalities are not supported
- finite-difference Jacobians are not used implicitly

#### Related

- utils.project_gradient
- utils.repair_newton

#### Example

```python
result = opt.optimizers.projected_gradient(system, controls, maxiter=50)
```

Source: `optimizer/optimizers/projected.py`.

### `optimizers.newton_cg`

Matrix-free Newton method using conjugate gradients and analytical Hessian-vector products.

Family: `olgs`. Root shortcut: `opt.newton_cg`.

```python
newton_cg(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, step_size: 'float' = 1.0, gradient_tolerance: 'float' = 1e-08, cg_tolerance: 'float' = 0.001, cg_maxiter: 'int' = 50, damping: 'float' = 0.0, curvature_tolerance: 'float' = 1e-12, max_step_norm: 'float | None' = None, reject_shrink: 'float' = 0.5, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Base OLGS plus hvp(controls, vector). |
| `cg_maxiter` | Inner linear-solve budget. |
| `gradient_tolerance` | Gradient-norm stop with curvature classification. |
| `reject_shrink` | Step reduction after rejection; the configured full step is restored after acceptance. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- system.hvp
- Controls

#### Best for

- smooth deterministic objectives
- large controls with matrix-free curvature

#### Watch out

- a finite-difference HVP is never selected implicitly

#### Related

- optimizers.trust_region
- optimizers.lbfgs

#### Example

```python
result = opt.optimizers.newton_cg(system, controls, maxiter=20)
```

Source: `optimizer/optimizers/newton_cg.py`.

### `optimizers.trust_region`

Steihaug trust-region method with model-ratio acceptance and matrix-free HVPs.

Family: `olgs`. Root shortcut: `opt.trust_region`.

```python
trust_region(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, initial_radius: 'float' = 1.0, max_radius: 'float' = 100.0, gradient_tolerance: 'float' = 1e-08, cg_tolerance: 'float' = 0.001, cg_maxiter: 'int' = 50, damping: 'float' = 0.0, curvature_tolerance: 'float' = 1e-12, acceptance_threshold: 'float' = 0.1, shrink_threshold: 'float' = 0.25, grow_threshold: 'float' = 0.75, shrink_factor: 'float' = 0.25, grow_factor: 'float' = 2.0, state: 'RunState | None' = None, warmstart: 'Any | None' = None, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Base OLGS plus hvp(controls, vector). |
| `initial_radius` | Initial L2 trust radius. |
| `gradient_tolerance` | Gradient-norm stop with curvature classification. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- system.hvp
- Controls

#### Best for

- indefinite or locally inaccurate quadratic models
- guarded second-order steps

#### Watch out

- currently a minimization method

#### Related

- optimizers.newton_cg

#### Example

```python
result = opt.optimizers.trust_region(system, controls, initial_radius=1.0, maxiter=20)
```

Source: `optimizer/optimizers/trust_region.py`.

### `optimizers.proximal_gradient`

Composite-objective descent using a system smooth gradient and proximal operator.

Family: `olgs`. Root shortcut: `opt.proximal_gradient`.

```python
proximal_gradient(system: 'Any', controls: 'Controls | None' = None, *, maxiter: 'int' = 100, step_size: 'float' = 0.1, mapping_tolerance: 'float' = 1e-08, max_step_norm: 'float | None' = None, reject_shrink: 'float' = 0.5, state: 'RunState | None' = None, warmstart: 'Any | None' = None, accept_metric: 'str | None' = None, accept_tolerance: 'float' = 0.0, **engine_kwargs: 'Any') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Base OLGS plus smooth_gradient(...) and prox(...). |
| `mapping_tolerance` | Proximal-gradient mapping threshold. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.smooth_gradient
- system.prox
- Controls

#### Best for

- L1 regularization
- total variation or other available proximal operators

#### Watch out

- evaluate must return the total composite J

#### Example

```python
result = opt.optimizers.proximal_gradient(system, controls, step_size=0.1, maxiter=50)
```

Source: `optimizer/optimizers/proximal.py`.

## guesses

Initial Controls generators and restart helpers.

### `guesses.constant_guess`

Selected-channel constant Controls.

Family: `olgs`. Root shortcut: `opt.constant_guess`.

```python
constant_guess(target: 'Any', *, value: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, endpoint: 'str | None' = 'free', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `value` | Scalar, channel mapping, or per-channel sequence. |
| `channels` | Optional channels to fill. |
| `endpoint` | free/hold or zero. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Constant selected channels. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- simple offsets
- channel sign tests
- known constant baselines

#### Example

```python
controls = opt.guesses.constant_guess(system, value={'ux': 0.2, 'uy': 0.0, 'uz': -0.1})
```

Source: `optimizer/guesses/simple.py`.

### `guesses.ramp_guess`

Monotone ramp between per-channel start and stop values.

Family: `olgs`. Root shortcut: `opt.ramp_guess`.

```python
ramp_guess(target: 'Any', *, start: 'float | Mapping[str, float] | Sequence[float]' = 0.0, stop: 'float | Mapping[str, float] | Sequence[float]' = 1.0, channels: 'str | Iterable[str] | None' = None, kind: 'str' = 'linear', endpoint: 'str | None' = 'free', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `start/stop` | Scalar, channel mapping, or sequence. |
| `kind` | linear, quadratic, or smoothstep. |
| `channels` | Optional channels to fill. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth monotone selected-channel ramp. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- testing endpoint behavior
- slow turn-on controls
- simple curriculum starts

#### Example

```python
controls = opt.guesses.ramp_guess(system, start=0.0, stop={'ux': 0.3})
```

Source: `optimizer/guesses/simple.py`.

### `guesses.sine_guess`

Sine-wave guess with channel-wise frequency and phase.

Family: `olgs`. Root shortcut: `opt.sine_guess`.

```python
sine_guess(target: 'Any', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, frequency: 'float | Mapping[str, float] | Sequence[float]' = 1.0, phase: 'float | Mapping[str, float] | Sequence[float]' = 0.0, offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `amplitude` | Output amplitude under the selected scale convention. |
| `frequency` | Cycles over the normalized control interval. |
| `phase` | Phase shift in radians. |
| `envelope/endpoint/scale` | Common shaping controls. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth periodic selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- single-frequency starts
- phase/frequency experiments
- smooth deterministic controls

#### Example

```python
controls = opt.guesses.sine_guess(system, amplitude=0.2, frequency=2, endpoint='zero')
```

Source: `optimizer/guesses/harmonic.py`.

### `guesses.cosine_guess`

Cosine-wave guess with channel-wise frequency and phase.

Family: `olgs`. Root shortcut: `opt.cosine_guess`.

```python
cosine_guess(target: 'Any', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, frequency: 'float | Mapping[str, float] | Sequence[float]' = 1.0, phase: 'float | Mapping[str, float] | Sequence[float]' = 0.0, offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `amplitude` | Output amplitude under the selected scale convention. |
| `frequency` | Cycles over the normalized control interval. |
| `phase` | Phase shift in radians. |
| `envelope/endpoint/scale` | Common shaping controls. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth periodic selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- even-symmetry pulse starts
- smooth deterministic controls

#### Example

```python
controls = opt.guesses.cosine_guess(system, amplitude=0.2, frequency=1)
```

Source: `optimizer/guesses/harmonic.py`.

### `guesses.gaussian_guess`

Localized Gaussian pulse guess.

Family: `olgs`. Root shortcut: `opt.gaussian_guess`.

```python
gaussian_guess(target: 'Any', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, center: 'float | Mapping[str, float] | Sequence[float]' = 0.5, width: 'float | Mapping[str, float] | Sequence[float]' = 0.15, offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `amplitude` | Peak scale under the selected convention. |
| `center` | Pulse center in normalized time. |
| `width` | Positive normalized width. |
| `envelope/endpoint/scale` | Common shaping controls. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Localized selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- localized pulses
- smooth starts with low endpoint disturbance

#### Watch out

- width must be positive

#### Example

```python
controls = opt.guesses.gaussian_guess(system, amplitude=0.2, width=0.15)
```

Source: `optimizer/guesses/harmonic.py`.

### `guesses.sinc_guess`

Centered sinc pulse with controllable side-lobe width.

Family: `olgs`. Root shortcut: `opt.sinc_guess`.

```python
sinc_guess(target: 'Any', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, center: 'float | Mapping[str, float] | Sequence[float]' = 0.5, width: 'float | Mapping[str, float] | Sequence[float]' = 4.0, offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `amplitude` | Output amplitude under the selected scale convention. |
| `center` | Pulse center in normalized time. |
| `width` | Positive side-lobe width parameter. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Sinc-shaped selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- band-limited style starts
- localized pulse experiments with side lobes

#### Watch out

- width must be positive

#### Example

```python
controls = opt.guesses.sinc_guess(system, amplitude=0.25, width=5.0)
```

Source: `optimizer/guesses/harmonic.py`.

### `guesses.fourier_guess`

Low-frequency Fourier-series deterministic guess.

Family: `olgs`. Root shortcut: `opt.fourier_guess`.

```python
fourier_guess(target: 'Any', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, modes: 'int' = 5, frequency_base: 'float' = 1.0, coefficients: 'Any' = None, phases: 'Any' = 0.0, decay: 'str' = '1/k', offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `modes` | Number of sine modes. |
| `frequency_base` | Base frequency multiplier. |
| `coefficients/phases` | Optional mode coefficients and phases. |
| `decay` | flat, 1/k, or 1/k2 coefficient template when coefficients are omitted. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth Fourier-series selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- structured smooth starts
- low-frequency pulse families
- basis-style experiments

#### Watch out

- modes must be >= 1
- frequency_base must be positive

#### Related

- guesses.random_fourier_guess
- utils.control_spectrum

#### Example

```python
controls = opt.guesses.fourier_guess(system, amplitude=0.2, modes=4, decay='1/k2')
```

Source: `optimizer/guesses/harmonic.py`.

### `guesses.random_guess`

Raw random control guess.

Family: `olgs`. Root shortcut: `opt.random_guess`.

```python
random_guess(target: 'Any', *, amplitude: 'float | tuple[float, float] | Mapping[str, float] | Sequence[float]' = 1.0, distribution: 'str' = 'uniform', offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, seed: 'int | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `distribution` | uniform, normal, or rademacher. |
| `amplitude` | Scalar/mapping/sequence, or two-number random amplitude range. |
| `seed` | Random seed for reproducibility. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Random selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- stress tests
- rough random baselines
- checking optimizer robustness

#### Not for

- physically smooth starts unless envelope/smoothing is added

#### Related

- guesses.random_smooth_guess
- guesses.random_fourier_guess

#### Example

```python
controls = opt.guesses.random_guess(system, amplitude=0.2, distribution='normal', seed=4)
```

Source: `optimizer/guesses/random.py`.

### `guesses.random_smooth_guess`

Random guess smoothed by a Gaussian correlation kernel.

Family: `olgs`. Root shortcut: `opt.random_smooth_guess`.

```python
random_smooth_guess(target: 'Any', *, amplitude: 'float | tuple[float, float] | Mapping[str, float] | Sequence[float]' = 1.0, correlation: 'float' = 0.15, distribution: 'str' = 'normal', offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, seed: 'int | None' = None, envelope: 'str | None' = 'hann', endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `correlation` | Smoothing width; <=1 is interpreted as fraction of control_dim. |
| `distribution` | uniform or normal raw samples before smoothing. |
| `envelope` | Defaults to hann. |
| `seed` | Random seed for reproducibility. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth random selected-channel pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- multi-start exploration
- smoother random pulses
- less high-frequency roughness than random_guess

#### Related

- utils.smoothness_report
- guesses.random_fourier_guess

#### Example

```python
controls = opt.guesses.random_smooth_guess(system, amplitude=0.2, correlation=0.2, seed=4)
```

Source: `optimizer/guesses/random.py`.

### `guesses.random_fourier_guess`

Random low-frequency Fourier-series guess.

Family: `olgs`. Root shortcut: `opt.random_fourier_guess`.

```python
random_fourier_guess(target: 'Any', *, amplitude: 'float | tuple[float, float] | Mapping[str, float] | Sequence[float]' = 1.0, modes: 'int' = 5, frequency_base: 'float' = 1.0, coefficient_scale: 'float' = 1.0, decay: 'str' = '1/k', offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, channels: 'str | Iterable[str] | None' = None, seed: 'int | None' = None, envelope: 'str | None' = None, endpoint: 'str | None' = 'free', scale: 'str' = 'max_abs', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `modes` | Number of random Fourier modes. |
| `coefficient_scale` | Standard deviation scale for random coefficients. |
| `decay` | flat, 1/k, or 1/k2 random coefficient damping. |
| `seed` | Random seed for reproducibility. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Smooth low-frequency random pulse. |

#### Requires

- ControlSpec or system.control_spec

#### Best for

- default multi-start guess
- smooth global exploration
- starting Adam or CMA-ES

#### Related

- optimizers.adam
- optimizers.cma_es
- utils.control_spectrum

#### Example

```python
controls = opt.guesses.random_fourier_guess(system, amplitude=0.1, modes=5, seed=1)
```

Source: `optimizer/guesses/random.py`.

### `guesses.scale_guess`

Rescale existing Controls without mutating them.

Family: `olgs`. Root shortcut: `opt.scale_guess`.

```python
scale_guess(controls: 'Controls', *, amplitude: 'float | Mapping[str, float] | Sequence[float]' = 1.0, offset: 'float | Mapping[str, float] | Sequence[float]' = 0.0, scale: 'str' = 'max_abs', channels: 'str | Sequence[str] | None' = None, endpoint: 'str | None' = 'free', name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Existing Controls to rescale. |
| `amplitude` | Target amplitude convention. |
| `offset` | Optional per-channel offset. |
| `channels` | Optional channels to retain/rescale. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Rescaled copy of existing controls. |

#### Requires

- Controls

#### Best for

- trying lower amplitudes
- normalizing a previous pulse
- restart preparation

#### Example

```python
scaled = opt.guesses.scale_guess(controls, amplitude=0.1)
```

Source: `optimizer/guesses/composite.py`.

### `guesses.mix_guess`

Weighted mix of compatible Controls objects.

Family: `olgs`. Root shortcut: `opt.mix_guess`.

```python
mix_guess(guesses: 'Sequence[Controls] | Controls', *more_guesses: 'Controls', weights: 'Sequence[float] | None' = None, name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `guesses` | Sequence of Controls or first Controls argument. |
| `weights` | Optional weights; defaults to equal average. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Weighted linear combination with shared ControlSpec. |

#### Requires

- Controls

#### Best for

- combining useful starts
- averaging nearby candidates
- manual basis experiments

#### Watch out

- all guesses must share keys and control_dim

#### Example

```python
mixed = opt.guesses.mix_guess([base, perturb], weights=[0.8, 0.2])
```

Source: `optimizer/guesses/composite.py`.

### `guesses.perturb_guess`

Add a generated perturbation to existing Controls.

Family: `olgs`. Root shortcut: `opt.perturb_guess`.

```python
perturb_guess(controls: 'Controls', *, amplitude: 'float | tuple[float, float] | Mapping[str, float] | Sequence[float]' = 0.01, kind: 'str' = 'random_fourier', seed: 'int | None' = None, channels: 'str | Sequence[str] | None' = None, name: 'str | None' = None, **kwargs: 'Any') -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Base Controls. |
| `kind` | random_fourier, random_smooth, random, or fourier. |
| `amplitude` | Perturbation amplitude. |
| `seed` | Random seed for random perturbation kinds. |
| `kwargs` | Forwarded to the selected perturbation generator. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Base controls plus generated perturbation. |

#### Requires

- Controls

#### Best for

- local restarts around a good pulse
- escaping a shallow local basin
- small random exploration

#### Related

- guesses.random_fourier_guess
- optimizers.adam

#### Example

```python
trial = opt.guesses.perturb_guess(best.controls, amplitude=0.01, kind='random_fourier', seed=2, modes=3)
```

Source: `optimizer/guesses/composite.py`.

## utils

Diagnostics, derivative checks, residual geometry, repair, guards, and spectrum tools.

### `utils.metric_report`

Evaluate and summarize current system metrics.

Family: `olgs`. Root shortcut: `opt.metric_report`.

```python
metric_report(system: 'Any', controls: 'Controls') -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...). |
| `controls` | Controls to evaluate. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | kind, metrics, raw_metric_keys |

#### Requires

- system.evaluate
- Controls

#### Best for

- quick metric snapshot
- checking objective names
- lightweight logs

#### Example

```python
report = opt.utils.metric_report(system, controls)
```

Source: `optimizer/utils/diagnostics.py`.

### `utils.diagnostic_report`

Metrics plus control, residual, and optional gradient summaries.

Family: `olgs`. Root shortcut: `opt.diagnostic_report`.

```python
diagnostic_report(system: 'Any', controls: 'Controls', *, residuals: 'str | None' = 'hard', include_gradient: 'bool' = True) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Usually provides evaluate(...), gradient(...), and optional residuals(...). |
| `controls` | Controls to inspect. |
| `residuals` | Residual hook name or None to skip residuals. |
| `include_gradient` | Whether to include gradient norm summaries. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | system_hooks, control_spec, controls, metrics, gradient, residuals |

#### Requires

- system.evaluate
- Controls

#### Best for

- getting unstuck
- post-run inspection
- checking residual availability and gradient scale

#### Related

- utils.geometry_probe
- utils.verify_gradient
- utils.control_spectrum

#### Example

```python
report = opt.utils.diagnostic_report(system, result.controls)
```

Source: `optimizer/utils/diagnostics.py`.

### `utils.geometry_probe`

Residual/Jacobian rank, conditioning, and repair feasibility summary.

Family: `olgs`. Root shortcut: `opt.geometry_probe`.

```python
geometry_probe(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', fallback: 'bool' = True, eps: 'float' = 1e-06, rcond: 'float | None' = None) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...); jacobian(...) is optional when fallback=True. |
| `controls` | Controls at which to inspect local geometry. |
| `fallback` | Use finite-difference Jacobian if analytical hook is missing. |
| `rcond` | Optional rank cutoff. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | residual_norm, jacobian_source, rank, nullspace_dimension, condition_number |

#### Requires

- system.residuals
- Controls

#### Best for

- checking if repair/projection is locally possible
- diagnosing rank deficiency
- understanding hard residual constraints

#### Related

- utils.repair_newton
- utils.nullspace_basis
- utils.project_gradient

#### Example

```python
geometry = opt.utils.geometry_probe(system, controls, eps=1e-6)
```

Source: `optimizer/utils/diagnostics.py`.

### `utils.finite_difference_gradient`

Full coordinate finite-difference gradient for a scalar metric.

Family: `olgs`. Root shortcut: `opt.finite_difference_gradient`.

```python
finite_difference_gradient(system: 'Any', controls: 'Controls', *, metric: 'str | None' = None, eps: 'float' = 1e-06, method: 'str' = 'central') -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...). |
| `controls` | Controls to perturb coordinate by coordinate. |
| `metric` | Scalar metric to differentiate, default J. |
| `method` | central or forward. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Finite-difference gradient in the same ControlSpec layout. |

#### Requires

- system.evaluate
- Controls

#### Best for

- small debugging systems
- checking full gradient arrays

#### Not for

- large control vectors in normal loops

#### Watch out

- cost is one or two evaluations per control coordinate

#### Related

- utils.verify_gradient

#### Example

```python
fd = opt.utils.finite_difference_gradient(system, controls, eps=1e-6)
```

Source: `optimizer/utils/derivatives.py`.

### `utils.verify_gradient`

Directional finite-difference check of the analytical gradient.

Family: `olgs`. Root shortcut: `opt.verify_gradient`.

```python
verify_gradient(system: 'Any', controls: 'Controls', *, metric: 'str | None' = None, eps: 'float' = 1e-06, directions: 'int | np.ndarray' = 8, seed: 'int | None' = 12345, rtol: 'float' = 0.0001, atol: 'float' = 1e-07) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and gradient(...). |
| `controls` | Controls at which to check the gradient. |
| `directions` | Number of random directions or explicit direction matrix. |
| `rtol/atol` | Pass thresholds for relative/absolute error. |
| `seed` | Random seed for generated directions. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | passed, max_absolute_error, max_relative_error, directions |

#### Requires

- system.evaluate
- system.gradient
- Controls

#### Best for

- first check on a new system
- debugging optimizer failures
- detecting sign/scale/channel-order mistakes

#### Watch out

- directional checks are cheaper than full finite differences but not exhaustive

#### Related

- utils.finite_difference_gradient
- optimizers.adam

#### Example

```python
check = opt.utils.verify_gradient(system, controls, eps=1e-6, directions=8)
```

Source: `optimizer/utils/derivatives.py`.

### `utils.finite_difference_jacobian`

Full finite-difference Jacobian of system residuals.

Family: `olgs`. Root shortcut: `opt.finite_difference_jacobian`.

```python
finite_difference_jacobian(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', eps: 'float' = 1e-06, method: 'str' = 'central') -> 'np.ndarray'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...). |
| `controls` | Controls to perturb coordinate by coordinate. |
| `residuals` | Residual hook name, default hard. |
| `method` | central or forward. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | np.ndarray |
| `shape` | (n_residuals, controls.spec.size) |

#### Requires

- system.residuals
- Controls

#### Best for

- small systems without analytical jacobian
- repair/projection debugging

#### Watch out

- expensive for large control vectors

#### Related

- utils.verify_jacobian
- utils.geometry_probe

#### Example

```python
jac = opt.utils.finite_difference_jacobian(system, controls, eps=1e-6)
```

Source: `optimizer/utils/derivatives.py`.

### `utils.verify_jacobian`

Directional finite-difference check of analytical residual Jacobian.

Family: `olgs`. Root shortcut: `opt.verify_jacobian`.

```python
verify_jacobian(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', eps: 'float' = 1e-06, directions: 'int | np.ndarray' = 8, seed: 'int | None' = 12345, rtol: 'float' = 0.0001, atol: 'float' = 1e-07, fallback: 'bool' = False) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...) and usually jacobian(...). |
| `controls` | Controls at which to check the Jacobian. |
| `fallback` | Allow finite-difference source when analytical jacobian is missing. |
| `directions` | Number of random directions or explicit direction matrix. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | passed, jacobian_source, max_absolute_error, max_relative_error |

#### Requires

- system.residuals
- Controls

#### Best for

- validating residual Jacobian hooks
- repair/projection debugging

#### Related

- utils.finite_difference_jacobian
- utils.geometry_probe

#### Example

```python
check = opt.utils.verify_jacobian(system, controls, eps=1e-6, directions=8)
```

Source: `optimizer/utils/derivatives.py`.

### `utils.nullspace_basis`

Basis columns for the local residual-Jacobian nullspace.

Family: `olgs`. Root shortcut: `opt.nullspace_basis`.

```python
nullspace_basis(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', rcond: 'float | None' = None, fallback: 'bool' = True, eps: 'float' = 1e-06) -> 'np.ndarray'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...); jacobian(...) optional with fallback. |
| `controls` | Controls at which to compute the local basis. |
| `rcond` | Optional rank cutoff. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | np.ndarray |
| `shape` | (controls.spec.size, nullspace_dimension) |

#### Requires

- system.residuals
- Controls

#### Best for

- understanding feasible first-order directions
- projected method experiments

#### Related

- utils.project_gradient
- utils.geometry_probe

#### Example

```python
basis = opt.utils.nullspace_basis(system, controls, eps=1e-6)
```

Source: `optimizer/utils/geometry.py`.

### `utils.project_gradient`

Remove local residual row-space component from a gradient-like Controls object.

Family: `olgs`. Root shortcut: `opt.project_gradient`.

```python
project_gradient(system: 'Any', controls: 'Controls', gradient: 'Controls', *, residuals: 'str' = 'hard', damping: 'float' = 0.0, fallback: 'bool' = True, eps: 'float' = 1e-06, return_info: 'bool' = False) -> 'Controls | dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...); jacobian(...) optional with fallback. |
| `controls` | Point defining the residual Jacobian. |
| `gradient` | Controls-shaped gradient/direction to project. |
| `return_info` | Return dict with geometry and projected Controls when True. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls \| dict |
| `meaning` | Projected gradient or info dict containing projected_gradient. |

#### Requires

- system.residuals
- Controls

#### Best for

- first-order constrained descent experiments
- checking if a gradient changes hard residuals

#### Watch out

- does not choose step size or run an optimizer

#### Related

- utils.nullspace_basis
- utils.repair_newton

#### Example

```python
info = opt.utils.project_gradient(system, controls, gradient, return_info=True)
```

Source: `optimizer/utils/geometry.py`.

### `utils.repair_newton`

Newton/LM-style residual repair that moves controls to reduce a named residual vector.

Family: `olgs`. Root shortcut: `opt.repair_newton`.

```python
repair_newton(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', method: 'str' = 'lm', maxiter: 'int' = 10, tolerance: 'float' = 1e-10, damping: 'float' = 1e-08, fallback: 'bool' = True, eps: 'float' = 1e-06, rcond: 'float | None' = None, max_step_norm: 'float | None' = None, line_search: 'bool' = True, shrink: 'float' = 0.5, max_backtracks: 'int' = 8) -> 'RepairResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide residuals(...); jacobian(...) optional with fallback. |
| `controls` | Controls to repair. |
| `method` | newton, lm, or damped. |
| `tolerance` | Residual-norm convergence threshold. |
| `damping` | LM/damped solve regularization. |
| `line_search` | Shrink repair delta until residual norm improves. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | RepairResult |
| `fields` | controls, residual_norm, converged, iterations, history, to_dict(...) |

#### Requires

- system.residuals
- Controls

#### Variants

- newton
- lm
- damped

#### Best for

- restoring hard residual feasibility
- post-optimizer repair
- local residual solves

#### Not for

- optimizing J, fidelity, or energy directly

#### Watch out

- finite-difference fallback can be expensive
- line search can fail when local model is poor

#### Related

- utils.geometry_probe
- utils.project_gradient
- utils.verify_jacobian

#### Example

```python
fixed = opt.utils.repair_newton(system, controls, maxiter=4, tolerance=1e-8)
```

Source: `optimizer/utils/repairs.py`.

### `utils.control_spectrum`

FFT-based channel-wise amplitude and power summaries.

Family: `olgs`. Root shortcut: `opt.control_spectrum`.

```python
control_spectrum(controls: 'Controls', *, dt: 'float | None' = None, high_frequency_cutoff: 'float | None' = None) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Controls to analyze. |
| `dt` | Sample spacing; defaults to controls.spec.dt or 1.0. |
| `high_frequency_cutoff` | Optional frequency threshold for high-power fraction. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | frequencies, dominant_frequency, total_power, amplitude, power |

#### Requires

- Controls

#### Best for

- seeing frequency content
- checking high-frequency artifacts
- comparing smooth/random guesses

#### Related

- utils.smoothness_report
- guesses.fourier_guess

#### Example

```python
spectrum = opt.utils.control_spectrum(controls, high_frequency_cutoff=5.0)
```

Source: `optimizer/utils/spectrum.py`.

### `utils.smoothness_report`

Finite-difference roughness, jump, and total-variation summaries.

Family: `olgs`. Root shortcut: `opt.smoothness_report`.

```python
smoothness_report(controls: 'Controls') -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Controls to analyze. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | first_difference_norm, second_difference_norm, total_variation, max_jump |

#### Requires

- Controls

#### Best for

- checking pulse roughness
- comparing raw vs smooth guesses
- spotting large jumps

#### Related

- utils.control_spectrum
- guesses.random_smooth_guess

#### Example

```python
smoothness = opt.utils.smoothness_report(controls)
```

Source: `optimizer/utils/spectrum.py`.

### `utils.RepairResult`

Result object returned by repair_newton.

Family: `olgs`. Root shortcut: `opt.RepairResult`.

```python
RepairResult(controls: 'Controls', residuals: 'np.ndarray', residual_norm: 'float', residual_max_abs: 'float', converged: 'bool', iterations: 'int', method: 'str', residual_name: 'str', jacobian_source: 'str', history: 'list[dict[str, Any]]' = <factory>, stop_reason: 'str' = 'unknown', run_record: 'RunRecord | None' = None, counters: 'dict[str, int]' = <factory>) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `constructor` | Usually not called directly; produced by repair_newton. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | RepairResult |
| `fields` | controls, residuals, residual_norm, converged, history, to_dict(...) |

#### Requires

- Controls

#### Best for

- inspecting repair outcome
- compact repair logs via to_dict(...)

#### Related

- utils.repair_newton

#### Example

```python
payload = fixed.to_dict(include_controls=False)
```

Source: `optimizer/utils/repairs.py`.

### `utils.diagnose_run`

Classify a short OLGS or OLSS history as converging, slow, stalled, oscillating, diverging, or repeating rejections.

Family: `olgs`, `olss`. Root shortcut: `opt.diagnose_run`.

```python
diagnose_run(result: 'Any', *, mode: 'str | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.curvature_probe`

Probe local curvature, negative-curvature evidence, and a suggested gradient step.

Family: `olgs`. Root shortcut: `opt.curvature_probe`.

```python
curvature_probe(system: 'Any', controls: 'Controls', *, directions: 'int' = 8, eps: 'float' = 1e-05, seed: 'int' = 0) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.compare_methods`

Compare several optimizers for a short in-memory budget from identical controls.

Family: `olgs`. Root shortcut: `opt.compare_methods`.

```python
compare_methods(system: 'Any', controls: 'Controls', methods: 'Iterable[str | Callable[..., Any]] | Mapping[str, Callable[..., Any]]', *, iterations: 'int' = 10, options: 'Mapping[str, Mapping[str, Any]] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.shooting_geometry_probe`

Report OLSS Jacobian rank, condition, weak variable blocks, and residual groups.

Family: `olss`. Root shortcut: `opt.shooting_geometry_probe`.

```python
shooting_geometry_probe(system: 'Any', variables: 'Any', *, options: 'Mapping[str, Any] | None' = None, rcond: 'float | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.uniform_control_grid`

Declare sample placement, times, physical interval and units.

Family: `olgs`, `olss`.

```python
uniform_control_grid(*, start, stop, count, placement, units=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlGrid |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.resample_controls`

Interpolate controls to an explicit target grid/layout; extrapolation is opt-in.

Family: `olgs`, `olss`.

```python
resample_controls(controls, *, source_grid, target_grid, target, method='linear', extrapolation='error', max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.smooth_controls`

Gaussian smoothing with sigma in samples and explicit endpoint policy.

Family: `olgs`, `olss`.

```python
smooth_controls(controls, *, sigma, boundary, grid=None, method='gaussian', channels=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.filter_controls`

Uniform-grid FFT filtering with explicit cutoff and endpoint policy.

Family: `olgs`, `olss`.

```python
filter_controls(controls, *, grid, kind, cutoff, boundary, channels=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.scale_controls`

Affine per-channel scaling and offsets without normalization.

Family: `olgs`, `olss`.

```python
scale_controls(controls, *, factors, offsets=0.0, channels=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.normalize_controls`

Explicit global/channel max-abs, L2, RMS or time-weighted normalization.

Family: `olgs`, `olss`.

```python
normalize_controls(controls, *, norm='max_abs', value=1.0, scope='global', channels=None, grid=None, zero='error', max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.mix_controls`

Explicit weighted sum of controls on identical declared grids.

Family: `olgs`, `olss`.

```python
mix_controls(inputs, *, weights, grids=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.select_channels`

Select and reorder arbitrary named control channels.

Family: `olgs`, `olss`.

```python
select_channels(controls, *, channels, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.replace_channel`

Replace a channel from compatible declared samples.

Family: `olgs`, `olss`.

```python
replace_channel(controls, *, channel, replacement, replacement_channel, grid=None, replacement_grid=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.assemble_channels`

Assemble a target layout from explicit input/channel pairs.

Family: `olgs`, `olss`.

```python
assemble_channels(inputs, *, channels, target, grids=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.join_controls_in_time`

Join contiguous declared segments with an explicit duplicate-node policy.

Family: `olgs`, `olss`.

```python
join_controls_in_time(inputs, *, grids, boundary='error', max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.apply_control_bounds`

Clip real channels to explicit bounds and report corrections.

Family: `olgs`, `olss`.

```python
apply_control_bounds(controls, *, lower, upper, channels=None, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.recover_shooting_controls`

Recover controls through the optional project-owned OLSS hook.

Family: `olgs`, `olss`.

```python
recover_shooting_controls(system, variables, *, grid, max_bytes=67108864)
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

#### Watch out

- No implicit activation or hard-constraint certificate.

#### Related

- utils.ControlGrid
- utils.ControlTransformResult

Source: `optimizer/utils/control_tools.py`.

### `utils.ControlGrid`

Explicit control sample grid and interval convention.

Family: `olgs`, `olss`.

```python
ControlGrid(times: 'tuple[float, ...]', start: 'float', stop: 'float', placement: 'str', units: 'str | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlGrid |

Source: `optimizer/utils/control_tools.py`.

### `utils.ControlTransformResult`

Owned transformed controls and numerical provenance.

Family: `olgs`, `olss`.

```python
ControlTransformResult(controls: 'Controls', operation: 'str', parameters: 'dict[str, Any]' = <factory>, source_grids: 'tuple[ControlGrid | None, ...]' = (), target_grid: 'ControlGrid | None' = None, facts: 'dict[str, Any]' = <factory>, source_refs: 'tuple[Any, ...]' = ()) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlTransformResult |

Source: `optimizer/utils/control_tools.py`.

## schedules

Small step-size policy objects for optimizer or repair workflows.

### `schedules.ConstantSchedule`

Schedule that always returns the same step size.

Family: `olgs`. Root shortcut: `opt.ConstantSchedule`.

```python
ConstantSchedule(value: 'float') -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `value` | Finite positive scalar step size. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ConstantSchedule |
| `methods` | initial() and update(...) |

#### Requires

- finite scalar value

#### Best for

- fixed-step experiments
- consistent repair/optimizer policy

#### Example

```python
schedule = opt.schedules.ConstantSchedule(0.25)
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.AdaptiveStepSchedule`

Shrink/grow step-size policy with optional min/max clamps.

Family: `olgs`. Root shortcut: `opt.AdaptiveStepSchedule`.

```python
AdaptiveStepSchedule(initial_step: 'float' = 0.001, shrink: 'float' = 0.5, grow: 'float' = 1.0, min_step: 'float' = 1e-12, max_step: 'float | None' = None) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `initial_step` | Initial positive step. |
| `shrink` | Factor in (0, 1) used after rejection. |
| `grow` | Positive multiplier used after acceptance. |
| `min_step/max_step` | Lower/upper clamps. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | AdaptiveStepSchedule |
| `methods` | initial() and update(current, accepted=...) |

#### Requires

- finite scalar parameters

#### Best for

- accept/reject driven step control
- manual closed-loop experiments

#### Example

```python
schedule = opt.schedules.AdaptiveStepSchedule(initial_step=1.0, shrink=0.5, grow=1.2)
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.ExponentialDecaySchedule`

Exponential accepted-iteration step decay with a floor.

Family: `olgs`. Root shortcut: `opt.ExponentialDecaySchedule`.

```python
ExponentialDecaySchedule(initial_step: 'float' = 0.001, decay: 'float' = 0.99, min_step: 'float' = 1e-12) -> None
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.LinearWarmupSchedule`

Linear warmup from an initial to a target step.

Family: `olgs`. Root shortcut: `opt.LinearWarmupSchedule`.

```python
LinearWarmupSchedule(initial_step: 'float', target_step: 'float', warmup_steps: 'int') -> None
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.CosineDecaySchedule`

Cosine step decay over a fixed iteration horizon.

Family: `olgs`. Root shortcut: `opt.CosineDecaySchedule`.

```python
CosineDecaySchedule(initial_step: 'float', total_steps: 'int', min_step: 'float' = 0.0) -> None
```

Source: `optimizer/schedules/step_size.py`.

## base

Base data objects and top-level entry points: Controls, results, warmstart state, diagnostics, and bound context utilities.

### `base.ControlSpec`

Named control layout: channel keys, control_dim, dtype, dt, and metadata.

Family: `olgs`.

```python
ControlSpec(keys: 'Sequence[str]', control_dim: 'int', dtype: 'Any' = <class 'float'>, dt: 'float | None' = None, meta: 'Mapping[str, Any]' = <factory>, scales: 'Mapping[str, float] | Sequence[float] | None' = None) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `keys` | Unique channel names. |
| `control_dim` | Number of time samples. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlSpec |
| `fields` | keys, shape, size, dt, meta, to_dict() |

#### Requires

- channel names
- control dimension

#### Best for

- declaring the control layout expected by a system

#### Example

```python
spec = opt.ControlSpec(('ux', 'uy', 'uz'), 33)
```

Source: `optimizer/controls.py`.

### `base.Controls`

Named control values backed by a dense (n_controls, control_dim) matrix.

Family: `olgs`.

```python
Controls(spec: 'ControlSpec', values: 'np.ndarray | Sequence[Sequence[float]]', *, name: 'str | None' = None, meta: 'Mapping[str, Any] | None' = None, copy: 'bool' = True) -> 'None'
```

#### Inputs

| Name | Meaning |
|---|---|
| `spec` | ControlSpec. |
| `values` | Matrix with shape spec.shape. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `methods` | as_matrix, flatten, channel, norm, max_abs, arithmetic |

#### Requires

- ControlSpec
- finite matrix

#### Best for

- all optimizer, guess, diagnostic, and repair calls

#### Example

```python
controls = opt.Controls.zeros(system.control_spec())
```

Source: `optimizer/controls.py`.

### `base.evaluate`

Evaluate a system/Controls pair and wrap metrics in Evaluation.

Family: `olgs`, `olss`.

```python
evaluate(system: '_Any', controls: '_Any', *, use_cache: 'bool' = True) -> '_Any'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...). |
| `controls` | Controls matching system.control_spec(). |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Evaluation |
| `fields` | controls, metrics, J, to_dict(...) |

#### Requires

- system.evaluate
- Controls

#### Best for

- quick objective/metric snapshots

#### Example

```python
evaluation = opt.evaluate(system, controls)
```

Source: `optimizer/core/evaluate.py`.

### `base.gradient`

Call and validate the system analytical gradient.

Family: `olgs`.

```python
gradient(system: '_Any', controls: '_Any', *, use_cache: 'bool' = True) -> 'Controls'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide gradient(...). |
| `controls` | Controls matching system.control_spec(). |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |
| `meaning` | Analytical gradient in the same layout. |

#### Requires

- system.gradient
- Controls

#### Best for

- manual gradient inspection
- custom StepProposal functions

#### Related

- utils.verify_gradient

#### Example

```python
grad = opt.gradient(system, controls)
```

Source: `optimizer/core/evaluate.py`.

### `base.value_and_gradient`

Call and validate a combined objective/analytical-gradient system capability.

Family: `olgs`.

```python
value_and_gradient(system: '_Any', controls: 'Controls', *, use_cache: 'bool' = True) -> 'tuple[Evaluation, Controls]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide value_and_gradient(...). |
| `controls` | Compatible Controls. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | tuple[Evaluation, Controls] |

#### Requires

- system.value_and_gradient
- Controls

#### Best for

- systems that reuse an expensive forward propagation

#### Example

```python
evaluation, gradient = opt.value_and_gradient(system, controls)
```

Source: `optimizer/core/evaluate.py`.

### `base.OptimizerResult`

Standard return object from optimizer calls.

Family: `olgs`.

```python
OptimizerResult(controls: 'Controls', metrics: 'dict[str, Any]', stop_reason: 'str', iterations: 'int', optimizer: 'str', objective_metric: 'str' = 'J', state: 'RunState | None' = None, system_params: 'dict[str, Any]' = <factory>, diagnostics: 'dict[str, Any]' = <factory>, run_record: 'RunRecord | None' = None) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `constructor` | Usually produced by optimizer methods. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, state, diagnostics, warmstart() |

#### Requires

- Controls
- metrics with J

#### Best for

- inspecting run outcome
- chaining via warmstart
- exporting to_dict(...)

#### Example

```python
warm = result.warmstart(target_optimizer='lbfgs')
```

Source: `optimizer/result.py`.

### `base.WarmStartState`

Safe handoff state for starting or continuing an optimizer.

Family: `olgs`.

```python
WarmStartState(controls: 'Controls', metrics: 'dict[str, Any]' = <factory>, step_size: 'float | None' = None, optimizer_state: 'dict[str, Any] | None' = None, source_optimizer: 'str | None' = None, target_optimizer: 'str | None' = None, system_params: 'dict[str, Any]' = <factory>) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `result_or_state` | Usually created from result.warmstart(...) or opt.warmstart(...). |

#### Returns

| Name | Meaning |
|---|---|
| `type` | WarmStartState |
| `fields` | controls, metrics, step_size, compatible optimizer_state |

#### Requires

- Controls

#### Best for

- chaining optimizer chunks
- transferring compatible optimizer memory

#### Example

```python
second = opt.optimizers.adam(system, warmstart=first.warmstart('adam'), maxiter=20)
```

Source: `optimizer/state.py`.

### `base.SystemRequest`

Family-aware operation request used by shared core and future CLI/MCP adapters.

Family: `olgs`, `olss`.

```python
SystemRequest(operation: 'str', payload: 'Mapping[str, Any]' = <factory>, options: 'Mapping[str, Any]' = <factory>, family: 'str | None' = None, request_id: 'str' = <factory>) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `operation` | Required OLGS or OLSS method name. |
| `payload/options` | Structured arguments. |
| `family` | Optional olgs/olss assertion. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | SystemRequest |
| `fields` | request_id, family, operation, payload, options, to_dict() |

#### Related

- base.SystemResponse
- base.transact

Source: `optimizer/transactions.py`.

### `base.SystemResponse`

Stable success/error response envelope for one system transaction.

Family: `olgs`, `olss`.

```python
SystemResponse(request_id: 'str', family: 'str | None', operation: 'str | None', ok: 'bool', result: 'Any' = None, error: 'Mapping[str, Any] | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | SystemResponse |
| `fields` | request_id, family, operation, ok, result, error, to_dict() |

#### Related

- base.SystemRequest
- base.transact

Source: `optimizer/transactions.py`.

### `base.SystemTransactionAdapter`

Bind one OLGS or OLSS system to the shared request/response execution shape.

Family: `olgs`, `olss`.

```python
SystemTransactionAdapter(system: 'Any') -> 'None'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Contract-valid OLGS or OLSS object. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | SystemTransactionAdapter |
| `methods` | execute(request) -> SystemResponse |

#### Related

- base.SystemRequest
- base.transact

Source: `optimizer/transactions.py`.

### `base.transact`

Execute a direct system operation through a family-aware structured envelope.

Family: `olgs`, `olss`.

```python
transact(system: 'Any', request: 'SystemRequest | Mapping[str, Any]') -> 'SystemResponse'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Contract-valid OLGS or OLSS object. |
| `request` | SystemRequest or equivalent mapping. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | SystemResponse |
| `meaning` | JSON-safe result or structured failure. |

#### Related

- base.SystemRequest
- base.SystemResponse

#### Example

```python
response = opt.transact(system, opt.SystemRequest('evaluate', {'controls': controls}))
```

Source: `optimizer/transactions.py`.

### `base.context`

Bind a system once for notebook/curriculum convenience.

Family: `olgs`.

```python
context(system: '_Any', **defaults: '_Any') -> 'OptimizerContext'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | System object to bind. |
| `defaults` | Default kwargs for bound runs. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerContext |
| `methods` | ctx.adam, ctx.evaluate, ctx.with_secondary, ctx.*_guess |

#### Requires

- system contract

#### Best for

- notebooks
- curriculum workflows
- reducing repeated system arguments

#### Example

```python
ctx = opt.context(system); result = ctx.adam(controls, maxiter=10)
```

Source: `optimizer/library.py`.

### `base.system_info`

Return declared system capabilities and optional static metadata without evaluating controls.

Family: `olgs`, `olss`.

```python
system_info(system: '_Any') -> 'dict[str, _Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Any system object; required and optional hooks are probed structurally. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `schema` | optimizer.system-info/v1 |

#### Best for

- checking a system API before an optimization run
- building generic tooling

#### Watch out

- describe() and system_spec() must themselves be static hooks.

#### Example

```python
system_info = opt.system_info(system)
```

Source: `optimizer/library.py`.

### `base.plot_data`

Request one explicit system-owned visualization data product without rendering it.

Family: `olgs`, `olss`.

```python
plot_data(system: '_Any', controls: '_Any', *, kind: 'str', **options: '_Any') -> 'PlotData'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLGS or OLSS system exposing the optional plot_data hook. |
| `controls` | Compatible OLGS Controls or OLSS ShootingVariables. |
| `kind` | Declared visualization kind such as bloch. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | PlotData |
| `fields` | kind, named layers, annotations, metadata |

#### Requires

- system.plot_data
- declared system_spec visualization

#### Best for

- explicit trajectory inspection
- renderer-independent system data

#### Watch out

- This can run a full trajectory and is never called automatically by tracking.

#### Example

```python
data = opt.plot_data(system, controls, kind='bloch')
```

Source: `optimizer/core/evaluate.py`.

### `base.context_info`

Return the current registered optimizer, OLGS contract, capability surface.

Family: `olgs`, `olss`.

```python
context_info(system: '_Any | None' = None) -> 'dict[str, _Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Optional system whose declared static metadata should be included. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `schema` | optimizer.context/v1 |

#### Best for

- agent handoff
- API discovery
- checking operation availability

#### Watch out

- It reports capabilities only; it does not run an optimization or evaluate a system.

#### Related

- base.system_info

#### Example

```python
snapshot = opt.context_info(system)
```

Source: `optimizer/introspection.py`.

### `base.continuation`

Run explicit system-parameter stages with safe control transfer and stage gates.

Family: `olgs`.

```python
continuation(system: '_Any', controls: 'Controls', *, stages: '_Iterable[ContinuationStage | dict[str, _Any]]', **kwargs: '_Any') -> 'ContinuationResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Initial base OLGS system. |
| `controls` | Initial Controls. |
| `stages` | ContinuationStage objects or equivalent mappings. |
| `control_transfer` | Required callable when ControlSpec layout or dt changes. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ContinuationResult |
| `fields` | stages, final_result, final_system, stop_reason |

#### Requires

- system.with_secondary
- one or more optimizer stages

#### Best for

- penalty curricula
- homotopy stages
- explicit grid refinement

#### Watch out

- optimizer-private memory is reset between stages
- new-stage controls are re-evaluated

#### Example

```python
result = opt.continuation(system, controls, stages=[opt.ContinuationStage(...)])
```

Source: `optimizer/continuation.py`.

### `base.to_plain`

Convert a library value to plain, strict-JSON data under the one serialization policy; non-finite floats become "nan", "inf", or "-inf".

Family: `olgs`, `olss`.

```python
to_plain(value: 'Any') -> 'Any'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | JSON data |

#### Related

- base.transact

#### Example

```python
payload = opt.to_plain(result)
```

Source: `optimizer/serialization.py`.

### `base.system_family`

Return "olgs" or "olss" from a system's static system_spec(); a missing family means OLGS.

Family: `olgs`, `olss`.

```python
system_family(system: 'Any') -> 'str'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | str |

#### Requires

- system.system_spec

#### Example

```python
family = opt.system_family(system)
```

Source: `optimizer/families.py`.

### `base.control_spec`

Build a ControlSpec from channel keys and a control dimension.

Family: `olgs`.

```python
control_spec(keys: '_Iterable[str]', control_dim: 'int', **kwargs: '_Any') -> 'ControlSpec'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ControlSpec |

#### Example

```python
spec = opt.control_spec(('ux', 'uy'), 33)
```

Source: `optimizer/library.py`.

### `base.controls`

Build Controls from a matrix-like payload for a ControlSpec.

Family: `olgs`.

```python
controls(spec: 'ControlSpec', values: '_Any', **kwargs: '_Any') -> 'Controls'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |

Source: `optimizer/library.py`.

### `base.zeros`

Build zero Controls for a ControlSpec.

Family: `olgs`.

```python
zeros(spec: 'ControlSpec', **kwargs: '_Any') -> 'Controls'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |

Source: `optimizer/library.py`.

### `base.constant`

Build constant Controls for a ControlSpec.

Family: `olgs`.

```python
constant(spec: 'ControlSpec', value: '_Any', **kwargs: '_Any') -> 'Controls'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | Controls |

Source: `optimizer/library.py`.

### `base.Evaluation`

Snapshot of one system evaluation: controls, metrics, and J.

Family: `olgs`.

```python
Evaluation(controls: 'Controls', metrics: 'dict[str, Any]', system_state: 'dict[str, Any] | None' = None, objective_metric: 'str' = 'J') -> None
```

Source: `optimizer/result.py`.

### `base.RunState`

Mutable state of one active optimization run: controls, metrics, step size, and optimizer memory.

Family: `olgs`.

```python
RunState(controls: 'Controls', metrics: 'dict[str, Any]' = <factory>, iteration: 'int' = 0, global_iteration: 'int' = 0, step_size: 'float | None' = None, optimizer_name: 'str | None' = None, optimizer_state: 'dict[str, Any]' = <factory>, best_controls: 'Controls | None' = None, best_metrics: 'dict[str, Any] | None' = None, stop_reason: 'str | None' = None, system_params: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/state.py`.

### `base.warmstart`

Create a safe WarmStartState from a result or active run state, optionally for a target optimizer.

Family: `olgs`.

```python
warmstart(result_or_state: '_Any', *, target_optimizer: 'str | None' = None) -> 'WarmStartState'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | WarmStartState |

Source: `optimizer/state.py`.

### `base.parallel_map`

Map a function over items with local parallel execution configured by ParallelConfig.

Family: `olgs`, `olss`.

```python
parallel_map(function: '_Callable[[_Any], _Any]', items: '_Iterable[_Any]', *, config: 'ParallelConfig | None' = None) -> 'list[_Any]'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | list |

Source: `optimizer/utils/parallel.py`.

### `base.ParallelConfig`

Configuration for local map-style parallel execution.

Family: `olgs`, `olss`.

```python
ParallelConfig(backend: 'str' = 'serial', workers: 'int | None' = None, chunksize: 'int' = 1) -> None
```

Source: `optimizer/utils/parallel.py`.

### `base.OptimizerContext`

Bound-system facade returned by context() for notebook and curriculum workflows.

Family: `olgs`.

```python
OptimizerContext(system: 'Any', library: "'OptimizerLibrary'", defaults: 'dict[str, Any] | None' = None) -> None
```

Source: `optimizer/library.py`.

### `base.OptimizerLibrary`

Facade class behind the root shortcuts; library is its default instance.

Family: `olgs`, `olss`.

```python
OptimizerLibrary() -> 'None'
```

Source: `optimizer/library.py`.

### `base.library`

Default OptimizerLibrary instance that serves the root shortcuts.

Family: `olgs`, `olss`.

Source: `optimizer/library.py`.

### `base.methods`

Return the status record of every public root function, built from the catalog declarations.

Family: `olgs`, `olss`.

```python
methods() -> 'dict[str, MethodInfo]'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict[str, MethodInfo] |

Source: `optimizer/library.py`.

### `base.MethodInfo`

Status record for one public root function: name, status, source module, and phase.

Family: `olgs`, `olss`.

```python
MethodInfo(name: 'str', status: 'str', module: 'str', phase: 'str') -> None
```

Source: `optimizer/library.py`.

### `base.ContinuationStage`

One declared system transition and bounded optimizer segment of an OLGS continuation plan.

Family: `olgs`.

```python
ContinuationStage(label: 'str', optimizer: 'str | OptimizerCallable', secondary_updates: 'Mapping[str, Any]' = <factory>, optimizer_options: 'Mapping[str, Any]' = <factory>, gate: 'StageGate' = <factory>, system_factory: 'SystemFactory | None' = None, control_transfer: 'ControlTransfer | None' = None) -> None
```

Source: `optimizer/continuation.py`.

### `base.StageGate`

Explicit evidence required before an OLGS continuation advances to the next stage.

Family: `olgs`.

```python
StageGate(objective_target: 'float | None' = None, objective_mode: 'str' = 'min', feasibility_tolerance: 'float | None' = None, residual_name: 'str' = 'hard', stationarity_tolerance: 'float | None' = None) -> None
```

Source: `optimizer/continuation.py`.

### `base.StageResult`

One completed continuation stage: its system, optimizer result, gate verdict, and control transfer.

Family: `olgs`.

```python
StageResult(index: 'int', stage: 'ContinuationStage', system: 'Any', result: 'OptimizerResult', gate_passed: 'bool', gate_metrics: 'Mapping[str, Any]', transfer: 'str') -> None
```

Source: `optimizer/continuation.py`.

### `base.ContinuationResult`

Result chain of an OLGS continuation plan: stages, final result, final system, and stop reason.

Family: `olgs`.

```python
ContinuationResult(stages: 'tuple[StageResult, ...]', final_result: 'OptimizerResult', final_system: 'Any', stop_reason: 'str', plan_id: 'str', attempts: 'list[dict[str, Any]]' = <factory>) -> None
```

Source: `optimizer/continuation.py`.

### `base.constant_schedule`

Create a ConstantSchedule.

Family: `olgs`.

```python
constant_schedule(value: 'float') -> 'ConstantSchedule'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ConstantSchedule |

Source: `optimizer/library.py`.

### `base.adaptive_step_schedule`

Create an AdaptiveStepSchedule from keyword arguments.

Family: `olgs`.

```python
adaptive_step_schedule(**kwargs: 'Any') -> 'AdaptiveStepSchedule'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | AdaptiveStepSchedule |

Source: `optimizer/library.py`.

### `base.exponential_decay_schedule`

Create an ExponentialDecaySchedule from keyword arguments.

Family: `olgs`.

```python
exponential_decay_schedule(**kwargs: 'Any') -> 'ExponentialDecaySchedule'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | ExponentialDecaySchedule |

Source: `optimizer/library.py`.

### `base.linear_warmup_schedule`

Create a LinearWarmupSchedule from keyword arguments.

Family: `olgs`.

```python
linear_warmup_schedule(**kwargs: 'Any') -> 'LinearWarmupSchedule'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | LinearWarmupSchedule |

Source: `optimizer/library.py`.

### `base.cosine_decay_schedule`

Create a CosineDecaySchedule from keyword arguments.

Family: `olgs`.

```python
cosine_decay_schedule(**kwargs: 'Any') -> 'CosineDecaySchedule'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | CosineDecaySchedule |

Source: `optimizer/library.py`.

## core

Shared chunk engine: the proposal contract, validated system evaluation, acceptance guards, and stopping rules.

### `core.run_chunk`

Shared optimizer engine for custom StepProposal functions.

Family: `olgs`, `olss`. Root shortcut: `opt.run_chunk`.

```python
run_chunk(system: 'Any', controls: 'Controls | None' = None, *, step: 'StepFunction', optimizer_name: 'str', maxiter: 'int', state: 'RunState | None' = None, warmstart: 'WarmStartState | None' = None, step_size: 'float | None' = None, stopping: 'StoppingConfig | None' = None, target_value: 'float | None' = None, target_metric: 'str | None' = None, target_mode: 'str | None' = None, stall_patience: 'int | None' = None, stall_tolerance: 'float' = 0.0, stall_metric: 'str | None' = None, stall_mode: 'str | None' = None, gradient_tolerance: 'float | None' = None, stationary_curvature_tolerance: 'float' = 1e-10, accept: 'AcceptFunction | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, system_params: 'Mapping[str, Any] | None' = None, stage: 'str | None' = None, use_cache: 'bool' = True, enforce_bounds: 'bool' = True, failed_trial_retries: 'int' = 3, failed_trial_shrink: 'float' = 0.5, gradient_source: 'str' = 'gradient', prefer_value_and_gradient: 'bool' = True, coordinate_mode: 'str' = 'auto') -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLGS or OLSS project contract; dispatches by family. |
| `controls/state/warmstart` | Starting point for the chunk. |
| `step` | Callable from StepContext to StepProposal or native variables. |
| `optimizer_name` | Name recorded in the numerical result. |
| `maxiter` | Chunk iteration budget. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, diagnostics |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- family contract
- native variables
- step callable

#### Best for

- custom optimizer experiments
- shared tracing/checkpoint behavior
- advanced users

#### Watch out

- attempted iterations are counted whether a trial is accepted or rejected
- pass either stopping=StoppingConfig(...) or the loose target/stall arguments, not both

#### Related

- core.StepContext
- core.StepProposal
- core.metric_guard

#### Example

```python
result = opt.run_chunk(system, controls, step=my_step, optimizer_name='custom', maxiter=5)
```

Source: `optimizer/core/engine.py`.

### `core.StepContext`

Read-only bundle the engine hands to a proposal function each iteration.

Family: `olgs`, `olss`. Root shortcut: `opt.StepContext`.

```python
StepContext(system: 'OptimizerSystem', evaluator: 'SystemEvaluator', state: 'RunState', evaluation: 'Evaluation | None', gradient: 'Controls | None', objective_direction: 'str', iteration: 'int', stage: 'str | None' = None, residuals: 'np.ndarray | None' = None, jacobian: 'np.ndarray | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `fields` | system, evaluator, state, variables, evaluation, gradient, residuals, jacobian, iteration, stage |
| `meaning` | Everything a method needs to propose the next trial controls. |

#### Requires

- run_chunk

#### Best for

- writing a custom step callable

#### Related

- core.StepProposal
- core.run_chunk

#### Example

```python
def my_step(ctx): return opt.StepProposal(ctx.state.controls - 0.01 * ctx.gradient)
```

Source: `optimizer/core/engine.py`.

### `core.StepProposal`

Trial update returned by one optimizer iteration.

Family: `olgs`, `olss`. Root shortcut: `opt.StepProposal`.

```python
StepProposal(controls: 'Any', step_size: 'float | None' = None, optimizer_state: 'Mapping[str, Any] | None' = None, optimizer_state_on_accept: 'Mapping[str, Any] | None' = None, optimizer_state_on_reject: 'Mapping[str, Any] | None' = None, step_size_on_accept: 'float | None' = None, step_size_on_reject: 'float | None' = None, technical: 'Mapping[str, Any]' = <factory>, reason: 'str | None' = None, stop_reason: 'str | None' = None) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Proposed native variables (Controls or ShootingVariables). |
| `step_size` | Optional step size to retain on the chunk state. |
| `optimizer_state` | Optimizer-private state update applied after the proposal. |
| `optimizer_state_on_accept/on_reject` | Branch-specific state so moments advance only on accepted moves. |
| `technical` | Free-form proposal evidence. |

#### Requires

- native variables

#### Best for

- returning a trial from a custom step callable

#### Watch out

- branch-specific fields win over the plain optimizer_state field

#### Related

- core.StepContext
- core.AcceptanceDecision

#### Example

```python
proposal = opt.StepProposal(trial_controls, step_size=0.05)
```

Source: `optimizer/core/engine.py`.

### `core.AcceptanceDecision`

Engine verdict on whether a trial replaces the current controls.

Family: `olgs`. Root shortcut: `opt.AcceptanceDecision`.

```python
AcceptanceDecision(accepted: 'bool', reason: 'str', technical: 'dict[str, Any]' = <factory>, optimizer_state_update: 'Mapping[str, Any] | None' = None, next_step_size: 'float | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `fields` | accepted, reason, technical |
| `meaning` | Recorded on the iteration log with the reason string. |

#### Requires

- Evaluation

#### Best for

- writing a custom accept function

#### Related

- core.metric_guard
- core.run_chunk

#### Example

```python
return opt.AcceptanceDecision(accepted=True, reason='accepted')
```

Source: `optimizer/core/engine.py`.

### `core.SystemEvaluator`

Validation and cache wrapper around an optimizer-facing system.

Family: `olgs`. Root shortcut: `opt.SystemEvaluator`.

```python
SystemEvaluator(system: 'Any', *, use_cache: 'bool' = True, cache_size: 'int | None' = 32) -> 'None'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Object satisfying the optimizer system contract. |
| `use_cache` | Reuse evaluations when numeric control content is unchanged. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | SystemEvaluator |
| `meaning` | evaluate/gradient raise; try_evaluate/try_gradient return structured outcomes. |

#### Requires

- system.evaluate
- system.gradient

#### Best for

- batch probes that should not repeat expensive propagation

#### Not for

- finite-difference fallbacks, which live in utils.derivatives

#### Related

- core.run_chunk
- utils.verify_gradient

#### Example

```python
ev = opt.SystemEvaluator(system); snapshot = ev.evaluate(controls)
```

Source: `optimizer/core/evaluate.py`.

### `core.StoppingConfig`

Declarative stopping rules for one chunk: budget, target, stall, and finiteness.

Family: `olgs`. Root shortcut: `opt.StoppingConfig`.

```python
StoppingConfig(maxiter: 'int', target_value: 'float | None' = None, target_metric: 'str | None' = None, target_mode: 'str | None' = None, stall_patience: 'int | None' = None, stall_tolerance: 'float' = 0.0, stall_metric: 'str | None' = None, stall_mode: 'str | None' = None, check_finite: 'bool' = True) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `maxiter` | Attempted engine iterations. |
| `target_value/target_metric/target_mode` | Stop once a metric reaches a threshold. |
| `stall_patience/stall_tolerance/stall_metric` | Stop when a metric stops improving. |
| `check_finite` | Stop before target/stall logic when metrics go non-finite. |

#### Requires

- scalar metrics

#### Best for

- reusing one stopping policy across chunks

#### Related

- core.run_chunk

#### Example

```python
result = opt.run_chunk(system, controls, step=my_step, optimizer_name='custom', maxiter=50, stopping=opt.StoppingConfig(maxiter=50, target_value=1e-6))
```

Source: `optimizer/core/stopping.py`.

### `core.metric_guard`

Build a reusable multi-metric accept/reject guard for run_chunk or optimizer accept hooks.

Family: `olgs`. Root shortcut: `opt.metric_guard`.

```python
metric_guard(*, improve: 'str' = 'J', mode: 'str' = 'min', tolerance: 'float' = 0.0, require: 'Mapping[str, RuleSpec] | None' = None) -> 'MetricGuard'
```

#### Inputs

| Name | Meaning |
|---|---|
| `improve` | Metric that should improve or not get worse, default J. |
| `mode` | min or max. |
| `require` | Mapping of metric -> (operator, threshold[, tolerance]). |
| `tolerance` | Default tolerance for improve and two-field requirements. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | MetricGuard |
| `meaning` | Callable accept function returning AcceptanceDecision. |

#### Requires

- scalar metrics

#### Best for

- preventing one metric from improving while another breaks
- custom acceptance in line_search/run_chunk

#### Related

- optimizers.line_search
- core.run_chunk
- core.AcceptanceDecision

#### Example

```python
guard = opt.core.metric_guard(improve='J', require={'fidelity': ('>=', 0.0)})
```

Source: `optimizer/core/guards.py`.

### `core.Variables`

Structural flat variables with scales, reconstruction and serialization; implemented by Controls and ShootingVariables.

Family: `olgs`, `olss`.

```python
Variables(*args, **kwargs)
```

Source: `optimizer/core/variables.py`.

### `core.RunRecord`

Bounded numerical samples and whole-run aggregates; distinct from frontend command history.

Family: `olgs`, `olss`.

```python
RunRecord(limit: 'int' = 256, rows: 'list[dict[str, Any]]' = <factory>, total: 'int' = 0, truncated: 'bool' = False, summary: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/core/record.py`.

### `core.CallOutcome`

Captured validated operation outcome with value, ok, reason and error.

Family: `olgs`, `olss`.

```python
CallOutcome(value: 'T | None', ok: 'bool', reason: 'str' = 'ok', error: 'str | None' = None) -> None
```

Source: `optimizer/core/runtime.py`.

### `core.StopReason`

Meaning, outcome class, families, and continuability of one stop reason; StopReason.lookup(reason) also resolves labelled reasons such as stage_gate_failed:<label>.

Family: `olgs`, `olss`.

```python
StopReason(name: 'str', meaning: 'str', families: 'tuple[str, ...]', outcome: 'str', continuable: 'bool', source: 'str') -> None
```

#### Related

- core.STOP_REASONS
- utils.diagnose_run

Source: `optimizer/core/stop_reasons.py`.

### `core.STOP_REASONS`

Read-only registry of every stop reason the library assigns, keyed by name.

Family: `olgs`, `olss`.

#### Returns

| Name | Meaning |
|---|---|
| `type` | Mapping[str, StopReason] |

#### Related

- core.StopReason

#### Example

```python
opt.core.STOP_REASONS['maxiter'].outcome
```

Source: `optimizer/core/stop_reasons.py`.

### `core.StopDecision`

Decision returned by stopping checks: whether to stop and the stop reason.

Family: `olgs`.

```python
StopDecision(stop: 'bool', reason: 'str | None' = None, details: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/core/stopping.py`.

### `core.StopTracker`

Stateful stopping helper for one optimizer chunk.

Family: `olgs`.

```python
StopTracker(config: 'StoppingConfig', *, initial_metrics: 'Mapping[str, Any] | None' = None) -> 'None'
```

Source: `optimizer/core/stopping.py`.

### `core.MetricGuard`

Callable multi-metric accept rule built by metric_guard().

Family: `olgs`. Root shortcut: `opt.MetricGuard`.

```python
MetricGuard(improve: 'str' = 'J', mode: 'str' = 'min', tolerance: 'float' = 0.0, require: 'Mapping[str, RuleSpec]' = <factory>) -> None
```

#### Related

- core.metric_guard

Source: `optimizer/core/guards.py`.

### `core.default_accept`

Default acceptance: accept trial controls when the selected metric does not get worse.

Family: `olgs`.

```python
default_accept(current: 'Evaluation', trial: 'Evaluation', proposal: 'StepProposal', state: 'RunState', *, metric: 'str' = 'J', mode: 'str' = 'min', tolerance: 'float' = 0.0) -> 'AcceptanceDecision'
```

Source: `optimizer/core/engine.py`.

### `core.EvaluationOutcome`

Structured result of an evaluation attempt that may fail; returned by SystemEvaluator.try_evaluate.

Family: `olgs`.

```python
EvaluationOutcome(evaluation: 'Evaluation | None', ok: 'bool', reason: 'str', error: 'str | None' = None) -> None
```

Source: `optimizer/core/evaluate.py`.

### `core.GradientOutcome`

Structured result of a gradient attempt that may fail; returned by SystemEvaluator.try_gradient.

Family: `olgs`.

```python
GradientOutcome(gradient: 'Controls | None', ok: 'bool', reason: 'str', error: 'str | None' = None) -> None
```

Source: `optimizer/core/evaluate.py`.

### `core.ValueGradientOutcome`

Structured result of a combined value and gradient attempt that may fail.

Family: `olgs`.

```python
ValueGradientOutcome(value_gradient: 'ValueGradient | None', ok: 'bool', reason: 'str', error: 'str | None' = None) -> None
```

Source: `optimizer/core/evaluate.py`.

## campaigns

Durable multi-stage coordination using objective and provenance declared by system_spec().

### `campaigns.Campaign`

Campaign object with durable stage, system-fingerprint, and selection evidence.

Family: `olgs`.

```python
Campaign(system: 'Any', root: 'str | Path | None' = None, *, system_factory: 'SystemFactory | None' = None, campaign_id: 'str | None' = None) -> 'None'
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.create`

Create a new campaign rooted at a new directory.

Family: `olgs`.

```python
create(system: 'Any', root: 'str | Path | None' = None, *, system_factory: 'SystemFactory | None' = None, campaign_id: 'str | None' = None) -> 'Campaign'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Declared OLGS system. |
| `root` | Nonexistent campaign path. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Campaign |

#### Requires

- system.system_spec

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.MethodSpec`

Declared optimizer name and bounded options for one campaign stage.

Family: `olgs`.

```python
MethodSpec(name: 'str', options: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.SelectionSpec`

System-owned metric and direction used for ranking; never inferred from J.

Family: `olgs`.

```python
SelectionSpec(metric_id: 'str', label: 'str', direction: 'str') -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.SystemReference`

Digest of the normalized system-owned semantic specification for one stage.

Family: `olgs`.

```python
SystemReference(system_id: 'str', fingerprint: 'str', system_spec: 'dict[str, Any]') -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.CandidateRef`

One selected stage result tied to an exact system and checkpoint.

Family: `olgs`.

```python
CandidateRef(candidate_id: 'str', mode: 'str', result: 'OptimizerResult', system: 'SystemReference', selection: 'SelectionSpec', source_id: 'str | None' = None) -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.ModeResult`

Candidates produced by a campaign mode, sorted by the system-owned objective.

Family: `olgs`.

```python
ModeResult(mode: 'str', candidates: 'tuple[CandidateRef, ...]', selected: 'CandidateRef') -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.ValidationResult`

Explicit validation of a candidate against the system that produced it.

Family: `olgs`.

```python
ValidationResult(source_id: 'str', system_fingerprint: 'str', metrics: 'Mapping[str, Any]', gradient_norm: 'float | None', counters: 'Mapping[str, int]') -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.random_smooth_guesses`

Generate count random smooth guesses with consecutive seeds.

Family: `olgs`.

```python
random_smooth_guesses(system: 'Any', count: 'int', *, amplitude: 'Any', seed: 'int' = 0, **options: 'Any') -> 'tuple[Controls, ...]'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | tuple[Controls, ...] |

#### Related

- guesses.random_smooth_guess

Source: `optimizer/campaigns/__init__.py`.

## visualization

Render explicit system-owned plot data; visualization never invents a trajectory.

### `visualization.PlotData`

Named layers and metadata for one explicit visualization product.

Family: `olgs`, `olss`. Root shortcut: `opt.PlotData`.

```python
PlotData(kind: 'str', layers: 'Sequence[PlotLayer]', annotations: 'Mapping[str, Any]' = <factory>, metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/visualization/models.py`.

### `visualization.plot_bloch`

Render validated Bloch layers without calling the system.

Family: `olgs`, `olss`.

```python
plot_bloch(value: 'Any', *, parameter: 'Any' = None, projections: 'bool' = True, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, figsize: 'tuple[float, float] | None' = None, colorbar: 'bool' = True, axes: 'bool' = False, appearance: 'str' = 'legacy', cmap: 'str | None' = None, color_by: 'str | None' = None, color_range: 'tuple[float, float] | None' = None, linewidth: 'float | None' = None, markers: 'bool | None' = None, arrows: 'int' = 0) -> 'tuple[Any, ...]'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.render_bloch`

Export Bloch PNG/PDF artifacts and return a serializable receipt without system calls; published files survive later history errors.

Family: `olgs`, `olss`.

```python
render_bloch(value: 'Any', output: 'str', *, parameter: 'Any' = None, options: 'Mapping[str, Any] | None' = None, formats: 'tuple[str, ...]' = ('png', 'pdf'), dpi: 'int' = 400) -> 'dict[str, Any]'
```

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `schema` | optimizer.bloch-render/1 |

Source: `optimizer/visualization/render.py`.

### `visualization.draw_timeseries`

Draw named per-component channels vs a scalar parameter onto an existing 2D axis; shared by plot_bloch's projections subplot and the standalone panel functions.

Family: `olgs`, `olss`.

```python
draw_timeseries(ax: 'Any', data: 'PlotData', *, palette: 'Sequence[str]' = ('#0072B2', '#E69F00', '#009E73'), component_labels: 'Sequence[str]' = ('x', 'y', 'z'), ylabel: 'str' = 'Bloch components', xlim: 'tuple[float, float] | None' = None, ylim: 'tuple[float, float] | None' = (-1.14, 1.14), prefix_labels: 'bool' = True, legend_ncols: 'int' = 2) -> 'None'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.draw_sphere`

Draw a light unit-sphere scaffold on an existing 3D axis.

Family: `olgs`, `olss`.

```python
draw_sphere(ax: 'Any', *, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, nu: 'int' = 25, nv: 'int' = 13, axes: 'bool' = False, axes_color: 'str' = '#555555') -> 'Any'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.style_context`

Apply the compact serif plotting style without mutating global rcParams.

Family: `olgs`, `olss`.

```python
style_context() -> 'Iterator[None]'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.validate_bloch_data`

Validate generic plot data for rendering inside the Bloch ball.

Family: `olgs`, `olss`.

```python
validate_bloch_data(value: 'PlotData | dict[str, Any]', *, tolerance: 'float' = 1e-08) -> 'PlotData'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.plot_bloch_panel`

Render one standalone Bloch-sphere panel (single figure, no projections subplot).

Family: `olgs`, `olss`.

```python
plot_bloch_panel(value: 'PlotData | Mapping[str, Any]', *, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, figsize: 'tuple[float, float] | None' = None, colorbar: 'bool' = True, axes: 'bool' = False, parameter: 'Any' = None, appearance: 'str' = 'legacy', cmap: 'str | None' = None, color_by: 'str | None' = None, color_range: 'tuple[float, float] | None' = None, linewidth: 'float | None' = None, markers: 'bool | None' = None, arrows: 'int' = 0) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.plot_projection_panel`

Render one standalone Bloch-component-vs-parameter panel from validated Bloch trajectory data.

Family: `olgs`, `olss`.

```python
plot_projection_panel(value: 'PlotData | Mapping[str, Any]', *, component_labels: 'Sequence[str]' = ('$r_x$', '$r_y$', '$r_z$'), xlim: 'tuple[float, float] | None' = None, figsize: 'tuple[float, float]' = (3.4, 3.0)) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.plot_controls_panel`

Render one standalone control-amplitude-vs-parameter panel; uses CONTROL_COLORS and skips the Bloch-ball check.

Family: `olgs`, `olss`.

```python
plot_controls_panel(value: 'PlotData | Mapping[str, Any]', *, component_labels: 'Sequence[str]' = ('$u_x$', '$u_y$', '$u_z$'), xlim: 'tuple[float, float] | None' = None, figsize: 'tuple[float, float]' = (4.5, 2.75)) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.validate_controls_data`

Validate generic plot data for a control-amplitude-vs-parameter panel (no Bloch-ball check).

Family: `olgs`, `olss`.

```python
validate_controls_data(value: 'PlotData | Mapping[str, Any]') -> 'PlotData'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.save_figure`

Save a renderer-produced figure to an explicit path.

Family: `olgs`, `olss`.

```python
save_figure(figure: 'Any', path: 'Any', *, dpi: 'int' = 400, close: 'bool' = False) -> 'None'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.save_panel`

Save one panel figure as .pdf and .png from a shared path stem at one dpi.

Family: `olgs`, `olss`.

```python
save_panel(figure: 'Any', out_stem: 'str | Path', *, formats: 'tuple[str, ...]' = ('pdf', 'png'), dpi: 'int' = 400, close: 'bool' = True) -> 'tuple[Path, ...]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.PlotLayer`

One named data layer with generic coordinates and presentation hints.

Family: `olgs`, `olss`. Root shortcut: `opt.PlotLayer`.

```python
PlotLayer(name: 'str', coordinates: 'Mapping[str, Any]', label: 'str | None' = None, style: 'Mapping[str, Any]' = <factory>, metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/visualization/models.py`.

### `visualization.coerce_plot_data`

Normalize a PlotData instance or an equivalent mapping into PlotData.

Family: `olgs`, `olss`.

```python
coerce_plot_data(value: 'PlotData | Mapping[str, Any]') -> 'PlotData'
```

Source: `optimizer/visualization/models.py`.

## system olgs

Mandatory static system semantics and capability-gated numerical hooks owned by each problem.

### `system_olgs.get_system_spec`

Return and validate the required static system semantic declaration.

Family: `olgs`. Root shortcut: `opt.get_system_spec`.

```python
get_system_spec(system: 'Any') -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.require_system`

Validate the required OLGS contract before an optimizer starts.

Family: `olgs`. Root shortcut: `opt.require_system`.

```python
require_system(system: 'Any') -> 'OLGSystem'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.with_secondary`

Request a system-owned parameter-updated copy for explicit continuation.

Family: `olgs`. Root shortcut: `opt.with_secondary`.

```python
with_secondary(system: 'Any', **updates: 'Any') -> 'Any'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.optional_control_bounds`

Read optional system-declared elementwise bounds.

Family: `olgs`. Root shortcut: `opt.optional_control_bounds`.

```python
optional_control_bounds(system: 'Any') -> 'ControlBounds'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.OLGS`

Base class for a self-contained Open Loop Gradient System.

Family: `olgs`. Root shortcut: `opt.OLGS`.

```python
OLGS()
```

Source: `optimizer/system_olgs/olgs.py`.

### `system_olgs.OLGSystem`

Protocol for the required OLGS public surface.

Family: `olgs`. Root shortcut: `opt.OLGSystem`.

```python
OLGSystem(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SYSTEM_SPEC_SCHEMA`

Schema identifier of the OLGS system_spec() declaration.

Family: `olgs`. Root shortcut: `opt.SYSTEM_SPEC_SCHEMA`.

Source: `optimizer/system_olgs/specs.py`.

### `system_olgs.basic_system_spec`

Build the minimum complete specification for a new OLGS system.

Family: `olgs`. Root shortcut: `opt.basic_system_spec`.

```python
basic_system_spec(system_id: 'str', *, name: 'str | None' = None, system_type: 'str' = 'generic', description: 'str' = '', control_spec: 'ControlSpec | Mapping[str, Any]', objective_direction: 'str' = 'minimize', metrics: 'Mapping[str, Any] | Iterable[str] | None' = None, parameters: 'Mapping[str, Any] | None' = None, derivatives: 'Mapping[str, Any] | None' = None, constraints: 'Mapping[str, Any] | None' = None, visualizations: 'Mapping[str, Any] | None' = None, allow_additional_metrics: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/specs.py`.

### `system_olgs.metric_spec`

Build one normalized OLGS metric declaration for system_spec().

Family: `olgs`. Root shortcut: `opt.metric_spec`.

```python
metric_spec(label: 'str', *, description: 'str | None' = None, role: 'str' = 'diagnostic', dtype: 'str' = 'float', shape: 'Iterable[int]' = (), unit: 'str | None' = None, required: 'bool' = False, allow_nonfinite: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/specs.py`.

### `system_olgs.validate_system_spec`

Validate and normalize a static OLGS system_spec() mapping.

Family: `olgs`. Root shortcut: `opt.validate_system_spec`.

```python
validate_system_spec(value: 'Mapping[str, Any]') -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/specs.py`.

### `system_olgs.validate_declared_metrics`

Validate metric names and declared shapes without changing system values.

Family: `olgs`.

```python
validate_declared_metrics(metrics: 'Mapping[str, Any]', system_spec: 'Mapping[str, Any]') -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/specs.py`.

### `system_olgs.TrackingSpec`

Type of the optional static tracking specification mapping.

Family: `olgs`. Root shortcut: `opt.TrackingSpec`.

```python
TrackingSpec(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.Metrics`

Type of an OLGS metrics mapping from metric name to value.

Family: `olgs`.

```python
Metrics(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.contract_info`

Return the generic OLGS contract without inspecting a concrete system.

Family: `olgs`. Root shortcut: `opt.contract_info`.

```python
contract_info() -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.describe_system`

Describe declared OLGS system capabilities without evaluating controls.

Family: `olgs`. Root shortcut: `opt.describe_system`.

```python
describe_system(system: 'Any') -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SystemProbe`

Presence of required, recommended, and optional OLGS hooks.

Family: `olgs`. Root shortcut: `opt.SystemProbe`.

```python
SystemProbe(has_system_spec: 'bool', has_control_spec: 'bool', has_evaluate: 'bool', has_gradient: 'bool', has_with_secondary: 'bool', has_forward_prop: 'bool', has_back_prop: 'bool', has_metrics: 'bool', has_describe: 'bool', has_constraint_spec: 'bool', has_value_and_gradient: 'bool', has_residuals: 'bool', has_jacobian: 'bool', has_control_bounds: 'bool', has_control_transfer: 'bool', has_hessian: 'bool', has_hvp: 'bool', has_smooth_gradient: 'bool', has_prox: 'bool', has_second_derivative: 'bool', has_simulate: 'bool', has_plot_data: 'bool', has_cache_reset: 'bool', has_cache_status: 'bool') -> None
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.probe_system`

Inspect OLGS hook presence without calling expensive methods.

Family: `olgs`. Root shortcut: `opt.probe_system`.

```python
probe_system(system: 'Any') -> 'SystemProbe'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.get_secondary_update_hook`

Return the system hook that applies secondary-parameter updates.

Family: `olgs`.

```python
get_secondary_update_hook(system: 'Any') -> 'Any'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.ControlBounds`

Elementwise lower and upper control limits in the system control layout.

Family: `olgs`. Root shortcut: `opt.ControlBounds`.

```python
ControlBounds(lower: 'Controls', upper: 'Controls') -> None
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.ValueGradient`

Validated system-owned objective metrics and analytical gradient pair.

Family: `olgs`. Root shortcut: `opt.ValueGradient`.

```python
ValueGradient(metrics: 'Mapping[str, Any]', gradient: 'Controls') -> None
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsValueAndGradient`

Optional capability for sharing work between value and gradient calls.

Family: `olgs`. Root shortcut: `opt.SupportsValueAndGradient`.

```python
SupportsValueAndGradient(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsBounds`

Optional elementwise control-bound capability.

Family: `olgs`. Root shortcut: `opt.SupportsBounds`.

```python
SupportsBounds(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsEqualityConstraints`

Optional hard equality-constraint capability: residuals and their Jacobian.

Family: `olgs`. Root shortcut: `opt.SupportsEqualityConstraints`.

```python
SupportsEqualityConstraints(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsHVP`

Optional Hessian-vector product capability for second-order methods.

Family: `olgs`. Root shortcut: `opt.SupportsHVP`.

```python
SupportsHVP(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsProximalObjective`

Optional composite-objective capability for proximal methods.

Family: `olgs`. Root shortcut: `opt.SupportsProximalObjective`.

```python
SupportsProximalObjective(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsControlTransfer`

Optional system-owned feasibility-preserving control transfer.

Family: `olgs`. Root shortcut: `opt.SupportsControlTransfer`.

```python
SupportsControlTransfer(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.SupportsPlotData`

Optional explicit data-product capability for visualization tools.

Family: `olgs`. Root shortcut: `opt.SupportsPlotData`.

```python
SupportsPlotData(*args, **kwargs)
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.validate_control_spec`

Call and validate system.control_spec().

Family: `olgs`.

```python
validate_control_spec(system: 'OLGSystem') -> 'ControlSpec'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.validate_controls_for_system`

Validate that controls match the system control specification.

Family: `olgs`. Root shortcut: `opt.validate_controls_for_system`.

```python
validate_controls_for_system(system: 'OLGSystem', controls: 'Controls') -> 'ControlSpec'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.validate_metrics`

Validate and normalize an evaluation metrics mapping.

Family: `olgs`. Root shortcut: `opt.validate_metrics`.

```python
validate_metrics(metrics: 'Metrics', *, system_spec: 'dict[str, Any] | None' = None, objective_metric: 'str | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.evaluate_system`

Validate controls, evaluate the system, and normalize its metrics.

Family: `olgs`. Root shortcut: `opt.evaluate_system`.

```python
evaluate_system(system: 'OLGSystem', controls: 'Controls', *, system_spec: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.gradient_system`

Validate controls and return a gradient matching the system layout.

Family: `olgs`. Root shortcut: `opt.gradient_system`.

```python
gradient_system(system: 'OLGSystem', controls: 'Controls') -> 'Controls'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.value_and_gradient_system`

Call and validate the optional combined objective and gradient hook.

Family: `olgs`. Root shortcut: `opt.value_and_gradient_system`.

```python
value_and_gradient_system(system: 'Any', controls: 'Controls', *, system_spec: 'dict[str, Any] | None' = None) -> 'ValueGradient'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_residuals`

Call the optional residual hook and validate a finite 1D vector.

Family: `olgs`. Root shortcut: `opt.optional_residuals`.

```python
optional_residuals(system: 'Any', controls: 'Controls', *, name: 'str' = 'hard') -> 'np.ndarray'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_jacobian`

Call the optional Jacobian hook and validate a finite 2D matrix.

Family: `olgs`. Root shortcut: `opt.optional_jacobian`.

```python
optional_jacobian(system: 'Any', controls: 'Controls', *, name: 'str' = 'hard') -> 'np.ndarray'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_hvp`

Call the optional Hessian-vector product hook and validate the Controls output.

Family: `olgs`. Root shortcut: `opt.optional_hvp`.

```python
optional_hvp(system: 'Any', controls: 'Controls', vector: 'Controls | np.ndarray') -> 'Controls'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_hessian`

Call the optional dense Hessian hook and validate its shape.

Family: `olgs`. Root shortcut: `opt.optional_hessian`.

```python
optional_hessian(system: 'Any', controls: 'Controls') -> 'np.ndarray'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_smooth_gradient`

Call and validate the smooth-part gradient of a composite objective.

Family: `olgs`. Root shortcut: `opt.optional_smooth_gradient`.

```python
optional_smooth_gradient(system: 'Any', controls: 'Controls') -> 'Controls'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_prox`

Call and validate the optional proximal operator.

Family: `olgs`. Root shortcut: `opt.optional_prox`.

```python
optional_prox(system: 'Any', controls: 'Controls', step_size: 'float') -> 'Controls'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_control_transfer`

Call one explicit system-owned feasibility-preserving transfer hook.

Family: `olgs`. Root shortcut: `opt.optional_control_transfer`.

```python
optional_control_transfer(system: 'Any', controls: 'Controls', *, operation: 'str', options: 'Mapping[str, Any] | None' = None) -> 'tuple[Controls, dict[str, Any]]'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.optional_plot_data`

Call and validate an explicit system-owned visualization data product.

Family: `olgs`. Root shortcut: `opt.optional_plot_data`.

```python
optional_plot_data(system: 'Any', controls: 'Controls', *, kind: 'str', **options: 'Any') -> 'Any'
```

Source: `optimizer/system_olgs/validation.py`.

### `system_olgs.finite_difference_jacobian`

Finite-difference Jacobian of system.residuals; the utils version adds named residual sets.

Family: `olgs`.

```python
finite_difference_jacobian(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', eps: 'float' = 1e-06, method: 'str' = 'central') -> 'np.ndarray'
```

#### Related

- utils.finite_difference_jacobian

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.finite_difference_hessian`

Dense finite-difference Hessian of a scalar metric.

Family: `olgs`. Root shortcut: `opt.finite_difference_hessian`.

```python
finite_difference_hessian(system: 'Any', controls: 'Controls', *, metric: 'str | None' = None, eps: 'float' = 1e-05, method: 'str' = 'central') -> 'np.ndarray'
```

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.finite_difference_hvp`

Finite-difference Hessian-vector product from gradient calls.

Family: `olgs`. Root shortcut: `opt.finite_difference_hvp`.

```python
finite_difference_hvp(system: 'Any', controls: 'Controls', vector: 'Controls | np.ndarray', *, eps: 'float' = 1e-06) -> 'Controls'
```

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.get_jacobian`

Return (jacobian, source) from the analytical hook or finite differences.

Family: `olgs`.

```python
get_jacobian(system: 'Any', controls: 'Controls', *, residuals: 'str' = 'hard', fallback: 'bool' = True, eps: 'float' = 1e-06) -> 'tuple[np.ndarray, str]'
```

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.get_hvp`

Return (hvp, source) from the analytical hook or finite differences.

Family: `olgs`.

```python
get_hvp(system: 'Any', controls: 'Controls', vector: 'Controls | np.ndarray', *, fallback: 'bool' = True, eps: 'float' = 1e-06) -> 'tuple[Controls, str]'
```

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.get_hessian`

Return (hessian, source) from the analytical hook or finite differences.

Family: `olgs`.

```python
get_hessian(system: 'Any', controls: 'Controls', *, metric: 'str | None' = None, fallback: 'bool' = True, eps: 'float' = 1e-05) -> 'tuple[np.ndarray, str]'
```

Source: `optimizer/system_olgs/derivatives.py`.

### `system_olgs.SystemEvaluation`

Optional structured evaluation output: controls, validated metrics, and system-owned data.

Family: `olgs`.

```python
SystemEvaluation(controls: 'Controls', metrics: 'dict[str, Any]', data: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olgs/results.py`.

### `system_olgs.ForwardResult`

Optional structured forward-propagation output: controls and system-owned data.

Family: `olgs`.

```python
ForwardResult(controls: 'Controls', data: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olgs/results.py`.

### `system_olgs.BackwardResult`

Optional structured backward-propagation output: controls and system-owned data.

Family: `olgs`.

```python
BackwardResult(controls: 'Controls', data: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olgs/results.py`.

### `system_olgs.GradientResult`

Optional structured gradient output: controls, gradient, optional metrics, and system-owned data.

Family: `olgs`.

```python
GradientResult(controls: 'Controls', gradient: 'Controls', metrics: 'dict[str, Any] | None' = None, data: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olgs/results.py`.

### `system_olgs.SimulationResult`

Optional structured simulation output: controls, metrics, trajectories, and system-owned data.

Family: `olgs`.

```python
SimulationResult(controls: 'Controls', metrics: 'dict[str, Any]', trajectories: 'dict[str, Any]' = <factory>, data: 'dict[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olgs/results.py`.

## system olss

Construction types, validation, and discovery for project-owned OLSS contracts.

### `system_olss.OLSS`

Base class for a self-contained Open Loop Shooting System.

Family: `olss`. Root shortcut: `opt.OLSS`.

```python
OLSS()
```

Source: `optimizer/system_olss/olss.py`.

### `system_olss.ShootingBlock`

One named, scaled block in a shooting vector.

Family: `olss`. Root shortcut: `opt.ShootingBlock`.

```python
ShootingBlock(name: 'str', shape: 'tuple[int, ...]' = (), meaning: 'str' = '', unit: 'str | None' = None, scale: 'float' = 1.0, role: 'str' = 'variable', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.ShootingSpec`

Ordered layout of shooting-variable blocks.

Family: `olss`. Root shortcut: `opt.ShootingSpec`.

```python
ShootingSpec(blocks: 'tuple[ShootingBlock, ...]', dtype: 'str' = 'float64', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.ShootingVariables`

Finite values attached to a ShootingSpec.

Family: `olss`. Root shortcut: `opt.ShootingVariables`.

```python
ShootingVariables(spec: 'ShootingSpec', values: 'np.ndarray', name: 'str' = 'shooting', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.VerificationCheck`

Named independent acceptance check included in OLSS qualification.

Family: `olss`. Root shortcut: `opt.VerificationCheck`.

```python
VerificationCheck(name: 'str', passed: 'bool', value: 'Any' = None, tolerance: 'Any' = None, message: 'str' = '') -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.basic_shooting_system_spec`

Build and validate the static OLSS declaration.

Family: `olss`. Root shortcut: `opt.basic_shooting_system_spec`.

```python
basic_shooting_system_spec(system_id: 'str', *, shooting_spec: 'ShootingSpec', residual_size: 'int', name: 'str | None' = None, system_type: 'str' = 'shooting', description: 'str' = '', residual_groups: 'Iterable[Mapping[str, Any]] | None' = None, metrics: 'Mapping[str, Any] | Iterable[str] | None' = None, parameters: 'Mapping[str, Any] | None' = None, jacobian_source: 'str' = 'analytical', capabilities: 'Mapping[str, Any] | None' = None, selection_metric: 'str | None' = None, selection_direction: 'str' = 'minimize', reconstruction: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None, verification_tolerance: 'float | None' = None, allow_additional_metrics: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/specs.py`.

### `system_olss.shooting_metric_spec`

Build one OLSS metric declaration for system_spec().

Family: `olss`. Root shortcut: `opt.shooting_metric_spec`.

```python
shooting_metric_spec(label: 'str', *, description: 'str | None' = None, role: 'str' = 'diagnostic', dtype: 'str' = 'float', shape: 'Iterable[int]' = (), unit: 'str | None' = None, required: 'bool' = False, allow_nonfinite: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/specs.py`.

### `system_olss.ShootingEvaluation`

Candidate, residual vector, declared metrics, and working options of one evaluation.

Family: `olss`. Root shortcut: `opt.ShootingEvaluation`.

```python
ShootingEvaluation(variables: 'ShootingVariables', residuals: 'np.ndarray', metrics: 'Mapping[str, Any]', options: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingVerification`

Independent verification residuals, metrics, options, independence source, and named checks.

Family: `olss`. Root shortcut: `opt.ShootingVerification`.

```python
ShootingVerification(residuals: 'np.ndarray', metrics: 'Mapping[str, Any]', options: 'Mapping[str, Any]' = <factory>, numerical_success: 'bool' = True, message: 'str' = 'verified', independent: 'bool' = False, independence_source: 'str' = 'unspecified', checks: 'tuple[VerificationCheck, ...]' = ()) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingResult`

One root solve: candidate, residuals, metrics, convergence, verification, qualification, stop reason, error, tolerance source, counters, and history.

Family: `olss`. Root shortcut: `opt.ShootingResult`.

```python
ShootingResult(variables: 'ShootingVariables', residuals: 'np.ndarray', metrics: 'dict[str, Any]', solver_converged: 'bool', verified: 'bool', qualified: 'bool', iterations: 'int', method: 'str', stop_reason: 'str', jacobian_source: 'str', verification: 'ShootingVerification | None' = None, counters: 'dict[str, int]' = <factory>, history: 'list[dict[str, Any]]' = <factory>, error: 'str | None' = None, verification_tolerance: 'float | None' = None, verification_tolerance_source: 'str | None' = None, run_record: 'RunRecord | None' = None, state: 'RunState | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.MultiStartResult`

All multistart results, qualified and unique roots, and the ranking with its selection source.

Family: `olss`. Root shortcut: `opt.MultiStartResult`.

```python
MultiStartResult(results: 'list[ShootingResult]', qualified: 'list[ShootingResult]', unique_qualified: 'list[ShootingResult]', ranked: 'list[ShootingResult]', selection_metric: 'str | None', selection_direction: 'str | None', selection_source: 'str | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ContinuationPoint`

One verified pseudo-arclength branch point with its tangent.

Family: `olss`. Root shortcut: `opt.ContinuationPoint`.

```python
ContinuationPoint(variables: 'ShootingVariables', residual_norm: 'float', tangent: 'np.ndarray', corrector_iterations: 'int', verified: 'bool', verification_independent: 'bool' = False) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingContinuationResult`

Pseudo-arclength branch points, step history, stop reason, and verification tolerance.

Family: `olss`. Root shortcut: `opt.ShootingContinuationResult`.

```python
ShootingContinuationResult(points: 'list[ContinuationPoint]', completed_steps: 'int', requested_steps: 'int', stop_reason: 'str', step_size: 'float', step_history: 'list[dict[str, Any]]' = <factory>, verification_tolerance: 'float | None' = None, verification_tolerance_source: 'str | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ParameterContinuationPoint`

One natural-continuation stage: parameter value, attempted step, and its shooting result.

Family: `olss`. Root shortcut: `opt.ParameterContinuationPoint`.

```python
ParameterContinuationPoint(parameter: 'float', result: 'ShootingResult', attempted_step: 'float') -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ParameterContinuationResult`

Natural-continuation stages, per-attempt step history, stop reason, and target status.

Family: `olss`. Root shortcut: `opt.ParameterContinuationResult`.

```python
ParameterContinuationResult(parameter: 'str', target: 'float', points: 'list[ParameterContinuationPoint]', stop_reason: 'str', reached_target: 'bool', step_history: 'list[dict[str, Any]]' = <factory>) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.OpenLoopShootingSystem`

Protocol for the seven required OLSS calls.

Family: `olss`. Root shortcut: `opt.OpenLoopShootingSystem`.

```python
OpenLoopShootingSystem(*args, **kwargs)
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.SHOOTING_SYSTEM_SPEC_SCHEMA`

Schema identifier of the OLSS system_spec() declaration.

Family: `olss`. Root shortcut: `opt.SHOOTING_SYSTEM_SPEC_SCHEMA`.

Source: `optimizer/system_olss/specs.py`.

### `system_olss.validate_shooting_system_spec`

Validate and normalize a static OLSS system_spec() mapping.

Family: `olss`. Root shortcut: `opt.validate_shooting_system_spec`.

```python
validate_shooting_system_spec(value: 'Mapping[str, Any]') -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/specs.py`.

### `system_olss.get_shooting_system_spec`

Return and validate the static OLSS system_spec() declaration.

Family: `olss`. Root shortcut: `opt.get_shooting_system_spec`.

```python
get_shooting_system_spec(system: 'Any') -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.validate_shooting_spec`

Call shooting_spec() and check it against the declared variable size.

Family: `olss`.

```python
validate_shooting_spec(system: 'Any') -> 'ShootingSpec'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.require_shooting_system`

Validate the required OLSS contract before a solver starts.

Family: `olss`. Root shortcut: `opt.require_shooting_system`.

```python
require_shooting_system(system: 'Any') -> 'OpenLoopShootingSystem'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.ShootingSystemProbe`

Presence of required and optional OLSS hooks, with required_ok.

Family: `olss`. Root shortcut: `opt.ShootingSystemProbe`.

```python
ShootingSystemProbe(has_system_spec: 'bool', has_shooting_spec: 'bool', has_residuals: 'bool', has_jacobian: 'bool', has_evaluate: 'bool', has_verify: 'bool', has_with_secondary: 'bool', has_residuals_and_jacobian: 'bool', has_shooting_bounds: 'bool', has_cost_gradient: 'bool', has_recover_controls: 'bool', has_forward_prop: 'bool', has_simulate: 'bool', has_plot_data: 'bool', has_sparse_jacobian: 'bool', has_jacobian_operator: 'bool', has_roots_equivalent: 'bool', has_cache_reset: 'bool', has_cache_status: 'bool') -> None
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.probe_shooting_system`

Inspect OLSS hook presence without calling numerical methods.

Family: `olss`. Root shortcut: `opt.probe_shooting_system`.

```python
probe_shooting_system(system: 'Any') -> 'ShootingSystemProbe'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.describe_shooting_system`

Describe an OLSS system's declarations and hook presence without evaluating it.

Family: `olss`. Root shortcut: `opt.describe_shooting_system`.

```python
describe_shooting_system(system: 'Any') -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.shooting_contract_info`

Return the generic OLSS contract: required and optional calls.

Family: `olss`. Root shortcut: `opt.shooting_contract_info`.

```python
shooting_contract_info() -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.ShootingOptions`

Type of the per-call OLSS numerical options mapping.

Family: `olss`.

```python
ShootingOptions(*args, **kwargs)
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.ShootingMetrics`

Type of an OLSS metrics mapping from metric name to value.

Family: `olss`.

```python
ShootingMetrics(*args, **kwargs)
```

Source: `optimizer/system_olss/contract.py`.

### `system_olss.validate_variables_for_shooting_system`

Check that ShootingVariables match the system's shooting_spec().

Family: `olss`.

```python
validate_variables_for_shooting_system(system: 'Any', variables: 'ShootingVariables') -> 'ShootingVariables'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.residuals_system`

Call residuals and validate a finite vector of the declared size.

Family: `olss`.

```python
residuals_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None) -> 'np.ndarray'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.jacobian_system`

Call jacobian and validate a finite dense matrix of the declared shape.

Family: `olss`.

```python
jacobian_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None) -> 'np.ndarray'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.residuals_and_jacobian_system`

Call the optional combined hook, or residuals and jacobian separately, and validate both.

Family: `olss`.

```python
residuals_and_jacobian_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None) -> 'tuple[np.ndarray, np.ndarray]'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.evaluate_shooting_system`

Call evaluate and check the metrics against the declaration.

Family: `olss`.

```python
evaluate_shooting_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.verify_shooting_system`

Call verify and normalize it into a ShootingVerification with its independence evidence.

Family: `olss`.

```python
verify_shooting_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None, independent: 'bool' = False, independence_source: 'str' = 'unspecified') -> 'ShootingVerification'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.roots_equivalent_system`

Call the optional system-owned root-equivalence predicate; None means use the numerical default.

Family: `olss`.

```python
roots_equivalent_system(system: 'Any', left: 'ShootingVariables', right: 'ShootingVariables', *, tolerance: 'float') -> 'bool | None'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.cost_gradient_system`

Call and validate the optional cost_gradient hook used by direct shooting.

Family: `olss`.

```python
cost_gradient_system(system: 'Any', variables: 'ShootingVariables', *, options: 'Mapping[str, Any] | None' = None) -> 'np.ndarray'
```

Source: `optimizer/system_olss/validation.py`.

### `system_olss.shooting_bounds_system`

Read and validate optional elementwise shooting-variable bounds.

Family: `olss`.

```python
shooting_bounds_system(system: 'Any') -> 'tuple[np.ndarray, np.ndarray] | None'
```

Source: `optimizer/system_olss/validation.py`.

## shooting

Root solving, verification, multistart, continuation, and direct shooting for OLSS systems.

### `shooting.solve_shooting`

Solve one OLSS residual equation and independently verify the candidate.

Family: `olss`. Root shortcut: `opt.solve_shooting`.

```python
solve_shooting(system: 'Any', initial: 'ShootingVariables', *, method: 'str' = 'lm', tolerance: 'float' = 1e-10, verification_tolerance: 'float | None' = None, maxiter: 'int' = 40, damping: 'float' = 1e-10, max_step_norm: 'float | None' = None, line_search: 'bool' = True, shrink: 'float' = 0.5, max_backtracks: 'int' = 12, options: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None, evaluator: 'ShootingEvaluator | None' = None, previous_result: 'ShootingResult | None' = None, on_iteration: 'Callable[[Mapping[str, Any]], None] | None' = None, on_event: 'Callable[[Mapping[str, Any]], None] | None' = None, step: 'Callable | None' = None, record_limit: 'int' = 256) -> 'ShootingResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLSS system. |
| `initial` | Initial ShootingVariables. |
| `verification_options` | Tighter system evaluation settings. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ShootingResult |

#### Requires

- OLSS required contract

#### Best for

- single shooting
- multiple shooting
- rectangular residual systems

#### Watch out

- solver convergence and verification are separate

#### Example

```python
result = opt.shooting.solve_shooting(system, z0, verification_options={'rtol': 1e-11})
```

Source: `optimizer/shooting/solvers.py`.

### `shooting.solve_shooting_multistart`

Solve independent seeds, deduplicate qualified roots, and rank a declared selection metric.

Family: `olss`. Root shortcut: `opt.solve_shooting_multistart`.

```python
solve_shooting_multistart(system: 'Any', starts: 'Iterable[ShootingVariables]', *, parallel: 'ParallelConfig | None' = None, system_factory: 'Callable[[], Any] | None' = None, dedup_tolerance: 'float' = 1e-07, solver_options: 'Mapping[str, Any] | None' = None) -> 'MultiStartResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `starts` | Iterable of ShootingVariables. |
| `system_factory` | Required for parallel workers. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | MultiStartResult |

#### Requires

- OLSS required contract

#### Best for

- multiple root branches
- seed clouds

#### Example

```python
campaign = opt.shooting.solve_shooting_multistart(system, seeds)
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.continue_shooting`

Follow a one-dimensional OLSS root branch with pseudo-arclength continuation.

Family: `olss`. Root shortcut: `opt.continue_shooting`.

```python
continue_shooting(system: 'Any', previous: 'ShootingVariables', current: 'ShootingVariables', *, steps: 'int', step_size: 'float', tolerance: 'float' = 1e-10, verification_tolerance: 'float | None' = None, max_corrector_iterations: 'int' = 15, options: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None, min_step: 'float | None' = None, max_step: 'float | None' = None, shrink: 'float' = 0.5, grow: 'float' = 1.5, fast_corrector_iterations: 'int' = 4, target_block: 'str | None' = None, target_value: 'float | None' = None, target_tolerance: 'float' = 1e-10) -> 'ShootingContinuationResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `previous/current` | Two verified neighboring branch points. |
| `step_size` | Arclength predictor size. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ShootingContinuationResult |

#### Requires

- Jacobian numerical rank = variable count - 1; redundant residual rows are allowed

#### Best for

- continuation through folds

#### Example

```python
branch = opt.shooting.continue_shooting(system, z_prev, z_now, steps=20, step_size=0.1)
```

Source: `optimizer/shooting/continuation.py`.

### `shooting.solve_direct_shooting`

Optimize a declared selection cost along the feasible OLSS residual set.

Family: `olss`. Root shortcut: `opt.solve_direct_shooting`.

```python
solve_direct_shooting(system: 'Any', initial: 'ShootingVariables', *, tolerance: 'float' = 1e-10, verification_tolerance: 'float | None' = None, maxiter: 'int' = 30, step_size: 'float' = 0.25, damping: 'float' = 1e-10, correction_maxiter: 'int' = 12, options: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None) -> 'ShootingResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLSS system declaring selection metric and cost_gradient. |
| `initial` | Initial ShootingVariables. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | ShootingResult |

#### Requires

- selection metric
- cost_gradient

#### Best for

- underdetermined shooting
- cost-based feasible selection

#### Watch out

- qualification is not a global optimality certificate

#### Example

```python
result = opt.shooting.solve_direct_shooting(system, z0)
```

Source: `optimizer/shooting/solvers.py`.

### `shooting.verify_shooting_jacobian`

Compare the declared OLSS Jacobian with explicit central finite differences.

Family: `olss`. Root shortcut: `opt.verify_shooting_jacobian`.

```python
verify_shooting_jacobian(system: 'Any', variables: 'ShootingVariables', *, eps: 'float' = 1e-06, atol: 'float' = 1e-07, rtol: 'float' = 1e-05, options: 'Mapping[str, Any] | None' = None) -> 'dict[str, Any]'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLSS system. |
| `variables` | Point to check. |
| `eps` | Relative perturbation scale. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | dict |
| `fields` | passed, errors, comparison_steps, step_max_abs_errors, analytical, numerical |

#### Requires

- OLSS residuals and Jacobian

#### Best for

- system bring-up
- sensitivity debugging

#### Example

```python
check = opt.shooting.verify_shooting_jacobian(system, z0)
```

Source: `optimizer/shooting/derivatives.py`.

### `shooting.continue_parameter`

Natural continuation over one secondary parameter with adaptive steps and exact target landing.

Family: `olss`. Root shortcut: `opt.continue_parameter`.

```python
continue_parameter(system: 'Any', initial: 'ShootingVariables', *, parameter: 'str', start: 'float', target: 'float', step_size: 'float', min_step: 'float | None' = None, max_step: 'float | None' = None, shrink: 'float' = 0.5, grow: 'float' = 1.5, max_steps: 'int' = 100, solver_options: 'Mapping[str, Any] | None' = None) -> 'ParameterContinuationResult'
```

Source: `optimizer/shooting/continuation.py`.

### `shooting.random_shooting_seeds`

Generate scale-aware random shooting seeds.

Family: `olss`. Root shortcut: `opt.random_shooting_seeds`.

```python
random_shooting_seeds(spec: 'ShootingSpec', count: 'int', *, seed: 'int | None' = None, center: 'ShootingVariables | None' = None, radius: 'float' = 1.0, bounds: 'tuple[np.ndarray, np.ndarray] | None' = None) -> 'list[ShootingVariables]'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.perturb_shooting_seeds`

Generate a scale-aware perturbation cloud around an anchor.

Family: `olss`. Root shortcut: `opt.perturb_shooting_seeds`.

```python
perturb_shooting_seeds(anchor: 'ShootingVariables', count: 'int', *, relative_radius: 'float' = 0.1, seed: 'int | None' = None, bounds: 'tuple[np.ndarray, np.ndarray] | None' = None) -> 'list[ShootingVariables]'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.refine_shooting_seeds`

Solve, select elites, perturb, and refine a shooting seed campaign.

Family: `olss`. Root shortcut: `opt.refine_shooting_seeds`.

```python
refine_shooting_seeds(system: 'Any', starts: 'Iterable[ShootingVariables]', *, elite_count: 'int' = 3, perturbations_per_elite: 'int' = 4, relative_radius: 'float' = 0.05, seed: 'int | None' = None, solver_options: 'Mapping[str, Any] | None' = None, parallel: 'ParallelConfig | None' = None, system_factory: 'Callable[[], Any] | None' = None) -> 'MultiStartResult'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.compare_shooting_replay`

Replay OLSS-recovered controls in OLGS and compare declared metrics.

Family: `olgs`, `olss`. Root shortcut: `opt.compare_shooting_replay`.

```python
compare_shooting_replay(shooting_system: 'Any', variables: 'ShootingVariables', gradient_system: 'Any', *, times: 'np.ndarray', metric_map: 'Mapping[str, str] | None' = None, shooting_options: 'Mapping[str, Any] | None' = None, interpolation: 'str' = 'system_owned', units: 'Mapping[str, str] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/shooting/replay.py`.

### `shooting.finite_difference_shooting_jacobian`

Central-difference reference Jacobian of the OLSS residual vector; does not replace the declared Jacobian.

Family: `olss`. Root shortcut: `opt.finite_difference_shooting_jacobian`.

```python
finite_difference_shooting_jacobian(system: 'Any', variables: 'ShootingVariables', *, eps: 'float' = 1e-06, options: 'Mapping[str, Any] | None' = None) -> 'np.ndarray'
```

Source: `optimizer/shooting/derivatives.py`.

### `shooting.ShootingEvaluator`

Validated, cached, and counted OLSS evaluation shared by the shooting tools.

Family: `olss`. Root shortcut: `opt.ShootingEvaluator`.

```python
ShootingEvaluator(system: 'Any', *, use_cache: 'bool' = True, cache_size: 'int | None' = 32) -> 'None'
```

Source: `optimizer/shooting/evaluator.py`.

## Stop reasons

Generated from `opt.core.STOP_REASONS`, the one registry of every stop reason the
library assigns. A labelled reason such as `stage_gate_failed:warmup` uses the entry
before the colon. Continuable means the final state is a valid start for another run.

| Reason | Outcome | Families | Continuable | Meaning | Assigned by |
|---|---|---|---|---|---|
| `maxiter` | stopped | olgs, olss | yes | The iteration budget was used up. | engine, root solvers, repair |
| `nonfinite_trial` | failed | olgs, olss | yes | Consecutive trials failed to evaluate: failed_trial_retries in the OLGS engine, every line-search trial of one iteration in OLSS. | engine, root solvers |
| `initial_evaluation_failed` | failed | olgs, olss | no | The starting point could not be evaluated; a coordinator records this for a seed or stage. | CMA-ES, multistart, natural continuation |
| `line_search_failed` | failed | olgs, olss | yes | No backtracking step reduced the residual norm. | residual repair, root solvers |
| `completed` | stopped | olgs, olss | yes | Every requested stage or continuation step finished. | continuation |
| `interrupted` | interrupted | olgs | yes | The run was interrupted; the last valid state is kept. | engine, CMA-ES |
| `target` | stopped | olgs | yes | The watched metric reached target_value. | stopping rules |
| `stall` | stopped | olgs | yes | No improvement for stall_patience iterations. | stopping rules |
| `nonfinite` | failed | olgs | no | The current metrics contain a non-finite value. | stopping rules |
| `gradient_converged` | converged | olgs | yes | The gradient norm is at or below gradient_tolerance. | engine |
| `stationary_nonminimum` | stopped | olgs | yes | The gradient tolerance is met but sampled curvature is negative: a saddle or a maximum. | engine |
| `gradient_failed` | failed | olgs | no | The system gradient failed at the current controls. | engine |
| `smooth_gradient_failed` | failed | olgs | no | The smooth-gradient call of a proximal run failed. | engine |
| `proposal_failed` | failed | olgs, olss | yes | The step function raised or proposed invalid variables. | engine, root solvers |
| `stopped` | stopped | olgs | yes | Fallback label for a stop without a named reason. | engine |
| `projected_kkt_converged` | converged | olgs | yes | Projected gradient met its feasibility and projected-stationarity tolerances. | projected_gradient |
| `restoration_failed` | failed | olgs | yes | Projected gradient could not restore feasibility. | projected_gradient |
| `proximal_mapping_converged` | converged | olgs | yes | The proximal-gradient mapping norm is below tolerance. | proximal_gradient |
| `no_usable_candidates` | failed | olgs | no | CMA-ES produced no finite candidate. | cma_es |
| `converged` | converged | olgs | yes | The residual norm is at or below the repair tolerance. | residual repair |
| `stage_gate_failed` | failed | olgs | yes | A continuation stage did not pass its gate; the stage label follows the colon. | continuation |
| `root_tolerance` | converged | olss | yes | The residual norm reached the working tolerance. | solve_shooting |
| `singular_step` | failed | olgs, olss | yes | The step could not be computed from the Jacobian. | solve_shooting, repair |
| `least_squares_stationary` | stopped | olss | yes | The residual cannot be reduced further: a least-squares stationary point, not a root. | solve_shooting |
| `bound_stalled` | stopped | olgs, olss | yes | Declared bounds blocked every step. | solve_shooting, repair |
| `jacobian_failed` | failed | olgs, olss | no | The Jacobian failed at an accepted point. | solve_shooting, direct shooting, repair |
| `evaluation_failed` | failed | olss | no | Declared metrics failed at the converged point. | solve_shooting |
| `verification_failed` | failed | olss | yes | Independent verification failed, raised, missed its tolerance, or failed a named check. | solve_shooting, continuation |
| `verification_not_independent` | failed | olss | yes | Verification reused the working options, so it cannot qualify the root. | solve_shooting |
| `initial_feasibility_failed` | failed | olss | no | Direct shooting could not correct its start onto the feasible set. | direct shooting |
| `cost_stationary` | converged | olss | yes | The projected cost gradient is below tolerance on the feasible set. | direct shooting |
| `cost_line_search_failed` | stopped | olss | yes | No corrected step reduced the cost. | direct shooting |
| `cost_gradient_failed` | failed | olss | no | The cost gradient failed at the current point. | direct shooting |
| `seed_verification_failed` | failed | olss | no | A pseudo-arclength seed failed verification. | pseudo-arclength continuation |
| `target_landing_failed` | failed | olss | yes | The corrector could not land exactly on the target value. | pseudo-arclength continuation |
| `minimum_step_reached` | stopped | olss | yes | The continuation step shrank below its minimum after failed attempts. | continuation |
| `target_reached` | stopped | olss | yes | The continuation reached its target value exactly. | continuation |
| `max_steps` | stopped | olss | yes | The continuation step budget was used up before the target. | natural continuation |

---

[⌂ Home](README.md)
