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
| [`optimizers`](#optimizers) | 13 | Methods that move Controls to improve system metrics. |
| [`guesses`](#guesses) | 13 | Initial Controls generators and restart helpers. |
| [`utils`](#utils) | 17 | Diagnostics, derivative checks, residual geometry, repair, guards, and spectrum tools. |
| [`schedules`](#schedules) | 5 | Small step-size policy objects for optimizer or repair workflows. |
| [`base`](#base) | 17 | Base data objects and top-level entry points: Controls, results, warmstart state, trace, and bound context utilities. |
| [`core`](#core) | 7 | Shared chunk engine: the proposal contract, validated system evaluation, acceptance guards, and stopping rules. |
| [`campaigns`](#campaigns) | 6 | Durable multi-stage coordination using objective and provenance declared by system_spec(). |
| [`tracker`](#tracker) | 5 | Passive durable run evidence; it records engine facts and never calls system numerics. |
| [`visualization`](#visualization) | 12 | Render explicit system-owned plot data; visualization never invents a trajectory. |
| [`system_olgs`](#system-olgs) | 4 | Mandatory static system semantics and capability-gated numerical hooks owned by each problem. |
| [`system_olss`](#system-olss) | 15 | Construction types, validation, and discovery for project-owned OLSS contracts. |
| [`shooting`](#shooting) | 12 | Root solving, verification, multistart, continuation, and direct shooting for OLSS systems. |
| [`optimizer_log`](#optimizer-log) | 3 | Standalone, read-only tracker reader package; it is not yet a portal UI or command-line application. |

## optimizers

Methods that move Controls to improve system metrics.

### `optimizers.adam`

Adaptive first-order optimizer with first and second moment state.

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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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

```python
cma_es(system: 'Any', controls: 'Controls | None' = None, *, variant: 'str' = 'diagonal', maxiter: 'int' = 100, population_size: 'int | None' = None, elite_fraction: 'float' = 0.5, sigma: 'float' = 0.1, min_sigma: 'float' = 1e-08, covariance_lr: 'float' = 0.2, seed: 'int | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, state: 'RunState | None' = None, warmstart: 'Any | None' = None, stopping: 'StoppingConfig | None' = None, target_value: 'float | None' = None, target_metric: 'str | None' = None, target_mode: 'str | None' = None, stall_patience: 'int | None' = None, stall_tolerance: 'float' = 0.0, stall_metric: 'str | None' = None, stall_mode: 'str | None' = None, trace: 'Trace | None' = None, create_trace: 'bool' = True, observers: 'Iterable[RunObserver] | None' = None, api_context: 'Mapping[str, Any] | None' = None, system_params: 'Mapping[str, Any] | None' = None, stage: 'str | None' = None, chunk: 'int' = 0, use_cache: 'bool' = True, checkpoint_start: 'bool' = True, checkpoint_best: 'bool' = True, checkpoint_final: 'bool' = True) -> 'OptimizerResult'
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
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

```python
RepairResult(controls: 'Controls', residuals: 'np.ndarray', residual_norm: 'float', residual_max_abs: 'float', converged: 'bool', iterations: 'int', method: 'str', residual_name: 'str', jacobian_source: 'str', history: 'list[dict[str, Any]]' = <factory>, stop_reason: 'str' = 'unknown') -> None
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

```python
diagnose_run(result: 'Any', *, mode: 'str | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.curvature_probe`

Probe local curvature, negative-curvature evidence, and a suggested gradient step.

```python
curvature_probe(system: 'Any', controls: 'Controls', *, directions: 'int' = 8, eps: 'float' = 1e-05, seed: 'int' = 0) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.compare_methods`

Compare several optimizers for a short in-memory budget from identical controls.

```python
compare_methods(system: 'Any', controls: 'Controls', methods: 'Iterable[str | Callable[..., Any]] | Mapping[str, Callable[..., Any]]', *, iterations: 'int' = 10, options: 'Mapping[str, Mapping[str, Any]] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

### `utils.shooting_geometry_probe`

Report OLSS Jacobian rank, condition, weak variable blocks, and residual groups.

```python
shooting_geometry_probe(system: 'Any', variables: 'Any', *, options: 'Mapping[str, Any] | None' = None, rcond: 'float | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/utils/run_diagnostics.py`.

## schedules

Small step-size policy objects for optimizer or repair workflows.

### `schedules.ConstantSchedule`

Schedule that always returns the same step size.

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

```python
ExponentialDecaySchedule(initial_step: 'float' = 0.001, decay: 'float' = 0.99, min_step: 'float' = 1e-12) -> None
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.LinearWarmupSchedule`

Linear warmup from an initial to a target step.

```python
LinearWarmupSchedule(initial_step: 'float', target_step: 'float', warmup_steps: 'int') -> None
```

Source: `optimizer/schedules/step_size.py`.

### `schedules.CosineDecaySchedule`

Cosine step decay over a fixed iteration horizon.

```python
CosineDecaySchedule(initial_step: 'float', total_steps: 'int', min_step: 'float' = 0.0) -> None
```

Source: `optimizer/schedules/step_size.py`.

## base

Base data objects and top-level entry points: Controls, results, warmstart state, trace, and bound context utilities.

### `base.ControlSpec`

Named control layout: channel keys, control_dim, dtype, dt, and metadata.

```python
ControlSpec(keys: 'Sequence[str]', control_dim: 'int', dtype: 'Any' = <class 'float'>, dt: 'float | None' = None, meta: 'Mapping[str, Any]' = <factory>) -> None
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

```python
OptimizerResult(controls: 'Controls', metrics: 'dict[str, Any]', stop_reason: 'str', iterations: 'int', optimizer: 'str', objective_metric: 'str' = 'J', state: 'RunState | None' = None, trace: 'Any | None' = None, system_params: 'dict[str, Any]' = <factory>, trace_id: 'str | None' = None, checkpoint_ids: 'dict[str, str]' = <factory>, tracking_failures: 'tuple[dict[str, str], ...]' = ()) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `constructor` | Usually produced by optimizer methods. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, state, trace, warmstart() |

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

```python
WarmStartState(controls: 'Controls', metrics: 'dict[str, Any]' = <factory>, step_size: 'float | None' = None, optimizer_state: 'dict[str, Any] | None' = None, source_optimizer: 'str | None' = None, target_optimizer: 'str | None' = None, trace_id: 'str | None' = None, checkpoint_ids: 'dict[str, str]' = <factory>, system_params: 'dict[str, Any]' = <factory>) -> None
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

```python
SystemResponse(request_id: 'str', family: 'str', operation: 'str', ok: 'bool', result: 'Any' = None, error: 'Mapping[str, Any] | None' = None) -> None
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

### `base.Trace`

In-memory run ledger for iteration records, chunk records, events, and checkpoints.

```python
Trace(run_id: 'str', iteration_records: 'list[IterationRecord]' = <factory>, chunk_records: 'list[ChunkRecord]' = <factory>, event_records: 'list[EventRecord]' = <factory>, checkpoints: 'dict[str, Checkpoint]' = <factory>, labels: 'dict[str, list[str]]' = <factory>) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `run_id` | String id for a run ledger. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Trace |
| `methods` | record_iteration, record_chunk, checkpoint, restore, to_dict |

#### Requires

- run id

#### Best for

- closed-loop inspection
- rollback checkpoints
- technical logs

#### Example

```python
trace = opt.trace('my-run')
```

Source: `optimizer/logs/trace.py`.

### `base.context`

Bind a system once for notebook/curriculum convenience.

```python
context(system: '_Any', *, trace: 'Trace | str | None' = None, **defaults: '_Any') -> 'OptimizerContext'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | System object to bind. |
| `trace` | Optional Trace or run id. |
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
ctx = opt.context(system, trace='run'); result = ctx.adam(controls, maxiter=10)
```

Source: `optimizer/library.py`.

### `base.system_info`

Return declared system capabilities and optional static metadata without evaluating controls.

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

- describe() and tracking_spec() must themselves be static hooks.

#### Example

```python
system_info = opt.system_info(system)
```

Source: `optimizer/library.py`.

### `base.plot_data`

Request one explicit system-owned visualization data product without rendering it.

```python
plot_data(system: '_Any', controls: 'Controls', *, kind: 'str', **options: '_Any') -> 'PlotData'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | OLGS system exposing the optional plot_data hook. |
| `controls` | Validated Controls. |
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

Return the current registered optimizer, OLGS contract, and tracker capability surface.

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
- checking tracker readiness

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

## core

Shared chunk engine: the proposal contract, validated system evaluation, acceptance guards, and stopping rules.

### `core.run_chunk`

Shared optimizer engine for custom StepProposal functions.

```python
run_chunk(system: 'Any', controls: 'Controls | None' = None, *, step: 'StepFunction', optimizer_name: 'str', maxiter: 'int', state: 'RunState | None' = None, warmstart: 'WarmStartState | None' = None, trace: 'Trace | None' = None, create_trace: 'bool' = True, observers: 'Iterable[RunObserver] | None' = None, step_size: 'float | None' = None, stopping: 'StoppingConfig | None' = None, target_value: 'float | None' = None, target_metric: 'str | None' = None, target_mode: 'str | None' = None, stall_patience: 'int | None' = None, stall_tolerance: 'float' = 0.0, stall_metric: 'str | None' = None, stall_mode: 'str | None' = None, gradient_tolerance: 'float | None' = None, stationary_curvature_tolerance: 'float' = 1e-10, accept: 'AcceptFunction | None' = None, accept_metric: 'str | None' = None, accept_mode: 'str | None' = None, accept_tolerance: 'float' = 0.0, system_params: 'Mapping[str, Any] | None' = None, api_context: 'Mapping[str, Any] | None' = None, random_state: 'RandomStateSource | None' = None, stage: 'str | None' = None, chunk: 'int' = 0, use_cache: 'bool' = True, enforce_bounds: 'bool' = True, failed_trial_retries: 'int' = 3, failed_trial_shrink: 'float' = 0.5, checkpoint_start: 'bool' = True, checkpoint_latest: 'bool' = True, checkpoint_accepted: 'bool' = True, checkpoint_best: 'bool' = True, checkpoint_final: 'bool' = True, gradient_source: 'str' = 'gradient', prefer_value_and_gradient: 'bool' = True) -> 'OptimizerResult'
```

#### Inputs

| Name | Meaning |
|---|---|
| `system` | Must provide evaluate(...) and usually gradient(...). |
| `controls/state/warmstart` | Starting point for the chunk. |
| `step` | Callable from StepContext to StepProposal or Controls. |
| `optimizer_name` | Name recorded in result and trace. |
| `maxiter` | Chunk iteration budget. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | OptimizerResult |
| `fields` | controls, metrics, J, stop_reason, iterations, optimizer, state, trace, checkpoint_ids |
| `handoff` | Call result.warmstart(target_optimizer=...) before chaining methods. |

#### Requires

- system.evaluate
- system.gradient
- Controls
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

```python
StepContext(system: 'OptimizerSystem', evaluator: 'SystemEvaluator', state: 'RunState', evaluation: 'Evaluation', gradient: 'Controls', objective_direction: 'str', iteration: 'int', stage: 'str | None' = None) -> None
```

#### Returns

| Name | Meaning |
|---|---|
| `fields` | system, evaluator, state, evaluation, gradient, iteration, stage |
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

```python
StepProposal(controls: 'Controls', step_size: 'float | None' = None, optimizer_state: 'Mapping[str, Any] | None' = None, optimizer_state_on_accept: 'Mapping[str, Any] | None' = None, optimizer_state_on_reject: 'Mapping[str, Any] | None' = None, step_size_on_accept: 'float | None' = None, step_size_on_reject: 'float | None' = None, technical: 'Mapping[str, Any]' = <factory>, reason: 'str | None' = None, stop_reason: 'str | None' = None) -> None
```

#### Inputs

| Name | Meaning |
|---|---|
| `controls` | Proposed trial controls. |
| `step_size` | Optional step size to retain on the chunk state. |
| `optimizer_state` | Optimizer-private state update applied after the proposal. |
| `optimizer_state_on_accept/on_reject` | Branch-specific state so moments advance only on accepted moves. |
| `technical` | Free-form payload recorded in the iteration log. |

#### Requires

- Controls

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

## campaigns

Durable multi-stage coordination using objective and provenance declared by system_spec().

### `campaigns.Campaign`

Campaign object with durable stage, system-fingerprint, and selection evidence.

```python
Campaign(system: 'Any', root: 'str | Path', *, system_factory: 'SystemFactory | None' = None, campaign_id: 'str | None' = None) -> 'None'
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.create`

Create a new campaign rooted at a new directory.

```python
create(system: 'Any', root: 'str | Path', *, system_factory: 'SystemFactory | None' = None, campaign_id: 'str | None' = None) -> 'Campaign'
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

### `campaigns.open`

Open saved campaign evidence with a caller-owned system resolver.

```python
open(root: 'str | Path', *, system_resolver: 'SystemResolver') -> 'Campaign'
```

#### Inputs

| Name | Meaning |
|---|---|
| `root` | Existing campaign path. |
| `system_resolver` | Callable from saved SystemReference to matching system. |

#### Returns

| Name | Meaning |
|---|---|
| `type` | Campaign |

#### Requires

- system resolver
- verified checkpoint artifacts

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.MethodSpec`

Declared optimizer name and bounded options for one campaign stage.

```python
MethodSpec(name: 'str', options: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.SelectionSpec`

System-owned metric and direction used for ranking; never inferred from J.

```python
SelectionSpec(metric_id: 'str', label: 'str', direction: 'str') -> None
```

Source: `optimizer/campaigns/__init__.py`.

### `campaigns.SystemReference`

Digest of the normalized system-owned semantic specification for one stage.

```python
SystemReference(system_id: 'str', fingerprint: 'str', system_spec: 'dict[str, Any]') -> None
```

Source: `optimizer/campaigns/__init__.py`.

## tracker

Passive durable run evidence; it records engine facts and never calls system numerics.

### `tracker.RunJournal`

Append-only receipt, timeline, checkpoint, live, and terminal writer.

```python
RunJournal(run_dir: 'Path', receipt: 'RunReceipt', *, _create_token: 'object | None' = None) -> 'None'
```

Source: `optimizer/tracker/journal.py`.

### `tracker.JournalObserver`

Passive bridge from already-computed core facts to a RunJournal.

```python
JournalObserver(journal: 'RunJournal', *, save_all_checkpoints: 'bool' = False, checkpoint_policy: 'CheckpointPolicy | None' = None) -> 'None'
```

Source: `optimizer/tracker/bridge.py`.

### `tracker.CheckpointPolicy`

Bound durable control artifacts while retaining every scalar iteration record.

```python
CheckpointPolicy(save_all: 'bool' = False, persist_start: 'bool' = True, persist_selected_best: 'bool' = True, persist_failure: 'bool' = True, persist_final: 'bool' = True) -> None
```

Source: `optimizer/tracker/bridge.py`.

### `tracker.read_timeline`

Validate and read append-only tracker events without optimizer imports.

```python
read_timeline(run_dir: 'str | Path') -> 'TimelineReadResult'
```

Source: `optimizer/tracker/reader.py`.

### `tracker.load_checkpoint`

Load a SHA-256 verified control artifact without a system object.

```python
load_checkpoint(run_dir: 'str | Path', reference: 'CheckpointReference | Mapping[str, Any]', *, verify: 'bool' = True) -> 'CheckpointData'
```

Source: `optimizer/tracker/artifacts.py`.

## visualization

Render explicit system-owned plot data; visualization never invents a trajectory.

### `visualization.PlotData`

Named layers and metadata for one explicit visualization product.

```python
PlotData(kind: 'str', layers: 'Sequence[PlotLayer]', annotations: 'Mapping[str, Any]' = <factory>, metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/visualization/models.py`.

### `visualization.plot_bloch`

Render validated Bloch layers without calling the system.

```python
plot_bloch(value: 'PlotData | dict[str, Any]', *, projections: 'bool' = True, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, figsize: 'tuple[float, float] | None' = None, colorbar: 'bool' = True, axes: 'bool' = False) -> 'tuple[Any, ...]'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.draw_timeseries`

Draw named per-component channels vs a scalar parameter onto an existing 2D axis; shared by plot_bloch's projections subplot and the standalone panel functions.

```python
draw_timeseries(ax: 'Any', data: 'PlotData', *, palette: 'Sequence[str]' = ('#0072B2', '#E69F00', '#009E73'), component_labels: 'Sequence[str]' = ('x', 'y', 'z'), ylabel: 'str' = 'Bloch components', xlim: 'tuple[float, float] | None' = None, ylim: 'tuple[float, float] | None' = (-1.14, 1.14), prefix_labels: 'bool' = True, legend_ncols: 'int' = 2) -> 'None'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.draw_sphere`

Draw a light unit-sphere scaffold on an existing 3D axis.

```python
draw_sphere(ax: 'Any', *, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, nu: 'int' = 25, nv: 'int' = 13, axes: 'bool' = False, axes_color: 'str' = '#555555') -> 'Any'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.style_context`

Apply the compact serif plotting style without mutating global rcParams.

```python
style_context() -> 'Iterator[None]'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.validate_bloch_data`

Validate generic plot data for rendering inside the Bloch ball.

```python
validate_bloch_data(value: 'PlotData | dict[str, Any]', *, tolerance: 'float' = 1e-08) -> 'PlotData'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.plot_bloch_panel`

Render one standalone Bloch-sphere panel (single figure, no projections subplot).

```python
plot_bloch_panel(value: 'PlotData | Mapping[str, Any]', *, equator: 'bool' = True, elevation: 'float' = 14.0, azimuth: 'float' = -54.0, figsize: 'tuple[float, float] | None' = None, colorbar: 'bool' = True, axes: 'bool' = False) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.plot_projection_panel`

Render one standalone Bloch-component-vs-parameter panel from validated Bloch trajectory data.

```python
plot_projection_panel(value: 'PlotData | Mapping[str, Any]', *, component_labels: 'Sequence[str]' = ('$r_x$', '$r_y$', '$r_z$'), xlim: 'tuple[float, float] | None' = None, figsize: 'tuple[float, float]' = (3.4, 3.0)) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.plot_controls_panel`

Render one standalone control-amplitude-vs-parameter panel; uses CONTROL_COLORS and skips the Bloch-ball check.

```python
plot_controls_panel(value: 'PlotData | Mapping[str, Any]', *, component_labels: 'Sequence[str]' = ('$u_x$', '$u_y$', '$u_z$'), xlim: 'tuple[float, float] | None' = None, figsize: 'tuple[float, float]' = (4.5, 2.75)) -> 'tuple[Any, Any]'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.validate_controls_data`

Validate generic plot data for a control-amplitude-vs-parameter panel (no Bloch-ball check).

```python
validate_controls_data(value: 'PlotData | Mapping[str, Any]') -> 'PlotData'
```

Source: `optimizer/visualization/panels.py`.

### `visualization.save_figure`

Save a renderer-produced figure to an explicit path.

```python
save_figure(figure: 'Any', path: 'Any', *, dpi: 'int' = 400, close: 'bool' = False) -> 'None'
```

Source: `optimizer/visualization/bloch.py`.

### `visualization.save_panel`

Save one panel figure as .pdf and .png from a shared path stem at one dpi.

```python
save_panel(figure: 'Any', out_stem: 'str | Path', *, formats: 'tuple[str, ...]' = ('pdf', 'png'), dpi: 'int' = 400, close: 'bool' = True) -> 'tuple[Path, ...]'
```

Source: `optimizer/visualization/panels.py`.

## system olgs

Mandatory static system semantics and capability-gated numerical hooks owned by each problem.

### `system_olgs.get_system_spec`

Return and validate the required static system semantic declaration.

```python
get_system_spec(system: 'Any') -> 'dict[str, Any]'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.require_system`

Validate the required OLGS contract before an optimizer starts.

```python
require_system(system: 'Any') -> 'OLGSystem'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.with_secondary`

Request a system-owned parameter-updated copy for explicit continuation.

```python
with_secondary(system: 'Any', **updates: 'Any') -> 'Any'
```

Source: `optimizer/system_olgs/contract.py`.

### `system_olgs.optional_control_bounds`

Read optional system-declared elementwise bounds.

```python
optional_control_bounds(system: 'Any') -> 'ControlBounds'
```

Source: `optimizer/system_olgs/validation.py`.

## system olss

Construction types, validation, and discovery for project-owned OLSS contracts.

### `system_olss.OLSS`

Base class for a self-contained Open Loop Shooting System.

```python
OLSS()
```

Source: `optimizer/system_olss/olss.py`.

### `system_olss.ShootingBlock`

One named, scaled block in a shooting vector.

```python
ShootingBlock(name: 'str', shape: 'tuple[int, ...]' = (), meaning: 'str' = '', unit: 'str | None' = None, scale: 'float' = 1.0, role: 'str' = 'variable', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.ShootingSpec`

Ordered layout of shooting-variable blocks.

```python
ShootingSpec(blocks: 'tuple[ShootingBlock, ...]', dtype: 'str' = 'float64', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.ShootingVariables`

Finite values attached to a ShootingSpec.

```python
ShootingVariables(spec: 'ShootingSpec', values: 'np.ndarray', name: 'str' = 'shooting', metadata: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/variables.py`.

### `system_olss.VerificationCheck`

Named independent acceptance check included in OLSS qualification.

```python
VerificationCheck(name: 'str', passed: 'bool', value: 'Any' = None, tolerance: 'Any' = None, message: 'str' = '') -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.basic_shooting_system_spec`

Build and validate the static OLSS declaration.

```python
basic_shooting_system_spec(system_id: 'str', *, shooting_spec: 'ShootingSpec', residual_size: 'int', name: 'str | None' = None, system_type: 'str' = 'shooting', description: 'str' = '', residual_groups: 'Iterable[Mapping[str, Any]] | None' = None, metrics: 'Mapping[str, Any] | Iterable[str] | None' = None, parameters: 'Mapping[str, Any] | None' = None, jacobian_source: 'str' = 'analytical', capabilities: 'Mapping[str, Any] | None' = None, selection_metric: 'str | None' = None, selection_direction: 'str' = 'minimize', reconstruction: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None, verification_tolerance: 'float | None' = None, allow_additional_metrics: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/specs.py`.

### `system_olss.shooting_metric_spec`

Build one OLSS metric declaration for system_spec().

```python
shooting_metric_spec(label: 'str', *, description: 'str | None' = None, role: 'str' = 'diagnostic', dtype: 'str' = 'float', shape: 'Iterable[int]' = (), unit: 'str | None' = None, required: 'bool' = False, allow_nonfinite: 'bool' = False) -> 'dict[str, Any]'
```

Source: `optimizer/system_olss/specs.py`.

### `system_olss.ShootingEvaluation`

Candidate, residual vector, declared metrics, and working options of one evaluation.

```python
ShootingEvaluation(variables: 'ShootingVariables', residuals: 'np.ndarray', metrics: 'Mapping[str, Any]', options: 'Mapping[str, Any]' = <factory>) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingVerification`

Independent verification residuals, metrics, options, independence source, and named checks.

```python
ShootingVerification(residuals: 'np.ndarray', metrics: 'Mapping[str, Any]', options: 'Mapping[str, Any]' = <factory>, numerical_success: 'bool' = True, message: 'str' = 'verified', independent: 'bool' = False, independence_source: 'str' = 'unspecified', checks: 'tuple[VerificationCheck, ...]' = ()) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingResult`

One root solve: candidate, residuals, metrics, convergence, verification, qualification, stop reason, error, tolerance source, counters, and history.

```python
ShootingResult(variables: 'ShootingVariables', residuals: 'np.ndarray', metrics: 'dict[str, Any]', solver_converged: 'bool', verified: 'bool', qualified: 'bool', iterations: 'int', method: 'str', stop_reason: 'str', jacobian_source: 'str', verification: 'ShootingVerification | None' = None, counters: 'dict[str, int]' = <factory>, history: 'list[dict[str, Any]]' = <factory>, error: 'str | None' = None, verification_tolerance: 'float | None' = None, verification_tolerance_source: 'str | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.MultiStartResult`

All multistart results, qualified and unique roots, and the ranking with its selection source.

```python
MultiStartResult(results: 'list[ShootingResult]', qualified: 'list[ShootingResult]', unique_qualified: 'list[ShootingResult]', ranked: 'list[ShootingResult]', selection_metric: 'str | None', selection_direction: 'str | None', selection_source: 'str | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ContinuationPoint`

One verified pseudo-arclength branch point with its tangent.

```python
ContinuationPoint(variables: 'ShootingVariables', residual_norm: 'float', tangent: 'np.ndarray', corrector_iterations: 'int', verified: 'bool', verification_independent: 'bool' = False) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ShootingContinuationResult`

Pseudo-arclength branch points, step history, stop reason, and verification tolerance.

```python
ShootingContinuationResult(points: 'list[ContinuationPoint]', completed_steps: 'int', requested_steps: 'int', stop_reason: 'str', step_size: 'float', step_history: 'list[dict[str, Any]]' = <factory>, verification_tolerance: 'float | None' = None, verification_tolerance_source: 'str | None' = None) -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ParameterContinuationPoint`

One natural-continuation stage: parameter value, attempted step, and its shooting result.

```python
ParameterContinuationPoint(parameter: 'float', result: 'ShootingResult', attempted_step: 'float') -> None
```

Source: `optimizer/system_olss/results.py`.

### `system_olss.ParameterContinuationResult`

Natural-continuation stages, per-attempt step history, stop reason, and target status.

```python
ParameterContinuationResult(parameter: 'str', target: 'float', points: 'list[ParameterContinuationPoint]', stop_reason: 'str', reached_target: 'bool', step_history: 'list[dict[str, Any]]' = <factory>) -> None
```

Source: `optimizer/system_olss/results.py`.

## shooting

Root solving, verification, multistart, continuation, and direct shooting for OLSS systems.

### `shooting.solve_shooting`

Solve one OLSS residual equation and independently verify the candidate.

```python
solve_shooting(system: 'Any', initial: 'ShootingVariables', *, method: 'str' = 'lm', tolerance: 'float' = 1e-10, verification_tolerance: 'float | None' = None, maxiter: 'int' = 40, damping: 'float' = 1e-10, max_step_norm: 'float | None' = None, line_search: 'bool' = True, shrink: 'float' = 0.5, max_backtracks: 'int' = 12, options: 'Mapping[str, Any] | None' = None, verification_options: 'Mapping[str, Any] | None' = None, evaluator: 'ShootingEvaluator | None' = None, previous_result: 'ShootingResult | None' = None, on_iteration: 'Callable[[Mapping[str, Any]], None] | None' = None, on_event: 'Callable[[Mapping[str, Any]], None] | None' = None) -> 'ShootingResult'
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

```python
continue_parameter(system: 'Any', initial: 'ShootingVariables', *, parameter: 'str', start: 'float', target: 'float', step_size: 'float', min_step: 'float | None' = None, max_step: 'float | None' = None, shrink: 'float' = 0.5, grow: 'float' = 1.5, max_steps: 'int' = 100, solver_options: 'Mapping[str, Any] | None' = None) -> 'ParameterContinuationResult'
```

Source: `optimizer/shooting/continuation.py`.

### `shooting.random_shooting_seeds`

Generate scale-aware random shooting seeds.

```python
random_shooting_seeds(spec: 'ShootingSpec', count: 'int', *, seed: 'int | None' = None, center: 'ShootingVariables | None' = None, radius: 'float' = 1.0, bounds: 'tuple[np.ndarray, np.ndarray] | None' = None) -> 'list[ShootingVariables]'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.perturb_shooting_seeds`

Generate a scale-aware perturbation cloud around an anchor.

```python
perturb_shooting_seeds(anchor: 'ShootingVariables', count: 'int', *, relative_radius: 'float' = 0.1, seed: 'int | None' = None, bounds: 'tuple[np.ndarray, np.ndarray] | None' = None) -> 'list[ShootingVariables]'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.refine_shooting_seeds`

Solve, select elites, perturb, and refine a shooting seed campaign.

```python
refine_shooting_seeds(system: 'Any', starts: 'Iterable[ShootingVariables]', *, elite_count: 'int' = 3, perturbations_per_elite: 'int' = 4, relative_radius: 'float' = 0.05, seed: 'int | None' = None, solver_options: 'Mapping[str, Any] | None' = None, parallel: 'ParallelConfig | None' = None, system_factory: 'Callable[[], Any] | None' = None) -> 'MultiStartResult'
```

Source: `optimizer/shooting/campaigns.py`.

### `shooting.compare_shooting_replay`

Replay OLSS-recovered controls in OLGS and compare declared metrics.

```python
compare_shooting_replay(shooting_system: 'Any', variables: 'ShootingVariables', gradient_system: 'Any', *, times: 'np.ndarray', metric_map: 'Mapping[str, str] | None' = None, shooting_options: 'Mapping[str, Any] | None' = None, interpolation: 'str' = 'system_owned', units: 'Mapping[str, str] | None' = None) -> 'dict[str, Any]'
```

Source: `optimizer/shooting/replay.py`.

### `shooting.finite_difference_shooting_jacobian`

Central-difference reference Jacobian of the OLSS residual vector; does not replace the declared Jacobian.

```python
finite_difference_shooting_jacobian(system: 'Any', variables: 'ShootingVariables', *, eps: 'float' = 1e-06, options: 'Mapping[str, Any] | None' = None) -> 'np.ndarray'
```

Source: `optimizer/shooting/derivatives.py`.

### `shooting.ShootingEvaluator`

Validated, cached, and counted OLSS evaluation shared by the shooting tools.

```python
ShootingEvaluator(system: 'Any', *, use_cache: 'bool' = True, cache_size: 'int | None' = 32) -> 'None'
```

Source: `optimizer/shooting/evaluator.py`.

## optimizer log

Standalone, read-only tracker reader package; it is not yet a portal UI or command-line application.

### `optimizer_log.read_timeline`

Read validated tracker events without importing optimizer numerics.

```python
read_timeline(run_dir: 'str | Path') -> 'TimelineReadResult'
```

Source: `optimizer_log/reader.py`.

### `optimizer_log.follow_timeline`

Incrementally follow appended complete JSONL events.

```python
follow_timeline(run_dir: 'str | Path', *, offset: 'int' = 0, previous_sequence: 'int | None' = None) -> 'TimelineReadResult'
```

Source: `optimizer_log/reader.py`.

### `optimizer_log.inspect_writer`

Inspect writer ownership/liveness without taking ownership.

```python
inspect_writer(run_dir: 'str | Path') -> 'dict[str, Any]'
```

Source: `optimizer_log/reader.py`.

---

[⌂ Home](README.md)
