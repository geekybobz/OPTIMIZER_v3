# OPTIMIZER v3 Auxiliary Tools Catalog

Status: current review draft.
Last updated: 2026-07-15.

This file lists library components that are not ordinary optimizers. Optimizers are
cataloged in [`OPTIMIZERS.md`](OPTIMIZERS.md). Architecture is summarized in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Categories

```text
guesses       initial controls
schedules     step-size, line-search, trust, and damping policies
constraints   generic control-space restrictions
diagnostics   measurements that do not move controls
repairs       tools that move controls to restore feasibility
logs          trace rows, checkpoints, reports, rollback
modes         later orchestration layer
```

User-facing calls should still be direct where possible:

```python
opt.fourier_guess(...)
opt.geometry_probe(...)
opt.verify_derivatives(...)
opt.repair_newton(...)
```

Internal modules may remain grouped by responsibility.

## Guesses

Guesses generate `Controls` from `system.control_spec()`. They should work for one,
two, three, six, or more control channels without special-case code.

| public call | idea | first wave |
|---|---|---|
| `opt.zero_guess(system)` | all channels zero | yes |
| `opt.constant_guess(system, amplitude=...)` | constant amplitude per channel | yes |
| `opt.sine_guess(system, amplitude=..., frequency=...)` | smooth sinusoidal start | yes |
| `opt.sinc_guess(system, amplitude=...)` | sinc-shaped start | yes |
| `opt.fourier_guess(system, n_terms=..., amplitude=...)` | low-dimensional Fourier random start | yes |
| `opt.random_guess(system, amplitude=...)` | random direct control start | yes |
| `opt.mix_controls(system, a, b, ratio=...)` | interpolate or perturb known controls | yes |

All guess functions should support amplitude control. Fourier/random guesses should
support seeds for reproducibility.

## Schedules

Schedules control update size and trust behavior. They should be reusable instead of
hard-coded inside each optimizer.

| module idea | purpose |
|---|---|
| constant step | fixed learning rate |
| adaptive shrink/grow | old line-search style step adaptation |
| backtracking line search | trial-and-accept step policy |
| strong Wolfe policy | infrastructure for L-BFGS and CG |
| trust-region radius | safe second-order step size |
| LM damping | Levenberg-Marquardt damping update |

Some schedules are public configuration options. Others are internal helpers.

## Constraints

Constraints are generic control-space restrictions, not physical system residuals.

Examples:

```text
box bounds
max norm
fixed norm
simple projections
smoothness or bandwidth caps
```

Physical equality conditions such as terminal leakage or robustness equations belong
in `system.residuals()`.

First constraints to implement:

| public/helper | idea |
|---|---|
| `opt.Bounds(low, high)` | box amplitude bounds |
| `opt.MaxNorm(value)` | cap total or channel norm |
| `opt.FixedNorm(value)` | project onto fixed norm |
| `opt.project_bounds(...)` | direct projection helper |

## Diagnostics

Diagnostics measure. They should not move controls.

| public call | idea | priority | needs |
|---|---|---|---|
| `opt.verify_derivatives` | compare analytical gradient/Jacobian to finite differences | core | `gradient`, optional `jacobian` |
| `opt.geometry_probe` | tangent ratio, rank, singular values, stiffness/stuck diagnosis | core | `gradient`, `jacobian` |
| `opt.stationarity_check` | projected or ordinary gradient stationarity | add | `gradient` |
| `opt.conditioning_monitor` | track Jacobian/Gram conditioning | add | `jacobian` |
| `opt.stall_detector` | objective and metric slope over a window | add | logs |
| `opt.step_acceptance_stats` | accepted/rejected step statistics | add | trace |
| `opt.control_spectrum` | frequency content and high-frequency warnings | add | controls |
| `opt.robustness_probe` | perturb controls/params and measure metric degradation | explore | `simulate` |
| `opt.curvature_spectrum` | Hessian or Gram spectral estimates | explore | `hvp` or FD |

Derivative checks are especially important because the normal path assumes analytical
gradients.

## Repairs

Repairs move controls to restore feasibility. They are not ordinary descent
optimizers.

| public call | idea | priority | needs |
|---|---|---|---|
| `opt.repair_newton` | min-norm Newton snap-back onto hard residuals | core | `residuals`, `jacobian` |
| `opt.repair_damped_newton` | damped Newton/LM repair for ill-conditioned residuals | add | `residuals`, `jacobian` |
| `opt.project_gradient` | project a gradient into a residual tangent space | core helper | `jacobian` |
| `opt.manifold_retract` | reusable snap-back after a raw step | add | `residuals`, `jacobian` |

These tools support projected descent, nullspace descent, and late-stage polishing.

## Logs And Checkpoints

Trace and checkpoint support are core, not optional.

### Trace Rows

Trace rows should capture:

```text
run_id
chunk
iter
global_iter
optimizer
stage
system_params
metrics
grad_norm
step_norm
control_norm
max_abs_control
accepted
reason
```

The metrics are whatever the system returns, for example:

```text
J
fidelity
infidelity
F_norm2
C_sym_norm2
energy
```

### Checkpoints

Checkpoints should save restorable state:

```text
controls
optimizer_state
system_params
metrics
iteration
stage label
random state
previous checkpoint id
```

Important labels:

```text
latest
stage_start
accepted
best_J
best_safe
```

`stage_start` is the key rollback point for curriculum.

## Curriculum And Stages

Curriculum is not an external objective wrapper. It is structured system-param tuning:

```text
set cost prefactors
run a short optimizer chunk
inspect logs
accept or rollback
change prefactors
continue
```

This can be represented by `Stage` metadata:

```text
name
system param updates
guards
target metric
accept/reject rule
checkpoint labels
```

The system still computes `J` and the analytical gradient.

## Modes

`modes/` remains a placeholder until the lower layers are stable.

Possible later modes:

```text
manual
benchmark
curriculum
auto
pipeline
```

Do not build these before controls, system contract, result/state, logs/checkpoints,
engine, basic optimizers, guesses, diagnostics, and repairs.

## Deferred Enhancer Ideas

These are useful ideas, but they should not shape the first skeleton:

| idea | where it belongs later |
|---|---|
| continuation/homotopy | curriculum or pipeline mode |
| adaptive regularization | stage/curriculum system-param update |
| preconditioning | schedule or optimizer helper |
| restart policies | optimizer helper or mode |
| perturbation kicks | later stall-recovery utility |
| iterate averaging | later engine option |
| negative-curvature escape | later curvature-based diagnostic/step tool |

