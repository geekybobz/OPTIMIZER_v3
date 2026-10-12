---
role: topic
parent: SHOOTING_HUB.md
summary: "Selection guide and semantic parameter reference for OLSS numerical tools."
read_when: "Read when choosing or configuring a shooting solver, continuation, seed campaign, or replay."
status: active
reviewed: 2026-10-12
---
# Shooting Methods

Exact signatures and defaults are generated in the [API Reference](../../API_REFERENCE.md).
This page owns method meaning, selection, and limitations.

## Root solving

### `solve_shooting`

Solves `residuals(z) = 0` with rectangular least-squares support.

| Parameter group | Main inputs |
|---|---|
| Method | `method`: `newton`, `lm`, or `damped`; `damping` |
| Convergence | `tolerance`, `verification_tolerance`, `maxiter` |
| Globalization | `line_search`, `shrink`, `max_backtracks`, `max_step_norm` |
| Accuracy | `options`, `verification_options`, optional shared `evaluator` |
| Continuation | `previous_result` |
| Observation | `on_iteration`, `on_event` |

`lm` and `damped` currently use fixed damped least squares, not an adaptive
Levenberg-Marquardt trust policy. Use this method for single or multiple
shooting root equations; inspect `qualified`, not only `solver_converged`.

Failures follow the OLGS engine rule. A start that cannot be evaluated raises
`ValueError`. After the start, a trial whose residuals fail (for example an
integration blow-up) is rejected and the line search shrinks. Failures end the
run with `nonfinite_trial`, `jacobian_failed`, `evaluation_failed`, or
`verification_failed` and the message in `ShootingResult.error`; coordinators
record a seed or stage whose start failed as `initial_evaluation_failed`. The
generated [stop-reason table](../../API_REFERENCE.md#stop-reasons) gives the
meaning, outcome, and continuability of every reason, including the successful
`root_tolerance` and `least_squares_stationary`.

The verification tolerance is resolved as described in
[OLSS Verification](../system_olss/VERIFICATION.md#verification-tolerance).

### `solve_direct_shooting`

Starts from residual feasibility and reduces a declared selection metric using
`cost_gradient`. It is appropriate for small constrained-cost refinements.
The current method is projected first-order descent rather than SQP and may
stop before a known stationary constrained optimum.

## Derivative and geometry inspection

- `finite_difference_shooting_jacobian` constructs a central-difference
  reference without changing the system's declared derivative.
- `verify_shooting_jacobian` compares the declared Jacobian with a numerical
  reference and reports strategy and error evidence; a numerical-declared
  Jacobian is compared at two separated steps.
- `shooting_geometry_probe` reports singular values, rank, condition, weak
  variable blocks, residual-group norms, and evaluation counters.

Use these when implementing equations, diagnosing singular correctors, or
identifying weakly determined variable blocks.

## Continuation

### `continue_shooting`

Pseudo-arclength continuation predicts along a Jacobian null direction and
corrects the augmented system. It accepts redundant residual rows when the
numerical rank is correct, adapts the step after success or failure, records
step history, and can land exactly on a target variable block. A corrector
point that fails to evaluate counts as a failed step: the step shrinks and the
error is kept in `step_history`.

### `continue_parameter`

Natural continuation changes one secondary parameter through
`with_secondary`, solves each stage, shrinks failed steps, grows successful
steps, and lands on the requested target. `step_history` holds one row per
attempt, including failed ones, with the stage's stop reason, residual and
verification residual, verification tolerance, and error, so a stop at
`minimum_step_reached` shows why the last attempts failed.

Use natural continuation when the project parameter remains outside the
shooting vector. Use pseudo-arclength near folds or when the continued
parameter is itself a shooting block.

## Seed campaigns

| Tool | Use |
|---|---|
| `random_shooting_seeds` | Reproducible scale-aware cloud around zero or a center |
| `perturb_shooting_seeds` | Local scale-aware cloud around one candidate |
| `solve_shooting_multistart` | Solve seeds, retain qualified roots, deduplicate, and rank on verified metrics; a seed that cannot be evaluated is recorded as `initial_evaluation_failed` instead of stopping the campaign |
| `refine_shooting_seeds` | Solve, select elites, perturb, and solve the expanded cloud |

Parallel multistart requires a caller-supplied `system_factory` so every worker
owns a system instance. Process execution is the useful default for Python-level
project code when the factory is pickle-compatible.

## Replay

`compare_shooting_replay` asks the OLSS system to recover controls on an
explicit grid, evaluates those controls through an OLGS system, and reports
grid, interpolation, units, metrics, and mapped differences. It is an explicit
cross-formulation check, not an automatic claim that both discretizations are
identical.

## Result choice

| Task | Read |
|---|---|
| One solve | `ShootingResult` |
| Seed cloud | `MultiStartResult.best`, `unique_qualified`, and all results |
| Pseudo-arclength | `ShootingContinuationResult.points` and `step_history` |
| Natural continuation | `ParameterContinuationResult.points` and `reached_target` |
| Short-run interpretation | `diagnose_run(result)` |

## Connections

- Related: [Shooting Hub](SHOOTING_HUB.md)
- Prerequisite: [OLSS Contract](../system_olss/CONTRACT.md)
- Related: [OLSS Verification](../system_olss/VERIFICATION.md)
- Related: [Generated API Reference](../../API_REFERENCE.md)
- Related: [OLSS scenario evidence](../../test_contained/olss_scenarios/REPORT.md)

---

[⌂ Home](SHOOTING_HUB.md)
