---
role: topic
parent: UTILS_HUB.md
summary: "Metric and diagnostic reports, and short-run diagnosis with its labels and thresholds."
read_when: "Read when deciding what to do after a short run."
status: active
reviewed: 2026-10-12
---
# Diagnostics

Diagnostics answer what is currently true about a system-control pair.

They do not change controls.

## Public Calls

```python
metrics = opt.utils.metric_report(system, controls)
report = opt.utils.diagnostic_report(system, controls)
diagnosis = opt.diagnose_run(result)
curvature = opt.curvature_probe(system, controls)
comparison = opt.compare_methods(system, controls, ("adam", "line_search"), iterations=5)
geometry = opt.shooting_geometry_probe(shooting_system, variables)
```

## `metric_report`

Use this for a lightweight metric snapshot.

Returns:

```text
kind
metrics
raw_metric_keys
```

Vector or complex metrics are summarized in JSON-friendly form instead of discarded.

## `diagnostic_report`

Use this when an optimizer run gets stuck or a control state needs inspection.

Returns:

```text
system_hooks
control_spec
controls
metrics
gradient
residuals
```

`include_gradient=False` can skip gradient evaluation when it is expensive.

`residuals=None` can skip residual evaluation.

## `diagnose_run`

Use this after a short run to decide whether to continue, change the step, or
switch methods. It reads the result's own samples and makes no system call.

Accepts an `OptimizerResult` or a `ShootingResult`; any other object raises
`TypeError`. `mode="max"` treats a larger objective as better.

```text
converging             improved by at least 1 % of the starting value
slow                   improved, but by less than 1 %
stalled                changed by at most 1e-6 relative, or worsened by at most 0.1 %
oscillating            the direction reversed in at least a third of the steps
diverging              worse than the start by more than 0.1 %
repeating_rejections   at least 80 % of at least three attempts were rejected
no_trace               the OLGS run kept no samples (create_trace=False)
```

Every threshold is relative to the run's starting value, so a run gets the same
label whether its objective is an infidelity near `1e-6` or an energy near
`1e3`. The OLGS trend starts at the objective before the first step, taken from
the run's chunk record. Oscillation is tested before divergence, so a run that
swings while improving is labelled `oscillating`. An OLSS run whose solver
converged is `converging`.

Returns the label, `objective_trend` or `residual_trend`, `relative_change`,
`direction_reversals`, gradient and step trends (OLGS), the rejection ratio,
the stop reason, and the `thresholds` that were applied.

## Best For

```text
checking objective names
checking control norms and finiteness
seeing gradient scale by channel
seeing whether residual hooks are available
building compact notebook logs
classifying a 5--10 iteration run before continuing
comparing methods from identical controls and budgets
checking OLSS Jacobian rank and weak variable blocks
```

## Watch Out

```text
diagnostic_report can call system.gradient(...)
missing residual hooks are reported as unavailable
diagnostics reveal problems but do not repair them
short-run classifications are evidence-based heuristics, not convergence proofs
compare_methods runs each selected method and may therefore be expensive
```

## Related Theory

- [Diagnostic Report Theory](../../Theory/utils/THEORY_DIAGNOSTIC_REPORTS.md)

## Related Notes

- [Util Methods](UTIL_METHODS.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Source](./diagnostics.py)

---

[⌂ Home](UTILS_HUB.md)
