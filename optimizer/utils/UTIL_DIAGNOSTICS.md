---
title: UTIL_DIAGNOSTICS
type: category_reference
module: optimizer/utils
source: optimizer/utils/diagnostics.py
methods:
  - metric_report
  - diagnostic_report
tags:
  - optimizer
  - utils
  - diagnostics
  - metrics
---

# Diagnostics

Diagnostics answer what is currently true about a system-control pair.

They do not change controls.

## Public Calls

```python
metrics = opt.utils.metric_report(system, controls)
report = opt.utils.diagnostic_report(system, controls)
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

## Best For

```text
checking objective names
checking control norms and finiteness
seeing gradient scale by channel
seeing whether residual hooks are available
building compact notebook logs
```

## Watch Out

```text
diagnostic_report can call system.gradient(...)
missing residual hooks are reported as unavailable
diagnostics reveal problems but do not repair them
```

## Related Theory

- [Diagnostic Report Theory](../../Theory/utils/THEORY_DIAGNOSTIC_REPORTS.md)

## Related Notes

- [Util Methods](UTIL_METHODS.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Source](./diagnostics.py)
