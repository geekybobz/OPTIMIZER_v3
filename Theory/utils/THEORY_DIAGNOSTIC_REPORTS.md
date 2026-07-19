---
title: THEORY_DIAGNOSTIC_REPORTS
type: theory_reference
module: Theory/utils
related_method: optimizer/utils/UTIL_DIAGNOSTICS.md
tags:
  - optimizer
  - utils
  - theory
  - diagnostics
  - metrics
---

# Diagnostic Report Theory

This note gives the mathematical context for:

- [Diagnostics API](../../optimizer/utils/UTIL_DIAGNOSTICS.md)

## What a Diagnostic Measures

At a control vector `u`, a system exposes metrics:

```text
M(u) = {J(u), fidelity(u), energy(u), ...}
```

Diagnostics summarize these values and the local numerical health of `u`.

## Control Norms

Controls are stored as a matrix by channel and time:

```text
U =
[ u_1(t_1) ... u_1(t_N)
  ...
  u_m(t_1) ... u_m(t_N) ]
```

The global norm is:

```text
||U||_2 = sqrt(sum_i sum_j U_ij^2)
```

Channel norms are:

```text
||u_i||_2 = sqrt(sum_j U_ij^2)
```

## Gradient Scale

If the system provides an analytical gradient, diagnostics report:

```text
||grad J(u)||_2
max |grad J(u)|
channel-wise gradient norms
```

Large gradient imbalance can indicate:

```text
different physical channel scales
wrong weights
missing normalization
wrong channel ordering
```

## Residual Norms

For hard residuals:

```text
r(u) in R^p
```

diagnostics report:

```text
||r(u)||_2
max_i |r_i(u)|
```

These are feasibility measures, not objective values unless the system explicitly
uses them inside `J`.

## Illustration

```text
controls u
  -> evaluate(u)  -> metrics
  -> gradient(u)  -> gradient norms
  -> residuals(u) -> residual norm and max violation
```

The diagnostic report is a snapshot of this state. It does not infer a correction
unless paired with projection or repair.

## API Reference

- [Diagnostics](../../optimizer/utils/UTIL_DIAGNOSTICS.md)
- [Util Lifecycle](../../optimizer/utils/UTIL_LIFECYCLE.md)
