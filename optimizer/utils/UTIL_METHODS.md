---
role: topic
parent: UTILS_HUB.md
summary: "Index of the public utilities by category."
read_when: "Read to find a utility quickly."
status: active
reviewed: 2026-10-12
---
# Util Methods

This note is the compact index of public utilities.

## Diagnostics

| Method | Purpose | Reference |
|---|---|---|
| `metric_report` | Evaluate and summarize current metrics. | [Diagnostics](UTIL_DIAGNOSTICS.md) |
| `diagnostic_report` | Summarize system hooks, controls, metrics, gradient, and residuals. | [Diagnostics](UTIL_DIAGNOSTICS.md) |
| `diagnose_run` | Classify a short OLGS or OLSS history without rerunning the system. | [Diagnostics](UTIL_DIAGNOSTICS.md) |
| `curvature_probe` | Estimate local curvature, negative-curvature evidence, and a suggested gradient step. | [Diagnostics](UTIL_DIAGNOSTICS.md) |
| `compare_methods` | Run selected optimizers briefly from identical controls and rank their declared objective. | [Diagnostics](UTIL_DIAGNOSTICS.md) |
| `shooting_geometry_probe` | Report OLSS Jacobian rank, conditioning, weak blocks, and residual groups. | [Diagnostics](UTIL_DIAGNOSTICS.md) |

## Derivative Checks

| Method | Purpose | Reference |
|---|---|---|
| `finite_difference_gradient` | Full coordinate finite-difference gradient for a scalar metric. | [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md) |
| `verify_gradient` | Directional check of analytical gradient against finite differences. | [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md) |
| `finite_difference_jacobian` | Full finite-difference residual Jacobian. | [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md) |
| `verify_jacobian` | Directional check of analytical residual Jacobian. | [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md) |

## Residual Geometry

| Method | Purpose | Reference |
|---|---|---|
| `geometry_probe` | Residual norm, Jacobian rank, singular values, and conditioning. | [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md) |
| `nullspace_basis` | Basis columns for first-order residual-preserving directions. | [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md) |
| `project_gradient` | Remove residual-changing row-space component from a gradient. | [Residual Geometry](UTIL_RESIDUAL_GEOMETRY.md) |

## Repair

| Method | Purpose | Reference |
|---|---|---|
| `repair_newton` | Newton/LM-style residual repair. | [Repair](UTIL_REPAIR.md) |
| `RepairResult` | Rich result object returned by repair. | [Repair](UTIL_REPAIR.md) |

## Spectrum and Smoothness

| Method | Purpose | Reference |
|---|---|---|
| `control_spectrum` | FFT amplitude, power, dominant frequency, and high-frequency fraction. | [Spectrum and Smoothness](UTIL_SPECTRUM_SMOOTHNESS.md) |
| `smoothness_report` | First/second difference, total variation, and jump summaries. | [Spectrum and Smoothness](UTIL_SPECTRUM_SMOOTHNESS.md) |

## Control Transformations

Grid declarations, resampling, smoothing/filtering, affine scaling, normalization,
mixing, channel assembly, temporal joins, bounds and shooting recovery return owned
candidates. See [Control transformations](UTIL_CONTROL_TRANSFORMS.md) for signatures,
validation and terminal examples. Activation and feasibility checks are explicit.

## Catalog Note

`metric_guard` is not a utils method.  It is implemented in `optimizer/core/guards.py`,
catalogued in the `core` group, and called as `opt.core.metric_guard(...)`.

- [Core Hub](../core/CORE_HUB.md)

## Related Notes

- [Utils Hub](UTILS_HUB.md)
- [Util Common API](UTIL_COMMON_API.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Util Boundary](UTIL_BOUNDARY.md)

---

[⌂ Home](UTILS_HUB.md)
