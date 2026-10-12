---
role: topic
parent: UTILS_HUB.md
summary: "Geometry probe, nullspace basis, and gradient projection for residual constraints."
read_when: "Read when examining or projecting onto a constraint surface."
status: active
reviewed: 2026-10-12
---
# Residual Geometry

Residual geometry tools study the local linear constraint surface defined by
system residuals.

## Public Calls

```python
geometry = opt.utils.geometry_probe(system, controls, residuals="hard")
basis = opt.utils.nullspace_basis(system, controls, residuals="hard")
projected = opt.utils.project_gradient(system, controls, gradient)
```

For richer projection details:

```python
info = opt.utils.project_gradient(system, controls, gradient, return_info=True)
```

## `geometry_probe`

Reports:

```text
residual_norm
residual_max_abs
jacobian_source
jacobian_shape
jacobian_norm
rank
nullspace_dimension
singular_values
condition_number
repair_locally_possible
```

## `nullspace_basis`

Returns basis columns in flattened control space:

```text
shape = (controls.spec.size, nullspace_dimension)
```

These columns represent first-order directions that do not change the named
residuals.

## `project_gradient`

Removes the local residual row-space component from a gradient-like `Controls`
object.

Stepping along `-projected_gradient` should reduce the first-order residual change
relative to the raw gradient.

## Internal Helpers

`jacobian_geometry` summarizes a Jacobian matrix but is not exported from
`optimizer.utils.__all__`.

`get_jacobian` chooses analytical Jacobians or finite-difference fallback and is
shared by diagnostics, projection, and repair.

## Related Theory

- [Residual Jacobian Geometry Theory](../../Theory/utils/THEORY_RESIDUAL_JACOBIAN_GEOMETRY.md)
- [Nullspace Projection Theory](../../Theory/utils/THEORY_NULLSPACE_PROJECTION.md)

## Related Notes

- [Derivative Checks](UTIL_DERIVATIVE_CHECKS.md)
- [Repair](UTIL_REPAIR.md)
- [Util Lifecycle](UTIL_LIFECYCLE.md)
- [Source](./geometry.py)

---

[⌂ Home](UTILS_HUB.md)
