"""Residual geometry and gradient projection helpers.

Why this file exists
--------------------
Hard analytical residuals define a local constraint surface.  To understand or move
near that surface, the library needs Jacobian geometry: rank, singular values,
nullspace dimension, and row-space projection.  These tools are the bridge between
diagnostics and later projected optimizers.

How it fits the architecture
----------------------------
- systems may expose ``residuals`` and optional ``jacobian``.
- ``derivatives.get_jacobian`` supplies either the analytical hook or an explicit
  finite-difference fallback.
- repair tools use the same Jacobian shape.
- the projected optimizer uses the same nullspace calculation while this helper
  remains available for explicit geometry inspection.

What this file deliberately does not do
---------------------------------------
It does not run a multi-iteration optimizer.  ``project_gradient`` returns a projected
gradient-like control object; it does not choose step sizes or accept/reject trials.

Reviewer invariants
-------------------
- Jacobian rows are residual dimensions and columns are flattened control values.
- nullspace basis columns live in flattened control space.
- projection solves in residual space using ``J J.T`` for stable underdetermined
  control problems.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.numerics import tangent_projection, jacobian_decomposition
from optimizer.system_olgs import optional_residuals, validate_controls_for_system
from optimizer.utils.derivatives import get_jacobian


def jacobian_geometry(
    jacobian: np.ndarray,
    *,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Return rank and conditioning information for a residual Jacobian."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a 2D matrix.")
    singular_values, _right, rank, _threshold, condition = jacobian_decomposition(matrix, rcond=rcond)
    return {
        "jacobian_shape": list(matrix.shape),
        "jacobian_norm": float(np.linalg.norm(matrix)),
        "rank": rank,
        "row_dimension": int(matrix.shape[0]),
        "control_dimension": int(matrix.shape[1]),
        "nullspace_dimension": int(matrix.shape[1] - rank),
        "singular_values": singular_values.tolist(),
        "condition_number": condition,
        "rcond": None if rcond is None else float(rcond),
    }


def nullspace_basis(
    system: Any,
    controls: Controls,
    *,
    residuals: str = "hard",
    rcond: float | None = None,
    fallback: bool = True,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Return a basis whose columns span the local Jacobian nullspace."""

    validate_controls_for_system(system, controls)
    jacobian, _source = get_jacobian(
        system,
        controls,
        residuals=residuals,
        fallback=fallback,
        eps=eps,
    )
    _singular, vh, rank, _threshold, _condition = jacobian_decomposition(jacobian, rcond=rcond, full=True)
    return vh[rank:, :].T.copy()


def project_gradient(
    system: Any,
    controls: Controls,
    gradient: Controls,
    *,
    residuals: str = "hard",
    damping: float = 0.0,
    fallback: bool = True,
    eps: float = 1.0e-6,
    return_info: bool = False,
) -> Controls | dict[str, Any]:
    """Project a gradient into the local nullspace of residual constraints.

    The returned direction is the original gradient with the Jacobian row-space
    component removed.  If a later optimizer steps along ``-projected_gradient``, the
    first-order change in hard residuals should be reduced.
    """

    validate_controls_for_system(system, controls)
    if gradient.spec.keys != controls.spec.keys or gradient.spec.control_dim != controls.spec.control_dim:
        raise ValueError("gradient controls must match controls layout.")
    damping = float(damping)
    if damping < 0.0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and nonnegative.")

    jacobian, source = get_jacobian(
        system,
        controls,
        residuals=residuals,
        fallback=fallback,
        eps=eps,
    )
    g = gradient.flatten(copy=False).astype(float, copy=False)
    projected_flat = tangent_projection(g, jacobian, damping=damping, scales=controls.scales)

    row_component = controls.scales ** 2 * g - projected_flat
    projected = Controls.from_flat(
        controls.spec,
        projected_flat,
        name=f"projected_{gradient.name or 'gradient'}",
    )

    if not return_info:
        return projected

    residual_before = optional_residuals(system, controls, name=residuals)
    return {
        "projected_gradient": projected,
        "jacobian_source": source,
        "residuals": residuals,
        "residual_norm": float(np.linalg.norm(residual_before)),
        "gradient_norm": float(np.linalg.norm(g)),
        "projected_gradient_norm": projected.norm(),
        "row_component_norm": float(np.linalg.norm(row_component)),
        "first_order_residual_change_norm": float(np.linalg.norm(jacobian @ projected_flat)),
        "geometry": jacobian_geometry(jacobian),
    }
