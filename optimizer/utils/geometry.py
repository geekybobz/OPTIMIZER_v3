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
- future projected optimizers can call ``project_gradient`` to remove hard-residual
  changing components from a descent direction.

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
from optimizer.system_olgs import optional_residuals, validate_controls_for_system
from optimizer.utils.derivatives import get_jacobian


def _rank_from_singular_values(singular_values: np.ndarray, rcond: float | None) -> int:
    """Return numerical rank using NumPy-style relative cutoff."""

    if singular_values.size == 0:
        return 0
    cutoff = (np.finfo(float).eps * max(singular_values.size, 1)) if rcond is None else float(rcond)
    threshold = float(np.max(singular_values)) * cutoff
    return int(np.sum(singular_values > threshold))


def jacobian_geometry(
    jacobian: np.ndarray,
    *,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Return rank and conditioning information for a residual Jacobian."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a 2D matrix.")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = _rank_from_singular_values(singular_values, rcond)
    nonzero = singular_values[singular_values > 0.0]
    condition = float(np.inf)
    if nonzero.size >= 2:
        condition = float(nonzero[0] / nonzero[-1])
    elif nonzero.size == 1:
        condition = 1.0
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
    _u, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)
    rank = _rank_from_singular_values(singular_values, rcond)
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
    gram = jacobian @ jacobian.T
    if damping > 0.0:
        gram = gram + damping * np.eye(gram.shape[0])

    # Solve in residual space because hard-control problems usually have many more
    # control variables than residual equations.  ``lstsq`` handles rank deficiency.
    rhs = jacobian @ g
    multiplier = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    row_component = jacobian.T @ multiplier
    projected_flat = g - row_component
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
