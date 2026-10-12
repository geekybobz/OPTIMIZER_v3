"""Array kernels shared by gradient constraints, shooting and continuation.

Coordinates are physical values x = scales * z. Residual tolerances retain
their system-owned units; scaling changes variable coordinates only.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def coordinate_scales(size: int, scales=None) -> np.ndarray:
    values = np.ones(size) if scales is None else np.asarray(scales, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("scales must contain one finite positive value per variable.")
    return values


def residual_step(jacobian, residuals, *, method="newton", damping=0.0,
                  scales=None, rcond=None, formulation="auto") -> np.ndarray:
    """Minimum-norm Newton or damped least-squares step in scaled coordinates.

    The residual-space formulation avoids an n-by-n identity for wide control
    grids. The augmented formulation avoids squaring the condition number for
    tall/square shooting systems. Callers may retain an existing formulation
    when preserving a numerical reference's arithmetic is necessary.
    """
    matrix = np.asarray(jacobian, dtype=float)
    residual = np.asarray(residuals, dtype=float)
    if matrix.ndim != 2 or residual.shape != (matrix.shape[0],):
        raise ValueError("Jacobian and residual shapes do not agree.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(residual)):
        raise ValueError("Jacobian and residuals must be finite.")
    if method not in {"newton", "lm", "damped"}:
        raise ValueError("method must be 'newton', 'lm', or 'damped'.")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative.")
    if formulation not in {"auto", "residual", "augmented"}:
        raise ValueError("Unknown residual-step formulation.")
    scale = coordinate_scales(matrix.shape[1], scales)
    scaled = matrix * scale
    if method == "newton":
        step = np.linalg.lstsq(scaled, -residual, rcond=rcond)[0]
    elif formulation == "residual" or (formulation == "auto" and matrix.shape[0] < matrix.shape[1]):
        gram = scaled @ scaled.T + damping * np.eye(matrix.shape[0])
        step = scaled.T @ np.linalg.lstsq(gram, -residual, rcond=rcond)[0]
    else:
        augmented = np.vstack((scaled, np.sqrt(damping) * np.eye(matrix.shape[1])))
        target = np.concatenate((-residual, np.zeros(matrix.shape[1])))
        step = np.linalg.lstsq(augmented, target, rcond=rcond)[0]
    return scale * step


def tangent_projection(gradient, jacobian, *, damping=0.0, scales=None, rcond=None):
    """Return the scaled-coordinate tangent gradient as a physical direction."""
    g = np.asarray(gradient, dtype=float).reshape(-1)
    matrix = np.asarray(jacobian, dtype=float)
    scale = coordinate_scales(g.size, scales)
    scaled = matrix * scale
    gradient_z = g * scale
    correction = residual_step(scaled, scaled @ gradient_z, method="lm",
                               damping=damping, rcond=rcond, formulation="residual")
    return scale * (gradient_z + correction)


def rank_threshold(singular_values, shape, *, rcond=None):
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite and nonnegative.")
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    return largest * (max(shape, default=0) * np.finfo(float).eps if rcond is None else float(rcond))


def jacobian_decomposition(jacobian, *, rcond=None, full=False):
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be a finite 2D matrix.")
    _left, singular_values, right = np.linalg.svd(matrix, full_matrices=full)
    threshold = rank_threshold(singular_values, matrix.shape, rcond=rcond)
    rank = int(np.sum(singular_values > threshold))
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values.size and singular_values[-1] > threshold else float("inf"))
    return singular_values, right, rank, threshold, condition


def difference_jacobian(function: Callable, values, *, eps=1e-6, method="central",
                        scales=None, relative=False):
    """Finite differences with one validated stencil for both system families."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")
    base = np.asarray(values, dtype=float).reshape(-1)
    scale = coordinate_scales(base.size, scales)
    if relative:
        scale = np.maximum(scale, np.abs(base))
    residual = np.asarray(function(base), dtype=float).reshape(-1)
    matrix = np.empty((residual.size, base.size))
    for column in range(base.size):
        step = float(eps) * scale[column]
        plus = base.copy()
        plus[column] += step
        r_plus = np.asarray(function(plus), dtype=float)
        if method == "central":
            minus = base.copy()
            minus[column] -= step
            r_minus = np.asarray(function(minus), dtype=float)
            matrix[:, column] = (r_plus - r_minus) / (2 * step)
        else:
            matrix[:, column] = (r_plus - residual) / step
    if not np.all(np.isfinite(matrix)):
        raise ValueError("finite-difference Jacobian produced non-finite values.")
    return matrix
