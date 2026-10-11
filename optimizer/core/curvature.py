"""Local curvature evidence shared by stopping and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.system_olgs import probe_system, validate_controls_for_system


def curvature_evidence(
    system: Any,
    controls: Controls,
    *,
    directions: int = 8,
    eps: float = 1.0e-5,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate extreme objective curvature in a deterministic local subspace."""

    validate_controls_for_system(system, controls)
    count = max(1, min(int(directions), controls.spec.size))
    evaluator = SystemEvaluator(system, use_cache=False)
    objective_direction = str(evaluator.system_spec["objective"]["direction"])
    rng = np.random.default_rng(int(seed))
    dimension = controls.spec.size
    if count == dimension:
        basis = np.eye(dimension)
    else:
        raw = rng.normal(size=(dimension, count))
        basis, _ = np.linalg.qr(raw)
        basis = basis[:, :count].T

    has_hvp = probe_system(system).has_hvp
    columns: list[np.ndarray] = []
    source = "hvp" if has_hvp else "finite_difference_gradient"
    center = controls.flatten(copy=False)
    for vector in basis:
        vector_controls = Controls.from_flat(controls.spec, vector, name="curvature_direction")
        if has_hvp:
            product = evaluator.hvp(controls, vector_controls).flatten(copy=False)
        else:
            plus = Controls.from_flat(controls.spec, center + float(eps) * vector)
            minus = Controls.from_flat(controls.spec, center - float(eps) * vector)
            plus_gradient = evaluator.gradient(plus).flatten(copy=False)
            minus_gradient = evaluator.gradient(minus).flatten(copy=False)
            product = (plus_gradient - minus_gradient) / (2.0 * float(eps))
        if objective_direction == "maximize":
            product = -np.asarray(product, dtype=float)
        columns.append(np.asarray(product, dtype=float))

    reduced = basis @ np.column_stack(columns)
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    minimum_index = int(np.argmin(eigenvalues))
    weak_direction = basis.T @ eigenvectors[:, minimum_index]
    gradient = evaluator.gradient(controls).flatten(copy=False).astype(float, copy=False)
    if objective_direction == "maximize":
        gradient = -gradient
    gradient_norm = float(np.linalg.norm(gradient))
    gradient_curvature = None
    suggested_step = None
    if gradient_norm > 0.0:
        unit = gradient / gradient_norm
        if has_hvp:
            unit_controls = Controls.from_flat(controls.spec, unit, name="gradient_direction")
            product = evaluator.hvp(controls, unit_controls).flatten(copy=False)
            if objective_direction == "maximize":
                product = -product
        else:
            plus = Controls.from_flat(controls.spec, center + float(eps) * unit)
            minus = Controls.from_flat(controls.spec, center - float(eps) * unit)
            product = (
                evaluator.gradient(plus).flatten(copy=False)
                - evaluator.gradient(minus).flatten(copy=False)
            ) / (2.0 * float(eps))
            if objective_direction == "maximize":
                product = -product
        gradient_curvature = float(unit @ product)
        if gradient_curvature > np.finfo(float).eps:
            suggested_step = gradient_norm / gradient_curvature

    return {
        "kind": "curvature_probe",
        "source": source,
        "directions": count,
        "dimension": dimension,
        "minimum_curvature": float(eigenvalues[0]),
        "maximum_curvature": float(eigenvalues[-1]),
        "negative_curvature": bool(eigenvalues[0] < 0.0),
        "weak_direction": weak_direction.tolist(),
        "gradient_norm": gradient_norm,
        "gradient_curvature": gradient_curvature,
        "suggested_step": suggested_step,
    }
