"""Explicit cross-family replay of OLSS-recovered controls in an OLGS system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from optimizer.controls import Controls
from optimizer.core.evaluate import SystemEvaluator
from optimizer.result import controls_to_dict
from optimizer.shooting.evaluator import ShootingEvaluator
from optimizer.system_olss import ShootingVariables


def compare_shooting_replay(
    shooting_system: Any,
    variables: ShootingVariables,
    gradient_system: Any,
    *,
    times: np.ndarray,
    metric_map: Mapping[str, str] | None = None,
    shooting_options: Mapping[str, Any] | None = None,
    interpolation: str = "system_owned",
    units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recover controls from OLSS, evaluate OLGS, and report declared metric differences."""

    hook = getattr(shooting_system, "recover_controls", None)
    if not callable(hook):
        raise TypeError("OLSS replay requires system.recover_controls(variables, *, times).")
    grid = np.asarray(times, dtype=float).reshape(-1)
    if grid.size < 1 or not np.all(np.isfinite(grid)):
        raise ValueError("times must be a non-empty finite one-dimensional grid.")
    recovered = hook(variables, times=grid)
    recovery_metadata: dict[str, Any] = {}
    if isinstance(recovered, Controls):
        controls = recovered
    elif isinstance(recovered, Mapping):
        if "values" not in recovered:
            raise TypeError("recover_controls() mappings must include 'values'.")
        spec_hook = getattr(gradient_system, "control_spec", None)
        if not callable(spec_hook):
            raise TypeError("Mapping recovery requires gradient_system.control_spec().")
        spec = spec_hook()
        values = np.asarray(recovered["values"], dtype=spec.dtype)
        if values.shape == (spec.control_dim, spec.n_controls):
            values = values.T
        controls = Controls.from_matrix(spec, values, name="shooting_replay")
        declared_times = recovered.get("times")
        if declared_times is not None and not np.allclose(np.asarray(declared_times), grid):
            raise ValueError("recover_controls() returned a grid different from the requested times.")
        recovery_metadata = {
            str(key): value
            for key, value in recovered.items()
            if key not in {"values", "times"}
        }
    else:
        raise TypeError("recover_controls() must return Controls or a mapping with 'values'.")
    shooting_evaluation = ShootingEvaluator(shooting_system).evaluate(
        variables,
        options=shooting_options,
    )
    gradient_evaluation = SystemEvaluator(gradient_system).evaluate(controls)
    comparisons = []
    for shooting_metric, gradient_metric in dict(metric_map or {}).items():
        left = float(shooting_evaluation.metrics[shooting_metric])
        right = float(gradient_evaluation.metrics[gradient_metric])
        comparisons.append(
            {
                "shooting_metric": str(shooting_metric),
                "gradient_metric": str(gradient_metric),
                "shooting_value": left,
                "gradient_value": right,
                "difference": right - left,
                "absolute_difference": abs(right - left),
            }
        )
    return {
        "kind": "shooting_replay_comparison",
        "controls": controls_to_dict(controls),
        "recovery_metadata": recovery_metadata,
        "grid": {
            "count": int(grid.size),
            "start": float(grid[0]),
            "stop": float(grid[-1]),
            "uniform": bool(grid.size < 3 or np.allclose(np.diff(grid), np.diff(grid)[0])),
        },
        "interpolation": str(interpolation),
        "units": dict(units or {}),
        "shooting_metrics": dict(shooting_evaluation.metrics),
        "gradient_metrics": dict(gradient_evaluation.metrics),
        "comparisons": comparisons,
    }
